import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from datathon_ml.semantic_vlm import (LocalStateObserver, SemanticVLMConfig, causal_windows,
    extract_semantic_states, parse_codes, perception_protocol)
from datathon_ml.model import semantic_feature_config
from datathon_ml.features import load_feature_cache, sha256_file


class SemanticVLMTests(unittest.TestCase):
    def test_only_exact_codes_are_valid_and_invalid_is_unknown(self):
        values, valid = parse_codes(json.dumps([0,1,2]*6))
        self.assertTrue(valid)
        np.testing.assert_allclose(values[:3], [.05,.95,.5])
        for text in ('[1]', json.dumps([True]*18), json.dumps([3]*18), 'explanation '+json.dumps([1]*18)):
            values, valid = parse_codes(text)
            self.assertFalse(valid)
            np.testing.assert_array_equal(values, np.full(18,.5))

    def test_windows_cover_endpoints_without_annotation_input(self):
        times=np.arange(0,11,.5)
        centers, indices=causal_windows(times,SemanticVLMConfig())
        self.assertEqual(centers[0],0)
        self.assertEqual(centers[-1],times[-1])
        self.assertTrue(np.all(np.diff(centers)>0))
        self.assertEqual(indices.shape,(len(centers),4))

    def test_extraction_ignores_label_and_anchors_and_reuses_cache(self):
        class Observer:
            provenance={"device":"test"}
            calls=0
            def observe(self,images):
                self.calls+=1
                return np.full(18,.5,dtype=np.float32),False
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            checkpoint=root/'checkpoint'
            checkpoint.mkdir()
            (checkpoint/'config.json').write_text('{}')
            (checkpoint/'model.safetensors').write_bytes(b'weights')
            source=root/'input.mov'
            source.write_bytes(b'video')
            observer=Observer()
            frames=[(float(i),np.zeros((64,32,3),np.uint8)) for i in range(7)]
            config=SemanticVLMConfig(image_size=64,device='cpu',allow_download=False)
            record={"path":str(source),"sha256":sha256_file(source),"label":1,"timestamps_sec":[12345]}
            with patch('datathon_ml.semantic_vlm.iter_video_frames',return_value=iter(frames)):
                result=extract_semantic_states([record],root/'cache',config,observer=observer,checkpoint=checkpoint)
            first=load_feature_cache(next(iter(result.values())))
            calls=observer.calls
            record.update(label=0,timestamps_sec=[])
            with patch('datathon_ml.semantic_vlm.iter_video_frames',side_effect=AssertionError('cache not reused')):
                result=extract_semantic_states([record],root/'cache',config,observer=observer,checkpoint=checkpoint)
            self.assertEqual(observer.calls,calls)
            self.assertEqual(first['features'].shape,(4,1,18))
            self.assertEqual(first['config']['precision'],'fp32')
            self.assertFalse(first['config']['continuous_window_coverage'])

    def test_auto_cpu_precision_protocol_changes_and_invalid_observer_evidence(self):
        class Observer:
            calls=0
            def observe(self,images):
                self.calls+=1
                # Invalid responses must stay unknown regardless of their values.
                return np.full(18,.95,dtype=np.float32),False
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            checkpoint=root/'checkpoint'
            checkpoint.mkdir()
            (checkpoint/'config.json').write_text('{}')
            (checkpoint/'model.safetensors').write_bytes(b'weights')
            source=root/'input.mov'
            source.write_bytes(b'video')
            record={'path':str(source),'sha256':sha256_file(source),'event_times':[9999], 'label':1}
            observer=Observer()
            config=SemanticVLMConfig(image_size=64,device='auto',allow_download=False)
            frames=[(float(i),np.zeros((64,32,3),np.uint8)) for i in range(7)]
            with patch('torch.cuda.is_available',return_value=False), patch('datathon_ml.semantic_vlm.iter_video_frames',side_effect=lambda *args:iter(frames)):
                paths=extract_semantic_states([record],root/'cache',config,observer=observer,checkpoint=checkpoint)
                first=load_feature_cache(next(iter(paths.values())))
                self.assertEqual(first['config']['precision'],'fp32')
                self.assertEqual(first['config']['device'],'cpu')
                np.testing.assert_array_equal(first['features'],np.full((4,1,18),.5,dtype=np.float32))
                before=observer.calls
                changed=SemanticVLMConfig(image_size=64,device='auto',allow_download=False,clip_seconds=6.)
                extract_semantic_states([record],root/'cache',changed,observer=observer,checkpoint=checkpoint)
                second=load_feature_cache(next(iter(paths.values())))
            self.assertGreater(observer.calls,before)
            self.assertNotEqual(semantic_feature_config(first['config']),semantic_feature_config(second['config']))
            self.assertEqual(first['provenance']['schema_valid_windows'],0)
            self.assertNotIn('label',first['config'])
            self.assertNotIn('event_times',first['config'])

    def test_generation_and_frame_count_are_part_of_the_fixed_protocol(self):
        first=perception_protocol(SemanticVLMConfig())
        self.assertNotEqual(first,perception_protocol(SemanticVLMConfig(frames_per_window=6)))
        self.assertNotEqual(first,perception_protocol(SemanticVLMConfig(max_new_tokens=120)))

    def test_batch_size_is_numeric_provenance_not_observation_protocol(self):
        self.assertEqual(SemanticVLMConfig().batch_size,1)
        self.assertEqual(perception_protocol(SemanticVLMConfig(batch_size=1)),
                         perception_protocol(SemanticVLMConfig(batch_size=2)))
        for invalid in (0,-1,1.5,True):
            with self.assertRaises(ValueError):
                SemanticVLMConfig(batch_size=invalid)

    def test_native_batch_left_padding_flat_images_and_individual_invalid_rows(self):
        import torch
        class Tokenizer:
            padding_side='right'
            def decode(self,tokens,skip_special_tokens):
                self.assert_suffix=tokens.tolist()
                return json.dumps([1]*18) if tokens.tolist()==[91] else '[1]'
        class Processor:
            tokenizer=Tokenizer()
            messages=[]
            def apply_chat_template(self,messages,**kwargs):
                self.messages.append(messages)
                return 'text'+str(len(self.messages))
            def __call__(self,**kwargs):
                self.arguments=kwargs
                return {'input_ids':torch.tensor([[0,11,12],[21,22,23]]),
                        'attention_mask':torch.tensor([[0,1,1],[1,1,1]])}
        class Model:
            def generate(self,**kwargs):
                self.arguments=kwargs
                return torch.tensor([[0,11,12,91],[21,22,23,92]])
        observer=LocalStateObserver.__new__(LocalStateObserver)
        observer.processor=Processor()
        observer.model=Model()
        observer.config=SemanticVLMConfig(image_size=64)
        observer.device=torch.device('cpu')
        images=[object() for _ in range(4)]
        observed=observer.observe_batch([images[:2],images[2:]])
        self.assertEqual(observer.processor.tokenizer.padding_side,'left')
        self.assertEqual(observer.processor.arguments['images'],images)
        self.assertEqual(observer.processor.arguments['text'],['text1','text2'])
        self.assertTrue(observer.processor.arguments['padding'])
        self.assertEqual([x['image'] for x in observer.processor.messages[1][1]['content'][:-1]],images[2:])
        self.assertTrue(observed[0][1])
        np.testing.assert_allclose(observed[0][0],np.full(18,.95))
        self.assertFalse(observed[1][1])
        np.testing.assert_array_equal(observed[1][0],np.full(18,.5))
        self.assertFalse(observer.model.arguments['do_sample'])

    def test_cuda_oom_retries_only_current_batch_and_preserves_window_order(self):
        import torch
        class Observer:
            calls=[]
            def observe_batch(self,clips):
                self.calls.append(len(clips))
                if len(self.calls)==2:
                    raise torch.cuda.OutOfMemoryError('synthetic CUDA OOM')
                return [(np.full(18,np.asarray(images[2])[0,0,0]/10,dtype=np.float32),True)
                        for images in clips]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            checkpoint=root/'checkpoint'
            checkpoint.mkdir()
            (checkpoint/'config.json').write_text('{}')
            (checkpoint/'model.safetensors').write_bytes(b'weights')
            source=root/'input.mov'
            source.write_bytes(b'video')
            config=SemanticVLMConfig(image_size=64,batch_size=2,device='cuda',allow_download=False)
            frames=[(float(i),np.full((64,32,3),i,np.uint8)) for i in range(9)]
            observer=Observer()
            with patch('datathon_ml.semantic_vlm._runtime_device',return_value=torch.device('cuda')), \
                 patch('datathon_ml.semantic_vlm.iter_video_frames',return_value=iter(frames)), \
                 patch('torch.cuda.empty_cache') as clear:
                paths=extract_semantic_states([{'path':str(source)}],root/'cache',config,
                                             observer=observer,checkpoint=checkpoint)
            cached=load_feature_cache(next(iter(paths.values())))
            centers,indices=causal_windows(np.arange(9),config)
            np.testing.assert_array_equal(cached['timestamps'],centers)
            np.testing.assert_allclose(cached['features'][:,0,0],indices[:,2]/10)
            self.assertEqual(observer.calls,[2,2,1,1,1])
            self.assertEqual(clear.call_count,1)
            self.assertEqual(cached['provenance']['batch_size_requested'],2)
            self.assertEqual(cached['provenance']['batch_size_effective'],1)
            self.assertEqual(cached['provenance']['successful_batch_sizes'],[1,2])
            self.assertTrue(cached['provenance']['mixed_generation_batch_sizes'])

    def test_legacy_sequential_cache_reuses_values_with_original_batch_provenance(self):
        class Observer:
            def observe(self,images):
                return np.full(18,.95,dtype=np.float32),True
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            checkpoint=root/'checkpoint'
            checkpoint.mkdir()
            (checkpoint/'config.json').write_text('{}')
            (checkpoint/'model.safetensors').write_bytes(b'weights')
            source=root/'input.mov'
            source.write_bytes(b'video')
            frames=[(float(i),np.zeros((64,32,3),np.uint8)) for i in range(5)]
            with patch('datathon_ml.semantic_vlm.iter_video_frames',return_value=iter(frames)):
                paths=extract_semantic_states([{'path':str(source)}],root/'cache',
                    SemanticVLMConfig(image_size=64,device='cpu'),observer=Observer(),checkpoint=checkpoint)
            destination=next(iter(paths.values()))
            original=load_feature_cache(destination)
            # Simulate an NPZ produced by the old sequential implementation.
            with np.load(destination,allow_pickle=False) as archive:
                payload={key:archive[key] for key in archive.files}
            old_provenance={key:value for key,value in original['provenance'].items()
                            if 'batch' not in key}
            payload['provenance_json']=np.array(json.dumps(old_provenance))
            np.savez_compressed(destination,**payload)
            with patch('datathon_ml.semantic_vlm.iter_video_frames',side_effect=AssertionError('cache not reused')):
                extract_semantic_states([{'path':str(source)}],root/'cache',
                    SemanticVLMConfig(image_size=64,device='cpu',batch_size=2),checkpoint=checkpoint)
            reused=load_feature_cache(destination)
            np.testing.assert_array_equal(reused['features'],original['features'])
            self.assertEqual(reused['config_json'],original['config_json'])
            self.assertEqual(reused['provenance']['batch_size_requested'],1)
            self.assertEqual(reused['provenance']['batch_size_effective'],1)
            self.assertEqual(reused['provenance']['last_reuse_batch_size_requested'],2)
            self.assertTrue(reused['provenance']['batch_size_reuse_mismatch'])
            self.assertIn('floating-point',reused['provenance']['numeric_batching_warning'])

    def test_non_cuda_memory_error_and_single_window_cuda_oom_propagate(self):
        import torch
        class Observer:
            def __init__(self,error):
                self.error=error
                self.calls=0
            def observe_batch(self,clips):
                self.calls+=1
                raise self.error
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            checkpoint=root/'checkpoint'
            checkpoint.mkdir()
            (checkpoint/'config.json').write_text('{}')
            (checkpoint/'model.safetensors').write_bytes(b'weights')
            source=root/'input.mov'
            source.write_bytes(b'video')
            for device,size,error in [('cuda',2,RuntimeError('unrelated failure')),
                                      ('cpu',2,torch.cuda.OutOfMemoryError('not CUDA runtime')),
                                      ('cuda',1,torch.cuda.OutOfMemoryError('single window'))]:
                observer=Observer(error)
                frames=[(float(i),np.zeros((64,32,3),np.uint8)) for i in range(5)]
                with self.subTest(device=device,size=size,error=type(error).__name__), \
                     patch('datathon_ml.semantic_vlm._runtime_device',return_value=torch.device(device)), \
                     patch('datathon_ml.semantic_vlm.iter_video_frames',return_value=iter(frames)), \
                     patch('torch.cuda.empty_cache') as clear, \
                     self.assertRaises(type(error)):
                    extract_semantic_states([{'path':str(source)}],root/'cache',
                        SemanticVLMConfig(image_size=64,device=device,batch_size=size),
                        observer=observer,checkpoint=checkpoint)
                self.assertEqual(observer.calls,1)
                clear.assert_not_called()


if __name__=='__main__':
    unittest.main()
