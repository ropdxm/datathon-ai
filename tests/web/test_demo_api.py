"""The source-data demo is private local media, never a model/jury job."""
import base64
from fractions import Fraction
import io
import json

import av
from fastapi.testclient import TestClient
from PIL import Image

from webapp.api import create_app
from webapp.config import Settings


def prepare_fixture(data_dir):
    directory = data_dir / 'demo'
    directory.mkdir(parents=True)
    jpeg = io.BytesIO()
    Image.new('RGB', (8, 8), 'blue').save(jpeg, format='JPEG')
    with av.open(str(directory / 'video.mp4'), 'w') as output:
        stream = output.add_stream('mpeg4', rate=5)
        stream.width, stream.height, stream.pix_fmt = 32, 32, 'yuv420p'
        stream.time_base = stream.codec_context.time_base = Fraction(1, 5)
        for index in range(5):
            frame = av.VideoFrame.from_image(Image.new('RGB', (32, 32), 'blue'))
            frame.pts, frame.time_base = index, Fraction(1, 5)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    result = {
        'job_id': 'dataset-demo', 'status': 'done', 'demo': True,
        'demo_source': 'provided_video', 'filename': 'Х-26.mov',
        'video_url': '/api/ui/demo/video', 'verdict': 'violation',
        'duration_sec': 1.0, 'elapsed_sec': 0,
        'model_name': 'Исходная разметка',
        'limitations': ['Модель не запускалась.'],
        'violations': [{
            'type': 'Фрагмент из разметки', 'timecode_sec': 0.2,
            'description': 'Отметка из предоставленных данных.',
            'evidence_frames': [{'timestamp_sec': 0.2,
                'image_base64': base64.b64encode(jpeg.getvalue()).decode()}],
        }],
    }
    (directory / 'result.json').write_text(json.dumps(result), encoding='utf8')
    return directory


def test_demo_requires_ui_session_and_is_separate_from_inference(tmp_path):
    directory = prepare_fixture(tmp_path)
    app = create_app(Settings(data_dir=tmp_path, api_key='test-only'))
    with TestClient(app) as client:
        assert client.get('/api/ui/demo').status_code == 401
        assert client.get('/api/ui/demo/video').status_code == 401
        config = client.get('/api/ui/config').json()
        assert config['ready'] is False
        assert config['demo_available'] is True
        response = client.get('/api/ui/demo')
        assert response.status_code == 200
        data = response.json()
        assert data['demo'] is True and data['demo_source'] == 'provided_video'
        assert data['video_url'] == '/api/ui/demo/video'
        assert data['violations'][0]['timecode_sec'] == 0.2
        assert response.headers['cache-control'] == 'no-store'
        assert client.get('/api/ui/demo', headers={'Origin':'https://unrelated.example'}).status_code == 403
        media = client.get('/api/ui/demo/video')
        assert media.status_code == 200
        assert media.headers['content-type'] == 'video/mp4'
        segment = client.get('/api/ui/demo/video', headers={'Range':'bytes=4-15'})
        assert segment.status_code == 206
        assert segment.content == media.content[4:16]
        assert client.get('/static/demo.mp4').status_code == 404
        with app.state.jobs._db() as database:
            assert database.execute('SELECT COUNT(*) FROM jobs').fetchone()[0] == 0


def test_absent_demo_reports_not_available_in_config(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        assert client.get('/api/ui/config').json()['demo_available'] is False
        assert client.get('/api/ui/demo').status_code == 404
        assert client.get('/api/ui/demo/video').status_code == 404


def test_invalid_demo_result_is_rejected_without_publishing_it(tmp_path):
    directory = prepare_fixture(tmp_path)
    (directory / 'result.json').write_text('{broken-json')
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        client.get('/api/ui/config')
        assert client.get('/api/ui/demo').status_code == 503
