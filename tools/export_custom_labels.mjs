/** Export simple, explicitly unverified custom-video label proposals with Artifact Tool. */
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
import assert from 'node:assert/strict';

function readArgs(argv) {
  const result = {};
  for (let index = 0; index < argv.length; index++) {
    const key = argv[index];
    if (['--self-test', '--validate-only', '--overwrite'].includes(key)) {
      result[key.slice(2)] = true;
    } else if (['--manifest', '--proposals', '--output', '--preview', '--runtime-dir', '--node-modules'].includes(key)) {
      if (!argv[index + 1] || argv[index + 1].startsWith('--')) throw new Error(`Missing value for ${key}`);
      result[key.slice(2)] = argv[++index];
    } else {
      throw new Error(`Unknown argument ${key}`);
    }
  }
  return result;
}

const recordsOf = value => Array.isArray(value) ? value : value.records;
const literal = value => value.startsWith('=') ? `'${value}` : value;

export function buildRows(manifest, proposals = { records: [] }) {
  const recordings = recordsOf(manifest);
  const entries = recordsOf(proposals);
  if (!Array.isArray(recordings) || !recordings.length || !Array.isArray(entries)) {
    throw new Error('Manifest must contain recordings and proposals must contain a records list');
  }
  const byHash = new Map();
  const byFilename = new Map();
  for (const record of recordings) {
    const filename = record.source_filename;
    if (typeof filename !== 'string' || !filename || /[/\\]/.test(filename)) {
      throw new Error('Every custom recording needs its original source_filename');
    }
    if (byFilename.has(filename) || byHash.has(record.source_sha256)) {
      throw new Error('Duplicate source filename or hash in manifest');
    }
    if (typeof record.source_sha256 !== 'string' || !/^[0-9a-f]{64}$/.test(record.source_sha256)) {
      throw new Error('Every custom recording needs a source SHA-256');
    }
    byHash.set(record.source_sha256, record);
    byFilename.set(filename, record);
  }
  const matched = new Map();
  for (const proposal of entries) {
    const recording = proposal.source_sha256 !== undefined
      ? byHash.get(proposal.source_sha256) : byFilename.get(proposal.source_filename);
    if (!recording || (proposal.source_filename !== undefined && proposal.source_filename !== recording.source_filename)) {
      throw new Error('Proposal does not match a custom recording; copied original labels cannot attach');
    }
    if (matched.has(recording.source_sha256)) throw new Error('Duplicate proposal for a source recording');
    const label = proposal.label ?? null;
    if (![null, 'violation', 'no violation'].includes(label)) throw new Error('Invalid binary proposal label');
    const points = proposal.event_times ?? [];
    const duration = Math.min(recording.duration_sec, recording.source_duration_sec ?? recording.duration_sec);
    if (!Number.isFinite(duration) || duration <= 0 || !Array.isArray(points)
        || points.some(point => typeof point !== 'number' || !Number.isFinite(point) || point < 0 || point >= duration)) {
      throw new Error('Invalid proposal timestamps or recording duration');
    }
    if ((label === 'violation' && !points.length) || (label !== 'violation' && points.length)) {
      throw new Error('Violation requires timestamps; no violation and unknown require none');
    }
    matched.set(recording.source_sha256, { label, points: [...new Set(points)].sort((a, b) => a - b) });
  }
  const rows = [];
  const summary = { recordings: recordings.length, violation: 0, no_violation: 0, unresolved: 0 };
  for (const recording of recordings) {
    const proposal = matched.get(recording.source_sha256);
    const filename = literal(recording.source_filename);
    if (proposal?.label === 'violation') {
      summary.violation++;
      for (const point of proposal.points) rows.push([filename, point, 'violation']);
    } else if (proposal?.label === 'no violation') {
      summary.no_violation++;
      rows.push([filename, null, 'no violation']);
    } else {
      summary.unresolved++;
      rows.push([filename, null, null]);
    }
  }
  return { rows, summary };
}

function selfTest() {
  const manifest = [{ source_filename: 'one.MOV', source_sha256: 'a'.repeat(64), duration_sec: 10 },
                    { source_filename: 'two.mp4', source_sha256: 'b'.repeat(64), duration_sec: 10 },
                    { source_filename: 'three.MOV', source_sha256: 'c'.repeat(64), duration_sec: 1.2 }];
  const proposals = { records: [{ source_sha256: 'a'.repeat(64), label: 'violation', event_times: [5, 0, 5] },
                                { source_filename: 'two.mp4', label: 'no violation', event_times: [] }] };
  const result = buildRows(manifest, proposals);
  assert.deepEqual(result.rows, [['one.MOV', 0, 'violation'], ['one.MOV', 5, 'violation'],
                               ['two.mp4', null, 'no violation'], ['three.MOV', null, null]]);
  assert.deepEqual(result.summary, { recordings: 3, violation: 1, no_violation: 1, unresolved: 1 });
  for (const row of [{ source_filename: 'X-1.mov', label: 'violation', event_times: [2] },
                     { source_filename: 'one.MOV', label: 'violation', event_times: [] },
                     { source_filename: 'one.MOV', label: 'violation', event_times: [10] },
                     { source_filename: 'one.MOV', label: 'no violation', event_times: [2] },
                     { source_filename: 'one.MOV', label: null, event_times: [2] },
                     { source_sha256: 'a'.repeat(64), source_filename: 'two.mp4', label: null }]) {
    assert.throws(() => buildRows(manifest, { records: [row] }));
  }
  assert.equal(buildRows(manifest).summary.unresolved, 3);
  console.log(JSON.stringify({ self_test: 'passed', cases: 9 }));
}

async function artifactTool(args) {
  const dependencyModules = path.resolve(args['node-modules'] ?? path.join(os.homedir(),
    '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules'));
  const runtimeDirectory = path.resolve(args['runtime-dir'] ?? 'artifacts/custom_spreadsheet_runtime');
  await fs.access(path.join(dependencyModules, '@oai/artifact-tool'));
  await fs.mkdir(runtimeDirectory, { recursive: true });
  const junction = path.join(runtimeDirectory, 'node_modules');
  try {
    const existing = await fs.realpath(junction);
    if (existing.toLowerCase() !== (await fs.realpath(dependencyModules)).toLowerCase()) {
      throw new Error('Runtime node_modules already points to a different dependency directory');
    }
  } catch (error) {
    if (error.code !== 'ENOENT') throw error;
    await fs.symlink(dependencyModules, junction, process.platform === 'win32' ? 'junction' : 'dir');
  }
  const require = createRequire(path.join(runtimeDirectory, 'package.json'));
  return import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
}

async function main() {
  const args = readArgs(process.argv.slice(2));
  if (args['self-test']) return selfTest();
  if (!args.manifest) throw new Error('Pass --manifest <custom manifest.json>');
  const manifest = JSON.parse(await fs.readFile(args.manifest, 'utf8'));
  const proposals = args.proposals ? JSON.parse(await fs.readFile(args.proposals, 'utf8')) : undefined;
  const { rows, summary } = buildRows(manifest, proposals);
  if (args['validate-only']) {
    console.log(JSON.stringify({ validated: true, ...summary, output_rows: rows.length }));
    return;
  }
  if (!args.output || !args.proposals) {
    throw new Error('Final export requires --proposals and --output; use --validate-only for pending proposals');
  }
  if (path.resolve(args.output) === path.resolve('open/customVideos/open_labels.xlsx')) {
    throw new Error('Do not overwrite the supplied annotation workbook');
  }
  if (!args.overwrite) {
    try {
      await fs.access(args.output);
      throw new Error('Output already exists; use --overwrite only for this generated proposal workbook');
    } catch (error) {
      if (error.code !== 'ENOENT') throw error;
    }
  }
  const { Workbook, SpreadsheetFile } = await artifactTool(args);
  const workbook = Workbook.create();
  const sheet = workbook.worksheets.add('Labels');
  const endRow = rows.length + 5;
  sheet.showGridLines = false;
  sheet.tabColor = '#24364B';
  sheet.getRange(`A1:C${endRow}`).format.font = { name: 'Arial', size: 10, color: '#18232F' };
  sheet.getRange(`A1:C${endRow}`).format.verticalAlignment = 'center';
  sheet.getRange(`A1:C${endRow}`).format.rowHeight = 21;
  sheet.getRange(`A1:A${endRow}`).format.columnWidthPx = 425;
  sheet.getRange(`B1:B${endRow}`).format.columnWidthPx = 145;
  sheet.getRange(`C1:C${endRow}`).format.columnWidthPx = 150;
  sheet.getRange('A1:C1').format.rowHeight = 10;
  sheet.getRange('A2').values = [['Unverified visual proposals']];
  sheet.getRange('A2').format.font = { name: 'Arial', size: 14, bold: true, color: '#18232F' };
  sheet.getRange('A2:C2').format.rowHeight = 27;
  sheet.getRange('A2:C2').format.borders = { bottom: { style: 'thin', color: '#AAB3BE' } };
  sheet.getRange('A3').values = [['Local visual proposals. Timestamps are seconds. Human review required. Blank labels are unresolved.']];
  sheet.getRange('A3').format.font = { name: 'Arial', size: 10, italic: true, color: '#526171' };
  sheet.getRange('A4:C4').format.rowHeight = 9;
  sheet.getRange('A5:C5').values = [['filename', 'timestamp', 'label']];
  sheet.getRange('A5:C5').format = {
    fill: '#24364B', font: { name: 'Arial', size: 10, bold: true, color: '#FFFFFF' },
    horizontalAlignment: 'center', verticalAlignment: 'center', rowHeight: 24,
    borders: { insideVertical: { style: 'thin', color: '#FFFFFF' } },
  };
  sheet.getRange(`A6:C${endRow}`).values = rows;
  sheet.getRange(`B6:B${endRow}`).setNumberFormat('0.0');
  sheet.getRange(`B6:B${endRow}`).format.horizontalAlignment = 'right';
  sheet.getRange(`C6:C${endRow}`).dataValidation = {
    rule: { type: 'list', values: ['violation', 'no violation'] },
  };
  sheet.freezePanes.freezeRows(5);
  workbook.recalculate();
  const inspected = await workbook.inspect({ kind: 'table', range: `Labels!A5:C${Math.min(endRow, 12)}`,
    include: 'values,formulas', tableMaxRows: 8, tableMaxCols: 3, maxChars: 2500 });
  const errors = await workbook.inspect({ kind: 'match',
    searchTerm: '#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',
    options: { useRegex: true, maxResults: 20 }, maxChars: 1200 });
  const actualRows = sheet.getRange(`A6:C${endRow}`).values;
  assert.equal(actualRows.length, rows.length);
  assert.deepEqual(actualRows, rows);
  const preview = await workbook.render({ sheetName: 'Labels', range: `A1:C${endRow}`, scale: 1, format: 'png' });
  const previewPath = args.preview ?? path.join(path.dirname(args.output), 'custom_labels_preview.png');
  await fs.mkdir(path.dirname(previewPath), { recursive: true });
  await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));
  await fs.mkdir(path.dirname(args.output), { recursive: true });
  const xlsx = await SpreadsheetFile.exportXlsx(workbook);
  await xlsx.save(args.output);
  console.log(JSON.stringify({ exported: path.resolve(args.output), preview: path.resolve(previewPath),
    ...summary, output_rows: rows.length, inspected: inspected.ndjson, error_scan: errors.ndjson }));
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  await main();
}
