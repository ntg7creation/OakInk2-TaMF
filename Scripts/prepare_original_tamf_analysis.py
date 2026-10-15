"""Prepare portable analysis of EXISTING TaMF evaluation outputs.

Run in the WSL TaMF environment. This script never imports or calls generation,
refinement, the four evaluators, MANO, or the FID encoder. It recovers motion values
from saved CR/SIV arrays only after source, cache, metadata and provenance checks.
Default output: common/analysis/test_one__0399/{analysis.json,motions.csv,plots/}.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

METRICS = ('cr', 'siv', 'psklj', 'fid')
NAMES = {'cr': 'CR', 'siv': 'SIV', 'psklj': 'PSKL-J', 'fid': 'FID'}

def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

def fmt(value, digits=6):
    return f'{value:,.{digits}f}' if number(value) else 'Missing / unavailable'

def require(condition, message):
    if not condition:
        raise ValueError(message)

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def analyze(repo_root, batch_dir, display_timezone='Asia/Jerusalem'):
    """Read and validate local saved artifacts, without evaluating motions."""
    REPO_ROOT, BATCH_DIR = Path(repo_root).resolve(), Path(batch_dir).resolve()
    DISPLAY_TIMEZONE = display_timezone
    USE_OPTIONAL_ARRAYS = True
    limitations = []
    PATHS = {
        'json': BATCH_DIR / 'results/original_tamf_metrics.json',
        'csv': BATCH_DIR / 'results/original_tamf_metrics.csv',
        'summary': BATCH_DIR / 'results/summary.md',
        'preflight': BATCH_DIR / 'results/preflight.json',
        'manifest': BATCH_DIR / 'manifest.json',
        'logs': BATCH_DIR / 'logs',
        # The actual runner/config currently live in the sibling Scripts directory.
        'runner': REPO_ROOT.parent / 'Scripts/run_original_tamf_evaluation.py',
        'config': REPO_ROOT.parent / 'Scripts/configs/partial_all.json',
        'official': {m: REPO_ROOT / f'script/compute_score/compute_score_{m}.py'
                     for m in ('cr', 'siv', 'psklj', 'fid')},
        'sdf_source': REPO_ROOT / 'src/dev_fn/util/sdf_util.py',
        'cr_arrays': (REPO_ROOT / 'tmp/compute_score/contact_ratio/gt_contact_dist.npy',
                      REPO_ROOT / 'tmp/compute_score/contact_ratio/refined_contact_dist.npy'),
        'siv_arrays': (REPO_ROOT / 'tmp/compute_score/solid_intersection_volume/gt.npy',
                       REPO_ROOT / 'tmp/compute_score/solid_intersection_volume/refined.npy'),
    }
    def read_text(path):
        try:
            return path.read_text(encoding='utf-8')
        except (OSError, UnicodeError) as exc:
            limitations.append(f'{path.name}: {exc}')
            return ''

    def read_json(path):
        text = read_text(path)
        try:
            return json.loads(text) if text else {}
        except json.JSONDecodeError as exc:
            limitations.append(f'{path.name}: invalid JSON: {exc}')
            return {}

    results = read_json(PATHS['json'])
    coverage = results.get('coverage') or {}
    entries = results.get('metrics') or {}
    preflight = read_json(PATHS['preflight'])
    manifest = read_json(PATHS['manifest'])
    summary_text = read_text(PATHS['summary'])
    runner_text = read_text(PATHS['runner'])  # read source, never import/run it
    config = read_json(PATHS['config'])
    sources = {m: read_text(p) for m, p in PATHS['official'].items()}
    sdf_source = read_text(PATHS['sdf_source'])
    try:
        with PATHS['csv'].open(encoding='utf-8', newline='') as stream:
            csv_rows = list(csv.DictReader(stream))
    except OSError as exc:
        csv_rows = []
        limitations.append(f'CSV unavailable: {exc}')
    run_id = results.get('run_id')
    logs = {}
    for metric in METRICS:
        entry = entries.get(metric) or {}
        # Use the directory recorded for each metric, so stale mixed runs are detectable.
        saved_name = Path(entry.get('stdout_log') or '').parent.name
        log_dir = PATHS['logs'] / (saved_name or str(run_id))
        logs[metric] = {kind: read_text(log_dir / f'{metric}_{kind}.txt')
                        for kind in ('stdout', 'stderr')}

    checks = []
    def check(label, condition, detail=''):
        checks.append({'Check': label, 'Result': 'PASS' if condition else 'UNVERIFIED / MISMATCH',
                       'Evidence': detail})

    generated = coverage.get('generated')
    paired = coverage.get('eligible_paired_motions')
    cr_frames = coverage.get('expected_cr_frames')
    siv_frames = coverage.get('expected_siv_frames')
    saved_samples = manifest.get('samples') or []
    unique = {}
    for sample in saved_samples:
        key = tuple(sample.get('info') or [])
        if key:
            unique.setdefault(key, sample)  # same first-occurrence deduplication as official code
    lengths = [s.get('len') for s in unique.values()]
    valid_lengths = bool(lengths) and all(isinstance(n, int) and n > 0 for n in lengths)
    check('Generated segment count', generated is not None and generated == len(saved_samples),
          f'coverage={generated}; manifest records={len(saved_samples)}')
    check('Deduplication accounting', number(generated) and number(paired)
          and coverage.get('deduplicated_segments') == generated - paired,
          f'deduplicated={coverage.get("deduplicated_segments")}')
    check('Paired refined motions', paired is not None and
          paired == len(unique) == coverage.get('official_unique_motions')
          == coverage.get('valid_refined') and coverage.get('missing_refined') == 0,
          f'manifest unique keys={len(unique)}; saved paired={paired}')
    check('CR frame coverage', valid_lengths and sum(lengths) == cr_frames
          == (entries.get('cr') or {}).get('official_stdout_shape_first_dim'),
          f'manifest sum={sum(lengths) if valid_lengths else "Missing"}; saved={cr_frames}')
    check('SIV sampled frame coverage', valid_lengths and
          sum((n + 19) // 20 for n in lengths) == siv_frames
          and (entries.get('siv') or {}).get('official_frame_counts') == [siv_frames, siv_frames],
          f'manifest stride-20 count={sum((n + 19)//20 for n in lengths) if valid_lengths else "Missing"}; saved={siv_frames}')
    for m in METRICS:
        e = entries.get(m) or {}
        check(f'{NAMES[m]} paired count / batch identity', paired is not None and
              e.get('evaluated_motions') == paired == e.get('sample_count') and
              bool(results.get('partial_batch_identity')) and
              e.get('partial_batch_identity') == results.get('partial_batch_identity'),
              f'paired={e.get("evaluated_motions")}')
    # Verify the source statements inspected during authoring; never execute source.
    source_tokens = {
        'cr': ('gt_contact_dist < 0.005', 'refined_contact_dist < 0.005',
               'print(gt_contact_ratio, refined_contact_ratio)'),
        'siv': ('el_vol = np.prod(_sdf.tick_unit)', 'el_vol * (10**6)',
                'range(0, avai_len, 20)', 'print(np.mean(gt_siv_list))',
                'print(np.mean(refined_siv_list))'),
        'psklj': ('np.diff(dataset_res_list[offset], n=2, axis=0)',
                  'dataset_mean_ps * np.log(dataset_mean_ps / model_mean_ps)',
                  'model_mean_ps * np.log(model_mean_ps / dataset_mean_ps)',
                  'print(pskl_1, pskl_2)'),
        'fid': ('calculate_activation_statistics(dataset_activation)',
                'calculate_activation_statistics(model_activation)', 'print(fid_value)'),
    }
    source_ok = {m: all(t in sources[m] for t in tokens) for m, tokens in source_tokens.items()}
    for m in METRICS:
        check(f'{NAMES[m]} official definition', source_ok[m],
              str(PATHS['official'][m].relative_to(REPO_ROOT)))
    check('SIV voxel spacing', 'obj_mesh_extent_expanded / resolution' in sdf_source,
          'Source mesh spacing is cubed, then multiplied by 10⁶ in SIV evaluator.')

    FIELDS = {'cr': ('gt_contact_ratio', 'generated_contact_ratio'),
              'siv': ('gt_siv', 'generated_siv'),
              'psklj': ('gt_to_generated', 'generated_to_gt'), 'fid': ('fid',)}
    def score(metric, field):
        e = entries.get(metric) or {}
        value = e.get(field)
        if e.get('status') != 'success' or e.get('exit_code') != 0:
            return None
        return value if number(value) else None

    scores = {m: [score(m, f) for f in FIELDS[m]] for m in METRICS}
    units = {'cr': 'ratio (0–1)', 'siv': 'cm³', 'psklj': 'unitless divergence',
             'fid': 'encoder feature-space score'}
    interpretations = {'cr': 'Compare with GT; higher is not always better',
                       'siv': 'Lower = less interpenetration',
                       'psklj': 'Lower = more similar motion spectra',
                       'fid': 'Lower = more similar feature distributions'}
    rows = []
    for m in METRICS:
        e = entries.get(m) or {}
        vals = scores[m]
        count = cr_frames if m == 'cr' else siv_frames if m == 'siv' else paired
        count_type = {'cr': 'pooled frames per population', 'siv': 'stride-20 frame samples per population',
                      'psklj': 'paired motions per population', 'fid': 'encoder activation rows per population'}[m]
        result_value = (fmt(vals[1]) if m in ('cr', 'siv') else fmt(vals[0]) if m == 'fid'
                        else f'GT → Generated: {fmt(vals[0])}\nGenerated → GT: {fmt(vals[1])}')
        rows.append({'Metric': NAMES[m],
                     'GT baseline': fmt(vals[0]) if m in ('cr', 'siv') else 'Not a separate baseline',
                     'Refined / comparison result': result_value,
                     'Units': units[m], 'Interpretation': interpretations[m],
                     'Runtime (s)': fmt(e.get('runtime_seconds'), 2),
                     'Evaluated count / type': f'{count:,}\n{count_type}' if number(count)
                         and e.get('status') == 'success' else 'Missing / unknown for unsuccessful metric',
                     'Status / error': e.get('status', 'Missing') + '\n' + (e.get('error') or
                         ('Missing: ' + ', '.join(f for f, v in zip(FIELDS[m], vals) if v is None)
                          if any(v is None for v in vals) else ''))})


    # Cross-check raw log order against the named JSON fields.
    def numeric_lines(text, width):
        parsed = []
        for line in text.splitlines():
            try:
                tokens = [float(t) for t in line.split()]
                if len(tokens) == width and all(math.isfinite(v) for v in tokens):
                    parsed.append(tokens)
            except ValueError:
                pass
        return parsed

    log_score_ok = {}
    log_count_ok = {}
    for m in METRICS:
        text = logs[m]['stdout']
        nums = numeric_lines(text, 2 if m in ('cr', 'psklj') else 1)
        logged = (nums[-1] if m in ('cr', 'psklj') else
                  [x[0] for x in nums[-2:]] if m == 'siv' else
                  nums[-1]) if nums else []
        vals = scores[m]
        log_score_ok[m] = len(logged) == len(vals) and all(
            number(v) and math.isclose(v, w, rel_tol=1e-10, abs_tol=1e-12)
            for v, w in zip(vals, logged))
        shapes = [tuple(int(v) for v in re.findall(r'\d+', s))
                  for s in re.findall(r'\([\d, ]+\)', text)]
        expected = cr_frames if m == 'cr' else paired
        log_count_ok[m] = (len(shapes) >= 2 and all(s[0] == expected for s in shapes[:2])) if m != 'siv' else True
        check(f'{NAMES[m]} stdout scores / order', log_score_ok[m], f'raw output={logged}')
        if m != 'siv':
            check(f'{NAMES[m]} stdout population counts', log_count_ok[m], f'first shapes={shapes[:2]}')
        matches = [r for r in csv_rows if r.get('Metric') == m and r.get('Run ID') == run_id]
        csv_ok = len(matches) == 1 and matches[0].get('Status') == (entries.get(m) or {}).get('status')
        if csv_ok:
            for column, field in [('Score', 'score'), ('Secondary score', 'secondary_score'),
                                   ('Runtime (s)', 'runtime_seconds')]:
                expected_value = (entries.get(m) or {}).get(field)
                try:
                    csv_ok &= (not matches[0].get(column)) if expected_value is None else math.isclose(
                        float(matches[0][column]), expected_value, rel_tol=1e-10, abs_tol=1e-12)
                except (ValueError, TypeError, KeyError):
                    csv_ok = False
        check(f'{NAMES[m]} CSV agrees with JSON', csv_ok, 'Run ID, status, both score fields, runtime')
    check('Summary agrees with run', bool(run_id) and run_id in summary_text and
          str(results.get('tamf_git_commit')) in summary_text, 'Saved summary run ID and commit')
    arrays = {}
    array_audit = []
    for m in ('cr', 'siv'):
        e = entries.get(m) or {}
        reasons = []
        pair = None
        timestamps = []
        if not USE_OPTIONAL_ARRAYS:
            reasons.append('Disabled in configuration')
        elif np is None:
            reasons.append('NumPy is unavailable')
        else:
            try:
                paths = PATHS[f'{m}_arrays']
                pair = [np.load(p, allow_pickle=False) for p in paths]
                start = datetime.fromisoformat(e['start_time']).timestamp()
                end = datetime.fromisoformat(e['end_time']).timestamp()
                expected = cr_frames if m == 'cr' else siv_frames
                for p, a in zip(paths, pair):
                    modified = p.stat().st_mtime
                    timestamps.append(datetime.fromtimestamp(modified, ZoneInfo('UTC')).isoformat())
                    if not start <= modified <= end:
                        reasons.append(f'{p.name}: modification time outside this metric run')
                    if a.ndim != 1 or a.size != expected:
                        reasons.append(f'{p.name}: unexpected shape {a.shape}; expected ({expected},)')
                    if not np.isfinite(a).all() or (a < 0).any():
                        reasons.append(f'{p.name}: nonfinite or negative values')
                if not source_ok[m] or not log_score_ok[m] or not log_count_ok[m]:
                    reasons.append('Official definition / saved log verification unavailable or mismatched')
                if e.get('status') != 'success' or e.get('exit_code') != 0:
                    reasons.append('Metric did not succeed')
                if not results.get('partial_batch_identity') or e.get('partial_batch_identity') != results.get('partial_batch_identity'):
                    reasons.append('Missing or mismatched batch identity')
                if Path(e.get('stdout_log') or '').parent.name != run_id:
                    reasons.append('Metric log belongs to a different run')
                reproduced = [float(np.mean(a < 0.005)) if m == 'cr' else float(a.mean()) for a in pair]
                if not all(number(saved) and math.isclose(observed, saved, rel_tol=1e-10, abs_tol=1e-12)
                           for observed, saved in zip(reproduced, scores[m])):
                    reasons.append('Array aggregates do not reproduce saved GT/refined scores')
            except (OSError, ValueError, TypeError, KeyError) as exc:
                reasons.append(f'Cannot verify arrays: {exc}')
        if not reasons:
            arrays[m] = pair
        array_audit.append({'Metric': NAMES[m], 'Distribution plots': 'Accepted with metadata/log evidence' if not reasons else 'SKIPPED',
                            'Evidence / reason': '; '.join(reasons) if reasons else
                                'Both shapes and aggregates match; timestamps lie inside metric window; same batch/log run',
                            'Array modification times (UTC)': '; '.join(timestamps) or 'Unavailable'})
    REVIEWED_SOURCES = {'CR loop': ('script/compute_score/compute_score_cr.py', '50a2df33e5a288f9f403d6cd343fe097e3393980d5ddd46dc1e8db0a0d63aaa6'), 'SIV loop': ('script/compute_score/compute_score_siv.py', '5ac8377fa5e0db05b61f9e65bb7a419106997469254dfde3c28374e68a950c2c'), 'Dataset cache order': ('src/oakink2_tamf/dataset/interaction_segment.py', '12becd2c312b75792ff8205e860030587b2fe0d031381f8699136008fe4e3993'), 'Original sample indices': ('src/oakink2_tamf/launch/sample.py', '7c86daa0435aba0ebe3d9e2a119d13da147404381cdea9d865aa250174ca40ca')}

    # These hashes pin the source inspected for this analysis, not the dirty historical
    # evaluator checkout. No historical array hashes were saved by the evaluator.
    mapping_reasons = []
    motion_rows, motion_values, boundary_audit = [], {}, []
    metadata_ok = False
    cache = None
    def local_saved_path(raw):
        # Resolve this run's WSL repository prefix without changing recorded strings
        # used in the official runner's fingerprint.
        saved_root = results['tamf_repository'].rstrip('/')
        require(raw.startswith(saved_root + '/'), f'Path is outside recorded repository: {raw}')
        return REPO_ROOT / raw[len(saved_root) + 1:]
    try:
        require(np is not None, 'NumPy unavailable')
        for label, (rel, expected_hash) in REVIEWED_SOURCES.items():
            actual_hash = hashlib.sha256((REPO_ROOT / rel).read_text(encoding='utf-8').encode()).hexdigest()
            require(actual_hash == expected_hash, f'{label}: author-reviewed source changed; ordering needs review')
            check(label + ' reviewed source hash', True, actual_hash)
        require('cache_dict["interaction_segment_info_list"]' in (REPO_ROOT / REVIEWED_SOURCES['Dataset cache order'][0]).read_text(), 'Cache-order source missing')
        # Loop: ascending dataset indices, seen-key skip, then file existence skip;
        # valid-length slicing; GT/refined extend together; SIV offsets 0,20,...
        for m in ('cr', 'siv'):
            tokens = ['for sample_id in tqdm(range(len(all_dataset)))',
                      'if info in duplicate_check:', 'duplicate_check.add(info)',
                      'if not os.path.exists(load_filepath):', 'avai_len = gt_sample["len"]',
                      'gt_hand_verts = gt_hand_verts[:avai_len, ...]',
                      'refined_hand_verts = refined_hand_verts[:avai_len, ...]']
            positions = [sources[m].find(t) for t in tokens]
            require(all(p >= 0 for p in positions) and positions == sorted(positions), f'{m}: official loop order not verified')
        cache_path = REPO_ROOT / manifest['subset_cache']
        require(digest(cache_path) == manifest['subset_cache_sha256'], 'Subset cache hash changed')
        require(digest(REPO_ROOT / manifest['source_cache']) == manifest['source_cache_sha256'], 'Original cache hash changed')
        check('Original and subset cache hashes', True, 'Both match preparation manifest')
        # CPU-only pickle loads of existing, local metadata; never import TaMF/models.
        with cache_path.open('rb') as stream:
            cache = pickle.load(stream)
        prefix = 'interaction_segment_'
        samples = manifest['samples']
        infos = cache[prefix + 'info_list']
        require(len(infos) == len(samples) == coverage['generated'], 'Manifest/cache segment counts differ')
        generated_root = REPO_ROOT / manifest['generated_dir']
        files = sorted(generated_root.glob('*.npy'))
        require([p.name for p in files] == [s['filename'] for s in samples], 'Generated file inventory changed')
        for pos, s in enumerate(samples):
            require(s['sample_id'] == int(Path(s['filename']).stem), f'Segment {pos}: original filename/index mismatch')
            require(tuple(s['info']) == tuple(infos[pos]), f'Segment {pos}: info mismatch')
            require(s['len'] == int(cache[prefix + 'len_list'][pos]), f'Segment {pos}: length mismatch')
            require(s['frame_id'] == list(cache[prefix + 'frame_id_list'][pos]), f'Segment {pos}: frame IDs mismatch')
            require(digest(files[pos]) == s['sha256'], f'Segment {pos}: generated bytes changed')
        require([s['sample_id'] for s in samples] == sorted(set(s['sample_id'] for s in samples)), 'Original indices not strictly ordered')
        check('All generated IDs and cached metadata', True, f'{len(samples)} filename hashes, keys, lengths and frame lists match')
        raw_refined_root = results['sample_refine_filepath'].rstrip('/')
        refined_root = local_saved_path(raw_refined_root)
        # Reproduce runner prepare_partial_results_state exactly, using original path strings.
        fingerprint = hashlib.sha256(PATHS['manifest'].read_bytes())
        fingerprint.update(raw_refined_root.encode())
        refined_files = sorted(refined_root.rglob('save_dict.pkl'))
        for p in refined_files:
            fingerprint.update((raw_refined_root + '/' + p.relative_to(refined_root).as_posix()).encode())
            fingerprint.update(str((p.stat().st_size, p.stat().st_mtime_ns)).encode())
        require(fingerprint.hexdigest() == results['partial_batch_identity'], 'Refined inventory/size/mtime fingerprint differs from evaluated run')
        check('Official runner refined snapshot fingerprint', True, fingerprint.hexdigest())
        for m in METRICS:
            e = entries[m]
            cmd = e['command']
            cache_flag = '--debug.cache_dict_filepath' if m == 'fid' else '--data.cache_dict_filepath'
            require(local_saved_path(cmd[cmd.index(cache_flag) + 1]) == cache_path, f'{m}: command used a different cache')
            require(cmd[cmd.index('--debug.sample_refine_filepath') + 1] == raw_refined_root, f'{m}: command used a different refined tree')
            require(e['partial_batch_identity'] == results['partial_batch_identity'], f'{m}: evaluated batch identity differs')
        seen = set()
        cr_offset = siv_offset = 0
        for pos, raw_info in enumerate(infos):
            info = tuple(raw_info)
            if info in seen:
                continue
            seen.add(info)  # Official adds before checking refined existence.
            path = refined_root / info[0].replace('/', '++') / str(info[1]) / str(info[2]) / 'save_dict.pkl'
            if not path.is_file():
                continue
            n = int(cache[prefix + 'len_list'][pos])
            frames = list(cache[prefix + 'frame_id_list'][pos])
            side = cache[prefix + 'hand_side_list'][pos]
            objects = sorted(cache[prefix + 'obj_traj_list'][pos])
            require(0 < n <= 160 and len(frames) == n and all(b > a for a, b in zip(frames, frames[1:])), f'Index {pos}: invalid frame metadata')
            require(side == info[2] and side in ('lh', 'rh'), f'Index {pos}: hand mismatch')
            with path.open('rb') as stream:
                refined = pickle.load(stream)
            require(tuple(refined['info']) == info and refined['hand_side'] == side, f'Index {pos}: refined info/hand mismatch')
            require(int(refined['len']) == n and list(refined['frame_id']) == frames, f'Index {pos}: refined length/frame mismatch')
            require(refined['obj_list'] == objects, f'Index {pos}: refined object mismatch')
            for name, shape in [('verts', (160, 778, 3)), ('joints', (160, 21, 3)), ('refine_pose_repr', (160, 99))]:
                a = np.asarray(refined[name])
                require(a.shape == shape and np.isfinite(a).all(), f'Index {pos}: invalid refined {name}')
            for name in ('pose', 'tsl', 'shape'):
                require(len(cache[prefix + name + '_list'][pos]) >= n, f'Index {pos}: insufficient GT {name}')
            ns = len(range(0, n, 20))
            motion_rows.append({'index': samples[pos]['sample_id'], 'cache_position': pos,
                                'sequence': info[0], 'primitive': info[1], 'hand': side,
                                'refined_output': path.relative_to(REPO_ROOT).as_posix(),
                                'valid_frames': n, 'siv_samples': ns, 'objects': objects,
                                'frame_ids': frames, 'siv_frame_offsets': list(range(0, n, 20)),
                                'cr_start': cr_offset, 'cr_end': cr_offset + n,
                                'siv_start': siv_offset, 'siv_end': siv_offset + ns})
            cr_offset += n
            siv_offset += ns
        require(len(motion_rows) == coverage['eligible_paired_motions'] == len(seen), 'Motion count/first-occurrence selection differs from saved batch')
        require(cr_offset == cr_frames and siv_offset == siv_frames, 'Recovered boundaries differ from official frame coverage')
        require(not [c for c in checks if c['Result'] != 'PASS' and ('coverage' in c['Check'] or 'paired count' in c['Check'])], 'Saved coverage/batch checks failed')
        metadata_ok = True
        check('First-occurrence motion metadata and boundaries', True, f'{len(motion_rows)} motions; {cr_offset} CR frames; {siv_offset} SIV samples')
    except (OSError, KeyError, TypeError, ValueError, pickle.UnpicklingError) as exc:
        mapping_reasons.append(str(exc))
        check('Exact per-motion reconstruction', False, str(exc))

    if metadata_ok:
        for m in ('cr', 'siv'):
            if m not in arrays:
                continue
            try:
                recovered = [[], []]
                weights = []
                for row in motion_rows:
                    start, end = row[f'{m}_start'], row[f'{m}_end']
                    weights.append(end - start)
                    for k, a in enumerate(arrays[m]):
                        require(end <= len(a) and end > start, f'{m}: invalid slice at index {row["index"]}')
                        chunk = a[start:end]
                        recovered[k].append(float((chunk < .005).mean()) if m == 'cr' else float(chunk.mean()))
                require(sum(weights) == len(arrays[m][0]) == len(arrays[m][1]), f'{m}: leftover array entries; no truncation allowed')
                pooled = [float(np.average(v, weights=weights)) for v in recovered]
                require(all(math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12) for a, b in zip(pooled, scores[m])), f'{m}: frame-weighted reconstructed scores mismatch')
                motion_values[m] = np.asarray(recovered)
                boundary_audit.append({'Metric': NAMES[m], 'GT weighted score': pooled[0], 'Refined weighted score': pooled[1],
                                        'Saved GT': scores[m][0], 'Saved refined': scores[m][1],
                                        'Weights': 'valid frames' if m == 'cr' else 'stride-20 samples',
                                        'Total weight': sum(weights), 'Equal-motion average (GT / refined)': str(np.mean(recovered, axis=1))})
                check(f'{NAMES[m]} frame-weighted reconstruction', True, str(pooled))
            except (ValueError, TypeError) as exc:
                mapping_reasons.append(str(exc))
                check(f'{NAMES[m]} frame-weighted reconstruction', False, str(exc))

    # Inventory actual saved metric artifacts, not inferred features from motion files.
    inventory_roots = [REPO_ROOT / 'tmp/compute_score', BATCH_DIR / 'results']
    metric_inventory = sorted(str(p.relative_to(REPO_ROOT)) for d in inventory_roots for p in d.rglob('*') if p.is_file())
    spectral_candidates = [p for p in metric_inventory if re.search(r'(psd|spectrum|spectra|activation|feature)', p, re.I)]
    fid = scores['fid'][0]
    fid_terms = {}
    for term in ('diff', 'tr1', 'tr2', 'tr_covmean'):
        matches = re.findall(rf'^{term}\s+([-+0-9.eE]+)\s*$', logs['fid']['stdout'], flags=re.M)
        if len(matches) == 1: fid_terms[term] = float(matches[0])
    fid_formula_ok = False
    if set(fid_terms) == {'diff', 'tr1', 'tr2', 'tr_covmean'} and number(fid):
        mean_term = fid_terms['diff']
        covariance_term = fid_terms['tr1'] + fid_terms['tr2'] - 2 * fid_terms['tr_covmean']
        fid_formula_ok = ('return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean' in sources['fid']
                          and math.isclose(mean_term + covariance_term, fid, rel_tol=1e-7, abs_tol=1e-6)
                          and source_ok['fid'] and log_score_ok['fid'])
    check('FID diagnostic formula and cancellation', fid_formula_ok,
          f'Log terms={fid_terms}; diff is printed with float32 precision; absolute tolerance 1e-6')
    runtime_items = []
    for m in METRICS:
        e = entries.get(m, {})
        try:
            start = datetime.fromisoformat(e['start_time']); end = datetime.fromisoformat(e['end_time'])
            t = e['runtime_seconds']
            require(e['status'] == 'success' and number(t) and end >= start, f'{m}: missing/invalid timestamps')
            runtime_items.append((m, t, start, end))
            check(f'{NAMES[m]} runtime versus timestamps', abs((end - start).total_seconds() - t) < .02, 'Recorded subprocess time versus UTC end − start')
        except (KeyError, TypeError, ValueError) as exc:
            limitations.append(f'{m} timeline unavailable: {exc}')
    mapping_table = []
    if metadata_ok:
        mapping_table = []
        for k, r in enumerate(motion_rows):
            row = {'Original index': r['index'], 'Cache position': r['cache_position'], 'Sequence': r['sequence'],
                   'Primitive': r['primitive'], 'Hand side': r['hand'], 'Refined output path (repo-relative)': r['refined_output'],
                   'Valid frames': r['valid_frames'], 'SIV samples': r['siv_samples'],
                   'CR slice [start,end)': f'[{r["cr_start"]}, {r["cr_end"]})',
                   'SIV slice [start,end)': f'[{r["siv_start"]}, {r["siv_end"]})', 'Objects': ', '.join(r['objects'])}
            for m in ('cr', 'siv'):
                if m in motion_values:
                    unit = '%' if m == 'cr' else 'cm³'; scale = 100 if m == 'cr' else 1
                    row[f'{NAMES[m]} GT / refined ({unit})'] = ' / '.join(f'{v:.4f}' for v in motion_values[m][:, k] * scale)
            mapping_table.append(row)


    # Export explicit motion values with units rather than encoding metrics in labels.
    for k, row in enumerate(motion_rows):
        row['metrics'] = {}
        for m in ('cr', 'siv'):
            if m in motion_values:
                gt, refined = (float(v) for v in motion_values[m][:, k])
                row['metrics'][m] = {'gt': gt, 'refined': refined,
                                     'refined_minus_gt': refined - gt,
                                     'unit': 'ratio' if m == 'cr' else 'cm3'}
                if m == 'cr':
                    row['metrics'][m]['difference_percentage_points'] = (refined - gt) * 100
    distribution_statistics = {}
    for m, pair in arrays.items():
        distribution_statistics[m] = []
        for a in pair:
            distribution_statistics[m].append({
                'count': int(a.size), 'mean': float(a.mean()), 'zero_count': int((a == 0).sum()),
                'zero_fraction': float((a == 0).mean()),
                'quantiles': {str(q): float(np.quantile(a, q)) for q in (0, .25, .5, .75, .9, .95, .99, 1)},
                'unit': 'm' if m == 'cr' else 'cm3'})
    unavailable = [
        {'view': 'PSKL-J detailed frequency spectra', 'reason': 'The official evaluator retained FFT/spectral arrays in memory; only shapes and the two aggregate KL scores were saved.'},
        {'view': 'FID joint exploratory PCA', 'reason': 'Neither GT nor refined encoder feature matrix was saved. The (91,64) log shapes are not feature arrays; the encoder is not rerun.'},
        {'view': 'Official per-motion PSKL-J / FID', 'reason': 'Both official metrics compare populations; there is no official individual score to recover from the aggregates.'},
    ]
    # Rendering is a separate plotting step; the notebook will only display its
    # saved results. Return primitive JSON-compatible state after the CLI serializes it.
    return {
        'run_id': run_id, 'results': results, 'coverage': coverage, 'entries': entries,
        'scores': scores, 'arrays': arrays, 'array_audit': array_audit,
        'metadata_ok': metadata_ok, 'motion_rows': motion_rows, 'motion_values': motion_values,
        'mapping_table': mapping_table, 'mapping_reasons': mapping_reasons,
        'boundary_audit': boundary_audit, 'checks': checks, 'limitations': limitations,
        'score_table': rows, 'distribution_statistics': distribution_statistics,
        'generated': generated, 'paired': paired, 'cr_frames': cr_frames, 'siv_frames': siv_frames,
        'fid': fid, 'fid_terms': fid_terms, 'fid_formula_ok': fid_formula_ok,
        'mean_term': mean_term if fid_formula_ok else None,
        'covariance_term': covariance_term if fid_formula_ok else None,
        'runtime_items': runtime_items, 'metric_inventory': metric_inventory,
        'spectral_candidates': spectral_candidates, 'unavailable': unavailable,
        'display_timezone': DISPLAY_TIMEZONE,
        'source_review_hashes': REVIEWED_SOURCES,
        'array_hashes_at_export': {m: [digest(p) for p in PATHS[f'{m}_arrays']] for m in arrays},
        'provenance_limits': [
            'No contemporaneous generation-cache checksum was saved. Original filename IDs rely on recorded one-worker settings and the unchanged prepared cache/manifest.',
            'The report saved no historical array/source hashes and recorded a dirty evaluator checkout. Association uses matching metadata, logs, timestamps, counts and scores; export hashes pin the currently validated bytes.',
            'The refined runner fingerprint includes Linux path strings and nanosecond mtimes. Run preparation in WSL; the display notebook performs no filesystem provenance checks.',
        ],
    }


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f'Unsupported JSON value: {type(value).__name__}')

def save_motion_csv(path, motion_rows):
    fields = ['original_index', 'cache_position', 'sequence', 'primitive', 'hand_side',
              'refined_output_path', 'valid_frames', 'siv_samples', 'objects',
              'cr_start', 'cr_end', 'siv_start', 'siv_end',
              'cr_gt_ratio', 'cr_refined_ratio', 'cr_difference_percentage_points',
              'siv_gt_mean_cm3', 'siv_refined_mean_cm3', 'siv_difference_cm3']
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for row in motion_rows:
            cr, siv = row['metrics'].get('cr', {}), row['metrics'].get('siv', {})
            writer.writerow({
                'original_index': row['index'], 'cache_position': row['cache_position'],
                'sequence': row['sequence'], 'primitive': row['primitive'], 'hand_side': row['hand'],
                'refined_output_path': row['refined_output'], 'valid_frames': row['valid_frames'],
                'siv_samples': row['siv_samples'], 'objects': ';'.join(row['objects']),
                **{k: row[k] for k in ('cr_start', 'cr_end', 'siv_start', 'siv_end')},
                'cr_gt_ratio': cr.get('gt'), 'cr_refined_ratio': cr.get('refined'),
                'cr_difference_percentage_points': cr.get('difference_percentage_points'),
                'siv_gt_mean_cm3': siv.get('gt'), 'siv_refined_mean_cm3': siv.get('refined'),
                'siv_difference_cm3': siv.get('refined_minus_gt'),
            })

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--batch-dir', type=Path, help='Saved partial batch directory; default common/partial_batch/test_one__0399')
    parser.add_argument('--output-dir', type=Path, help='Analysis-only destination; default common/analysis/test_one__0399')
    parser.add_argument('--no-plots', action='store_true', help='Save JSON/CSV values only; needs no Matplotlib. The notebook displays PNGs when exported.')
    parser.add_argument('--display-timezone', default='Asia/Jerusalem')
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    batch = (args.batch_dir or root / 'common/partial_batch/test_one__0399').resolve()
    output = (args.output_dir or root / 'common/analysis/test_one__0399').resolve()
    protected = [batch, root / 'tmp/compute_score', root / 'script', root / 'src',
                 root / 'common/sample', root / 'common/sample_refine', root / 'common/save_cache_dict']
    if output == root or any(output == p or p in output.parents or output in p.parents for p in protected):
        parser.error('Output directory would overlap existing evaluation inputs. Choose a separate analysis directory.')
    print('Reading and validating existing artifacts; no evaluator or model is called.', flush=True)
    state = analyze(root, batch, args.display_timezone)
    state['schema_version'] = 1
    state['prepared_at_utc'] = datetime.now(timezone.utc).isoformat()
    state['analysis_script_sha256'] = digest(Path(__file__))
    state['definitions'] = {
        'cr': {'threshold_m': .005, 'operator': '<', 'aggregation': 'pooled valid frames', 'motion_unit': 'ratio', 'plot_distance_unit': 'mm'},
        'siv': {'frame_step': 20, 'first_offset': 0, 'aggregation': 'pooled stride-20 samples', 'unit': 'cm3', 'arrays_already_converted': True},
        'psklj': {'population_metric': True, 'frequency_convention': 'full unshifted 158-bin FFT of second differences of 160 padded frames; no assumed Hz'},
        'fid': {'population_metric': True, 'unit': 'official encoder feature space', 'formula': 'squared mean distance + tr1 + tr2 - 2*tr_covmean'},
    }
    output.mkdir(parents=True, exist_ok=True)
    state['figures'] = []
    if not args.no_plots:
        try:
            from tamf_result_plots import render
            state['figures'] = render(state, output)
        except ImportError as exc:
            state['limitations'].append(f'PNG export unavailable: {exc}. No package installed; values still saved.')
    save_motion_csv(output / 'motions.csv', state['motion_rows'] if state['metadata_ok'] else [])
    # Last write publishes a complete portable snapshot. The notebook only reads
    # this JSON and referenced PNGs; original arrays/logs/caches are never written.
    tmp = output / 'analysis.json.tmp'
    tmp.write_text(json.dumps(state, default=json_value, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
    tmp.replace(output / 'analysis.json')
    failures = [c for c in state['checks'] if c['Result'] != 'PASS']
    print(f'Saved {output / "analysis.json"}')
    print(f'Saved {output / "motions.csv"}')
    print(f'{len(state["motion_values"])} recovered metrics; {len(state["motion_rows"])} motions; '
          f'{len(state["figures"])} figures; {len(state["checks"]) - len(failures)}/{len(state["checks"])} checks passed.')
    for failure in failures:
        print(f'SKIPPED / UNVERIFIED: {failure["Check"]}: {failure["Evidence"]}')
    return 0 if not failures else 2

if __name__ == '__main__':
    sys.exit(main())
