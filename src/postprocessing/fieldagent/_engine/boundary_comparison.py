"""Frozen-pool boundary selection experiment. No labels in this module.

Candidates are generated once from raw predictions. All controllers receive the
same measurements and verification. Replay rechecks neighbours against the
current scene. This is an A1 selection pilot, not free-coordinate A2 editing.
"""
import base64
from collections import Counter
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

import cv2
import numpy as np
from shapely.geometry import LineString
from boundary_executor import BoundaryExecutor
from boundary_tools import REGISTRY, catalog, sample_ring, normals
from cvr_agent import dump, save_items, scene_metrics, sha, state_key, read_items
from fieldbqc_geometry import vertices, polygons
from fieldbqc_planner import APIPlanner

VERSION = 'boundary-selection-pilot-v1'
TOOLS = [k for k, v in REGISTRY.items() if v.task != 'legalize']
RULE = dict(reduction_weight=.65, support_gain_weight=2.5,
            shift_penalty=.015, area_penalty=.15, curved_line_penalty=.05,
            long_segment_px=12., straight_fraction=.5, max_rounds=2)
PROMPT = Path(__file__).with_name('BOUNDARY_SELECTION_PROMPT.md').read_text(encoding='utf-8')
SCHEMA = dict(type='object', additionalProperties=False, properties={
    'selections': dict(type='array', items=dict(type='object', additionalProperties=False,
        properties={'target_id': dict(type='string'),
                    'candidate_id': dict(type=['string', 'null']),
                    'reason': dict(type='string')},
        required=['target_id', 'candidate_id', 'reason'])),
    'summary': dict(type='string')}, required=['selections', 'summary'])


def profile(box, item, items):
    g = item.geometry
    segments = []
    for p in polygons(g):
        q = np.asarray(p.exterior.simplify(1.).coords)
        segments.extend(np.linalg.norm(np.diff(q, axis=0), axis=1).tolist())
    total = max(sum(segments), 1e-9)
    shared = sum(g.boundary.intersection(p.geometry.boundary).length for p in items if p.identifier != item.identifier)
    return dict(vertices=vertices(g), perimeter=g.length, area=g.area,
        points_per_pixel=vertices(g)/max(g.length, 1.),
        long_straight_fraction=sum(v for v in segments if v >= RULE['long_segment_px'])/total,
        image_support=box.evidence(g), shared_length=shared,
        bbox=list(g.bounds), holes=sum(len(p.interiors) for p in polygons(g)),
        components=len(polygons(g)))


def features(box, origin, candidate):
    a, b = origin.geometry, candidate.geometry
    n0, n1 = vertices(a), vertices(b)
    return dict(vertices_before=n0, vertices=n1, reduction=1-n1/max(n0, 1),
        support_before=box.evidence(a), support_after=box.evidence(b),
        support_gain=box.evidence(b)-box.evidence(a),
        shift=a.boundary.hausdorff_distance(b.boundary),
        area_change=abs(b.area-a.area)/max(a.area, 1.))


def score(candidate, diagnostic):
    f = candidate['features']; tools = [a['tool'] for a in candidate['chain']]
    penalty = RULE['curved_line_penalty'] if 'boundary_line_realign' in tools and diagnostic['long_straight_fraction'] < RULE['straight_fraction'] else 0.
    return (RULE['reduction_weight']*f['reduction'] + RULE['support_gain_weight']*f['support_gain']
            - RULE['shift_penalty']*f['shift'] - RULE['area_penalty']*f['area_change'] - penalty)


def specifications():
    actions = []
    for tool in TOOLS:
        if REGISTRY[tool].task == 'simplify':
            actions.extend(dict(tool=tool, parameters=dict(tolerance=t)) for t in (.5, 1., 2.))
        else:
            actions.extend(dict(tool=tool, parameters=dict(radius=r)) for r in (1.5, 3.))
    return actions


def verify_chain(executor, stages, uid):
    """Apply each typed gate in memory; restore all executor state afterwards."""
    current, anchors = executor.current, executor.simplify_anchors
    checks = []
    try:
        for step, candidate in stages:
            task = REGISTRY[step['tool']].task
            verdict = executor.verify(candidate, [uid], task)
            checks.append(verdict)
            if not verdict['accept']:
                break
            executor.current = candidate
            if task == 'realign':
                executor.simplify_anchors = candidate
        return dict(accept=len(checks) == len(stages) and all(c['accept'] for c in checks),
                    reasons=sorted({r for c in checks for r in c['reasons']}), stages=checks)
    finally:
        executor.current, executor.simplify_anchors = current, anchors


def build_pool(image, items, folder, config):
    began = time.monotonic(); e = BoundaryExecutor(image, items, folder, config)
    diagnostics = {p.identifier: profile(e.box, p, items) for p in items}
    entries = []; geometries = {}; index = {}; rejected = Counter()
    for item in items:
        uid = item.identifier; good_realign = []
        for action in specifications():
            chain = [action]
            cid = f'p{len(entries):05d}'; started = time.monotonic()
            candidate = None; verdict = None
            with e.audit.call('pool_candidate_generation', dict(candidate_id=cid, target_id=uid, chain=chain), REGISTRY[action['tool']].methods) as event:
                try:
                    candidate, info = e.box.propose(action['tool'], items, [uid], action['parameters'])
                    with e.audit.call('pool_typed_verification'):
                        verdict = verify_chain(e, [(action, candidate)], uid)
                    if verdict['accept'] and REGISTRY[action['tool']].task == 'realign':
                        good_realign.append((action, candidate))
                except Exception as error:
                    verdict = dict(accept=False, reasons=['tool_error'], error=f'{type(error).__name__}: {error}')
                row = pool_row(e, items, item, candidate, chain, cid, verdict, started)
                entries.append(row); event.update(candidate_id=cid, accept=verdict['accept'], reasons=verdict['reasons'])
                if candidate is not None:
                    geometries[cid] = candidate
        # Joint correction + compression; both actual steps use their own gate.
        for first, intermediate in good_realign:
            for tolerance in (1., 2.):
                second = dict(tool='contour_dp', parameters=dict(tolerance=tolerance))
                chain = [first, second]; cid = f'p{len(entries):05d}'; started = time.monotonic(); candidate = None
                with e.audit.call('pool_joint_candidate_generation', dict(candidate_id=cid, target_id=uid, chain=chain), ['verified_realign_then_DP']) as event:
                    try:
                        with e.audit.call('joint_DP_generation', methods=REGISTRY['contour_dp'].methods):
                            candidate, info = e.box.propose('contour_dp', intermediate, [uid], second['parameters'])
                        with e.audit.call('pool_joint_typed_verification'):
                            verdict = verify_chain(e, [(first, intermediate), (second, candidate)], uid)
                        if verdict['accept']:
                            overall = e.verify(candidate, [uid], 'realign')
                            if not overall['accept']:
                                verdict = overall
                    except Exception as error:
                        verdict = dict(accept=False, reasons=['tool_error'], error=f'{type(error).__name__}: {error}')
                    row = pool_row(e, items, item, candidate, chain, cid, verdict, started)
                    entries.append(row); event.update(accept=verdict['accept'], reasons=verdict['reasons'])
                    if candidate is not None:
                        geometries[cid] = candidate
        print(f'POOL {uid}: {sum(r["verdict"]["accept"] for r in entries if r["target_id"] == uid)} passing candidates', flush=True)
    # Deduplicate only within the same target/family, so every single tool keeps
    # its own baseline even when another algorithm produces identical geometry.
    for row in entries:
        if not row['verdict']['accept']:
            rejected.update(row['verdict']['reasons']); continue
        target = next(p for p in geometries[row['candidate_id']] if p.identifier == row['target_id'])
        family = '+'.join(a['tool'] for a in row['chain'])
        key = (row['target_id'], family, target.geometry.normalize().wkb_hex)
        row['duplicate_of'] = index.get(key)
        if key not in index:
            index[key] = row['candidate_id']
        row['rule_score'] = score(row, diagnostics[row['target_id']])
    elapsed = time.monotonic()-began
    result = dict(version=VERSION, raw_state=state_key(items), runtime_gt_access=False,
        diagnostics=diagnostics, candidates=entries, seconds=elapsed,
        generated=len(entries), passing=sum(r['verdict']['accept'] for r in entries),
        tool_errors=sum('tool_error' in r['verdict']['reasons'] for r in entries),
        reject_reasons=dict(rejected), rule=RULE, protocol='Frozen raw-state candidates; current-neighbour revalidation at replay')
    dump(Path(folder)/'pool.json', result)
    dump(Path(folder)/'methods.json', dict(actual_method_calls=dict(e.audit.methods)))
    return result


def pool_row(e, items, item, candidate, chain, cid, verdict, started):
    target = next((p for p in candidate if p.identifier == item.identifier), None) if candidate is not None else None
    path = save_items(e.root/'candidates'/cid/'candidate.pixel.geojson', [target]) if target is not None else None
    row = dict(candidate_id=cid, target_id=item.identifier, chain=chain, verdict=verdict,
        candidate_file=path, features=features(e.box, item, target) if target is not None and target.geometry.is_valid and not target.geometry.is_empty else None,
        seconds=time.monotonic()-started)
    dump(e.root/'candidates'/cid/'result.json', row)
    return row


def available(pool, pending=None, excluded=(), tool=None):
    return [r for r in pool['candidates'] if r['verdict']['accept'] and not r.get('duplicate_of')
        and r['candidate_id'] not in excluded and (pending is None or r['target_id'] in pending)
        and (tool is None or (len(r['chain']) == 1 and r['chain'][0]['tool'] == tool))]


def rule_plan(pool, candidates, pending):
    decisions = []
    for uid in pending:
        rows = [r for r in candidates if r['target_id'] == uid]
        best = max(rows, key=lambda r: (r['rule_score'], -r['features']['vertices'], r['candidate_id'])) if rows else None
        chosen = best if best and best['rule_score'] > 0 else None
        decisions.append(dict(target_id=uid, candidate_id=chosen['candidate_id'] if chosen else None,
            reason=f'Frozen proxy score={chosen["rule_score"]:.5f}; not a GT quality estimate' if chosen else 'No positive passing candidate'))
    return dict(selections=decisions, summary='Deterministic profile-aware proxy ranking')


def validate_plan(plan, candidates, pending):
    if not isinstance(plan, dict) or set(plan) != {'selections', 'summary'} or not isinstance(plan['summary'], str):
        raise ValueError('Invalid selection plan fields')
    if not isinstance(plan['selections'], list):
        raise ValueError('Selections must be an array')
    choices = {r['candidate_id']: r['target_id'] for r in candidates}; seen = set()
    for r in plan['selections']:
        if not isinstance(r, dict) or set(r) != {'target_id', 'candidate_id', 'reason'}:
            raise ValueError('Invalid selection fields')
        uid, cid = r['target_id'], r['candidate_id']
        if not isinstance(uid, str) or uid not in pending or uid in seen:
            raise ValueError('Unknown or duplicate target')
        if not isinstance(r['reason'], str) or not r['reason'].strip():
            raise ValueError('Missing reason')
        if cid is not None and (not isinstance(cid, str) or cid not in choices or choices[cid] != uid):
            raise ValueError('Unknown or mismatched candidate')
        seen.add(uid)
    if seen != set(pending):
        raise ValueError('Each pending target must have an explicit choice or keep')
    return plan


class SelectionExecutor(BoundaryExecutor):
    def replay(self, row, source, reason):
        if len(self.history) >= self.config.budget or time.monotonic()-self.started >= self.config.seconds:
            raise RuntimeError('Selection replay budget exhausted')
        reference = row['candidate_file']
        if sha(reference['path']) != reference['sha256']:
            raise ValueError('Frozen candidate changed')
        frozen = read_items(reference['path']); uid = row['target_id']
        replacement = next(p for p in frozen if p.identifier == uid)
        candidate = [replacement if p.identifier == uid else p for p in self.current]
        task = 'realign' if any(REGISTRY[a['tool']].task == 'realign' for a in row['chain']) else 'simplify'
        tid = f't{len(self.history):05d}'; folder = self.root/'candidates'/tid
        before = scene_metrics(self.current); base = self.state_id; began = time.monotonic()
        with self.audit.call('replay_frozen_candidate', dict(candidate_id=row['candidate_id'], chain=row['chain'],
                source=source, reason=reason), ['frozen_candidate_retrieval']) as event:
            with self.audit.call('typed_boundary_verifier', methods=['common_scene_safety', 'typed_action_gate', 'cumulative_origin_guard']):
                verdict = self.verify(candidate, [uid], task)
            path = save_items(folder/'attempted.pixel.geojson', candidate)
            if verdict['accept']:
                with self.audit.call('commit_boundary_candidate', methods=['verified_atomic_commit']):
                    self.current = candidate
                    if task == 'realign':
                        self.simplify_anchors = [replacement if p.identifier == uid else p for p in self.simplify_anchors]
                    save_items(self.root/'checkpoints'/(tid+'.pixel.geojson'), self.current)
            result = dict(trial_id=tid, candidate_id=row['candidate_id'], base_state=base, output_state=self.state_id,
                tool='+'.join(a['tool'] for a in row['chain']), target_ids=[uid], parameters=row['chain'],
                source=source, reason=reason, committed=verdict['accept'], verdict=verdict,
                candidate_file=path, frozen_candidate=reference, before=before, after=scene_metrics(self.current),
                seconds=time.monotonic()-began)
            dump(folder/'result.json', result); event.update(committed=verdict['accept'], reasons=verdict['reasons'])
        self.history.append(result)
        return result


def overlay(image, items, bounds=None, title='', side=320):
    h, w = image.shape[:2]
    x0, y0, x1, y1 = (0, 0, w, h) if bounds is None else bounds
    x0, y0 = max(0, int(x0)), max(0, int(y0)); x1, y1 = min(w, int(math.ceil(x1))), min(h, int(math.ceil(y1)))
    frame = image[y0:y1, x0:x1].copy(); scale = min(side/max(x1-x0, 1), side/max(y1-y0, 1))
    frame = cv2.resize(frame, (max(1, round((x1-x0)*scale)), max(1, round((y1-y0)*scale))), interpolation=cv2.INTER_NEAREST)
    out = np.full((side+32, side, 3), 245, dtype=np.uint8); out[32:32+frame.shape[0], :frame.shape[1]] = frame
    for p in items:
        for poly in polygons(p.geometry):
            for r in [poly.exterior, *poly.interiors]:
                xy = (np.asarray(r.coords)-[x0, y0])*scale+[0, 32]
                cv2.polylines(out, [np.round(xy).astype(np.int32)], True, (40, 255, 50), 1)
        if bounds is None:
            c = p.geometry.representative_point()
            cv2.putText(out, p.identifier, (int((c.x-x0)*scale), 32+int((c.y-y0)*scale)), cv2.FONT_HERSHEY_SIMPLEX, .35, (0, 220, 255), 1)
    cv2.putText(out, title[:42], (3, 20), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 0), 1)
    return out


def visual_bundle(e, pool, candidates, pending, folder):
    images = [('raw', overlay(e.box_image, [], title='Raw image', side=512)),
              ('current', overlay(e.box_image, e.current, title='Current prediction IDs', side=512))]
    # Identical deterministic observation construction is available to both
    # controllers. Render top score, best simplify, and best correction for each
    # target, but retain EVERY passing pool entry in the structured context.
    targets = {p.identifier: p for p in e.original}; panels = []
    for uid in pending:
        p = targets[uid]; x0, y0, x1, y1 = p.geometry.bounds; bounds = (x0-6, y0-6, x1+6, y1+6)
        rows = [r for r in candidates if r['target_id'] == uid]
        top = sorted(rows, key=lambda r: r['rule_score'], reverse=True)
        simple = next((r for r in top if len(r['chain']) == 1 and REGISTRY[r['chain'][0]['tool']].task == 'simplify'), None)
        correct = next((r for r in top if any(REGISTRY[a['tool']].task == 'realign' for a in r['chain'])), None)
        chosen = [top[0] if top else None, simple, correct]
        tiles = [overlay(e.box_image, [], bounds, uid+' raw crop', 160),
                 overlay(e.box_image, [p], bounds, uid+' original', 160)]
        for r in chosen:
            if r:
                candidate = next(q for q in read_items(r['candidate_file']['path']) if q.identifier == uid)
                tiles.append(overlay(e.box_image, [candidate], bounds, r['candidate_id']+' N='+str(r['features']['vertices']), 160))
            else:
                tiles.append(overlay(e.box_image, [p], bounds, 'No candidate', 160))
        panels.append(np.hstack(tiles))
        if len(panels) == 4:
            images.append((f'candidate_atlas_{len(images)}', np.vstack(panels))); panels = []
    if panels:
        images.append((f'candidate_atlas_{len(images)}', np.vstack(panels)))
    urls = []
    for name, im in images:
        ok, blob = cv2.imencode('.png', im)
        if not ok:
            raise ValueError('Failed image encoding')
        (folder/(name+'.png')).write_bytes(blob.tobytes())
        urls.append('data:image/png;base64,'+base64.b64encode(blob).decode('ascii'))
    return urls


def gpt_plan(e, pool, candidates, pending, settings, call_number):
    folder = e.root/'gpt_calls'/f'{call_number:03d}'; folder.mkdir(parents=True)
    context = dict(goal='Use substantially fewer necessary vertices AND more accurate image-supported boundary location; no fixed point target.',
        protocol=VERSION, state_id=e.state_id, raw_pool_state=pool['raw_state'],
        pending_ids=pending, tools=catalog(), diagnostics={k: pool['diagnostics'][k] for k in pending},
        candidates=[dict(candidate_id=r['candidate_id'], target_id=r['target_id'], chain=r['chain'],
                         features=r['features'], rule_score=r['rule_score']) for r in candidates],
        rule_formula=RULE, feedback=[dict(target_ids=h['target_ids'], candidate_id=h['candidate_id'],
            committed=h['committed'], reasons=h['verdict']['reasons']) for h in e.history],
        image_order='raw scene, current IDs, atlases: raw crop / original / top proxy / best simplifier / best correction',
        atlas_limitation='Only deterministic representative candidates rendered; all passing candidates listed in JSON. Rule sees same numeric features, no GPT-specific candidate generation.',
        ground_truth_available=False, budget=dict(max_rounds=2, round=call_number),
        constraints=asdict(e.config))
    dump(folder/'context.json', context)
    builder = APIPlanner(settings, schema=SCHEMA, system_prompt=PROMPT)
    payload = builder._payload(context, visual_bundle(e, pool, candidates, pending, folder))
    dump(folder/'request.json', payload)
    key = os.environ.get(settings.get('api_key_env', 'OPENAI_API_KEY'))
    if not key:
        raise RuntimeError('OPENAI_API_KEY unavailable')
    endpoint = settings['base_url'].rstrip('/')+('/responses' if settings['api'] == 'responses' else '/chat/completions')
    meta = dict(endpoint=endpoint, requested_model=settings['model'], request_sha256=sha(folder/'request.json'), retries=0)
    started = time.monotonic()
    try:
        with e.audit.call('gpt_api_request', meta, ['HTTP POST', 'GPT visual frozen-pool selection']) as event:
            req = urllib.request.Request(endpoint, data=json.dumps(payload, allow_nan=False).encode('utf-8'),
                headers={'Authorization': 'Bearer '+key, 'Content-Type': 'application/json'}, method='POST')
            with urllib.request.urlopen(req, timeout=settings.get('timeout_seconds', 90)) as response:
                data = json.load(response)
            dump(folder/'response.json', data)
            if settings['api'] == 'responses':
                raw = ''.join(p.get('text', '') for item in data.get('output', []) if item.get('type') == 'message' for p in item.get('content', []) if p.get('type') == 'output_text')
                complete = data.get('status') == 'completed'
            else:
                choice = data['choices'][0]; raw = choice['message']['content']; complete = choice.get('finish_reason') == 'stop'
            meta.update(seconds=time.monotonic()-started, usage=data.get('usage', {}),
                response_model=data.get('model'), response_id=data.get('id'), completed=complete)
            if not complete:
                raise ValueError('Incomplete model response')
            plan = validate_plan(json.loads(raw), candidates, pending)
            dump(folder/'decision.json', plan); event.update(meta, decision=plan)
        dump(folder/'metadata.json', meta)
        return plan, meta
    except Exception as error:
        message = f'HTTP {error.code}' if isinstance(error, urllib.error.HTTPError) else type(error).__name__+': '+str(error)
        meta.update(error=message, seconds=time.monotonic()-started)
        dump(folder/'error.json', meta)
        raise RuntimeError(message) from None


def run_selection(image, items, pool, folder, config, method, settings=None, max_rounds=2):
    e = SelectionExecutor(image, items, folder, config); e.box_image = image
    allids = [p.identifier for p in items]; pending = list(allids); excluded = set(); records = []; calls = 0; error = None
    stop = 'all_targets_decided'; started = time.monotonic(); requested = []
    tool = method if method in TOOLS else None
    with e.audit.call('controller_protocol', dict(method=method, max_rounds=max_rounds, rule=RULE), ['frozen_pool_selection_protocol']):
        pass
    for round_number in range(1, max_rounds+1):
        if not pending:
            break
        candidates = available(pool, pending, excluded, tool)
        if method == 'gpt_selector':
            calls += 1
            try:
                plan, record = gpt_plan(e, pool, candidates, pending, settings, round_number)
                records.append(record)
            except RuntimeError as exc:
                error = str(exc); stop = 'external_api_or_response_error'; break
        else:
            with e.audit.call('deterministic_rule_selection', dict(round=round_number, pending=pending), ['profile_aware_proxy_ranking']):
                plan = rule_plan(pool, candidates, pending)
        validate_plan(plan, candidates, pending)
        dump(e.root/'decisions'/f'{round_number:03d}.json', plan)
        selected = {r['target_id']: r for r in plan['selections']}; lookup = {r['candidate_id']: r for r in candidates}; retry = []
        for uid in pending:  # Common input order, never model-dependent priority.
            decision = selected[uid]; requested.append(dict(round=round_number, **decision))
            cid = decision['candidate_id']
            if cid is None:
                continue
            result = e.replay(lookup[cid], method, decision['reason']); excluded.add(cid)
            if not result['committed']:
                retry.append(uid)
        pending = retry
    if pending and not error:
        stop = 'revision_budget_exhausted'
    trace = e.finish(stop)
    candidates_used = available(pool, tool=tool)
    # Computational attribution of a shared precomputed pool, not measured
    # end-to-end latency from independently regenerating it for every arm.
    generation_cost = pool['seconds'] if tool is None else sum(r['seconds'] for r in pool['candidates'] if len(r['chain']) == 1 and r['chain'][0]['tool'] == tool)
    trace.update(version=VERSION, selection_mode='A1 frozen-pool GPT selection' if method == 'gpt_selector' else 'Deterministic frozen-pool ranking',
        model_calls=calls, model_records=records, gpt_error=error, selections=requested,
        shared_pool_generated=pool['generated'], accessible_candidates=len(candidates_used),
        candidate_generation_seconds_attributed=generation_cost,
        replay_controller_seconds=time.monotonic()-started,
        attributed_total_seconds=generation_cost+time.monotonic()-started,
        cost_limitation='Pool generated once and attributed; not independent natural latency per arm',
        limitations=['Development pilot; thresholds and rule not independently validated',
            'Frozen candidates; no new tool/code/coordinates from GPT',
            'Shared-edge movement constructor absent', 'Image support is an uncertain proxy'])
    dump(e.root/'trace.json', trace)
    contract = json.loads((e.root/'tool_catalog.json').read_text(encoding='utf-8'))
    contract.update(model_calls=calls, controller=method, pool_generation='Shared and separately audited under pools/',
                    capabilities='Frozen-pool selection, no free-coordinate edits')
    dump(e.root/'tool_catalog.json', contract)
    return trace
