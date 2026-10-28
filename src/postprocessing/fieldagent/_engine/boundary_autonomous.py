"""A2 model-authored geometry with explicit preview -> feedback -> commit.

Runtime never reads labels. Model edits are reconstructed literally: no hidden
simplify, snap, make_valid, nearest-point correction or heuristic finishing.
The same typed BoundaryExecutor gates protect pool, tool and direct proposals.
"""
import base64
from copy import deepcopy
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
from boundary_comparison import SelectionExecutor, available, features, profile, overlay
from boundary_tools import REGISTRY, catalog, parameters
from cvr_agent import dump, read_items, save_items, scene_metrics, sha
from fieldbqc_direct_edit import ring_records, replace_ring
from fieldbqc_geometry import polygons, vertices
from fieldbqc_planner import APIPlanner

VERSION = 'boundary-autonomous-v1'
PROMPT = Path(__file__).with_name('BOUNDARY_AUTONOMOUS_PROMPT.md').read_text(encoding='utf-8')
MINIMUM_VERTICES_PROMPT = Path(__file__).with_name('BOUNDARY_MINIMUM_VERTICES_PROMPT.md').read_text(encoding='utf-8')


def obj(fields):
    return dict(type='object', additionalProperties=False, properties=fields, required=list(fields))


STRING = dict(type='string')
INTEGER = dict(type='integer')
XY = dict(type='array', items=dict(type='number'), minItems=2, maxItems=2)
EDIT = [
    obj(dict(kind=dict(type='string', enum=['retain_vertices']), ring_id=STRING,
        keep_indices=dict(type='array', items=INTEGER))),
    obj(dict(kind=dict(type='string', enum=['move_vertices']), ring_id=STRING,
        moves=dict(type='array', items=obj(dict(index=INTEGER, position=XY))))),
    obj(dict(kind=dict(type='string', enum=['replace_boundary_arc']), ring_id=STRING,
        start_index=INTEGER, end_index=INTEGER, path=dict(type='array', items=XY))),
]
COMMON = dict(target_id=STRING, reason=STRING,
    evidence_refs=dict(type='array', items=STRING))
PARAMETERS = dict(type='array', items=obj(dict(name=STRING, value=dict(type='number'))))
COMMANDS = [
    obj(dict(kind=dict(type='string', enum=['direct_edit']), **COMMON,
        task=dict(type='string', enum=['simplify', 'realign']), edits=dict(type='array', items=dict(anyOf=EDIT)))),
    obj(dict(kind=dict(type='string', enum=['use_tool']), **COMMON,
        tool_id=dict(type='string', enum=list(REGISTRY)), parameters=PARAMETERS)),
    obj(dict(kind=dict(type='string', enum=['use_pool']), **COMMON, candidate_id=STRING)),
    obj(dict(kind=dict(type='string', enum=['commit', 'discard']), **COMMON, proposal_id=STRING)),
    obj(dict(kind=dict(type='string', enum=['keep', 'review']), **COMMON)),
]
SCHEMA = obj(dict(state_id=STRING, diagnosis=STRING,
    commands=dict(type='array', items=dict(anyOf=COMMANDS)), done=dict(type='boolean')))


def point(value, shape):
    if (not isinstance(value, list) or len(value) != 2 or
        any(type(v) not in (int, float) or not math.isfinite(v) for v in value)):
        raise ValueError('Expected finite original-image [x,y]')
    if not 0 <= value[0] <= shape[1] or not 0 <= value[1] <= shape[0]:
        raise ValueError('coordinate_outside_image')
    return [float(v) for v in value]


def index(value, n):
    if type(value) is not int or not 0 <= value < n:
        raise ValueError('Unknown vertex index')
    return value


def reconstruct(item, edits, task, shape):
    if task not in ('simplify', 'realign') or not isinstance(edits, list) or not 1 <= len(edits) <= 16:
        raise ValueError('Invalid typed edit list')
    records = {r['ring_id']: r['coordinates'] for r in ring_records(item.geometry)}
    g = item.geometry; touched = set()
    for edit in edits:
        kind = edit.get('kind') if isinstance(edit, dict) else None
        spec = next((s for s in EDIT if s['properties']['kind']['enum'] == [kind]), None)
        if not spec or set(edit) != set(spec['required']):
            raise ValueError('Unknown edit or incorrect fields')
        rid = edit['ring_id']
        if not isinstance(rid, str) or rid not in records or rid in touched:
            raise ValueError('Unknown ring or more than one edit per state ring')
        touched.add(rid); coords = deepcopy(records[rid]); n = len(coords)
        if task == 'realign' and not rid.endswith('/outer'):
            raise ValueError('Position correction edits exterior rings only')
        if kind == 'retain_vertices':
            keep = edit['keep_indices']
            if (not isinstance(keep, list) or not 3 <= len(keep) < n or
                any(type(v) is not int or not 0 <= v < n for v in keep) or len(set(keep)) != len(keep) or
                sum((b-a) % n for a, b in zip(keep, keep[1:]+keep[:1])) != n):
                raise ValueError('Keep >=3 and fewer than original vertices in forward cyclic order')
            replacement = [coords[k] for k in keep]
        elif kind == 'move_vertices':
            if task != 'realign':
                raise ValueError('Vertex movement requires realign task')
            moves = edit['moves']; seen = set()
            if not isinstance(moves, list) or not 1 <= len(moves) <= n:
                raise ValueError('Empty or excessive moves')
            for move in moves:
                if not isinstance(move, dict) or set(move) != {'index', 'position'}:
                    raise ValueError('Invalid vertex movement')
                k = index(move['index'], n)
                if k in seen:
                    raise ValueError('Repeated moved index')
                seen.add(k); coords[k] = point(move['position'], shape)
            replacement = coords
        else:
            start = index(edit['start_index'], n); end = index(edit['end_index'], n)
            path = edit['path']
            if start == end or not isinstance(path, list) or not 2 <= len(path) <= 256:
                raise ValueError('Specify a non-full-ring forward arc and a path of 2..256 points')
            path = [point(p, shape) for p in path]
            if task == 'simplify' and (path[0] != coords[start] or path[-1] != coords[end]):
                raise ValueError('Simplify arc endpoints must exactly match state vertices; use realign to move them')
            if not LineString(path).is_simple:
                raise ValueError('Replacement path self-intersects')
            rest = []; k = (end+1) % n
            while k != start:
                rest.append(coords[k]); k = (k+1) % n
            replacement = path+rest
        if len(replacement) < 3 or len({tuple(p) for p in replacement}) != len(replacement):
            raise ValueError('Collapsed ring or duplicate coordinates')
        old_area = sum(a[0]*b[1]-b[0]*a[1] for a, b in zip(records[rid], records[rid][1:]+records[rid][:1]))
        new_area = sum(a[0]*b[1]-b[0]*a[1] for a, b in zip(replacement, replacement[1:]+replacement[:1]))
        if old_area*new_area <= 0:
            raise ValueError('Ring orientation reversed or collapsed')
        g = replace_ring(g, rid, replacement)
    return item.with_geometry(g)


class AutonomousExecutor(SelectionExecutor):
    def __init__(self, image, items, root, config=None):
        super().__init__(image, items, root, config)
        self.image = image; self.pending_proposals = {}; self.proposal_counter = 0
        self.decisions = []; self.preview_attempts = 0
        self.author = 'GPT'

    def propose(self, command, *, expected_state, pool=None):
        # A failed proposal is part of the record and never changes geometry.
        if expected_state != self.state_id:
            raise ValueError('Stale state; observe again')
        if self.preview_attempts >= self.config.budget or time.monotonic()-self.started >= self.config.seconds:
            raise RuntimeError('Candidate or time budget exhausted')
        self.preview_attempts += 1; self.proposal_counter += 1
        pid = f'a{self.proposal_counter:05d}'; folder = self.root/'proposals'/pid
        candidate = None; task = None; base = self.state_id; started = time.monotonic()
        uid = command.get('target_id'); lookup = {p.identifier: p for p in self.current}
        methods = []; source = command.get('kind'); key = None
        with self.audit.call('autonomous_proposal', dict(proposal_id=pid, state_id=base, command=command), ['explicit_geometry_proposal_protocol']) as event:
            try:
                if uid not in lookup:
                    raise ValueError('Unknown target ID')
                if source == 'direct_edit':
                    task = command['task']
                    prefix = 'model' if self.author == 'GPT' else 'deterministic'
                    methods = [prefix+'_vertex_index_selection' if e['kind'] == 'retain_vertices' else
                               prefix+'_explicit_vertex_movement' if e['kind'] == 'move_vertices' else
                               prefix+'_explicit_arc_replacement' for e in command['edits']]
                    with self.audit.call('literal_direct_geometry_reconstruction', methods=methods+['polygon_ring_reconstruction']):
                        target = reconstruct(lookup[uid], command['edits'], task, self.box.shape)
                    candidate = [target if p.identifier == uid else p for p in self.current]
                elif source == 'use_tool':
                    tool = command['tool_id']; cfg = parameters(tool, {p['name']: p['value'] for p in command['parameters']})
                    task = REGISTRY[tool].task; methods = list(REGISTRY[tool].methods)
                    with self.audit.call('autonomous_local_tool_generation', dict(tool=tool, parameters=cfg), methods):
                        candidate, info = self.box.propose(tool, self.current, [uid], cfg)
                elif source == 'use_pool':
                    row = next((r for r in available(pool) if r['candidate_id'] == command['candidate_id'] and r['target_id'] == uid), None)
                    if row is None:
                        raise ValueError('Unknown passing pool candidate')
                    f = row['candidate_file']
                    if sha(f['path']) != f['sha256']:
                        raise ValueError('Changed frozen pool candidate')
                    with self.audit.call('autonomous_pool_retrieval', dict(candidate_id=row['candidate_id']), ['frozen_candidate_retrieval']):
                        target = next(p for p in read_items(f['path']) if p.identifier == uid)
                        candidate = [target if p.identifier == uid else p for p in self.current]
                    task = 'realign' if any(REGISTRY[a['tool']].task == 'realign' for a in row['chain']) else 'simplify'
                    methods = ['frozen_candidate_retrieval']
                else:
                    raise ValueError('Not a proposal command')
                key = (base, uid, task, next(p for p in candidate if p.identifier == uid).geometry.normalize().wkb_hex)
                if key in self.tried:
                    raise ValueError('Repeated same-state geometry; change the proposal')
                self.tried.add(key)
                with self.audit.call('autonomous_typed_verifier', dict(task=task), ['common_scene_safety', 'typed_action_gate', 'cumulative_origin_guard']):
                    verdict = self.verify(candidate, [uid], task)
            except (ValueError, KeyError, TypeError, IndexError) as error:
                verdict = dict(accept=False, reasons=['proposal_invalid'], error=f'{type(error).__name__}: {error}', metrics={})
            except Exception as error:
                verdict = dict(accept=False, reasons=['proposal_error'], error=f'{type(error).__name__}: {error}', metrics={})
            path = save_items(folder/'candidate.pixel.geojson', candidate) if candidate is not None else None
            measurement = None
            if candidate is not None:
                target = next(p for p in candidate if p.identifier == uid)
                if target.geometry.is_valid and not target.geometry.is_empty:
                    measurement = features(self.box, lookup[uid], target)
            duplicate_pool = []
            if candidate is not None and pool:
                target = next(p for p in candidate if p.identifier == uid)
                for row in available(pool, [uid]):
                    f = row['candidate_file']
                    if sha(f['path']) != f['sha256']:
                        raise ValueError('Changed pool file during novelty check')
                    other = next(p for p in read_items(f['path']) if p.identifier == uid)
                    if target.geometry.equals_exact(other.geometry, 1e-8, normalize=True):
                        duplicate_pool.append(row['candidate_id'])
            record = dict(proposal_id=pid, base_state=base, target_id=uid, task=task,
                source=source, author=self.author, command=command, actual_methods=methods, candidate_file=path,
                verdict=verdict, features=measurement, duplicate_raw_pool=duplicate_pool,
                novel_relative_to_raw_pool=(source == 'direct_edit' and candidate is not None and not duplicate_pool) if pool is not None else None,
                novelty_note='Geometry novelty only, not an accuracy or global novelty claim',
                seconds=time.monotonic()-started, status='previewed')
            dump(folder/'result.json', record); self.pending_proposals[pid] = record
            event.update(proposal_id=pid, accept=verdict['accept'], reasons=verdict['reasons'])
        return record

    def commit_proposal(self, pid, *, expected_state, reason):
        if expected_state != self.state_id:
            raise ValueError('Stale commit state')
        if time.monotonic()-self.started >= self.config.seconds:
            raise RuntimeError('Scene time budget exhausted before commit')
        if pid not in self.pending_proposals:
            raise ValueError('Unknown proposal ID')
        proposal = self.pending_proposals[pid]
        if proposal['status'] != 'previewed' or proposal['base_state'] != self.state_id:
            raise ValueError('Proposal stale or already consumed')
        if not proposal['verdict']['accept'] or proposal['candidate_file'] is None:
            raise ValueError('Cannot commit rejected proposal')
        f = proposal['candidate_file']
        if sha(f['path']) != f['sha256']:
            raise ValueError('Proposal file changed')
        candidate = read_items(f['path']); uid = proposal['target_id']; before = scene_metrics(self.current)
        tid = f't{len(self.history):05d}'; started = time.monotonic()
        with self.audit.call('autonomous_commit', dict(proposal_id=pid, reason=reason), ['explicit_observed_feedback_commit']) as event:
            with self.audit.call('autonomous_commit_reverification', methods=['same_typed_action_gate']):
                verdict = self.verify(candidate, [uid], proposal['task'])
            if verdict['accept']:
                self.current = candidate
                if proposal['task'] in ('realign', 'legalize'):
                    target = next(p for p in candidate if p.identifier == uid)
                    self.simplify_anchors = [target if p.identifier == uid else p for p in self.simplify_anchors]
                save_items(self.root/'checkpoints'/(tid+'.pixel.geojson'), self.current)
            row = dict(trial_id=tid, proposal_id=pid, base_state=expected_state, output_state=self.state_id,
                tool=proposal['source'], target_ids=[uid], parameters=proposal['command'],
                source=self.author+'_autonomous', reason=reason, committed=verdict['accept'], verdict=verdict,
                candidate_file=f, frozen_candidate=f, before=before, after=scene_metrics(self.current),
                seconds=time.monotonic()-started)
            dump(self.root/'candidates'/tid/'result.json', row); self.history.append(row)
            proposal['status'] = 'committed' if verdict['accept'] else 'commit_rejected'
            event.update(committed=verdict['accept'], reasons=verdict['reasons'])
        return row


def shared_boundary_context(item, items):
    """Locate protected shared segments, including subsegments of long edges.

    This is geometric evidence, not a simplification or an acceptance bypass.
    The model still authors its keep indices/path and the common gate checks it.
    """
    neighbours=[]
    for other in items:
        if other.identifier == item.identifier:
            continue
        shared=item.geometry.boundary.intersection(other.geometry.boundary)
        if shared.length>1e-7:
            neighbours.append((other.identifier,shared))
    result=[]
    for ring in ring_records(item.geometry):
        coords=ring['coordinates'];edges=[];protected=set()
        for index,point in enumerate(coords):
            segment=LineString([point,coords[(index+1)%len(coords)]])
            matches=[dict(neighbour_id=uid,shared_length=float(segment.intersection(shared).length))
                for uid,shared in neighbours if segment.intersection(shared).length>1e-7]
            if matches:
                edges.append(dict(start_index=index,end_index=(index+1)%len(coords),neighbours=matches))
                protected.update([index,(index+1)%len(coords)])
        result.append(dict(ring_id=ring['ring_id'],shared_edges=edges,
            protected_endpoint_indices=sorted(protected),
            policy='For a local simplification retain these endpoints and their connecting original segments; simplify other arcs. Gates remain unchanged.'))
    return result


def build_context(e, pool, ids, allow_direct, evidence, direct_only=False):
    lookup = {p.identifier: p for p in e.current}
    pending = [dict(proposal_id=r['proposal_id'], target_id=r['target_id'], task=r['task'],
        base_state=r['base_state'], status=r['status'], verdict=r['verdict'], features=r['features'],
        source=r['source'], command=r['command'], duplicate_raw_pool=r['duplicate_raw_pool'])
        for r in e.pending_proposals.values() if r['target_id'] in ids][-24:]
    return dict(version=VERSION, state_id=e.state_id, image_shape=list(e.box.shape),
        goal_mode=getattr(e, 'goal_mode', 'balanced'),
        remaining_target_calls_including_current=getattr(e, 'remaining_target_calls', None),
        coordinate_system='Original-image pixel-cell coordinates [x,y], top-left origin; closing duplicate omitted',
        raw_input_state=__import__('cvr_agent').state_key(e.original), runtime_gt_access=False,
        allowed_target_ids=ids, allow_direct=allow_direct, direct_only=direct_only,
        instances=[dict(target_id=uid, rings=ring_records(lookup[uid].geometry),
            diagnostics=profile(e.box, lookup[uid], e.current),
            shared_boundary_protection=shared_boundary_context(lookup[uid], e.current)) for uid in ids],
        available_tools=catalog(),
        pool_candidates=[dict(candidate_id=r['candidate_id'], target_id=r['target_id'], chain=r['chain'], features=r['features'])
                         for r in available(pool, ids)] if pool else [],
        proposals=pending, feedback=e.history[-12:], evidence_ids=evidence,
        limits=asdict(e.config), protocol='One command per model response. Proposal must be visually observed in a later response before explicit commit; any commit invalidates other state-bound proposals.',
        direct_geometry_policy='Literal model indices/coordinates. No hidden simplification or legalization. Correct exterior rings only; no shared-boundary reconstruction in v1.')


def validate_packet(packet, context):
    if not isinstance(packet, dict) or set(packet) != set(SCHEMA['required']):
        raise ValueError('Invalid packet fields')
    if packet['state_id'] != context['state_id']:
        raise ValueError('Stale model state')
    if not isinstance(packet['diagnosis'], str) or not packet['diagnosis'].strip() or type(packet['done']) is not bool:
        raise ValueError('Missing diagnosis or invalid done flag')
    commands = packet['commands']
    if not isinstance(commands, list) or len(commands) > 1 or (not commands and not packet['done']):
        raise ValueError('One command per response, or done with no command')
    lookup = {p['target_id']: p for p in context['instances']}
    props = {p['proposal_id']: p for p in context['proposals']}
    for command in commands:
        kind = command.get('kind') if isinstance(command, dict) else None
        spec = next((s for s in COMMANDS if kind in s['properties']['kind']['enum']), None)
        if not spec or set(command) != set(spec['required']):
            raise ValueError('Unknown command or incorrect fields')
        uid = command['target_id']
        if uid not in lookup or not isinstance(command['reason'], str) or not command['reason'].strip():
            raise ValueError('Unknown target or missing reason')
        refs = command['evidence_refs']
        if not isinstance(refs, list) or not refs or any(not isinstance(r, str) or r not in context['evidence_ids'] for r in refs):
            raise ValueError('Must reference observed images')
        if kind == 'direct_edit':
            if not context['allow_direct']:
                raise ValueError('A1 mode forbids direct geometry')
            # Geometry syntax errors become proposal feedback, not API failures.
            if command['task'] not in ('simplify', 'realign') or not isinstance(command['edits'], list):
                raise ValueError('Invalid direct action type')
        elif kind == 'use_tool':
            if context.get('direct_only'):
                raise ValueError('Direct-only ablation forbids local tool proposals')
            rows = command['parameters']
            if not isinstance(rows, list) or any(not isinstance(r, dict) or set(r) != {'name', 'value'} for r in rows):
                raise ValueError('Invalid tool parameters')
            names = [r['name'] for r in rows]
            if any(not isinstance(n, str) for n in names) or len(set(names)) != len(names):
                raise ValueError('Duplicate parameter names')
            parameters(command['tool_id'], {r['name']: r['value'] for r in rows})
        elif kind == 'use_pool':
            if context.get('direct_only'):
                raise ValueError('Direct-only ablation forbids pool selection')
            if not any(r['candidate_id'] == command['candidate_id'] and r['target_id'] == uid for r in context['pool_candidates']):
                raise ValueError('Unknown pool candidate')
        elif kind in ('commit', 'discard'):
            proposal = props.get(command['proposal_id'])
            if not proposal or proposal['target_id'] != uid or proposal['status'] != 'previewed' or proposal['base_state'] != context['state_id']:
                raise ValueError('Unknown, consumed or stale proposal')
            if kind == 'commit' and not proposal['verdict']['accept']:
                raise ValueError('Cannot commit a rejected proposal')
        if packet['done'] and kind in ('direct_edit', 'use_tool', 'use_pool'):
            raise ValueError('Proposal requires a later observation; cannot finish with unseen preview')
    return packet


def image_bundle(e, ids, folder):
    h, w = e.box.shape; frames = [('scene', overlay(e.image, e.current, title='Current IDs', side=512), [0, 0, w, h])]
    lookup = {p.identifier: p for p in e.current}
    for uid in ids:
        item = lookup[uid]; x0, y0, x1, y1 = item.geometry.bounds
        bounds = (max(0, x0-8), max(0, y0-8), min(w, x1+8), min(h, y1+8))
        frames.append((f'{uid}_raw', overlay(e.image, [], bounds, uid+' raw', 512), bounds))
        numbered = overlay(e.image, [item], bounds, uid+' current vertex indices', 512)
        scale = min(512/(math.ceil(bounds[2])-int(bounds[0])), 512/(math.ceil(bounds[3])-int(bounds[1])))
        # Label up to 64 uniformly spaced indices per ring; ALL coordinates and
        # indices are present in context, and crops retain original coordinates.
        for ring in ring_records(item.geometry):
            coords = ring['coordinates']; stride = max(1, math.ceil(len(coords)/64))
            for i in range(0, len(coords), stride):
                x, y = coords[i]; px = round((x-int(bounds[0]))*scale); py = 32+round((y-int(bounds[1]))*scale)
                cv2.circle(numbered, (px, py), 2, (0, 230, 255), -1)
                cv2.putText(numbered, str(i), (px+2, py-2), cv2.FONT_HERSHEY_SIMPLEX, .32, (0, 230, 255), 1)
        frames.append((f'{uid}_indices', numbered, bounds))
        for proposal in list(e.pending_proposals.values())[-8:]:
            if proposal['target_id'] != uid or proposal['status'] != 'previewed' or not proposal['candidate_file']:
                continue
            f = proposal['candidate_file']
            if sha(f['path']) != f['sha256']:
                raise ValueError('Changed preview image geometry')
            candidate = next(p for p in read_items(f['path']) if p.identifier == uid)
            if not candidate.geometry.is_empty:
                frames.append((proposal['proposal_id']+'_preview', overlay(e.image, [candidate], bounds,
                    proposal['proposal_id']+' accept='+str(proposal['verdict']['accept']), 512), bounds))
    urls = []; mapping = []
    for number, (name, frame, bounds) in enumerate(frames):
        eid = f'image_{number:02d}'; cv2.putText(frame, eid, (max(1, frame.shape[1]-90), 20), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 0), 1)
        ok, blob = cv2.imencode('.png', frame)
        if not ok:
            raise ValueError('Failed image encoding')
        path = folder/(eid+'.png'); path.write_bytes(blob.tobytes())
        bx0, by0, bx1, by1 = int(bounds[0]), int(bounds[1]), math.ceil(bounds[2]), math.ceil(bounds[3])
        factor = min(512/(bx1-bx0), 512/(by1-by0))
        mapping.append(dict(evidence_id=eid, name=name, path=str(path), sha256=sha(path),
            original_bounds=[bx0, by0, bx1, by1], scale_xy=[round((bx1-bx0)*factor)/(bx1-bx0), round((by1-by0)*factor)/(by1-by0)],
            display_offset=[0,32], label_policy='At most 64 indices per ring rendered, all coordinates in context'))
        urls.append('data:image/png;base64,'+base64.b64encode(blob).decode('ascii'))
    dump(folder/'images.json', mapping)
    return urls, mapping


def request_packet(e, pool, ids, settings, number, allow_direct):
    folder = e.root/'gpt_calls'/f'{number:03d}'; folder.mkdir(parents=True)
    urls, mapping = image_bundle(e, ids, folder)
    context = build_context(e, pool, ids, allow_direct, [r['evidence_id'] for r in mapping], getattr(e, 'direct_only', False)); context['images'] = mapping
    dump(folder/'context.json', context)
    actual_prompt = PROMPT + ('\n\n'+MINIMUM_VERTICES_PROMPT if context['goal_mode'] == 'minimize_vertices' else '')
    (folder/'system_prompt.md').write_text(actual_prompt, encoding='utf-8')
    payload = APIPlanner(settings, schema=SCHEMA, system_prompt=actual_prompt)._payload(context, urls)
    dump(folder/'request.json', payload)
    endpoint = settings['base_url'].rstrip('/')+('/responses' if settings['api'] == 'responses' else '/chat/completions')
    metadata = dict(endpoint=endpoint, requested_model=settings['model'], request_sha256=sha(folder/'request.json'), retries=0)
    started = time.monotonic()
    try:
        with e.audit.call('autonomous_gpt_request', metadata, ['HTTP POST', 'visual_geometry_diagnosis_proposal_and_revision']) as event:
            key = os.environ.get(settings.get('api_key_env', 'OPENAI_API_KEY'))
            if not key:
                raise RuntimeError('Configured API key unavailable')
            remaining = e.config.seconds-(time.monotonic()-e.started)
            if remaining <= 0:
                raise TimeoutError('Scene budget exhausted before model request')
            req = urllib.request.Request(endpoint, data=json.dumps(payload, allow_nan=False).encode('utf-8'),
                headers={'Authorization': 'Bearer '+key, 'Content-Type': 'application/json'}, method='POST')
            with urllib.request.urlopen(req, timeout=min(settings.get('timeout_seconds', 90), remaining)) as response:
                data = json.load(response)
            dump(folder/'response.json', data)
            if settings['api'] == 'responses':
                raw = ''.join(p.get('text', '') for item in data.get('output', []) if item.get('type') == 'message' for p in item.get('content', []) if p.get('type') == 'output_text')
                complete = data.get('status') == 'completed'
            else:
                choice = data['choices'][0]; raw = choice['message']['content']; complete = choice.get('finish_reason') == 'stop'
            metadata.update(seconds=time.monotonic()-started, usage=data.get('usage', {}),
                response_id=data.get('id'), response_model=data.get('model'), completed=complete)
            if not complete:
                raise ValueError('Incomplete model response')
            packet = validate_packet(json.loads(raw), context)
            dump(folder/'decision.json', packet); event.update(metadata, decision=packet)
        dump(folder/'metadata.json', metadata)
        return packet, metadata
    except Exception as error:
        metadata.update(error=f'HTTP {error.code}' if isinstance(error, urllib.error.HTTPError) else type(error).__name__+': '+str(error), seconds=time.monotonic()-started)
        dump(folder/'error.json', metadata)
        raise RuntimeError(metadata['error']) from None


def run_autonomous(image, items, folder, config, settings, *, ids=None, pool=None,
                   max_calls=6, allow_direct=True, direct_only=False, planner=None, calls_per_target=None,
                   goal_mode='balanced'):
    e = AutonomousExecutor(image, items, folder, config)
    if goal_mode not in ('balanced', 'minimize_vertices'):
        raise ValueError('Unknown goal mode')
    e.goal_mode = goal_mode
    if direct_only and not allow_direct:
        raise ValueError('Incompatible capability switches')
    e.direct_only = direct_only
    ids = [p.identifier for p in items] if ids is None else list(ids)
    if not ids or len(ids) != len(set(ids)) or not set(ids) <= {p.identifier for p in items}:
        raise ValueError('Explicit distinct known target IDs required')
    if type(max_calls) is not int or max_calls < 1:
        raise ValueError('Positive call budget required')
    if calls_per_target is not None and (type(calls_per_target) is not int or calls_per_target < 1):
        raise ValueError('Positive per-target call cap required')
    # One target at a time limits image/coordinate ambiguity. A target can be
    # revisited in later calls; callers may select a small pilot or a full scene.
    target_index = 0; calls = 0; records = []; stop = 'model_call_budget'; error = None
    per_target_calls = {uid:0 for uid in ids}; unfinished = []
    planner = request_packet if planner is None else planner
    while calls < max_calls and target_index < len(ids):
        if time.monotonic()-e.started >= config.seconds or e.preview_attempts >= config.budget:
            stop = 'time_or_candidate_budget'; break
        current_ids = [ids[target_index]]
        if calls_per_target is not None and per_target_calls[current_ids[0]] >= calls_per_target:
            unfinished.append(current_ids[0]); target_index += 1; continue
        calls += 1; per_target_calls[current_ids[0]] += 1
        e.remaining_target_calls = min(max_calls-calls+1,
            calls_per_target-per_target_calls[current_ids[0]]+1 if calls_per_target is not None else max_calls-calls+1)
        try:
            packet, meta = planner(e, pool, current_ids, settings, calls, allow_direct)
            records.append(meta)
        except RuntimeError as exc:
            error = str(exc); stop = 'external_api_or_response_error'; break
        # A paid response can arrive after the scene deadline. Keep its usage
        # and response record, but export the last committed state unchanged.
        if time.monotonic()-e.started >= config.seconds:
            stop = 'time_or_candidate_budget'; break
        # Also validate injected planners used by offline behavioural tests.
        observed = build_context(e, pool, current_ids, allow_direct,
            [f'image_{i:02d}' for i in range(32)], direct_only)
        validate_packet(packet, observed)
        for command in packet['commands']:
            with e.audit.call('autonomous_command', dict(state_id=packet['state_id'], command=command), ['structured_action_dispatch']) as event:
                kind = command['kind']
                if kind in ('direct_edit', 'use_tool', 'use_pool'):
                    proposal = e.propose(command, expected_state=packet['state_id'], pool=pool)
                    event.update(proposal_id=proposal['proposal_id'], verdict=proposal['verdict'])
                elif kind == 'commit':
                    result = e.commit_proposal(command['proposal_id'], expected_state=packet['state_id'], reason=command['reason'])
                    event.update(committed=result['committed'])
                elif kind == 'discard':
                    e.pending_proposals[command['proposal_id']]['status'] = 'discarded'
                else:
                    event.update(kept=True, review=kind == 'review')
                    target_index += 1
            e.decisions.append(dict(call=calls, diagnosis=packet['diagnosis'], command=command, state_after=e.state_id))
        if packet['done']:
            # done ends the current target, not silently all remaining targets.
            if target_index < len(ids) and ids[target_index] == current_ids[0]:
                target_index += 1
        dump(e.root/'progress.json', dict(model_calls=calls, target_ids=ids,
            processed_targets=target_index-len(unfinished), last_target=current_ids[0],
            current=scene_metrics(e.current), commits=sum(h['committed'] for h in e.history),
            per_target_calls=per_target_calls, unfinished_target_ids=unfinished+ids[target_index:]))
    if target_index == len(ids):
        stop = 'some_targets_hit_call_budget' if unfinished else 'all_targets_decided'
    trace = e.finish(stop)
    direct = [p for p in e.pending_proposals.values() if p['source'] == 'direct_edit']
    trace.update(version=VERSION, model_calls=calls, model_records=records, gpt_error=error, goal_mode=goal_mode,
        mode='Direct-only ablation' if direct_only else ('A2 direct geometry + local tools' if allow_direct else 'A1 local tools only, same observation/commit protocol'),
        target_ids=ids, processed_targets=target_index-len(unfinished), decisions=e.decisions,
        per_target_calls=per_target_calls, unfinished_target_ids=unfinished+ids[target_index:],
        proposals=list(e.pending_proposals.values()), preview_attempts=e.preview_attempts,
        direct_proposals=len(direct), direct_validated=sum(p['verdict']['accept'] for p in direct),
        direct_committed=sum(p['status'] == 'committed' for p in direct),
        novel_direct_committed=(sum(p['status'] == 'committed' and p['novel_relative_to_raw_pool'] is True for p in direct) if pool is not None else None),
        candidates=e.preview_attempts, commit_checks=len(e.history),
        raw_pool_provided=pool is not None, runtime_gt_access=False,
        limitations=['Literal geometry proposals are not semantic correctness guarantees',
            'Shared-edge joint movement absent; protected by common gates',
            'No network discovery/code execution', 'Independent evaluation and matched deterministic construction control needed'])
    dump(e.root/'trace.json', trace)
    contract = json.loads((e.root/'tool_catalog.json').read_text(encoding='utf-8'))
    contract.update(model_calls=calls, direct_schema=SCHEMA, allow_direct=allow_direct, version=VERSION)
    dump(e.root/'tool_catalog.json', contract)
    return trace


def run_deterministic_geometry(image, items, folder, config, *, ids, pool=None):
    """Same explicit edit representation, deterministic generator control.

    This control exposes the existing DP and image-correction constructors in
    the SAME retain/path representation, not a new optimal direct-edit solver.
    Full generator calls and proposal checks are charged and logged.
    """
    e = AutonomousExecutor(image, items, folder, config)
    e.author = 'deterministic_geometry'
    for uid in ids:
        item = next(p for p in e.current if p.identifier == uid)
        proposals = []
        specs = [('contour_dp', dict(tolerance=t)) for t in (.5, 1., 2.)]
        specs += [('boundary_line_realign', dict(radius=r)) for r in (1.5, 3.)]
        for tool, cfg in specs:
            with e.audit.call('deterministic_explicit_geometry_generation', dict(tool=tool, parameters=cfg),
                    list(REGISTRY[tool].methods)+['same_representation_geometry_encoding']):
                candidate, info = e.box.propose(tool, e.current, [uid], cfg)
                target = next(p for p in candidate if p.identifier == uid)
                old_rings = {r['ring_id']: r['coordinates'] for r in ring_records(item.geometry)}
                new_rings = {r['ring_id']: r['coordinates'] for r in ring_records(target.geometry)}
                edits = []
                for rid, coords in old_rings.items():
                    result = new_rings.get(rid)
                    if result is None or result == coords:
                        continue
                    if tool == 'contour_dp':
                        lookup = {tuple(p): k for k, p in enumerate(coords)}
                        if not all(tuple(p) in lookup for p in result) or len(result) >= len(coords):
                            continue
                        edits.append(dict(kind='retain_vertices', ring_id=rid,
                            keep_indices=[lookup[tuple(p)] for p in result]))
                    elif rid.endswith('/outer') and len(result) <= 256:
                        edits.append(dict(kind='replace_boundary_arc', ring_id=rid,
                            start_index=0, end_index=len(coords)-1, path=result))
                if not edits:
                    continue
                command = dict(kind='direct_edit', target_id=uid, task=REGISTRY[tool].task,
                    reason=f'Deterministic {tool} encoded as explicit model-compatible geometry',
                    evidence_refs=['runtime_image'], edits=edits)
                p = e.propose(command, expected_state=e.state_id, pool=pool)
                p['generator'] = dict(tool=tool, parameters=cfg, methods=list(REGISTRY[tool].methods))
                if p['verdict']['accept']:
                    f = p['features']
                    utility = .65*f['reduction']+2.5*f['support_gain']-.015*f['shift']-.15*f['area_change']
                    proposals.append((utility, p))
        if proposals:
            value, best = max(proposals, key=lambda p: (p[0], p[1]['proposal_id']))
            if value > 0:
                result = e.commit_proposal(best['proposal_id'], expected_state=e.state_id,
                    reason='Deterministic positive proxy maximum, same explicit geometry interface')
    trace = e.finish('deterministic_candidates_complete')
    trace.update(version=VERSION, mode='Deterministic same-representation control',
        model_calls=0, model_records=[], gpt_error=None, target_ids=list(ids),
        processed_targets=len(ids), proposals=list(e.pending_proposals.values()),
        preview_attempts=e.preview_attempts, candidates=e.preview_attempts, commit_checks=len(e.history),
        direct_proposals=0, direct_validated=0, direct_committed=0, novel_direct_committed=0,
        deterministic_explicit_proposals=len(e.pending_proposals), runtime_gt_access=False,
        limitations=['Existing DP/straight-correction generator encoded in explicit geometry representation',
            'Not a complete deterministic search of all possible model paths',
            'Shared-edge joint movement absent', 'Development data only'])
    dump(e.root/'trace.json', trace)
    return trace
