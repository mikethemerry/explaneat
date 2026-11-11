"""Read-only independent checks using the frozen target and verification APIs."""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from explaneat.verify.artefacts import resolve
from explaneat.verify.model import load
from explaneat.verify.probe import ablate, probe, query_mask, subgroup

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--target', type=Path, default=Path(__file__).resolve().parent.parent / 'item')
parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent)
args = parser.parse_args()
TARGET = args.target
OUT = args.out
OUT.mkdir(exist_ok=True)
target = resolve(str(TARGET))
vm = load(target.genome_id, target.operations, cache=False)
queries = {q['id']: q for q in json.loads((TARGET / 'queries.json').read_text())}
result = {'started_at': datetime.now().astimezone().isoformat(), 'parts': {}, 'probes': {}}
for i in range(1, 11):
    name = f'positional_gate_{i}'
    group = subgroup(vm, f'node_{i} > 0')
    off = ablate(vm, name, to='zero')
    pid = vm.part(name).id
    result['parts'][name] = {'id': pid, 'n': group['subgroup']['n'],
                             'active_mean_contribution': group['subgroup']['parts'][pid]['mean_contribution'],
                             'ablation': off['data']}
    print('checked', name, flush=True)
result['parts']['direct_square_values'] = ablate(vm, 'direct_square_values', to='zero')
for qid in ['q_81c0264d79', 'q_ba499e9531', 'q_9289b16fa8', 'q_214151afce',
            'q_2c67ef71a7', 'q_4c308cc6d9', 'q_9749a2307d', 'q_0ce0ed2acf',
            'q_e1d08e0127', 'q_a991f7e6af', 'q_9ee81a792f']:
    args = queries[qid]['arguments']
    result['probes'][qid] = probe(vm, hold=args['hold'], sweeps=json.loads(args['sweeps']))
for hold in ['row:1', 'row:17', 'row:100']:
    result['probes']['independent_' + hold] = probe(vm, hold=hold, sweeps=['c_c1=-1,0,1', 'c_d1=-1,0,1'])

# Independently enumerate all four-square lines, then distinguish an open line
# from an immediately playable completion square. Use only real training rows.
raw = vm.base.raw_train
names = vm.base.feature_names
col = {name: j for j, name in enumerate(names)}
lines = []
for c in range(7):
    for r in range(6):
        for dc, dr in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            pts = [(c + k * dc, r + k * dr) for k in range(4)]
            if all(0 <= cc < 7 and 0 <= rr < 6 for cc, rr in pts):
                lines.append(pts)
opened = np.zeros(len(raw), dtype=bool)
playable = np.zeros(len(raw), dtype=bool)
for pts in lines:
    indices = [col[f'c_{chr(97 + c)}{r + 1}'] for c, r in pts]
    values = raw[:, indices]
    is_open = (np.sum(values == -1, axis=1) == 3) & (np.sum(values == 0, axis=1) == 1)
    opened |= is_open
    for k, (c, r) in enumerate(pts):
        supported = True if r == 0 else np.all(raw[:, [col[f'c_{chr(97 + c)}{rr + 1}'] for rr in range(r)]] != 0, axis=1)
        playable |= is_open & (values[:, k] == 0) & supported
stated_mask = query_mask(vm, queries['q_b16d796df5']['arguments']['where'])
risk = vm.m.risk(vm.m.logit(vm.m.X))
result['threats'] = {'lines': len(lines), 'open_line_n': int(opened.sum()),
                      'playable_threat_n': int(playable.sum()),
                      'stated_n': int(stated_mask.sum()),
                      'stated_matches_open': bool(np.array_equal(stated_mask, opened)),
                      'stated_matches_playable': bool(np.array_equal(stated_mask, playable))}
for label, mask in [('open', opened), ('playable', playable), ('stated', stated_mask)]:
    result['threats'][label] = {'n': int(mask.sum()),
                                'outcome_rate': float(vm.m.y[mask].mean()) if mask.any() else None,
                                'mean_predicted_probability': float(risk[mask].mean()) if mask.any() else None}
result['finished_at'] = datetime.now().astimezone().isoformat()
(OUT / 'checks.json').write_text(json.dumps(result, indent=2, default=str) + '\n')
print(json.dumps(result['threats'], indent=2), flush=True)
