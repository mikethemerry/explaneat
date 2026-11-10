"""Apply the v2 errata of 2026-11-11 (E8/E9 relabelled, E13 withdrawn, E16 added). Text fields only.

    uv run python experiments/errata/2026-11-11-v2/apply_errata_v2.py --dry   # check every match
    uv run python experiments/errata/2026-11-11-v2/apply_errata_v2.py         # apply

The DB edits run under the explanation row lock (``lock_explanation``) and are refused unless the
structural fields of every operation are unchanged.
"""
import copy
import json
import sys
from pathlib import Path

DRY = '--dry' in sys.argv
REPO = Path(__file__).resolve().parents[3]
REF = 'experiments/errata/2026-11-11/ERRATA.md (v2)'
DATE = '2026-11-11'
STRUCT_KEYS = {'subgraph_nodes', 'subgraph_connections', 'entry_nodes', 'exit_nodes', 'child_annotation_ids',
               'node_id', 'parent_annotation_id'}

E16_OLD = 'oldpeak and thal:7 influence the output ONLY through these gates (oldpeak has no direct edge).'
E16_NEW = ('oldpeak influences the output ONLY through these gates (it has no direct edge); thal:7 acts mainly '
           'through them (raising the risk) but also has a small direct edge in the linear core (-0.07), which '
           'lowers it and is its only effect where the gates are off.')
E89_OLD = ('international_plan +0.77/SD (+2.62 logit for plan holders), voice_mail_plan -0.86/SD (-1.96 for holders)')
E89_NEW = ('international_plan +0.77/SD (direct term +2.62 logit for plan holders; average whole-model effect '
           'for holders +2.67), voice_mail_plan -0.86/SD (direct term -1.96 for holders; average whole-model '
           'effect for holders -2.37)')

# (label, annotation name, field, old, new): DB hypothesis edits, mirrored in the layer's copies
DB = [
    ('heart-s5', 'heart_disease_logit', 'hypothesis', E16_OLD, E16_NEW),
    ('churn100-s1', 'churnModelRoot', 'hypothesis', E89_OLD, E89_NEW),
]
REC = [  # (label, field, old, new)
    ('churn100-s1', 'summary', '(3) linear terms (international plan +2.62, voice-mail plan -1.96, per-state offsets)',
     '(3) linear terms (international plan: direct +2.62, average whole-model effect for holders +2.67; '
     'voice-mail plan: direct -1.96, average whole-model effect for holders -2.37; per-state offsets)'),
]
PROSE = [  # (label, old, new, expected sentences)
    ('heart100-s0', 'the model predicts a risk of heart disease of about 19%;',
     'the model predicts a risk of heart disease of about 20%;', 2),
]
ERRATA = {
    'heart-s5': [{'location': 'operations.heart_disease_logit.hypothesis (DB; layer parts[].original_hypothesis copy)',
                  'was': E16_OLD, 'now': E16_NEW, 'id': 'E16'}],
    'churn100-s1': [{'location': 'operations.churnModelRoot.hypothesis (DB; layer copy) and record.summary',
                     'was': 'E8/E9 correction gave only the direct terms (-1.96, +2.62); the original -2.37 and +2.67 '
                            'were right numbers with the wrong label (whole-model averages for holders, not the linear term)',
                     'now': 'both stated and labelled: direct term -1.96 / +2.62; average whole-model effect for holders '
                            '-2.37 / +2.67', 'id': 'E8, E9 (relabelled)'}],
    'heart100-s0': [{'location': 'drafted.lead[0] (and summary copy)',
                     'was': 'about 19% (the E13 correction)',
                     'now': 'about 20% restored: E13 withdrawn, not an error (lint false positive: the typical '
                            "patient's risk is 19.80%; the 0.193 the lint matched is a category share in C55)",
                     'id': 'E13 (withdrawn)'}],
}


def rec_path(label):
    return REPO / f'experiments/10seed/explanations/{label}.json'


def struct(ops):
    return [(op.get('type'), {k: v for k, v in (op.get('params') or {}).items() if k in STRUCT_KEYS}) for op in ops]


def db_stage():
    from sqlalchemy.orm.attributes import flag_modified
    from mcp_server.server import get_db
    from explaneat.db.explanation_lock import lock_explanation
    db = get_db()
    for (label, name, field, old, new) in DB:
        gid = json.loads(rec_path(label).read_text())['genome_id']
        with db.session_scope() as s:
            ex = lock_explanation(s, gid)
            ops = copy.deepcopy(list(ex.operations))
            before = copy.deepcopy(ops)
            anns = [op for op in ops if op.get('type') == 'annotate' and op['params'].get('name') == name]
            assert len(anns) == 1, (label, name)
            val = anns[0]['params'][field]
            assert val.count(old) == 1, (label, name, val.count(old))
            anns[0]['params'][field] = val.replace(old, new)
            assert struct(ops) == struct(before), label
            print(f'{label}: DB {name}.{field} edited' + (' (dry run)' if DRY else ''))
            if DRY:
                s.rollback()
            else:
                ex.operations = ops
                flag_modified(ex, 'operations')


def walk(o):
    if isinstance(o, dict):
        if isinstance(o.get('text'), str):
            yield o
        for v in o.values():
            yield from walk(v)
    elif isinstance(o, list):
        for v in o:
            yield from walk(v)


def layer_stage():
    from explaneat.analysis import plain_language_store as pls
    d = REPO / 'experiments/10seed/plain_language'
    for label in sorted({r[0] for r in DB} | {r[0] for r in PROSE}):
        layer = pls.load_layer(label, d)
        n = 0
        for (l, _name, _f, old, new) in DB:
            if l != label:
                continue
            for p in layer['parts']:
                if old in (p.get('original_hypothesis') or ''):
                    p['original_hypothesis'] = p['original_hypothesis'].replace(old, new); n += 1
            for h in (layer.get('drafted') or {}).get('hypothesis_revisions', {}).values():
                if old in (h.get('original') or ''):
                    h['original'] = h['original'].replace(old, new); n += 1
        for (l, old, new, k) in PROSE:
            if l != label:
                continue
            hits = [s for s in walk(layer['drafted']) if old in s['text']]
            assert len(hits) == k, (label, len(hits))
            for s in hits:
                s['text'] = s['text'].replace(old, new); n += 1
        assert n >= 1, label
        errs = pls.lint_draft(layer, pls.draft_input(layer['drafted']))
        print(f'{label}: {n} layer edit(s); lint: {"ok" if not errs else errs}')
        if not DRY:
            pls.save_layer(layer, d)


def record_stage():
    for label, items in ERRATA.items():
        p = rec_path(label)
        rec = json.loads(p.read_text())
        mv = copy.deepcopy(rec.get('mike_verification'))
        for (l, field, old, new) in REC:
            if l == label:
                assert rec[field].count(old) == 1, (label, field)
                rec[field] = rec[field].replace(old, new)
        rec['errata'] = list(rec.get('errata') or []) + [{'date': DATE, **it, 'ref': REF} for it in items]
        assert rec.get('mike_verification') == mv
        print(f'{label}: record errata +{len(items)}')
        if not DRY:
            p.write_text(json.dumps(rec, indent=2))


if __name__ == '__main__':
    stages = [a for a in sys.argv[1:] if not a.startswith('--')] or ['layer', 'record', 'db']
    for st in stages:
        {'db': db_stage, 'layer': layer_stage, 'record': record_stage}[st]()
