"""Apply the 2026-11-11 errata. Text fields only; no structural operation is touched.

Run with --dry to check every replacement matches exactly the expected number of times."""
import copy, json, sys
from pathlib import Path

DRY = '--dry' in sys.argv
STAGE = [a for a in sys.argv[1:] if not a.startswith('--')]
REPO = Path.cwd()
REF = 'experiments/errata/2026-11-11/ERRATA.md'
DATE = '2026-11-11'
STRUCT_KEYS = {'subgraph_nodes', 'subgraph_connections', 'entry_nodes', 'exit_nodes', 'child_annotation_ids',
               'node_id', 'parent_annotation_id'}

# ---------------------------------------------------------------- DB operations
# (label, annotation name, path inside params, old, new, expected count)
DB = [
    ('heart-s5', 'ischaemia_pathway', 'hypothesis',
     '35 open from low BP alone, mean predicted risk 0.13)', '35 open from low BP alone, mean predicted risk 0.18)', 1),
    ('heart-s5', 'ischaemia_pathway', 'evidence.records[1].narrative',
     'these patients have low predicted risk (mean 0.13).', 'these patients have low predicted risk (mean 0.18).', 1),
    ('adult-s5', 'age_profile', 'hypothesis',
     'for Thai respondents (27 rows)', 'for Thai respondents (26 rows)', 1),
    ('adult-s5', 'age_profile', 'hypothesis',
     '2990=relu(1.446 age-0.48)', '2990=relu(1.446 age+0.137*2134-0.48)', 1),
    ('churn100-s4', 'area415_state47', 'hypothesis',
     'small positive offset (~0.15) for rows outside area code 415, larger (~+3) for state 47 rows outside area 415, zero in area 415.',
     'small positive offset (~0.11) for rows outside area code 415, larger (~+3) for state 47 rows outside area 415, '
     'zero in area 415 except for its 30 state 47 (Washington) rows (1.03).', 1),
    ('churn100-s4', 'area415_state47', 'hypothesis',
     'Zero in area 415 (2,016 rows); 0.16 for other areas (logit +0.03);',
     'Zero for 1,986 of the 2,016 area-415 rows, 1.03 (logit +0.21) for the 30 state 47 rows in area 415; '
     '0.11 for other areas (logit +0.02);', 1),
    ('churn100-s1', 'churnModelRoot', 'hypothesis',
     'international_plan +0.77/SD (+2.67 logit for plan holders), voice_mail_plan -0.86/SD (-2.37 for holders)',
     'international_plan +0.77/SD (+2.62 logit for plan holders), voice_mail_plan -0.86/SD (-1.96 for holders)', 1),
    ('monk2-s0', 'upper_edge_count_classifier', 'evidence.records[2].payload.positive_rate_by_a1_and_first_value_count.a1=3.all counts',
     '108/108 pos', '107/107 pos', 1),
]
ROOT_OF = {'churn100-s1': 'churnModelRoot', 'monk2-s0': None}

# ---------------------------------------------------------------- records
REC = [  # (label, study, field, old, new)
    ('churn100-s1', '10seed', 'summary', 'international plan +2.67, voice-mail plan -2.37',
     'international plan +2.62, voice-mail plan -1.96'),
    ('monk2-s0', '10seed', 'resisted', "(correct: 107/107; narrative corrected, payload not)",
     "(correct: 107/107; narrative corrected, payload corrected by the 2026-11-11 errata)"),
]

# ---------------------------------------------------------------- layers
# copies of the DB hypotheses inside the layer (parts[].original_hypothesis, hypothesis_revisions[].original)
LAYER_COPIES = [(l, old, new) for (l, _a, path, old, new, _n) in DB if path == 'hypothesis']
# drafted prose: (label, study, old, new); replaced in every drafted sentence (blocks, readdown, summary...)
PROSE = [
    ('churn100-s1', '10seed', 'from 112 to 249 minutes the chance goes from about 2% to about 5% without one',
     'from 112 to 249 minutes the chance goes from about 2% to about 4% without one'),
    ('churn100-s1', '10seed', 'then rises steadily from about 2% to about 5%.', 'then rises steadily from about 2% to about 4%.'),
    ('churn100-s1', '10seed', 'from about 2% at 112 to about 5% at 249 minutes', 'from about 2% at 112 to about 4% at 249 minutes'),
    ('adult-s5', '10seed', 'who work less than about 50 hours a week, 25290 of whom',
     'who work less than about 50 hours a week (no further effect above that), 25290 of whom'),
    ('adult-s5', '10seed', 'those working less than about 50 hours a week, 25290 of whom',
     'those working less than about 50 hours a week (no further effect above that), 25290 of whom'),
    ('heart100-s7', '10seed', 'It does so only with a normal resting ECG, nothing until about 168 beats per minute, then steadily',
     'It does so only with a normal resting ECG, nothing until about 168 beats per minute (between 160 and 176 beats '
     'per minute for different patients), then steadily'),
    ('churn100-s7', '10seed', 'takes a typical customer from about 1% to about 15%.', 'takes a typical customer from about 1% to about 14%.'),
    ('churn100-s7', '10seed', 'from about 1% to about 15% between 0 and 4 calls.', 'from about 1% to about 14% between 0 and 4 calls.'),
    ('heart100-s0', '10seed', 'the model predicts a risk of heart disease of about 20%;',
     'the model predicts a risk of heart disease of about 19%;'),
    ('heart-s5-c1', 'species_leads', 'a higher maximum heart rate lowers it steadily, from about 28% at 118 to about 18% at 175',
     'a higher maximum heart rate lowers it steadily, from about 27% at 118 to about 18% at 175'),
    ('heart-s5-c1', 'species_leads', 'from about 18% at 42 to about 27% at 66 years.', 'from about 18% at 42 to about 26% at 66 years.'),
    ('heart-s2', '10seed',
     'then no further for most patients, but for patients with a reversible defect on the thallium scan it rises in a jump: '
     'for a typical patient from about 12% at 42 to about 28% at 66 years.',
     'then no further: for a typical patient from about 12% at 42 to about 28% at 66 years, but for a typical patient with a '
     'reversible defect on the thallium scan this part is off at every age, so age changes nothing through it.'),
]
# citations: (label, study, text fragment identifying the sentence, claim to add, claims it must already have)
CITE = [
    ('adult-s5', '10seed', 'who work less than about 50 hours a week', 'C142', ['C143']),
    ('adult-s5', '10seed', 'those working less than about 50 hours a week', 'C142', ['C143']),
    ('heart100-s7', '10seed', 'nothing until about 168 beats per minute', 'C83', ['C82']),
    ('heart-s2', '10seed', 'this part is off at every age, so age changes nothing through it', 'C33', ['C30', 'C37']),
]
EXPECT_PROSE = {  # number of drafted sentences each PROSE entry must change (blocks + readdown + summary copies)
}

ERRATA = {  # label -> list of {location, was, now}
    'heart-s5': [
        {'location': 'operations.ischaemia_pathway.hypothesis (DB; layer parts[].original_hypothesis copy)',
         'was': '35 open from low BP alone, mean predicted risk 0.13', 'now': 'mean predicted risk 0.18'},
        {'location': 'operations.ischaemia_pathway.evidence[1].narrative (DB)',
         'was': 'these patients have low predicted risk (mean 0.13)', 'now': '(mean 0.18)'}],
    'adult-s5': [
        {'location': 'operations.age_profile.hypothesis (DB; layer copy)', 'was': 'for Thai respondents (27 rows)', 'now': '(26 rows)'},
        {'location': 'operations.age_profile.hypothesis (DB; layer copy)', 'was': '2990=relu(1.446 age-0.48)',
         'now': '2990=relu(1.446 age+0.137*2134-0.48) (the Thailand indicator 2134 also feeds 2990)'},
        {'location': 'drafted.blocks[3][0], drafted.readdown.ann_125.evidence[0], drafted.summary[11] (claims)',
         'was': "'about 50 hours a week' cited only C143", 'now': "cites C143 and C142 (the 50-hour threshold claim); "
         "'(no further effect above that)' added, the shape phrase the lint requires for C142"}],
    'churn100-s4': [
        {'location': 'operations.area415_state47.hypothesis (DB; layer copy)',
         'was': "zero in area 415; 'Zero in area 415 (2,016 rows)'; offset ~0.15 / 0.16 (logit +0.03) for other areas",
         'now': 'zero in area 415 except its 30 Washington (state 47) rows, where 595 = 1.03 (logit +0.21); 0.11 (logit +0.02) for other areas'}],
    'churn100-s1': [
        {'location': 'operations.churnModelRoot.hypothesis (DB; layer copy) and record.summary',
         'was': 'voice-mail plan -2.37 logit for holders; international plan +2.67', 'now': '-1.96; +2.62'},
        {'location': 'drafted.blocks[2][2], drafted.readdown.ann_79.evidence[2], drafted.readdown.ann_83.evidence[2], drafted.summary[11]',
         'was': 'about 5% (cited value 4.45%, rounded twice)', 'now': 'about 4%'}],
    'monk2-s0': [
        {'location': 'operations.<root>.evidence[2].payload.positive_rate_by_a1_and_first_value_count (DB) and record.resisted',
         'was': "a1=3: '108/108 pos'", 'now': "'107/107 pos'"}],
    'heart-s2': [
        {'location': 'drafted.blocks[1][2], drafted.readdown.ann_31.evidence[1], drafted.summary[7]',
         'was': "'for patients with a reversible defect on the thallium scan it rises in a jump: for a typical patient from about 12% ... to about 28%'",
         'now': 'the 12% to 28% rise is for the typical patient (normal scan); for a typical patient with a reversible defect the part is off at every age (now also cites C33)'}],
    'churn100-s7': [
        {'location': 'drafted.blocks[0][2], drafted.readdown.ann_80.evidence[2] (and summary copy)',
         'was': 'about 15% (cited value 14.48%, rounded twice)', 'now': 'about 14%'}],
    'heart100-s0': [
        {'location': 'drafted.lead[0] (and summary copy)', 'was': 'about 20% (cited value 19.3%, rounded twice)', 'now': 'about 19%'}],
    'heart100-s7': [
        {'location': 'drafted.blocks[2][1], drafted.readdown.ann_38.evidence[2] (and summary copy) (claims)',
         'was': "'about 168 beats per minute' cited only C82", 'now': "cites C82 and C83 (the 168 bpm threshold claim), with C83's spread phrase "
         "'(between 160 and 176 beats per minute for different patients)' that the lint requires"}],
    'heart-s5-c1': [
        {'location': 'drafted.blocks[0][0] (and summary copy)', 'was': 'about 28% at 118 (cited value 27.50%, rounded twice)', 'now': 'about 27%'},
        {'location': 'drafted.readdown.ann_35.evidence[3]', 'was': 'about 27% at 66 years (cited value 26.45%, rounded twice)', 'now': 'about 26%'}],
}

STUDY = {'heart-s5-c1': 'species_leads'}


def rec_path(label):
    return REPO / f"experiments/{STUDY.get(label, '10seed')}/explanations/{label}.json"


def layer_dir(label):
    return REPO / f"experiments/{STUDY.get(label, '10seed')}/plain_language"


def get_path(obj, path):
    import re
    toks = re.findall(r'[^.\[\]]+|\[\d+\]', path) if '=' not in path else None
    if toks is None:   # keys with '=' or spaces: split on the known structure
        parts = path.split('.')
        out = []
        for p in parts:
            m = re.match(r'^(.*)\[(\d+)\]$', p)
            if m:
                out += [m.group(1), f'[{m.group(2)}]']
            else:
                out.append(p)
        toks = out
    cur = obj
    for t in toks[:-1]:
        cur = cur[int(t[1:-1])] if t.startswith('[') else cur[t]
    last = toks[-1]
    return cur, (int(last[1:-1]) if last.startswith('[') else last)


def db_stage():
    from sqlalchemy.orm.attributes import flag_modified
    from mcp_server.server import get_db
    from explaneat.db.explanation_lock import lock_explanation
    db = get_db()
    by_label = {}
    for row in DB:
        by_label.setdefault(row[0], []).append(row)
    for label, rows in by_label.items():
        gid = json.loads(rec_path(label).read_text())['genome_id']
        with db.session_scope() as s:
            ex = lock_explanation(s, gid)
            ops = copy.deepcopy(list(ex.operations))
            before = json.dumps(ops, sort_keys=True)
            for (_l, name, path, old, new, n) in rows:
                anns = [op for op in ops if op.get('type') == 'annotate' and
                        (op['params'].get('name') == name if name else not op['params'].get('parent_annotation_id')
                         and old in json.dumps(op['params']))]
                anns = [op for op in anns if old in json.dumps(op['params'])] if name is None else anns
                assert len(anns) == 1, (label, name, len(anns))
                container, key = get_path(anns[0]['params'], path)
                val = container[key]
                assert isinstance(val, str) and val.count(old) == n, (label, path, val.count(old) if isinstance(val, str) else type(val))
                container[key] = val.replace(old, new)
            # only text changed: the structural fields are identical
            def struct(o):
                return [(op.get('type'), {k: v for k, v in (op.get('params') or {}).items() if k in STRUCT_KEYS})
                        for op in o]
            assert struct(ops) == struct(json.loads(before)), label
            print(f'{label}: DB {len(rows)} text edit(s)' + (' (dry run)' if DRY else ''))
            if not DRY:
                ex.operations = ops
                flag_modified(ex, 'operations')
            else:
                s.rollback()


def walk_sentences(o):
    if isinstance(o, dict):
        if 'text' in o and isinstance(o.get('text'), str):
            yield o
        for v in o.values():
            yield from walk_sentences(v)
    elif isinstance(o, list):
        for v in o:
            yield from walk_sentences(v)


def layer_stage():
    from explaneat.analysis import plain_language_store as pls
    labels = sorted({r[0] for r in LAYER_COPIES} | {r[0] for r in PROSE} | {r[0] for r in CITE})
    for label in labels:
        d = layer_dir(label)
        layer = pls.load_layer(label, d)
        n = 0
        for (l, old, new) in LAYER_COPIES:
            if l != label:
                continue
            k = 0
            for p in layer['parts']:
                if old in (p.get('original_hypothesis') or ''):
                    p['original_hypothesis'] = p['original_hypothesis'].replace(old, new); k += 1
            for h in (layer.get('drafted') or {}).get('hypothesis_revisions', {}).values():
                if old in (h.get('original') or ''):
                    h['original'] = h['original'].replace(old, new); k += 1
            assert k >= 1, (label, old)
            n += k
        for (l, _st, old, new) in PROSE:
            if l != label:
                continue
            k = 0
            for s in walk_sentences(layer['drafted']):
                if old in s['text']:
                    s['text'] = s['text'].replace(old, new); k += 1
            assert k >= 1, (label, old)
            print(f'  {label}: prose {k} sentence(s): {old[:60]!r}')
            n += k
        for (l, _st, frag, add, must) in CITE:
            if l != label:
                continue
            k = 0
            for s in walk_sentences(layer['drafted']):
                if frag in s['text'] and isinstance(s.get('claims'), list) and all(c in s['claims'] for c in must):
                    if add not in s['claims']:
                        s['claims'].append(add); k += 1
            assert k >= 1, (label, frag)
            print(f'  {label}: citation +{add} in {k} sentence(s)')
            n += k
        errs = pls.lint_draft(layer, pls.draft_input(layer['drafted']))
        print(f'{label}: {n} layer edit(s); lint: {"ok" if not errs else errs}')
        assert not errs, errs
        if not DRY:
            pls.save_layer(layer, d)


def record_stage():
    for label, items in ERRATA.items():
        p = rec_path(label)
        rec = json.loads(p.read_text())
        mv = copy.deepcopy(rec.get('mike_verification'))
        for (l, _st, field, old, new) in REC:
            if l == label:
                assert rec[field].count(old) == 1, (label, field)
                rec[field] = rec[field].replace(old, new)
        rec['errata'] = list(rec.get('errata') or []) + [
            {'date': DATE, **it, 'ref': REF} for it in items]
        assert rec.get('mike_verification') == mv
        print(f'{label}: record errata +{len(items)}')
        if not DRY:
            p.write_text(json.dumps(rec, indent=2))


if __name__ == '__main__':
    for st in (STAGE or ['layer', 'record', 'db']):
        {'db': db_stage, 'layer': layer_stage, 'record': record_stage}[st]()
