"""Rebuild explanations.csv (pre-registered 30, in verification order) and
explanations_heart100.csv (additional analysis) from explanations/*.json.

    uv run python experiments/10seed/build_explanations_csv.py [--dir explanations_v4 --suffix _v4]
--dir reads the records from another folder; --suffix writes <name><suffix>.csv (defaults unchanged)."""
import csv, datetime as dt, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
_a = sys.argv[1:]
REC_DIR = _a[_a.index('--dir') + 1] if '--dir' in _a else 'explanations'
SUFFIX = _a[_a.index('--suffix') + 1] if '--suffix' in _a else ''
PRE = ['adult-s0','heart-s5','monk2-s2','heart-s0','heart-s1','monk2-s3','monk2-s4','adult-s1','monk2-s5','adult-s2',
       'monk2-s9','heart-s6','adult-s7','monk2-s7','monk2-s8','monk2-s0','adult-s4','heart-s9','heart-s4','heart-s2',
       'adult-s3','adult-s9','adult-s5','heart-s8','heart-s7','monk2-s6','heart-s3','adult-s8','adult-s6','monk2-s1']
H100 = [f'heart100-s{s}' for s in (5, 0, 1, 6, 9, 4, 2, 8, 7, 3)]
C100 = [f'churn100-s{s}' for s in (4, 1, 0, 7, 8, 2, 9, 3, 6, 5)]
C200 = [f'churn200-s{s}' for s in (1, 2, 4, 7, 9, 5, 3, 8, 0, 6)]

def mike_verified(r):
    mv = r.get('mike_verification') or {}
    return bool(mv.get('verified')) and mv.get('protocol') == 'signoff-v1'


def signoff_columns(r):
    """The signoff-v1 summary (explaneat review signoff --write); empty until a genome is signed off.
    signoff_wrong = sentences marked wrong (all fixed before the sign-off could pass);
    signoff_error_bound = one-sided 95% Clopper-Pearson upper bound on the wrong share in the
    sampled strata (signoff_sample_wrong of signoff_sample_n)."""
    so = (r.get('mike_verification') or {}).get('signoff') or {}
    c = so.get('counts') or {}
    return dict(signoff_protocol=so.get('protocol', ''), signoff_date=(so.get('checked_at') or '')[:10],
                signoff_confirmed=c.get('confirmed', ''),
                signoff_wrong=(c.get('wrong', 0) + so.get('wrong_fixed', 0)) if so else '',
                signoff_unsure=c.get('unsure', ''), signoff_sample_n=so.get('sample_n', ''),
                signoff_sample_wrong=so.get('sample_wrong', ''), signoff_error_bound=so.get('error_bound_95', ''))


LEVELS_DIR = os.environ.get('EXPLANEAT_LEVELS_DIR') or os.path.join(HERE, '..', 'explanation_levels', 'levels')
LEVELS_COLUMNS = ('faithful_max_err', 'n_leaves', 'granularity', 'understood_fraction', 'closed_form_only_fraction',
                  'understood_fraction_verifier')


def levels_columns(r, l):
    """The three levels of explanation (appended after every existing column; 2026-11-12).
    faithful_max_err and n_leaves come from the record itself; granularity (share of hidden
    units in a part at most 6 units wide, depth not penalised), understood_fraction (share of
    leaves whose concept probe grades meaningful) and closed_form_only_fraction come from
    experiments/explanation_levels/levels/<label>.json (compute_levels.py, read only) and are
    blank when that file is missing or was computed for another record set (its records_dir
    must be this run's --dir: the older records in explanations/ are other explanations of the
    same genomes). understood_fraction_verifier (2026-11-13, after the existing columns) is the
    blind verifier's share of leaves graded meaningful where a verdict exists, else blank."""
    out = dict(faithful_max_err=r.get('function_preservation_max_abs_err', ''), n_leaves=r.get('n_leaf_annotations', ''),
               granularity='', understood_fraction='', closed_form_only_fraction='', understood_fraction_verifier='')
    p = os.path.join(LEVELS_DIR, f'{l}.json')
    if os.path.exists(p):
        with open(p) as f:
            data = json.load(f)
        lv = data.get('levels') or {}
        if data.get('records_dir') != os.path.basename(REC_DIR) or data.get('genome_id') != r.get('genome_id'):
            lv = {}
        d, u = lv.get('decomposed') or {}, lv.get('understood') or {}
        out.update(granularity=d.get('granularity', ''), understood_fraction=u.get('understood_fraction', ''),
                   closed_form_only_fraction=u.get('closed_form_only_fraction', ''),
                   understood_fraction_verifier=u.get('understood_fraction_verifier', ''))
    return {k: ('' if v is None else v) for k, v in out.items()}


def row(i, l):
    r = json.load(open(os.path.join(HERE, REC_DIR, f'{l}.json')))
    v = r['verified_by_orchestrator']; oc = v['op_counts']
    mins = (dt.datetime.fromisoformat(r['end']) - dt.datetime.fromisoformat(r['start'])).total_seconds() / 60
    return dict(order=i + 1, label=l, dataset=l.rsplit('-', 1)[0], seed=int(l.rsplit('s', 1)[1]), genome_id=r['genome_id'],
                protocol=r.get('protocol_version', 'pilot' if r.get('pilot') else 'v2'),
                structural_coverage=v['structural_coverage'], compositional_coverage_tool=v['compositional_coverage'],
                hierarchy_complete=v['hierarchy']['complete'], full_coverage=v['full_coverage'], within_budget=r['within_budget'],
                n_leaf=r['n_leaf_annotations'], n_composition=r['n_composition_annotations'],
                n_splits=oc.get('split_node', 0), n_identity=oc.get('add_identity_node', 0), n_consolidate=oc.get('consolidate_node', 0),
                hidden_nodes=r['hidden_nodes'], annot_monosemanticity=round(r['hidden_nodes_single_leaf_no_split'] / r['hidden_nodes'], 3),
                hidden_in_any_annotation=r.get('hidden_nodes_in_any_annotation'),
                function_preservation_max_abs_err=v['function_preservation_max_abs_err'],
                closed_form_max_abs_err=v['closed_form_max_abs_err'], numeric_max_abs_err=v['numeric_max_abs_err'],
                claude_minutes=round(mins, 1),
                # Mike's verification lives in the JSON record (key "mike_verification":
                # {"verified": bool, "minutes": float, "changes": str, "protocol", "signoff"}) so rebuilds
                # keep it. PREREG amendment 2026-11-11: verified_by_mike = signoff-v1 passed.
                verified_by_mike='yes' if mike_verified(r) else 'no',
                mike_minutes=r.get('mike_verification', {}).get('minutes', ''),
                mike_changes=r.get('mike_verification', {}).get('changes', ''),
                pilot=r.get('pilot', False), rule_agreement=r.get('rule_agreement', ''), vs_40gen=r.get('vs_40gen', ''),
                resisted=r['resisted'],
                **signoff_columns(r),
                **levels_columns(r, l))

for name, labels in (('explanations.csv', PRE), ('explanations_heart100.csv', H100),
                     ('../churn/explanations_churn100.csv', C100), ('../churn/explanations_churn200.csv', C200)):
    rows = [row(i, l) for i, l in enumerate(labels) if os.path.exists(os.path.join(HERE, REC_DIR, f'{l}.json'))]
    if not rows:
        continue
    with open(os.path.join(HERE, name[:-4] + SUFFIX + '.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(name, len(rows), 'rows;', sum(r['full_coverage'] for r in rows), 'full coverage')
