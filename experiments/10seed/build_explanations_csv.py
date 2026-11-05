"""Rebuild explanations.csv (pre-registered 30, in verification order) and
explanations_heart100.csv (additional analysis) from explanations/*.json."""
import csv, datetime as dt, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
PRE = ['adult-s0','heart-s5','monk2-s2','heart-s0','heart-s1','monk2-s3','monk2-s4','adult-s1','monk2-s5','adult-s2',
       'monk2-s9','heart-s6','adult-s7','monk2-s7','monk2-s8','monk2-s0','adult-s4','heart-s9','heart-s4','heart-s2',
       'adult-s3','adult-s9','adult-s5','heart-s8','heart-s7','monk2-s6','heart-s3','adult-s8','adult-s6','monk2-s1']
H100 = [f'heart100-s{s}' for s in (5, 0, 1, 6, 9, 4, 2, 8, 7, 3)]
C100 = [f'churn100-s{s}' for s in (4, 1, 0, 7, 8, 2, 9, 3, 6, 5)]
C200 = [f'churn200-s{s}' for s in (1, 2, 4, 7, 9, 5, 3, 8, 0, 6)]

def row(i, l):
    r = json.load(open(os.path.join(HERE, 'explanations', f'{l}.json')))
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
                claude_minutes=round(mins, 1), verified_by_mike='no', mike_minutes='', mike_changes='',
                pilot=r.get('pilot', False), rule_agreement=r.get('rule_agreement', ''), vs_40gen=r.get('vs_40gen', ''),
                resisted=r['resisted'])

for name, labels in (('explanations.csv', PRE), ('explanations_heart100.csv', H100),
                     ('../churn/explanations_churn100.csv', C100), ('../churn/explanations_churn200.csv', C200)):
    rows = [row(i, l) for i, l in enumerate(labels) if os.path.exists(os.path.join(HERE, 'explanations', f'{l}.json'))]
    with open(os.path.join(HERE, name), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(name, len(rows), 'rows;', sum(r['full_coverage'] for r in rows), 'full coverage')
