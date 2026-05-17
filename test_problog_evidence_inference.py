import os
from pathlib import Path

from problog import get_evaluatable
from problog.program import PrologString

from branch_schema import Branch, Condition
from problog_export import export_branches_to_problog_latent


def test_problog_evidence_changes_latent_posterior():
    branches = [
        Branch(
            branch_id='b0',
            tree_id=0,
            parent_node_id=1,
            conditions=[
                Condition(feature_idx=0, threshold=1.7, direction='le', node_id=10),
                Condition(feature_idx=1, threshold=0.2, direction='gt', node_id=12),
            ],
        )
    ]

    branch_probs = {
        0: [0.75],
        1: [0.75],
    }
    observed_data = [
        [1.0, 0.5],
        [2.0, 0.1],
    ]

    output_path = 'test_kb_latent_inference.pl'
    try:
        path = export_branches_to_problog_latent(
            branches,
            branch_probs,
            observed_data=observed_data,
            output_path=output_path,
        )
        text = Path(path).read_text(encoding='utf-8')
        text += '\nquery(z(b0,0)).\nquery(z(b0,1)).\n'

        result = get_evaluatable().create_from(PrologString(text)).evaluate()
        result_by_name = {str(key): float(value) for key, value in result.items()}
        posterior_true = result_by_name['z(b0,0)']
        posterior_false = result_by_name['z(b0,1)']

        assert posterior_true > 0.75
        assert posterior_false < 0.75

        print('posterior_true', posterior_true)
        print('posterior_false', posterior_false)
        print('test_problog_evidence_changes_latent_posterior OK')
    finally:
        if os.path.exists(output_path):
            os.remove(output_path)


if __name__ == '__main__':
    test_problog_evidence_changes_latent_posterior()
