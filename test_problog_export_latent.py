import os
from pathlib import Path
from branch_schema import Branch, Condition
from problog_export import export_branches_to_problog_latent


def test_problog_export_latent_content():
    branches = [
        Branch(
            branch_id='b0',
            tree_id=0,
            parent_node_id=1,
            conditions=[
                Condition(feature_idx=0, threshold=1.7, direction='le', node_id=10),
                Condition(feature_idx=1, threshold=0.2, direction='gt', node_id=12),
            ],
            class_proportions=[0.2, 0.8],
        )
    ]

    branch_probs = {
        0: [0.75],
        1: [0.2],
    }
    observed_data = [
        [1.0, 0.5],
        [2.0, 0.1],
    ]

    output_path = 'test_kb_latent.pl'
    try:
        path = export_branches_to_problog_latent(
            branches,
            branch_probs,
            observed_data=observed_data,
            output_path=output_path,
        )

        text = Path(path).read_text(encoding='utf-8')

        assert 'branch_struct(b0, X) :-' in text
        assert 'threshold(t0_10,1.7)' in text
        assert 'threshold(t0_12,0.2)' in text

        assert '0.75000000::z(b0,0).' in text
        assert '0.20000000::z(b0,1).' in text

        assert 'branch_struct(b0, X) :- le(b0,f0,t0_10,X), gt(b0,f1,t0_12,X).' in text
        assert 'not_z(b0,X) :- \\+ z(b0,X).' in text
        assert '0.97467943::le(b0,f0,t0_10,X) :- z(b0,X).' in text
        assert '0.22360680::le(b0,f0,t0_10,X) :- not_z(b0,X).' in text

        assert '0.97467943::gt(b0,f1,t0_12,X) :- z(b0,X).' in text
        assert '0.22360680::gt(b0,f1,t0_12,X) :- not_z(b0,X).' in text
        assert 'evidence(le(b0,f0,t0_10,0)).' in text
        assert 'evidence(gt(b0,f1,t0_12,0)).' in text
        assert 'evidence(le(b0,f0,t0_10,1), false).' in text
        assert 'evidence(gt(b0,f1,t0_12,1), false).' in text

        print('test_problog_export_latent_content OK')

    finally:
        if os.path.exists(output_path):
            os.remove(output_path)


if __name__ == '__main__':
    test_problog_export_latent_content()
