import os
from pathlib import Path

from branch_schema import Branch, Condition
from problog_export import export_branches_to_problog_latent


def test_problog_latent_branch_scoping_for_shared_prefix():
    branches = [
        Branch(
            branch_id='b0',
            tree_id=0,
            parent_node_id=1,
            conditions=[
                Condition(feature_idx=0, threshold=1.7, direction='le', node_id=10),
                Condition(feature_idx=1, threshold=0.2, direction='gt', node_id=12),
            ],
        ),
        Branch(
            branch_id='b1',
            tree_id=0,
            parent_node_id=2,
            conditions=[
                Condition(feature_idx=0, threshold=1.7, direction='le', node_id=10),
                Condition(feature_idx=2, threshold=0.9, direction='le', node_id=13),
            ],
        ),
    ]

    branch_probs = {
        0: [0.75, 0.40],
    }

    output_path = 'test_kb_latent_scoped.pl'
    try:
        path = export_branches_to_problog_latent(branches, branch_probs, output_path=output_path)
        text = Path(path).read_text(encoding='utf-8')

        assert 'branch_struct(b0, X) :- le(b0,f0,t0_10,X), gt(b0,f1,t0_12,X).' in text
        assert 'branch_struct(b1, X) :- le(b1,f0,t0_10,X), le(b1,f2,t0_13,X).' in text

        assert '0.97467943::le(b0,f0,t0_10,X) :- z(b0,X).' in text
        assert '0.97467943::le(b1,f0,t0_10,X) :- z(b1,X).' in text

        assert '::le(f0,t0_10,X) :- z(b0,X).' not in text
        assert '::le(f0,t0_10,X) :- z(b1,X).' not in text

        assert 'not_z(b0,X) :- \\+ z(b0,X).' in text
        assert 'not_z(b1,X) :- \\+ z(b1,X).' in text

        print('test_problog_latent_branch_scoping_for_shared_prefix OK')
    finally:
        if os.path.exists(output_path):
            os.remove(output_path)


if __name__ == '__main__':
    test_problog_latent_branch_scoping_for_shared_prefix()
