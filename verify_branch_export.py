from pathlib import Path
import json
import argparse
from typing import Dict, List


def count_branch_rules(problog_path: str) -> int:
    return sum(1 for line in Path(problog_path).read_text(encoding='utf-8').splitlines() if line.startswith('branch_struct('))


def count_branches_in_json(json_path: str) -> int:
    return len(json.loads(Path(json_path).read_text(encoding='utf-8')))


def _expected_rule(branch: Dict) -> str:
    branch_id = branch['branch_id']
    conditions: List[Dict] = branch.get('conditions', [])
    if not conditions:
        return f"branch_struct({branch_id}, X)."
    atoms = []
    tree_id = branch['tree_id']
    for cond in conditions:
        atoms.append(
            f"{cond['direction']}(f{cond['feature_idx']},t{tree_id}_{cond['node_id']},X)"
        )
    return f"branch_struct({branch_id}, X) :- {', '.join(atoms)}."


def spot_check_first_branch(json_path: str, problog_path: str) -> bool:
    branches = json.loads(Path(json_path).read_text(encoding='utf-8'))
    if not branches:
        return True
    expected = _expected_rule(branches[0])
    rules = set(Path(problog_path).read_text(encoding='utf-8').splitlines())
    return expected in rules


def threshold_symbol_collisions(json_path: str) -> int:
    branches = json.loads(Path(json_path).read_text(encoding='utf-8'))
    seen = {}
    collisions = 0
    for branch in branches:
        tree_id = branch['tree_id']
        for cond in branch.get('conditions', []):
            key = (tree_id, cond['node_id'])
            thr = cond['threshold']
            if key in seen and seen[key] != thr:
                collisions += 1
            seen[key] = thr
    return collisions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('json_path')
    parser.add_argument('problog_path')
    args = parser.parse_args()
    json_count = count_branches_in_json(args.json_path)
    rule_count = count_branch_rules(args.problog_path)
    print('branches in json:', json_count)
    print('branch_struct rules:', rule_count)
    print('first branch spot-check:', 'ok' if spot_check_first_branch(args.json_path, args.problog_path) else 'FAILED')
    print('tree/node threshold collisions in json:', threshold_symbol_collisions(args.json_path))
    if json_count != rule_count:
        raise SystemExit('Mismatch between JSON branches and ProbLog rules')
    if not spot_check_first_branch(args.json_path, args.problog_path):
        raise SystemExit('Spot-check failed: first branch rule not found or malformed')
