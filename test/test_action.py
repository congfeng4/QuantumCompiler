"""
Make sure action decoding and encoding are correct
"""
import pytest

from contrib.action import ActionAsPolicyInt, ActionAsPolicyTuple, AcionAsPolicy
from itertools import product

NUM_QUBITS = 20

ACTION_LIST = [
    dict(action='SWAP', left=10, right=2),
    dict(action='BRIDGE', left=5, right=19),
]

@pytest.mark.parametrize('action, converter', product(ACTION_LIST, [ActionAsPolicyTuple(), ActionAsPolicyInt()]))
def test_action_as_int_policy(action, converter: AcionAsPolicy):
    policy = converter.to_policy(action, NUM_QUBITS)
    print(f'{policy=}')
    print(f'{action=}')
    print(f'{converter.from_policy(policy, NUM_QUBITS)}')
    assert action == converter.from_policy(policy, NUM_QUBITS)
