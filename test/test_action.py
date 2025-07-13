"""
Make sure action decoding and encoding are correct
"""
import pytest

from contrib.action import ActionAsIntPolicy, ActionAsTuplePolicy, AcionAsPolicyABC
from itertools import product

NUM_QUBITS = 20

ACTION_LIST = [
    dict(action='MAP', logical=10, physical=2),
    dict(action='SWAP', left=10, right=2),
    dict(action='BRIDGE', left=5, right=19, middle=9),
]

@pytest.mark.parametrize('action, converter', product(ACTION_LIST, [ActionAsTuplePolicy]))
def test_action_as_int_policy(action, converter: type[AcionAsPolicyABC]):
    policy = converter.to_policy(action, NUM_QUBITS)
    print(f'{policy=}')
    print(f'{action=}')
    print(f'{converter.from_policy(policy, NUM_QUBITS)}')
    assert action == converter.from_policy(policy, NUM_QUBITS)
