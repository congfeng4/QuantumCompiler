"""
Make sure action decoding and encoding are correct
"""
import pytest

from contrib.action import from_policy, to_policy

NUM_QUBITS = 20

ACTION_LIST = [
    dict(action='MAP', logical=10, physical=2),
    dict(action='SWAP', left=10, right=2),
    dict(action='BRIDGE', left=5, right=19),
]

@pytest.mark.parametrize('action', ACTION_LIST)
def test_action_policy(action):
    policy = to_policy(action, NUM_QUBITS)
    assert action == from_policy(policy, NUM_QUBITS)
