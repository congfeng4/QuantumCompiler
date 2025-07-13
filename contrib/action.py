"""
Define how the action (dict) is encoded and decoded from policy output.
"""
from typing import TypeVar

import gymnasium as gym

ActionType = dict[str, int | str]


ACTION_STR_TO_INT = {
    'SWAP': 0, 'BRIDGE': 1
}
ACTION_INT_TO_STR = {val: key for key, val in ACTION_STR_TO_INT.items()}


class AcionAsPolicy:
    PolicyType = TypeVar('PolicyType')

    def from_policy(cls, policy, num_qubits: int) -> ActionType:
        raise NotImplementedError

    def to_policy(cls, action: ActionType, num_qubits: int) -> PolicyType:
        raise NotImplementedError

    def action_space(self, num_qubits: int):
        raise NotImplementedError

    def __repr__(self):
        return str(self)


class ActionAsPolicyInt(AcionAsPolicy):
    """
    Policy as a single int.
    """
    PolicyType = int

    def from_policy(cls, policy: int, num_qubits: int) -> ActionType:
        num_params = num_qubits * num_qubits
        type, params = policy // num_params, policy % num_params
        type = ACTION_INT_TO_STR[type]
        bit1 = params // num_qubits
        bit2 = params % num_qubits
        return dict(action=type, left=bit1, right=bit2)
    
    def to_policy(cls, action: ActionType, num_qubits: int) -> PolicyType:
        num_params = num_qubits * num_qubits
        type = ACTION_STR_TO_INT[action['action']]
        bit1, bit2 = action['left'], action['right']
        return type * num_params + bit1 * num_qubits + bit2

    def action_space(self, num_qubits: int):
        return gym.spaces.Discrete(2 * num_qubits * num_qubits)

    def __str__(self):
        return 'int'


class ActionAsPolicyTuple(AcionAsPolicy):
    """
    Policy as a tuple.
    """
    PolicyType = tuple[int, int, int, int]
    
    def from_policy(cls, policy: PolicyType, *args):
        action = ACTION_INT_TO_STR[policy[0]]
        if action == 'SWAP':
            return dict(action='SWAP', left=policy[1], right=policy[2])
        if action == 'BRIDGE':
            return dict(action='BRIDGE', left=policy[1], right=policy[2])
        raise ValueError(policy)
    
    def to_policy(cls, action: ActionType, *args):
        action_type = action['action']
        type = ACTION_STR_TO_INT[action_type]
        if action_type == 'SWAP':
            return type, action['left'], action['right']
        if action_type == 'BRIDGE':
            return type, action['left'], action['right']
        raise ValueError(action)

    def action_space(self, num_qubits: int):
        return gym.spaces.MultiDiscrete([2, num_qubits, num_qubits])

    def __str__(self):
        return 'tuple'
