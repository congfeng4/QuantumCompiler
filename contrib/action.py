"""
Define how the action (dict) is encoded and decoded from policy output.
"""
from typing import TypeVar

import gymnasium as gym

ActionType = dict[str, int | str]


ACTION_STR_TO_INT = {
    'MAP': 0, 'SWAP': 1, 'BRIDGE':2
}
ACTION_INT_TO_STR = {val: key for key, val in ACTION_STR_TO_INT.items()}


class AcionAsPolicyABC:
    PolicyType = TypeVar('PolicyType')

    @classmethod
    def from_policy(cls, policy: int, *args) -> ActionType:
        raise NotImplementedError

    @classmethod
    def to_policy(cls, action: ActionType, num_qubits: int) -> PolicyType:
        raise NotImplementedError

    @classmethod
    def action_space(self):
        raise NotImplementedError


class ActionAsIntPolicy:
    """
    Policy as a single int.
    """
    PolicyType = int

    @classmethod
    def from_policy(cls, policy: int, num_qubits: int) -> ActionType:
        num_params = num_qubits * num_qubits
        type, params = policy // num_params, policy % num_params
        type = ACTION_INT_TO_STR[type]
        bit1 = params // num_qubits
        bit2 = params % num_qubits
        if type == 'MAP':
            return dict(action=type, logical=bit1, physical=bit2)
        return dict(action=type, left=bit1, right=bit2)
    
    @classmethod
    def to_policy(cls, action: ActionType, num_qubits: int) -> PolicyType:
        num_params = num_qubits * num_qubits
        type = ACTION_STR_TO_INT[action['action']]
        if action['action'] == 'MAP':
            bit1, bit2 = action['logical'], action['physical']
        else:
            bit1, bit2 = action['left'], action['right']
    
        return type * num_params + bit1 * num_qubits + bit2


class ActionAsTuplePolicy:
    """
    Policy as a tuple.
    """
    PolicyType = tuple[int, int, int, int]
    
    @classmethod
    def from_policy(cls, policy: PolicyType, *args):
        action = ACTION_INT_TO_STR[policy[0]]
        if action == 'MAP':
            return dict(action='MAP', logical=policy[1], physical=policy[2])
        if action == 'SWAP':
            return dict(action='SWAP', left=policy[1], right=policy[2])
        if action == 'BRIDGE':
            return dict(action='BRIDGE', left=policy[1], right=policy[2], middle=policy[3])
        raise ValueError(policy)
    
    @classmethod
    def to_policy(cls, action: ActionType, *args):
        action_type = action['action']
        type = ACTION_STR_TO_INT[action_type]
        if action_type == 'MAP':
            return type, action['logical'], action['physical'], 0
        if action_type == 'SWAP':
            return type, action['left'], action['right'], 0
        if action_type == 'BRIDGE':
            return type, action['left'], action['right'], action['middle']
        raise ValueError(action)

    @classmethod
    def action_space(self, num_qubits: int):
        return gym.spaces.MultiDiscrete([2, num_qubits, num_qubits, num_qubits])
