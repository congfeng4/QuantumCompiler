ActionType = dict[str, int | str]


ACTION_STR_TO_INT = {
    'MAP': 0, 'SWAP': 1, 'BRIDGE':2
}
ACTION_INT_TO_STR = {val: key for key, val in ACTION_STR_TO_INT.items()}



def from_policy(policy: int, num_qubits: int) -> ActionType:
    num_params = num_qubits * num_qubits
    type, params = policy // num_params, policy % num_params
    type = ACTION_INT_TO_STR[type]
    bit1 = params // num_qubits
    bit2 = params % num_qubits
    if type == 'MAP':
        return dict(action=type, logical=bit1, physical=bit2)
    return dict(action=type, left=bit1, right=bit2)


def to_policy(action: ActionType, num_qubits: int) -> int:
    num_params = num_qubits * num_qubits
    type = ACTION_STR_TO_INT[action['action']]
    if action['action'] == 'MAP':
        bit1, bit2 = action['logical'], action['physical']
    else:
        bit1, bit2 = action['left'], action['right']

    return type * num_params + bit1 * num_qubits + bit2
