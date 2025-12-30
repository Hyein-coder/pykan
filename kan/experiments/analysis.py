import numpy as np

def find_index_sign_revert(data_list, epsilon=5e-3):
    """
    Finds the first index where the sign changes and persists for
    more than one element.

    Values with absolute magnitude smaller than epsilon are treated as having the same sign as the previous element, effectively ignoring small fluctuations around zero.

    Args:
        data_list (list): A list of numbers.
        epsilon (float): The threshold below which a value is considered to have "no sign change" relative to the previous value.

    Returns:
        int or None: The first index of a persistent sign change,
                     or None if no such change is found.
    """
    if len(data_list) < 3:
        return None

    effective_signs = [np.sign(data_list[0])]

    for i in range(1, len(data_list)):
        val = data_list[i]
        prev_sign = effective_signs[-1]

        if abs(val) < epsilon:
            effective_signs.append(prev_sign)
        else:
            effective_signs.append(np.sign(val))
    print(effective_signs)

    # Now search for the persistent sign change using these cleaned signs
    for i in range(1, len(effective_signs) - 1):
        if effective_signs[i] != effective_signs[i-1]:
            if effective_signs[i+1] == effective_signs[i]:
                return i

    return None