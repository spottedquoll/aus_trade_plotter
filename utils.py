import pyexcel as pe
from numpy import sign, isscalar


def flatten_list(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


def is_empty(some_list):
    if some_list is None or some_list == [] or len(some_list) == 0:
        return True
    else:
        return False


def clean_string(raw_str, bad_characters, replacement, case=None):
    for k in bad_characters:
        raw_str = raw_str.replace(k, replacement)

    if case is not None:
        if case == 'lower':
            raw_str = raw_str.lower()
        elif case == 'upper':
            raw_str = raw_str.upper()
        else:
            raise ValueError('Unknown case')

    return raw_str


def is_within_tolerance(val_1, val_2, tol):
    assert tol <= 1
    assert tol > 0
    assert isscalar(val_1)
    assert isscalar(val_2)

    if sign(val_1) == sign(val_2) and sign(val_1) == -1:
        val_1 = abs(val_1)
        val_2 = abs(val_2)

    if val_1 == val_2:
        return True
    elif val_2 * (1 - tol) < val_1 < val_2 * (1 + tol):
        return True
    else:
        return False
