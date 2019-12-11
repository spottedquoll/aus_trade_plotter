import pyexcel as pe


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
