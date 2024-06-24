import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2tuple(v):
    """
    >>> str2tuple('(1, 2, 3)')
    (1, 2, 3)
    """
    pattern = re.compile(r'\((.*?)\)')
    match = pattern.search(v)

    if match:
        values_str = match.group(1)
        try:
            values = tuple(map(int, values_str.split(',')))
            return values
        except ValueError:
            raise argparse.ArgumentTypeError('Invalid tuple format. Must be comma-separated integers.')
    else:
        raise argparse.ArgumentTypeError('Invalid tuple format. Must be enclosed in parentheses.')
    