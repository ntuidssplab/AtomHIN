import os
from typing import Literal
from typing_extensions import deprecated


def find_data_file(target_name: str,
                   raw_dir: str) -> tuple[str, Literal['.dat', '.csv']]:
    """

    Args:
        target_name (str): e.g. `node`, `link`, ...
        raw_dir (str): 

    Returns:
        (file, ext)
    """
    names = os.listdir(raw_dir)
    csv_name = f'{target_name}.csv'
    dat_name = f'{target_name}.dat'

    assert (csv_name in names) ^ (dat_name in names), (
        f'Expect one of {csv_name} or {dat_name} in raw_dir: {raw_dir}, '
        f'but got {"both" if (csv_name in names) and (dat_name in names) else "none"}'
        ' of them.'
    )
    file = csv_name if csv_name in names else dat_name
    return os.path.join(raw_dir, file), file[-4:]


@deprecated('Use find_data_file instead')
def _find_extension(raw_dir: str) -> Literal['.dat', '.csv']:
    """Check whether the raw_dir contains data in .dat format or .csv format

    Args:
        raw_dir (str): path to directory contains data files

    Returns:
        Literal['.dat' or '.csv']
    """

    CANDIDATES = ['.dat', '.csv']
    which = None
    for name in os.listdir(raw_dir):
        _, ext = os.path.splitext(name)

        if ext in CANDIDATES:
            idx = CANDIDATES.index(ext)
            if which is None:
                which = idx
            else:
                if which != idx:
                    raise Exception(
                        f'Expect all file in raw_dir: {raw_dir} has extention either '
                        '"dat" or "csv". But both "dat" and "csv" exist: \n'
                        f'{os.listdir(raw_dir)}'
                    )

    if which is None:
        raise Exception(
            f'Expect all file in raw_dir: {raw_dir} has extention either '
            '"dat" or "csv". But none of "dat" and "csv" exist: \n'
            f'{os.listdir(raw_dir)}'
        )
    return CANDIDATES[which]
