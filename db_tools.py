import magic
import os.path as path
from glob import glob
from pathlib import Path
from itertools import islice, chain
from typing import List, Tuple, Optional
from sphfile import SPHFile

DEFAULT_DATA_DIR = "TIMIT"

def convert_sph_to_wav(in_path: str, out_path: str) -> None:    
    """
    Converts a given SPH file at `in_path` to WAV and writes it to `out_path`.
    """
    sph = SPHFile(in_path)
    sph.write_wav(out_path)

def convert_dir_to_wav(in_path: str) -> None:
    """
    Converts all SPH files in directory `in_path` to WAV and writes them to the same directory with file extension .x-wav.
    """
    sph_files = glob(in_path + "/**/*.WAV", recursive=True)
    for sph_file in sph_files:
        out_file_path = sph_file.replace("WAV", "x-wav")
        # Don't convert the file if it's already been converted in the past
        if magic.from_file(sph_file, mime=True) != "audio/x-wav" and not path.exists(out_file_path):
            convert_sph_to_wav(sph_file, out_file_path)

def get_n_pairs(n: Optional[int] = None, train: bool = True, test: bool = True, data_dir: str = "TIMIT",) -> List[Tuple[str, str]]:
    """
    Returns ``n`` pairs of WAV/PHN paths for each of the specified data categories (``"train"`` and/or ``"test"``).

    Keyword Arguments:
        n {Optional[int]} -- the number of pairs to return (default: {None})
        train {bool} -- whether to use the training data category (default: {True})
        test {bool} -- whether to use the test data category (default: {True})
        data_dir {str} -- the path to the TIMIT database (default: {"TIMIT"})

    Returns:
        List[Tuple[str, str]] -- a list of ``n`` pairs of WAV/PHN paths for each of the specified data categories
    """
    train_wav = Path(f"{data_dir}/TRAIN").glob("**/*.x-wav") if train else list()
    train_phn = Path(f"{data_dir}/TRAIN").glob("**/*.PHN") if train else list()
    test_wav = Path(f"{data_dir}/TEST").glob("**/*.x-wav") if test else list()
    test_phn = Path(f"{data_dir}/TEST").glob("**/*.PHN") if test else list()
    return [(str(wav), str(phn)) for wav, phn in chain(zip(islice(train_wav, n),
        islice(train_phn, n)), zip(islice(test_wav, n), islice(test_phn, n)))]

if __name__ == "__main__":
    convert_dir_to_wav(DEFAULT_DATA_DIR)
