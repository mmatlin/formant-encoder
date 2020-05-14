from sphfile import SPHFile
from glob import glob
import magic, pathlib, os.path
from typing import List

def convert_sph_to_wav(in_path: str, out_path: str) -> None:
    """
    Converts a given SPH file to WAV and writes it to the same relative location in a new directory.
    """
    sph = SPHFile(in_path)
    pathlib.Path(out_path).parents[0].mkdir(parents=True, exist_ok=True)
    sph.write_wav(out_path)

def convert_dir_to_wav(in_path: str, out_path_prefix: str = "WAV") -> List[str]:
    """
    Converts all SPH files in a directory to WAV and writes them to the same relative location in a new directory.
    Returns a list of paths to WAV files in the new directory, whether the WAVs are new or not.
    """
    out_file_paths = []
    sph_files = glob(in_path + "/**/*.WAV", recursive=True)

    for sph_file in sph_files:
        out_file_path = sph_file.replace("WAV", "wav")
        out_file_path = f"{out_path_prefix}\\{sph_file}"
        # Don't convert the file if it's already been converted in the past
        if magic.from_file(sph_file, mime=True) != "audio/x-wav" and not os.path.exists(out_file_path):
            convert_sph_to_wav(sph_file, out_file_path)
        out_file_paths.append(out_file_path)
    
    return out_file_paths