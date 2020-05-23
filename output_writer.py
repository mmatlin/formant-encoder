import csv
import os.path as path
from pathlib import Path
from typing import List, Tuple, Optional, Iterator
from extract_formants import extract_formants
from process_phn import get_num_monophthongs, ipa_class_index
from db_tools import get_n_pairs

class OutputWriter:
    """
    An iterator that returns ``n`` sets of calculated formants of either train data, test data, or both.
    Can also write the data to CSVs or return extant CSV file objects.

    Keyword Arguments:
        n {Optional[int]} -- the number of audio files of each data category to process (default: {None})
        train {bool} -- whether to use the training data category (default: {True})
        test {bool} -- whether to use the test data category (default: {True})
        data_dir {str} -- the path to the TIMIT database (default: {"TIMIT"})
        csv_dir {str} -- the path to which CSVs should be written (default: {"out"})
    """

    def __init__(self, n: Optional[int] = None, train: bool = True, test: bool = True, data_dir: str ="TIMIT", csv_dir: str = "out") -> None:
        self.n = n
        self.train = train
        self.test = test
        self.data_dir = data_dir
        self.csv_dir = csv_dir

    def __iter__(self) -> Iterator[Tuple[List[Tuple[str, Tuple[int, int, int, int, int]]], str, str]]:
        """
        For each audio file requested to be processed, this yields a tuple containing:
            —a list of tuples, each of which has a phone and a tuple of the phone's calculated formants
            —the path to the WAV file
            —the data category of the WAV file
        """
        return ((extract_formants(wav, phn), wav, "TEST" if "TEST" in wav else "TRAIN") for wav, phn in get_n_pairs(self.n, self.train, self.test))

    def write_to_csv(self, verbose: bool = False) -> None:
        """
        Writes a CSV for the each of the ``OutputWriter`` instance's data categories in ``self.csv_dir`` containing the sets of calculated formants for the data category.
        """        
        Path(self.csv_dir).mkdir(exist_ok=True)
        with open(f"{self.csv_dir}/train.csv", "wt", encoding="utf-8", newline="") as train_file:
            with open(f"{self.csv_dir}/test.csv", "wt", encoding="utf-8", newline="") as test_file:
                csv_header = ("phone", "phone_class_index", "f1", "f2", "f3", "f4", "f5")
                train_csvwriter = csv.writer(train_file)
                test_csvwriter = csv.writer(test_file)
                train_csvwriter.writerow(csv_header)
                test_csvwriter.writerow(csv_header)
                for vowels_and_formants, wav_path, category in self:
                    if verbose:
                        print(f"File: {wav_path} (category: {category})")
                    writer = train_csvwriter if category == "TRAIN" else test_csvwriter
                    for vowel_and_formants in vowels_and_formants:
                        phone, formants = vowel_and_formants
                        row = (phone, ipa_class_index[phone]) + tuple(formants)
                        writer.writerow(row)
                        if verbose:
                            print(row)

    def get_cached_csv(self, category: str) -> str:
        """
        Returns the path to the cached CSV for the given data category ``category``.

        Arguments:
            category {str} -- the requested data category (``"train"`` or ``"test"``)

        Returns:
            str -- the path to the cached CSV for ``category``, if one exists

        Raises:
            FileNotFoundError: if there is no CSV written yet for the requested data category
        """
        csv_path = f"{self.csv_dir}/{category.lower()}.csv"
        if path.exists(csv_path):
            return csv_path
        raise FileNotFoundError(f"There is no {category.lower()} CSV written yet.")
