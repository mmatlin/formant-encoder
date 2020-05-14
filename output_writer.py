from extract_formants import extract_formants
from process_phn import get_num_monophthongs
from convert_to_wav import convert_dir_to_wav
import unicodecsv as csv
from typing import List, Tuple

data_dir = "TIMIT"
wav_data_dir_prefix = "WAV"

# Get paths to each WAV file and its corresponding PHN file from the TIMIT database
wav_file_paths = convert_dir_to_wav(data_dir, wav_data_dir_prefix)
phn_file_paths = [wav_file_path[len(wav_data_dir_prefix) + 1:].replace("WAV", "PHN") for wav_file_path in wav_file_paths]

# Set up Unicode CSV writers
with open("train_ind_formants.csv", "wb") as train_ind_formants_file:
    with open("train_dep_formants.csv", "wb") as train_dep_formants_file:
        with open("test_ind_formants.csv", "wb") as test_ind_formants_file:
            with open("test_dep_formants.csv", "wb") as test_dep_formants_file:
                train_ind_csvwriter = csv.writer(train_ind_formants_file)
                train_dep_csvwriter = csv.writer(train_dep_formants_file)
                test_ind_csvwriter = csv.writer(test_ind_formants_file)
                test_dep_csvwriter = csv.writer(test_dep_formants_file)
                # For each WAV file and its respective PHN file, extract the formants for each monophthong
                for wav, phn in zip(wav_file_paths, phn_file_paths):
                    vowels_and_formants = extract_formants(wav, phn)
                    for vowel_and_formants in vowels_and_formants:
                        vowel = vowel_and_formants[0]
                        formants = vowel_and_formants[1]
                        ind_row = tuple(str(formant) for formant in formants)
                        dep_row = vowel
                        print(ind_row)
                        print(dep_row)
                        if "TEST" in wav:
                            test_ind_csvwriter.writerow(ind_row)
                            test_dep_csvwriter.writerow(dep_row)
                        else:
                            train_ind_csvwriter.writerow(ind_row)
                            train_dep_csvwriter.writerow(dep_row)