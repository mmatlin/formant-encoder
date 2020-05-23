from scipy.fft import fft
from scipy.io import wavfile
from scipy.interpolate import interp1d
from db_tools import convert_dir_to_wav
import process_phn, numpy as np
from smooth import smooth
from typing import List, Tuple
import matplotlib.pyplot as plt

FFT_SLICE_RADIUS = 50 # number of samples on either side of the center sample of the phone which are used for FFT
MAX_FREQ = 8000 # assumed max frequency of voice data, since TIMIT data is at 16 kHz, so the Nyquist frequency is 8 kHz
FORMANT_OVERLAP = 250 # number of minimum Hz between frequencies which are decided to be formants

def extract_formants(wav_path: str, phn_path: str, plot = False) -> List[Tuple[str, Tuple[int, int, int, int, int]]]:
    """
    Takes a path to a file storing data about the phones in an audio file as well as a path to the audio file.
    Returns a list of tuples, each of which contain (a) the phone for which a set of formants were extracted and (b) a tuple containing the formants.
    """
    out = []
    phn_data = process_phn.extract_monophthong_times(phn_path)
    _, wav_data = wavfile.read(wav_path)

    # If plot argument is True, print the transcript of the recording to provide context for the plots
    if plot:
        txt_data_path = phn_path.replace("PHN", "TXT")
        with open(txt_data_path) as transcript:
            print(transcript.read())

    for phn_instance in phn_data:
        # Determine start and end times of phone, then determine middle sample index
        vowel_phone, vowel_start, vowel_end = phn_instance
        fft_slice_middle = vowel_start + (vowel_end - vowel_start) // 2
        
        # Set vowel_data to be a slice of wav_data at the middle sample index ± FFT_SLICE_RADIUS
        vowel_data = wav_data[fft_slice_middle - FFT_SLICE_RADIUS : fft_slice_middle + FFT_SLICE_RADIUS]

        # Take the absolute value of the real part of the FFT of vowel_data, then stretch it out to MAX_FREQ length and smooth it out a bit
        vowel_data_fft = fft(vowel_data).real
        vowel_data_fft = abs(vowel_data_fft)[:len(vowel_data_fft) // 2] # Get rid of needless second half of FFT, it's symmetrical
        vowel_data_fft_interp = interp1d(np.arange(vowel_data_fft.size), vowel_data_fft)
        vowel_data_fft = vowel_data_fft_interp(np.linspace(0, vowel_data_fft.size - 1, MAX_FREQ))
        vowel_data_fft = smooth(vowel_data_fft, 20)

        # If plot argument is True, plot the spectral slice for each vowel
        if plot:
            print(vowel_phone)
            plt.semilogy(vowel_data_fft,'r')
            plt.suptitle(f"[{vowel_phone}]", fontsize=20)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Log-amplitude")
            plt.show()

        # Associate the frequencies with their amplitudes and sort them in increasing amplitude
        formant_candidates = [(freq, amplitude) for freq, amplitude in enumerate(vowel_data_fft)]
        formant_candidates = sorted(formant_candidates, key=lambda x: x[1])

        # Until there are 5 chosen formants, keep adding the remaining most intense frequency unless it overlaps with an already chosen formant
        # Then sort the chosen formants in order of increasing frequency
        formants = []
        while len(formants) < 5:
            if len(list(filter(lambda x: abs(x[0] - formant_candidates[-1][0]) <= FORMANT_OVERLAP, formants))) == 0:
                formants.append(formant_candidates[-1])
            formant_candidates.pop()
        formants = sorted(formants, key=lambda x: x[0])
        out.append(tuple([vowel_phone, tuple(formant[0] for formant in formants)]))
    return out
    