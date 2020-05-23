# Formant encoder and phone classifier
Summary: an encoder which compresses audio data based on prominent acoustic features in order to create a lightweight phonetic classifier.

This project aims to test the viability of a feedforward neural network alternative for simple monophthong recognition in place of (commonly used) recurrent neural networks. A feedforward NN would be beneficial in that it is more lightweight (possibly in exchange for classification accuracy). By estimating the formants of phones being spoken, it should be possible to train a simple neural network on the most prominent frequencies of phones to determine the phone being spoken. The formants can be passed into the network as the input layer and the output layer will contain the estimated likelihood of each possible phone being spoken. However, this would be extremely lossy in regard to the information in the audio file, so the objective of this project is to create an encoder for audio data which captures formants and approximates the regions of the instantaneous spectral slice between formants as piecewise polynomials. The coefficients of the polynomials and their intervals will be passed into the neural network. This way, the neural network will have a compressed representation of the phone being spoken at any given point, and the encoder, although lossy, will retain a good amount of the original information from the audio file.

This project uses the [TIMIT](https://github.com/philipperemy/timit) corpus of phonetically transcribed speech for its training and test data.

Checklist:
- [x] Create modules for managing the TIMIT database (e.g., converting audio from the [NIST SPHERE](https://www.isip.piconepress.com/projects/speech/software/tutorials/production/fundamentals/v1.0/section_02/s02_01_p04.html) format to WAV)
  - [ ] Update the database management tools to write the state of the database to the working directory, so it can be automatically determined whether or not the database needs to be converted to WAV, for example
- [x] Create a module which manages processed audio data and writes output to CSV
  - [ ] Write a generator function for retrieving written data (first, determine if necessary)
  - [ ] Add functionality so needless audio processing is avoided (check if CSV is already written and with which version of audio processing algorithm)
- [x] Begin writing an audio processing algorithm (current algorithm vaguely estimates formants)
  - [ ] Estimate formants more accurately
  - [ ] Approximate inter-formant regions with polynomials
  - [ ] Write a function which calculates how lossy the spectral slice approximation is
- [x] Begin a phone classifier (current classifier is a stand-in and needs to be revised)
- [ ] Add a dependencies management file for easy installation of required modules (PyTorch's stable releases for installation using `pip` are placed on their [stable release repository](https://download.pytorch.org/whl/torch_stable.html) instead of PyPI, which [does not](https://github.com/pytorch/pytorch/issues/25639) conform to [PEP 503](https://www.python.org/dev/peps/pep-0503/) specifications, so [it is not yet possible](https://github.com/python-poetry/poetry/issues/1391) to have `poetry` install PyTorch without using the direct link to the wheel)
