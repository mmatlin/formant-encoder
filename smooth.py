import numpy

def smooth(x, window_len=11):
    """
    Smooths a given signal using the Hann window (https://en.wikipedia.org/wiki/Hann_function).
    Adapted from https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html.
    """
    s = numpy.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]
    w = numpy.hanning(window_len)
    y = numpy.convolve(w / w.sum(), s, mode='valid')
    return y