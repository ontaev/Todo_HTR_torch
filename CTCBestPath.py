from __future__ import division
from __future__ import print_function
from itertools import groupby
import numpy as np


def ctcBestPath(mat, classes):
    "implements best path decoding as shown by Graves (Dissertation, p63)"

    # get char indices along best path
    best_path = np.argmax(mat, axis=1)
    #print('bp', best_path)
    # collapse best path (using itertools.groupby), map to chars, join char list to string
    blank_idx = 0
    best_chars_collapsed = [classes[k] for k, _ in groupby(best_path) if k != blank_idx]
    res = ''.join(best_chars_collapsed)
    return res