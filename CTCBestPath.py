# ----------------------------------------------------------------- #
#            This code was taken from github repo:                  #
# "Connectionist Temporal Classification (CTC) decoding algorithms" #
#                developed by Harald Scheidl                        #
#         https://github.com/githubharald/CTCDecoder/               #
#               Copyright (c) 2018 Harald Scheidl                   #
# ----------------------------------------------------------------- #

from __future__ import division
from __future__ import print_function
from itertools import groupby
import numpy as np


def ctcBestPath(mat, classes):
    "implements best path decoding as shown by Graves (Dissertation, p63)"

    # get char indices along best path
    best_path = np.argmax(mat, axis=1)
    
    # collapse best path (using itertools.groupby), map to chars, join char list to string
    blank_idx = 0 # index of CTC BLANK character (in original repo: blank_idx = len(classes))
    best_chars_collapsed = [classes[k] for k, _ in groupby(best_path) if k != blank_idx]
    res = ''.join(best_chars_collapsed)
    return res