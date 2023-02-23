import numpy as np


def cmu_data_slice(meta, slice_len, overlap=0, shuffle=False):

    slices = []
    for i, amc in enumerate(meta.amc_names):
        if i >= len(meta.amc_lengths):
            break
        if meta.amc_lengths[i] > slice_len:
            start_idx = 0
            stop_idx = slice_len - 1

            while stop_idx < meta.amc_lengths[i]:
                slices.append((i, amc, start_idx, stop_idx))
                start_idx += slice_len - overlap
                stop_idx += slice_len - overlap

    if shuffle:
        np.random.shuffle(slices)

    return slices
