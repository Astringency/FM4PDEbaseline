import math
import torch
import torch.nn as nn
from einops import rearrange, repeat
import numpy as np


def UnstructuredPositionalEncoderWbath(image_shape, num_frequency_bands, row_coords, col_coords, ocn_floor, max_frequencies=None):
    *spatial_shape, _ = image_shape

    #normalize col/row coords to be in -1,1
    row_coords = row_coords - np.min(row_coords)
    row_len = np.max(row_coords)
    row_coords = row_coords - row_len / 2
    row_scale = np.max(np.abs(row_coords))
    norm_row_coords = row_coords / row_scale
    norm_row_coords = torch.from_numpy(norm_row_coords)

    col_coords = col_coords - np.min(col_coords)
    col_len = np.max(col_coords)
    col_coords = col_coords - col_len/2
    col_scale = np.max(np.abs(col_coords))
    norm_col_coords = col_coords/col_scale
    norm_col_coords =torch.from_numpy(norm_col_coords)

    ocn_floor_len = np.max(ocn_floor) - np.min(ocn_floor)
    ocn_floor = ocn_floor - ocn_floor_len / 2 - np.min(ocn_floor)
    norm_ocn_floor = ocn_floor / np.max(ocn_floor)
    norm_ocn_floor = torch.from_numpy(norm_ocn_floor)

    pos = torch.stack((norm_row_coords, norm_col_coords, norm_ocn_floor)).T.to(torch.float32)

    encodings = []


    if max_frequencies is None:
        max_frequencies = pos.shape[-1]

    frequencies = [torch.linspace(1.0, max_freq / 2.0, num_frequency_bands)
                   for max_freq in max_frequencies]

    frequency_grids = []
    for i, frequencies_i in enumerate(frequencies):
        frequency_grids.append(pos[..., i:i + 1] * frequencies_i[None, ...])

    encodings.extend([torch.sin(math.pi * frequency_grid) for frequency_grid in frequency_grids])
    encodings.extend([torch.cos(math.pi * frequency_grid) for frequency_grid in frequency_grids])
    enc = torch.cat(encodings, dim=-1)

    # flatten encodings along spatial dimensions
    # don't think this is needed
    enc = rearrange(enc, "... c -> (...) c")

    return enc




