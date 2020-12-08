import numpy as np
from PIL import Image as im
from pathlib import Path


def dice_score(a: np.ndarray, b: np.ndarray):
    a, b = a.flatten(), b.flatten()
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return 2. * intersection/union

orig_paths = list(Path('testB/').iterdir())
gen_paths = list(Path('test20-25/').iterdir())

if len(orig_paths) != len(orig_paths):
    raise Exception('Different count of original and generated images!')

for o, g in zip(orig_paths, gen_paths):
    orig = im.open(o).resize((256, 256))
    orig = np.asarray(orig, dtype=np.uint8)[:,:,0]
    gen = im.open(g)
    gen = np.asarray(gen, dtype=np.uint8)[:,:,0]

    bin_orig = orig > 120
    bin_gen = gen > 120
    print(dice_score(bin_orig, bin_gen))
