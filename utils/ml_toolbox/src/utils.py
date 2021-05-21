import math
import numpy as np

def real_to_grid(x, y, z, o, c, s):
    """ Translate real coordinates x, y and z with origin o, cell size c and sampling s to grid coordinates."""
    x = (x - o.z) / (c.z / s[2])
    y = (y - o.y) / (c.y / s[1])
    z = (z - o.x) / (c.x / s[0])
    return x, y, z


def grid_to_real(x, y, z, o, c, s):
    """ Translate grid coordinates x, y and z to real space coordinates with origin o, cell size c and sampling s."""
    x = (x + 0) * (c.z / s[2]) + o.z
    y = (y + 0) * (c.y / s[1]) + o.y
    z = (z + 0) * (c.x / s[0]) + o.x
    return x, y, z


def divx(x, d=8, ):
    """ Ensure the number is divisible (to integer) by x (to ensure it can pool and concatenate max 3 times (2^3))."""
    if x % d != 0:
        y = math.ceil(x / d)
        x = y * d
    return x


def normalise(data):
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    return data


def progress(i, lim, desc="Progress", length=25):
    """
    An alteration of code taken from:
    https://www.reddit.com/r/learnpython/comments/9xktge/how_to_use_the_tqdm_progress_bar_on_a_pandas/
    """
    if i == 0:
        print()
    i = i+1
    print(f"\r{desc}: {('='*(length*i//lim)).ljust(length)} {100*i//lim}%", end="")
    if i == lim:
        print()
        print()
