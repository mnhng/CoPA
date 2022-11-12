# Adapted from https://github.com/dccastro/Morpho-MNIST.git
import gzip
import pathlib
import struct

import numpy as np


def _load_uint8(f):
    idx_dtype, ndim = struct.unpack('BBBB', f.read(4))[2:]
    shape = struct.unpack('>' + 'I' * ndim, f.read(4 * ndim))
    buffer_length = int(np.prod(shape))
    data = np.frombuffer(f.read(buffer_length), dtype=np.uint8).reshape(shape)
    return data


def _save_uint8(data, f):
    data = np.asarray(data, dtype=np.uint8)
    f.write(struct.pack('BBBB', 0, 0, 0x08, data.ndim))
    f.write(struct.pack('>' + 'I' * data.ndim, *data.shape))
    f.write(data.tobytes())


def save_idx(data: np.ndarray, path: str):
    """Writes an array to disk in IDX format.

    Parameters
    ----------
    data : array_like
        Input array of dtype ``uint8`` (will be coerced if different dtype).
    path : str
        Path of the output file. Will compress with `gzip` if path ends in '.gz'.

    References
    ----------
    http://yann.lecun.com/exdb/mnist/
    """
    path = pathlib.Path(path).expanduser()
    open_fcn = gzip.open if path.suffix == '.gz' else open
    with open_fcn(path, 'wb') as f:
        _save_uint8(data, f)


def load_idx(path: str) -> np.ndarray:
    """Reads an array in IDX format from disk.

    Parameters
    ----------
    path : str
        Path of the input file. Will uncompress with `gzip` if path ends in '.gz'.

    Returns
    -------
    np.ndarray
        Output array of dtype ``uint8``.

    References
    ----------
    http://yann.lecun.com/exdb/mnist/
    """
    path = pathlib.Path(path).expanduser()
    open_fcn = gzip.open if path.suffix == '.gz' else open
    with open_fcn(path, 'rb') as f:
        return _load_uint8(f)
