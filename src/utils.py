import time
from enum import IntEnum
from typing import Callable, Dict, Iterable, Union
from unicodedata import numeric

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyBigWig as pbw
from numpy.core.numeric import normalize_axis_tuple
from numpy.lib.stride_tricks import as_strided
from scipy import signal, sparse
from scipy.signal import correlate, correlation_lags
from scipy.stats import levene, mannwhitneyu, normaltest, shapiro, ttest_ind
from inspect import signature
import os 
from pathlib import Path

BASES = np.array(["A", "C", "G", "T", "N"]) # N last


eps = np.finfo(float).eps


# DOCSTRING TO REVIEW
def init_tf(tf):
    tfver = tf.__version__
    physical_devices = tf.config.list_physical_devices("GPU")

    # default
    strategy = tf.distribute.get_strategy()  # works for CPU or single GPU

    #memory config
    for gpu in physical_devices:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    # multi-GPU only
    if len(physical_devices) > 1:
        strategy = tf.distribute.MirroredStrategy()

    return {"ver": tfver, "strategy": strategy, "devices": physical_devices}

def get_arg_number(fun):
    sig = str(signature(fun))
    sig = sig.lstrip("(")
    sig = sig.rstrip(")")
    sig = sig.split(',')
    return len(sig)

def loadnp(l: str):
    if l[-1] == "y":
        return np.load(l)
    elif l[-1] == "z":
        return np.load(l)["arr_0"]
    else:
        raise ValueError("file: '{}' is nor .npz nor .npy".format(l))





def cor_lag(x, y, mode="full"):
    """
    Return the offset which leads to the highest  correlation between two signals.
    """
    correlation = signal.correlate(x, y, mode="full")
    lags = signal.correlation_lags(x.size, y.size, mode="full")
    return lags[np.argmax(correlation)]


def autocorr(a: Iterable[numeric]) -> Iterable[np.array]:
    """
    Return the autocorrelation (from pandas) of a in range(inf, sup, step).

    ### Parameters
    1. a : np.array
        - 1D array to perform autocorrelation onto
    2. inf : int
        lower border (included) to perform autocorrelation
    3. sup : int
        - upper border (exluded) to perform autocorrelation
    4. step : int
        - distance between steps to performa autocorrelation

    ### Returns
    - np.array:
        1D Array containing the autocorrelation for the given offsets
    """
    a = zscore(a)
    x1 = a / len(a)

    return correlation_lags(a.size, x1.size), correlate(a, x1)


def zscore(a: Iterable[numeric]) -> np.array:
    """
    return Zscored array

    ### Parameters
    1. a: np.array
        - 1D array to zscore

    ### Returns
    - np.array
        Same array zscored
    """

    return (a - np.mean(a)) / np.std(a)


def is_normal(data, alpha=0.05):
    if len(data) > 5000:
        return normaltest(data).pvalue > alpha
    else:
        return shapiro(data).pvalue > alpha


def choose_test_stat(data1, data2, alpha_normality=0.05, alpha_variance=0.05):
    result = {}

    # Normality
    normal1 = is_normal(data1, alpha=alpha_normality)
    normal2 = is_normal(data2, alpha=alpha_normality)

    # Levene's variance test
    p_levene = levene(data1, data2).pvalue
    equal_var = p_levene > alpha_variance

    if normal1 and normal2:
        if equal_var:
            stat, p = ttest_ind(data1, data2, equal_var=True)
            test_used = "Student t-test (equal variance)"
        else:
            stat, p = ttest_ind(data1, data2, equal_var=False)
            test_used = "Welch t-test (unequal variance)"
    else:
        stat, p = mannwhitneyu(data1, data2, alternative="two-sided")
        test_used = "Mann-Whitney U test (non-parametric)"

    if p < 0.0001:
        stars = "****"
    elif p < 0.001:
        stars = "***"
    elif p < 0.01:
        stars = "**"
    elif p < 0.05:
        stars = "*"
    else:
        stars = "ns"

    # Résumé
    result["test"] = test_used
    result["p_value"] = p
    result["stars"] = stars
    result["normal1"] = normal1
    result["normal2"] = normal2
    result["levene_p"] = p_levene

    return result


def normalize(a: Iterable[numeric], inf: numeric = 0.0, sup: numeric = 1.0) -> np.array:
    """
    normalize array between [inf; sup]
    :param np.array a: array
    :param num inf: lower bound
    :param num sup: upper bound

    """
    assert inf < sup, "lower bound >= higher bound"
    return ((a - np.min(a)) / (np.max(a) - np.min(a)) * (sup - inf)) + inf


def sliding_window_view(x, window_shape, axis=None, *, subok=False, writeable=False):
    window_shape = tuple(window_shape) if np.iterable(window_shape) else (window_shape,)
    # first convert input to array, possibly keeping subclass
    x = np.array(x, copy=False, subok=subok)

    window_shape_array = np.array(window_shape)
    if np.any(window_shape_array < 0):
        raise ValueError("`window_shape` cannot contain negative values")

    if axis is None:
        axis = tuple(range(x.ndim))
        if len(window_shape) != len(axis):
            raise ValueError(
                f"Since axis is `None`, must provide "
                f"window_shape for all dimensions of `x`; "
                f"got {len(window_shape)} window_shape elements "
                f"and `x.ndim` is {x.ndim}."
            )
    else:
        axis = normalize_axis_tuple(axis, x.ndim, allow_duplicate=True)
        if len(window_shape) != len(axis):
            raise ValueError(
                f"Must provide matching length window_shape and "
                f"axis; got {len(window_shape)} window_shape "
                f"elements and {len(axis)} axes elements."
            )

    out_strides = x.strides + tuple(x.strides[ax] for ax in axis)

    # note: same axis can be windowed repeatedly
    x_shape_trimmed = list(x.shape)
    for ax, dim in zip(axis, window_shape):
        if x_shape_trimmed[ax] < dim:
            raise ValueError("window shape cannot be larger than input array shape")
        x_shape_trimmed[ax] -= dim - 1
    out_shape = tuple(x_shape_trimmed) + window_shape
    return as_strided(
        x, strides=out_strides, shape=out_shape, subok=subok, writeable=writeable
    )


def reshape_bin(
    a: Iterable, binsize: int, func: Callable[[], Iterable] = None, keepdim=False
) -> np.array:
    """
    return array shaped in bin (convinient to apply func)
    if the bin size do not fit perfectly, discard some values.
    ex1: reshape_bin(np.arange(10), 3)
    // [[0,1,2],
        [3,4,5],
        [6,7,8]]

    ex2: reshape_bin(np.arange(10),3, np.sum)
    //[3,12,21]

    """
    a = a[: len(a) - len(a) % binsize]
    a = a.reshape((-1, binsize))
    if func is not None:
        a = func(a, axis=1).ravel()

    if keepdim:
        a = np.repeat(a.ravel(), binsize)

    return a


def consecutive(data: Iterable[numeric], stepsize: int = 1) -> np.array:
    return np.split(data, np.where(np.diff(data) > stepsize)[0] + 1)


def string2fasta(a: str, header: Union[numeric, str] = "", size: int = 120) -> str:
    """
    Transform string to fasta formatted string

    :param str a: string to transform
    :param header: The name of the sequence as required by fasta format (">header")
    :param int size: number of residue by line (usually fasta required 60 or 120 as a maximum).
    """
    if header == "":
        f = header
    else:
        f = ">" + str(header) + "\n"

    while len(a) > 0:
        try:
            f += a[:size]
            a = a[size:]
        except IndexError:
            f += a
            a = ""
        f += "\n"

    return f


def space_random_opt(n: int, space: int, upper_bound: int, lower_bound: int = 0):
    """
    Generate n random numbers with a minimal space between.

    n:int number of random numbers
    space:int minimum space between two consecutives random numbers
    upper_bound:int maximal number
    lower_bound:int minimal number
    """
    range_n = upper_bound - lower_bound
    assert range_n >= space * n, "sample too tiny"  # check if sample is large enough
    coeffs = np.random.rand(n)  # generate coefficients of spacing
    spacer = (range_n - n * space) / np.sum(coeffs)  # normalize coeffs
    spacerz = (spacer * coeffs) + space  # get distances
    idx = np.cumsum(spacerz).astype(int) + lower_bound  # get indices
    return idx


def loadbar(x, n, t0=0, strmore=""):
    print(
        f"|{'=' * (int(30 * x / (n - 1)) - 1)}>{'.' * int(30 * (1 - (x / (n - 1))))}|\
    {x + 1}/{n} {time.time() - t0:.2f}s"
        + strmore,
        end="\r",
        flush=True,
    )


def vcorrcoef(X, Y):
    # Alex Westbrook
    Xm = np.reshape(np.mean(X, axis=1), (X.shape[0], 1))
    Ym = np.reshape(np.mean(Y, axis=1), (Y.shape[0], 1))
    r_num = np.sum((X - Xm) * (Y - Ym), axis=1)
    r_den = np.sqrt(np.sum((X - Xm) ** 2, axis=1) * np.sum((Y - Ym) ** 2, axis=1))
    r = r_num / r_den
    return r


def gauss(sigma, mu=0, maxone=False):
    # Support for float sigma and generate x range with np.arange
    x_range = np.arange(-3 * sigma, 3 * sigma + 1)

    # Gaussian scaling factor
    tmps = 1 / (sigma * np.sqrt(2 * np.pi))

    # Define the Gaussian function
    def f(x):
        return 0.5 * ((x - mu) / sigma) ** 2

    # Generate the Gaussian kernel
    g = np.array([tmps * np.exp(-f(x)) for x in x_range])

    # Normalize if maxone flag is set, or return the original kernel
    if maxone:
        return g / np.max(g)  # Normalizes the kernel so its maximum value is 1
    else:
        return g


def create_ranges(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """from https://stackoverflow.com/questions/47125697/concatenate-range-arrays-given-start-stop-numbers-in-a-vectorized-way-numpy
        Answer by Divakar

    Create ranges element wise from two 1D arrays (starts, ends)
    Parameters
    ----------
    starts : np.ndarray[int]
        array of int same length as ends, where each starts[i]<ends[i]
    ends : np.ndarray[int]
        array of int same length as starts, where each starts[i]<ends[i]

    Returns
    -------
    np.ndarray[int]
        flat array of all ranges. equivalent to :
        tmp = []
        for i in range(len(start)):
            tmp.append(np.arange(start[i], end[i]))

        return np.hstack(tmp)
    """
    lenghts = ends - starts
    clens = lenghts.cumsum()
    ids = np.ones(clens[-1], dtype=int)
    ids[0] = starts[0]
    ids[clens[:-1]] = starts[1:] - ends[:-1] + 1
    out = ids.cumsum()
    return out


def create_bw(fname: str, chrom_sizes_path: str = "mm10"):
    assert not os.path.exists(fname), "File already exists, please remove it manually"
    if isinstance(chrom_sizes_path, str) or isinstance(chrom_sizes_path, Path):
        bw = pbw.open(fname, "wb")
        chrom_sizes = pd.read_csv(
            chrom_sizes_path,
            sep="\t",
            header=None,
        )
        order_dict = {f"chr{i}": i for i in range(1, 20)}
        chrom_sizes = chrom_sizes[
            np.isin(chrom_sizes[0], [f"chr{i}" for i in range(1, 20)])
        ]
        chrom_sizes = chrom_sizes.sort_values(by=0, key=lambda x: x.map(order_dict))
        header = list((zip(chrom_sizes[0].values, chrom_sizes[1].values)))
        bw.addHeader(header)
        return bw


    else:
        raise NotImplementedError


def kmer_frequencies(
    one_hot: Iterable[Iterable], k: int, order: str = "ACGT"
) -> Dict[str, int]:
    letters = np.array(list(order + "N"))
    arr = np.argmax(one_hot, axis=1) + 4 * (np.sum(one_hot, axis=1) != 1)
    kmer, count = np.unique(
        sliding_window_view(arr, k).reshape(-1, k), return_counts=True, axis=0
    )
    return {"".join(km): count[i] for i, km in enumerate(letters[kmer])}


def H(x):
    a = np.log2(x)
    a[np.isnan(a)] = 0
    a[np.isinf(a)] = 0
    return -np.sum(x * a, axis=1)


def meta_idx(rm, motif, chr, size):
    rttmp = rm[(rm.repName == motif) & (rm.genoName == chr)].copy()
    try:
        assert len(rttmp) > 0
    except AssertionError:
        return None
    strand = rttmp.strand.values == "-"
    offset = rttmp[["repStart", "repLeft"]].values
    offset = offset[np.arange(len(offset)), np.array(strand, dtype=int)]
    offset[strand] *= -1
    pos = rttmp[["genoStart", "genoEnd"]].values
    pos = pos[np.arange(pos.shape[0]), np.array(strand, dtype=int)]
    pos -= offset
    x = create_ranges(pos - size, pos + size).reshape((-1, 2 * size))
    x[strand, :] = x[strand, ::-1]
    return x, strand, np.abs(offset), (rttmp.genoEnd - rttmp.genoStart).values


def identify_axes(ax_dict, fontsize=48):
    """
    matplotlib mosaic (somplex semantic...) documentation
    Helper to identify the Axes in the examples below.

    Draws the label in a large font in the center of the Axes.

    Parameters
    ----------
    ax_dict : dict[str, Axes]
        Mapping between the title / label and the Axes.
    fontsize : int, optional
        How big the label should be.
    """
    kw = dict(ha="center", va="center", fontsize=fontsize, color="darkgrey")
    for k, ax in ax_dict.items():
        ax.text(0.5, 0.5, k, transform=ax.transAxes, **kw)


def period(cor):
    lag, cor = autocorr(cor)
    st = int(np.where(lag == 0)[0])
    lag = lag[st:]
    cor = cor[st:]
    return lag[np.argmin(cor) + np.argmax(cor[np.argmin(cor) :])]


def chrom_sizes_cumsum(chrom_sizes_path):
    chrom_sizes = pd.read_csv(chrom_sizes_path, sep="\t", header=None)
    chrom_sizes = chrom_sizes[chrom_sizes[0].isin([f"chr{i}" for i in range(1, 20)])]
    chrom_sizes = chrom_sizes.iloc[
        np.argsort(chrom_sizes[0].str.lstrip("chr").astype(int))
    ]
    chrom_sizes[2] = [0] + list(np.cumsum(chrom_sizes[1])[:-1] + 1)
    return chrom_sizes


def make_sparse_ones(
    df,
    start_column="p0",
    end_column="p1",
    chrom_column="chrom",
    chrom_sizes_path="/home/maxime/data/sequences/mm10/mm10.chrom.sizes",
):
    dftmp = df.copy()

    # coordinates + cum_sum
    chrom_sizes = chrom_sizes_cumsum(chrom_sizes_path)
    genome_size = np.sum(chrom_sizes[1])
    dict_chrom_sizes = dict(zip(list(chrom_sizes[0]), list(chrom_sizes[2].astype(int))))

    dftmp[start_column] += dftmp[chrom_column].map(dict_chrom_sizes)
    dftmp = dftmp[pd.isna(dftmp).sum(axis=1) == 0]

    dftmp[end_column] += dftmp[chrom_column].map(dict_chrom_sizes)
    dftmp[start_column] = dftmp[start_column].astype(int)
    dftmp[end_column] = dftmp[end_column].astype(int)

    lengths = dftmp[end_column] - dftmp[start_column]
    rows = np.repeat(np.arange(len(dftmp)), lengths)
    columns = create_ranges(dftmp[start_column].values, dftmp[end_column].values)
    return sparse.csr_matrix(
        (np.ones(len(rows)), (rows, columns)), shape=(len(dftmp), genome_size)
    )


def smith_waterman(seq1, seq2):
    """
    modified from https://github.com/slavianap/Smith-Waterman-Algorithm
    """

    # Assigning the constants for the scores
    class Score(IntEnum):
        MATCH = 1
        MISMATCH = -1
        GAP = -10

    # Assigning the constant values for the traceback
    class Trace(IntEnum):
        STOP = 0
        LEFT = 1
        UP = 2
        DIAGONAL = 3

    # Generating the empty matrices for storing scores and tracing
    row = len(seq1) + 1
    col = len(seq2) + 1
    matrix = np.zeros(shape=(row, col), dtype=np.int)
    tracing_matrix = np.zeros(shape=(row, col), dtype=np.int)

    # Initialising the variables to find the highest scoring cell
    max_score = -1
    max_index = (-1, -1)

    # Calculating the scores for all cells in the matrix
    for i in range(1, row):
        for j in range(1, col):
            # Calculating the diagonal score (match score)
            match_value = Score.MATCH if seq1[i - 1] == seq2[j - 1] else Score.MISMATCH
            diagonal_score = matrix[i - 1, j - 1] + match_value

            # Calculating the vertical gap score
            vertical_score = matrix[i - 1, j] + Score.GAP

            # Calculating the horizontal gap score
            horizontal_score = matrix[i, j - 1] + Score.GAP

            # Taking the highest score
            matrix[i, j] = max(0, diagonal_score, vertical_score, horizontal_score)

            # Tracking where the cell's value is coming from
            if matrix[i, j] == 0:
                tracing_matrix[i, j] = Trace.STOP

            elif matrix[i, j] == horizontal_score:
                tracing_matrix[i, j] = Trace.LEFT

            elif matrix[i, j] == vertical_score:
                tracing_matrix[i, j] = Trace.UP

            elif matrix[i, j] == diagonal_score:
                tracing_matrix[i, j] = Trace.DIAGONAL

            # Tracking the cell with the maximum score
            if matrix[i, j] >= max_score:
                max_index = (i, j)
                max_score = matrix[i, j]

    # Initialising the variables for tracing
    aligned_seq1 = ""
    aligned_seq2 = ""
    current_aligned_seq1 = ""
    current_aligned_seq2 = ""
    (max_i, max_j) = max_index
    aligned_seq1 = "-" * (len(seq1) - (max_i))
    aligned_seq2 = "-" * (len(seq2) - (max_j))
    # Tracing and computing the pathway with the local alignment
    while tracing_matrix[max_i, max_j] != Trace.STOP:
        if tracing_matrix[max_i, max_j] == Trace.DIAGONAL:
            current_aligned_seq1 = seq1[max_i - 1]
            current_aligned_seq2 = seq2[max_j - 1]
            max_i = max_i - 1
            max_j = max_j - 1

        elif tracing_matrix[max_i, max_j] == Trace.UP:
            current_aligned_seq1 = seq1[max_i - 1]
            current_aligned_seq2 = "-"
            max_i = max_i - 1

        elif tracing_matrix[max_i, max_j] == Trace.LEFT:
            current_aligned_seq1 = "-"
            current_aligned_seq2 = seq2[max_j - 1]
            max_j = max_j - 1

        aligned_seq1 = aligned_seq1 + current_aligned_seq1
        aligned_seq2 = aligned_seq2 + current_aligned_seq2

    # Reversing the order of the sequences
    aligned_seq1 = aligned_seq1[::-1]
    aligned_seq2 = aligned_seq2[::-1]

    return aligned_seq1, aligned_seq2


def localign(ref, sequences, n):
    """
    sequences must be 'ATGC' one-hot encoded
    """
    offsets = [0]
    ref = "".join(BASES[np.argmax(ref, axis=1)])
    for i, t in enumerate(sequences):
        if i <= n or i >= len(sequences) - n:
            aseq1, aseq2 = smith_waterman(ref, "".join(BASES[np.argmax(t, axis=1)]))
            off1 = len(ref) - len(aseq1) - 1
            off2 = len(t) - len(aseq2) - 1
            offsets.append(off2 - off1)

        else:
            offsets.append(0)
        loadbar(i, len(sequences))

    return offsets

def savefig(fname, figure=None):
    if figure is None:
        figure = plt
    
    figure.savefig(fname +".svg")
    figure.savefig(fname + ".pdf", dpi=300)