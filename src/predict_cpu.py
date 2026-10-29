"""
Minimal CPU prediction of nucleosome occupancy from a FASTA sequence.

Usage:
    python src/predict_cpu.py --model chemical --fasta tests/example_chr18.fa --out pred.tsv

--model is "chemical" or "mnase" (published model downloaded from Zenodo record 22013958,
cached in ~/.keras/models/) or a path to a local .h5 file.
Writes one line per predicted base (position<TAB>prediction, 1-based position in the
input sequence). The first and last winsize//2 bases are not predicted (incomplete context).
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force CPU

import argparse
import numpy as np
import utils as mf
import tensorflow as tf
from tensorflow.keras.mixed_precision import LossScaleOptimizer

# Published models (file name, md5) in Zenodo record 22013958
ZENODO_URL = "https://zenodo.org/records/22013958/files/{}?download=1"
ZENODO_MODELS = {
    "chemical": ("chemical_cleavage.h5", "b66853cf80a3fb5266c9ba34e2ea78e9"),
    "mnase": ("MNaseseq.h5", "788edfeb3edc81e12e1e113cf892b9b1"),
}


def read_fasta(path):
    with open(path) as f:
        return "".join(line.strip() for line in f if not line.startswith(">")).upper()


def one_hot(seq):
    # A, C, G, T columns; any other base (N) is all zeros
    arr = np.zeros((len(seq), 4), dtype=np.float32)
    b = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    for col, base in enumerate("ACGT"):
        arr[b == ord(base), col] = 1
    return arr


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="chemical",
                        help='"chemical", "mnase" (downloaded from Zenodo) or path to a local .h5')
    parser.add_argument("--fasta", required=True, help="FASTA file with a single sequence")
    parser.add_argument("--out", required=True, help="Output TSV file")
    parser.add_argument("--batch", type=int, default=1024)
    args = parser.parse_args()

    if args.model in ZENODO_MODELS:
        fname, md5 = ZENODO_MODELS[args.model]
        model_path = tf.keras.utils.get_file(fname, ZENODO_URL.format(fname), file_hash=md5)
    else:
        model_path = args.model

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "mae_cor": None,
            "correlate": None,
            "LossScaleOptimizer": LossScaleOptimizer,
            "Adam": None,
        },
        compile=False,
    )
    winsize = model.input_shape[1]

    seq = one_hot(read_fasta(args.fasta))
    if len(seq) < winsize:
        raise ValueError(f"sequence shorter than the model window ({winsize} bp)")

    windows = mf.sliding_window_view(seq, (winsize, 4), axis=(0, 1)).squeeze(1)
    pred = np.concatenate(
        [
            model.predict_on_batch(windows[i : i + args.batch])[:, 2]  # central output (step 0)
            for i in range(0, len(windows), args.batch)
        ]
    )

    positions = np.arange(len(pred)) + winsize // 2 + 1
    np.savetxt(args.out, np.c_[positions, pred], fmt=["%d", "%.5f"], delimiter="\t",
               header="position\tprediction", comments="")
    print(f"{len(pred)} predictions written to {args.out}")
