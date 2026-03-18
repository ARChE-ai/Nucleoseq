import pyBigWig as pbw
import numpy as np
from pathlib import Path
from utils import create_bw, loadnp, loadbar
import argparse
from time import time



parser = argparse.ArgumentParser(
    prog="npToBW",
    description= "Transform a set of numpy file to bigwig, saved in the same directory as the numpy file"
    )

parser.add_argument("filename", help="name of the saved bw")
parser.add_argument("np_format", help="path to the numpy file with {} to substitute chrom number : path/to/files/chrom{}.npz")
parser.add_argument("--chromsizes_file", help = "path to chrom.sizes", default="../demo/labels/mm10.chrom.sizes")
parser.add_argument("--chromosomes", help = "chromosomes numbers")
parser.add_argument("-m", "--mean", help = "mean on axis 1", action="store_true", default=False)

args = parser.parse_args()
filename = Path(args.filename)
np_format = args.np_format
run_path = Path(np_format).parent
chromsizes = Path(args.chromsizes_file)
take_mean = args.mean
t0 = time()
file_path_str = str(run_path / filename)
bw = create_bw(file_path_str, chromsizes)
print(f"Create {file_path_str}")
for chrom in range(1, 20):
    vals = loadnp(np_format.format(chrom))
    if take_mean:
        vals = np.mean(vals, axis=1)
    try:
        bw.addEntries(
            f"chr{chrom}",
            0,
            values = vals,
            span = 1,
            step = 1
        )
    except RuntimeError as e:
        bw.close()
        raise e
    loadbar(chrom, 20, t0)
print("SAVING...")
bw.close()
print(f"DONE - {time()-t0:2f}s")


