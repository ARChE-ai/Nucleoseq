# Nucleoseq 🧬

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow 2.5](https://img.shields.io/badge/TensorFlow-2.5-orange.svg)
![License CC-BY 4.0](https://img.shields.io/badge/License-CC--BY_4.0-green.svg)

**Deep-learning models and genome-wide nucleosome occupancy predictions**

Nucleoseq provides trained convolutional neural network (CNN) models and genome-wide prediction
tracks of nucleosome occupancy and in-silico mutagenesis (ISM) in mouse embryonic stem cells
(mESC, genome assembly *mm10*). The models were trained on MNase-seq and chemical cleavage
datasets to predict nucleosome occupancy directly from DNA sequence.

This README documents the full pipeline end to end — training, ISM, and motif
discovery/enrichment — so it can be reproduced from scratch on your own machine.

---

## 📖 Table of Contents
- [Repository structure](#️-repository-structure)
- [Installation](#-installation)
- [Getting the data](#-getting-the-data)
- [Pipeline overview](#-pipeline-overview)
- [Phase 0 — Data preparation](#phase-0--data-preparation)
- [Phase 1 — Training a model](#phase-1--training-a-model)
- [Phase 2 — In-silico mutagenesis (ISM) & genome-wide predictions](#phase-2--in-silico-mutagenesis-ism--genome-wide-predictions)
- [Phase 3 — Motif discovery (STREME/XSTREME) & motif-vs-ISM analysis (FIMO)](#phase-3--motif-discovery-stremexstreme--motif-vs-ism-analysis-fimo)
- [Figures & notebooks](#-figures--notebooks)
- [Methods & architectures](#-methods--architectures)
- [Source datasets](#-source-datasets)
- [Known limitations / things to double-check before you rely on them](#-known-limitations--things-to-double-check-before-you-rely-on-them)
- [Citation, License & Contact](#-citation-license--contact)

---

## 🗂️ Repository structure

```text
.
├── config/                  # YAML configs for every phase below (templates — copy & edit, don't run as-is)
├── data/                    # Reference genome, labels, trained models, prediction/ISM tracks (gitignored, see below)
├── demo/                    # Run outputs: training runs, ISM outputs, XSTREME/FIMO runs (gitignored, see below)
├── notebooks/                # Jupyter notebooks used to build the paper's figures
├── src/                     # Core pipeline scripts (training, ISM, motif analysis)
├── environment.yml          # Unified conda + pip environment definition
├── LICENSE                  # CC-BY 4.0
└── CITATION.cff             # Citation metadata
```

`data/` and `demo/` are excluded from git (see `.gitignore`) because they hold multi-GB
binaries — reference genome, bigWig tracks, trained models, run outputs. See
[Getting the data](#-getting-the-data) for how to populate them.

### Scripts in `src/`

| Script | Role |
|---|---|
| `fasta_ohe.py` | One-hot encode a multi-contig FASTA into per-chromosome `.npy` arrays |
| `train.py` | Train a model from a YAML config |
| `modeles.py` | CNN architecture definitions (`CNN_simple5H`, `Chemical_5H`) |
| `generator_opt_determinist.py` | Keras data generator used by `train.py` (`KDNAmulti_bw`) |
| `losses.py` | Custom loss (`MAE + (1 - Pearson r)`) and metrics |
| `ism.py` | In-silico mutagenesis + genome-wide prediction, per chromosome |
| `npzToBw.py` | Assemble per-chromosome `.npz` arrays (predictions or ISM scores) into one genome-wide bigWig |
| `build_sequence_bw.py` | Build the genome-wide integer-encoded sequence bigWig used by `train.py` |
| `build_nprs_fasta.py` | Build the candidate-region FASTA (`nprs.fasta`) fed to XSTREME, from two models' ISM tracks |
| `fimo_by_motif.sh` | Run FIMO separately for every motif in a combined MEME file (parallelised) |
| `fimo_by_motif_plots.py` | Correlate each motif's FIMO hits against the ISM tracks, one logo+scatter plot per motif |
| `synthetic_kmers.py`, `synthetic_CTCF.py`, `synthetic_satellites.py` | Synthetic-sequence experiments (k-mer conservation, CTCF background variation, satellite repeats) |
| `wavelet_transform.py` | Wavelet transforms on genomic tracks |
| `utils.py` | Shared helpers (bigWig I/O, one-hot handling, plotting helpers, stats) |

---

## 🛠️ Installation

```bash
conda env create -f environment.yml
conda activate nucleoseq
```

You'll also need the **MEME Suite** (for `streme`, `xstreme`, `fimo` — used in Phase 3), which is
not a Python package. Easiest via conda:
```bash
conda install -c bioconda meme
```
or follow the instructions at [meme-suite.org](https://meme-suite.org/meme/doc/install.html).

A GPU (with CUDA 11.2 / cuDNN 8.1, matching `environment.yml`) is strongly recommended for
training and ISM — both run a CNN forward pass over the entire genome.

---

## 📦 Getting the data

Three kinds of external data feed this pipeline, and they live in three different places —
don't try to route any of them through git:

1. **Public reference data** (mm10 genome, RepeatMasker, JASPAR, ENCODE tracks, refTSS) —
   download scripts / instructions, not redistributed here.
2. **Our trained models + genome-wide prediction/ISM tracks** — published as a separate Zenodo
   data deposit (DOI: *TODO once published*), because they're tens of GB.
3. **Anything you regenerate yourself** by running the phases below on your own sequencing data.

If a `setup_download.sh` / Zenodo record isn't set up yet in your copy of this repo, ask the
maintainer, or see `GIT_ZENODO_SUBMISSION_PLAN.md` for the publishing plan.

Expected layout once populated (paths referenced throughout this README and in
`notebooks/figures_new_clean.ipynb`'s `PATHS` dict):

```text
data/
├── sequences/
│   ├── mm10.fa                       # reference genome (public download)
│   ├── mm10.chrom.sizes              # chromosome sizes (public download)
│   ├── mm10.bw                       # genome-wide integer-encoded sequence track (see Phase 0)
│   ├── chr*.npy                      # per-chromosome one-hot arrays (see Phase 0)
│   ├── repeat_masker                 # UCSC RepeatMasker track (public download)
│   └── JASPAR2026_vertebrates.meme   # JASPAR motif database (public download)
├── modeles/                          # trained models (Zenodo)
├── ISM_tracks/                       # genome-wide ISM bigWigs (Zenodo, or regenerate via Phase 2)
└── predicted_nucleosome_occupancy/   # genome-wide prediction bigWigs (Zenodo, or regenerate via Phase 2)

demo/
├── labels/                           # training label bigWigs (MNase-seq / chemical cleavage, public GEO accessions)
└── test/                             # every training/ISM/XSTREME run lands here, one subdirectory per run
```

---

## 🚀 Pipeline overview

```text
  reference FASTA
        │  Phase 0 (fasta_ohe.py + build_sequence_bw.py)
        ▼
  per-chr one-hot .npy/.npz  +  genome-wide sequence .bw
        │                              │
        │  Phase 1 (train.py)          │  Phase 2 (ism.py) — run once per model
        ▼                              ▼
   trained model (best.h5)  ──────►  per-chr prediction + ISM .npz
        (MNase model AND               │  npzToBw.py
         Chemical model)               ▼
                          genome-wide prediction.bw + ism.bw   (one pair per model)
                                       │
                                       │  Phase 3, step 1: build_nprs_fasta.py
                                       │  (needs BOTH models' ism.bw)
                                       ▼
                            nprs.fasta + crossmut.bw
                                       │
                                       │  Phase 3, steps 2-4: external XSTREME → fimo_by_motif.sh → fimo_by_motif_plots.py
                                       ▼
                    motif discovery + per-motif FIMO hits + motif-vs-ISM correlation plots
```

---

## Phase 0 — Data preparation

**1. One-hot encode the reference genome**, one array per chromosome:
```bash
python src/fasta_ohe.py --fasta data/sequences/mm10.fa --out data/sequences --mm10_canonical_only
```
This writes `data/sequences/chr1.npy` … `chr19.npy` (`chrX`/`chrY`/`chrM` too, without the
`--mm10_canonical_only` flag), each a `(L, 4)` uint8 one-hot array.

**2. Wrap each `.npy` into a `.npz`** (`ism.py` expects `.npz` with the array stored under the
key `arr_0`, i.e. a plain `np.savez`, not `.npy` directly):
```python
import numpy as np
for c in list(range(1, 20)) + ["X", "Y", "M"]:
    arr = np.load(f"data/sequences/chr{c}.npy")
    np.savez(f"data/sequences/chr{c}.npz", arr)
```
(This is why you'll see both `.npy` and `.npz` per chromosome in `data/sequences/` — it's not a
mistake, `ism.py` and other scripts consume the two formats differently. If you don't need the
`.npy` copy afterwards, you can delete it to save space — a compressed `.npz`
(`np.savez_compressed`) would also work and roughly halve the size, but isn't required.)

**3. Build the genome-wide sequence bigWig** used by `train.py` (`data.seq_path` in
`config_train.yaml`). `train.py`'s data generator (`KDNAmulti_bw` in
`generator_opt_determinist.py`) reads sequence as a **single bigWig covering the whole genome**,
where each position's value is the base index (0=A, 1=C, 2=G, 3=T, **-1**=N/unmapped) — not the
one-hot `.npy`/`.npz` arrays from steps 1–2:
```bash
python src/build_sequence_bw.py \
    --seq_dir data/sequences \
    --chrom_sizes data/sequences/mm10.chrom.sizes \
    --out data/sequences/mm10.bw
```

---

## Phase 1 — Training a model

1. Copy a template config and edit the paths (`config_train.yaml` currently has an absolute
   `/home/maxime/...`-style path baked in from the original run — replace with yours):
   ```bash
   cp config/config_train.yaml config/my_train.yaml
   ```
   Key fields:
   | Key | Meaning |
   |---|---|
   | `data.seq_path` | path to the genome-wide sequence bigWig from Phase 0 step 3 |
   | `data.label_path` | path to the training-label bigWig (e.g. `demo/labels/sparse_A_16_gaussiansmoothed.bw` for MNase-seq, or the chemical-cleavage equivalent) |
   | `data.mask_train` / `mask_val` | list of bigWig paths defining which genomic positions are usable (e.g. mappability, blacklist) |
   | `model.name` | `CNN_simple5H` or `Chemical_5H` (see [Methods](#-methods--architectures)) |
   | `model.winsize` | input window size, default 2001 bp |
   | `training.train_chr` / `val_chr` | chromosome numbers for train/validation split |
   | `output_dir` | where run directories are created, e.g. `demo/test` |
   | `experiment_name` | prefix for the run directory name |

2. Launch training:
   ```bash
   python src/train.py --config config/my_train.yaml
   ```
   This creates `demo/test/<experiment_name>_<timestamp>/` containing:
   - `best.h5` — best checkpoint by validation loss (this is the file every downstream phase
     expects, e.g. `<run_dir>/best.h5`)
   - `epochs/epoch_NNN.h5` — per-epoch weight snapshots
   - `history` — CSV of the training history
   - `used_train_index.csv` — the exact training windows sampled
   - `train_config_used.yaml` — a copy of the config used, for provenance

---

## Phase 2 — In-silico mutagenesis (ISM) & genome-wide predictions

`ism.py` does two things in the same pass, per chromosome: it computes the **baseline
prediction** at every position, and the **ISM score** (mean-squared prediction shift when each
base is mutated to the 3 alternatives).

1. Edit a config (`config_ism.yaml` is a template — its `model`/`seq_dir` paths are examples,
   not real defaults):
   ```yaml
   model: demo/test/<your_training_run_dir>       # directory containing best.h5
   chr: [1, 2, 3]                                  # chromosomes to process — start small, this is expensive
   seq_dir: data/sequences/                        # directory with chr{N}.npz from Phase 0
   output_file: "ISM_run1"
   pos: 3000000                                    # start position on each chromosome
   # size: <leave commented to run to the end of the chromosome>
   batch_size: 2048
   gpu: 0
   ```
2. Run it:
   ```bash
   python src/ism.py --config config/my_ism.yaml
   ```
   Outputs land in `<model_dir>/_ism_outputs/`:
   - `{chrom}_{output_file}.npz` — ISM scores, shape `(L, 3)` (one column per alternative base)
   - `{chrom}_{output_file}PREDICTION.npz` — baseline predictions, shape `(L,)`

3. Assemble the per-chromosome arrays into genome-wide bigWigs:
   ```bash
   # Predictions
   python src/npzToBw.py predictions.bw \
       "<model_dir>/_ism_outputs/{}_ISM_run1PREDICTION.npz" \
       --chromsizes_file data/sequences/mm10.chrom.sizes

   # ISM scores (mean over the 3 alternative-base columns)
   python src/npzToBw.py ism.bw \
       "<model_dir>/_ism_outputs/{}_ISM_run1.npz" \
       --chromsizes_file data/sequences/mm10.chrom.sizes \
       --mean
   ```
   These `predictions.bw` / `ism.bw` files are exactly what `notebooks/figures_new_clean.ipynb`'s
   `PATHS` dict expects at `mnase_pred_bw`/`mnase_train_mutasome_bw` (or the `chem_*` equivalents)
   and what Phase 3's `fimo_by_motif_plots.py` reads as `mnase_ism_path`/`chem_ism_path`.

   > Note: `src/npzToBw.py` takes a **positional** `filename` and `np_format` argument (not
   > `--flags`) — the `{}` in `np_format` is substituted with chromosome numbers 1–19.

---

## Phase 3 — Motif discovery (STREME/XSTREME) & motif-vs-ISM analysis (FIMO)

This phase asks: *which sequence motifs does the model's ISM signal concentrate on?* It needs ISM
tracks from **two** trained models (MNase-seq and chemical-cleavage — run Phase 1 + Phase 2 once
for each), then discovers motifs and correlates them against both ISM tracks.

**1. Build candidate regions (`nprs.fasta`)** from the two models' ISM tracks. This combines the
MNase and chemical ISM signal (geometric mean of z-scored, smoothed tracks) into a single
"cross-mutasome" track, finds its local peaks, and extracts a fixed-width window of sequence
around each one — these peak regions are the input XSTREME will search for motifs in:
```bash
python src/build_nprs_fasta.py \
    --mnase_ism_bw demo/test/<mnase_run_dir>/_ism_outputs/ism.bw \
    --chem_ism_bw  demo/test/<chem_run_dir>/_ism_outputs/ism.bw \
    --seq_bw data/sequences/mm10.bw \
    --chrom_sizes data/sequences/mm10.chrom.sizes \
    --out_crossmut_bw demo/test/crossmut.bw \
    --out_fasta demo/test/nprs.fasta \
    --out_csv demo/test/nprs.csv
```
(This also produces `crossmut.bw`, referenced as `crossmut_bw` in
`notebooks/figures_new_clean.ipynb`'s `PATHS` dict — it's fully regeneratable from the two ISM
tracks, so it doesn't need to be archived separately.)

**2. Motif discovery with XSTREME** (external tool, not a repo script):
```bash
xstreme --p demo/test/nprs.fasta \
        --oc demo/test/xstreme_$(date +%y%m%d)_maxw30_JASPAR2026 \
        --meme-maxw 30 \
        --m data/sequences/JASPAR2026_vertebrates.meme
```
This produces the standard XSTREME output layout (`streme_out/`, `meme_out/`, `*_tomtom_out/`,
`xstreme.html`) plus `combined.meme` — the single, sequentially-renumbered MEME file
(`MOTIF 1 MA0139.2-CTCF`, `MOTIF 2 ...-STREME-1`, etc.) that `fimo_by_motif.sh` and
`fimo_by_motif_plots.py` read directly; no separate merge step needed.

**3. Scan every motif genome-wide with FIMO** (parallelised, one motif per job):
```bash
src/fimo_by_motif.sh \
    --meme demo/test/xstreme_.../combined.meme \
    --fa   data/sequences/mm10.fa \
    --out  demo/test/xstreme_.../fimo_by_motif \
    --thresh 5e-2
```
This creates one subdirectory per motif (named by the motif's numeric ID from `combined.meme`,
e.g. `fimo_by_motif/1/`, `fimo_by_motif/2/`, …), each containing `fimo.tsv`. Requires `fimo` on
your `PATH` (part of the MEME Suite). Re-running is safe — it skips motifs that already have a
non-empty `fimo.tsv`.

**4. Correlate each motif's hits against the ISM tracks:**
```bash
cd src
python fimo_by_motif_plots.py
```
Edit the constants at the top of the script first — `XSTREME_DIR`, `MNASE_RUN_DIR`, `CHEM_RUN_DIR`
(pointing at your Phase 1 training run directories) and `motif_range` in `__main__` (defaults to
`range(1, 629)`, matching the example run's 628 motifs — adjust to however many motifs your
`combined.meme` actually has, i.e. `wc -l demo/test/xstreme_.../fimo_by_motif/motifs.txt`).
Outputs land in `<xstreme_dir>/fimo_by_motif/_plots/`: a logo+scatter SVG/PDF per motif, a
per-motif single-hit CSV, and `_fimo_correlations.csv` summarising the Spearman correlation
between each motif's sequence information content and ISM score.

---

## 📊 Figures & notebooks

The figures in the paper are built from the tracks/models produced above, in
`notebooks/figures_new_clean.ipynb` (main figures) and `notebooks/microc_ecdf_clean.ipynb` (Micro-C
ECDF panel). Both notebooks resolve all paths relative to the repo root via a `PATHS` dict defined
in an early cell — edit the values there (not scattered through the notebook) if your data layout
differs from the one described above. `notebooks/genome_research.mplstyle` is the shared plotting
style; both notebooks load it automatically.

---

## 🧠 Methods & Architectures

Two CNN architectures (`src/modeles.py`):
1. **CNN_simple5H** — lightweight 3-layer CNN (32 filters, kernel lengths 3/10/20 bp), used for
   MNase-seq.
2. **Chemical_5H** — deeper (256→64→64 filters, kernel lengths 5/11/21 bp), used for
   chemical-cleavage data.

Both: each convolution is followed by batch normalisation and max-pooling; multi-head output via
a dense layer + sigmoid; loss `L = MAE + (1 − Pearson r)` with bin-wise re-weighting
(`src/losses.py`); trained on mm10 unique-mapping regions, genome-wide predictions inferred over
sliding 2001 bp windows.

---

## 📊 Source Datasets

- **MNase-seq**: GSE122589 (Pablo Navarro laboratory, Institut Pasteur)
- **Chemical cleavage**: GSE82127 (Voong et al., 2017)

---

## ⚠️ Known limitations / things to double-check before you rely on them

- `config/config_ism.yaml` and `config/config.yaml` contain example/stale absolute paths from the
  original development machine — always copy and edit, never run a template config as-is.
- Phase 3 needs ISM tracks from **two** separately trained models (MNase-seq and chemical
  cleavage) — run Phases 1–2 twice, once per architecture/dataset, before `build_nprs_fasta.py`.
- `src/build_sequence_bw.py` and `src/build_nprs_fasta.py` are new additions, script-ified from
  the original development notebook/generator logic — sanity-check their output (e.g. spot-check
  a few `mm10.bw` positions against the FASTA) before relying on them for a full run.

---

## 📜 Citation, License & Contact

- **License**: [CC-BY 4.0](LICENSE)
- **Citation**: see [`CITATION.cff`](CITATION.cff)
- **Contact**: maxime.christophe1@gmail.com (Maxime Christophe)
