# -*- coding: utf-8 -*-
"""
@author: maxime christophe
"""

from typing import Iterable, Callable
import numpy as np
import pyBigWig as pbw
import tensorflow as tf

from utils import reshape_bin
from sklearn.utils.class_weight import compute_sample_weight
import pandas as pd


class KDNAmulti_bw(tf.keras.utils.Sequence):
    """
    Generate batches for keras.fit() from bigWig and one-hot sequences.

    Parameters
    ----------
    seq : list[str]
        Paths to npz/np arrays (one-hot, one per chromosome; loaded via loadnp).
    lab : str
        Path to bigWig label file (coverage/signal).
    chr : Iterable[int]
        Chromosome numbers to use (e.g. [1,2,3,19]).
    winsize : int
        Window size (odd), default 2001.
    frac : int
        Number of training windows per epoch (NOT a fraction).
    reverse : str
        "", "random", "reverse", or "augmentation".
    truncate : int
        Number of bp to discard from start (e.g. 3_000_000).
    batch_size : int
    mask : callable or None
        A function that returns positions to mask for a chromosome (partial prefilled).
    headsteps : np.ndarray
        Offsets for multi-head predictions (default [-500, -250, 0, 250, 500]).
    weights : bool
        If True, return sample weights; else return ones.
    index_selection : str
        "", "local", "no-overlap", or "validation".
    process_fun : callable or None
        Optional preprocessing function applied to label array for each chr.
    """

    def __init__(
        self,
        seq: str,
        lab: str = None,
        chr: Iterable[str] = ["chr1"],
        winsize: int = 2001,
        dist_between_windows: int = 300,
        reverse: str = "",
        batch_size: int = 2048,
        mask: int = None,
        headsteps: Iterable[int] = np.array([-500, -250, 0, 250, 500]),
        weights: str = "auto",
        rundir: str = "",
        workers: int = 10,
        training: bool = True,
    ):
        self.bw = lab
        self.chr = [f"chr{x}" for x in chr]
        print(f"------------{self.chr}")

        self.one_hot = seq
        self.winsize = winsize
        self.half = winsize // 2
        self.odd = winsize % 2
        self.reverse = reverse
        self.batch_size = batch_size
        self.step = headsteps
        self.weights = weights
        self.mask_bw = mask
        self.rundir = rundir
        self.workers = workers
        self.win_dist = dist_between_windows
        self.training = training

        self.select_chr()

    def select_chr(self, end=None):
        if self.weights not in ["auto", "batch"]:
            print(f"No weights will be applied as {self.weights} option is unknown")

        if end is None:
            self.index_epoch = []

        for chrom in self.chr:
            if self.mask_bw is not None:
                mask_tmp = []
                for m in self.mask_bw:
                    with pbw.open(m) as mask:
                        mask_local = mask.values(chrom, 0, -1, numpy=True)

                    mask_local = np.nan_to_num(mask_local, nan=0).astype(bool)
                    mask_local[: self.half + self.odd] = False
                    mask_local[-self.half - self.odd :] = False
                    mask_tmp.append(mask_local)

                used_mask = np.logical_and.reduce(mask_tmp)
            else:
                raise NotImplementedError

            index_list = np.flatnonzero(used_mask)
            keep_pos = [0]
            last_val = index_list[0]

            while True:
                j = np.searchsorted(index_list, last_val + self.win_dist, side="left")
                if j >= len(index_list):
                    break
                keep_pos.append(j)
                last_val = index_list[j]

            self.index_epoch.extend(
                [(chrom, x) for x in index_list[np.asarray(keep_pos)]]
            )
            self.frac = len(self.index_epoch)

        # Save idx
        self.index_epoch = pd.DataFrame(self.index_epoch, columns=["chrom", "center"])
        if self.training:
            self.index_epoch.to_csv(f"{self.rundir}/used_train_index.csv", index=False)
            self.index_epoch = self.index_epoch.sample(frac=1).reset_index(drop=True)
        
        if self.weights == "auto":
            vals = []
            step = np.asarray(self.step)
            min_head, max_head = step.min(), step.max()

            with pbw.open(self.bw) as bw:
                for chrom, dfc in self.index_epoch.groupby("chrom", sort=False):
                    centers = dfc["center"].to_numpy(np.int64)
                    if centers.size == 0:
                        continue

                    # On lit un bloc unique couvrant tous les centres (+heads)
                    start = int(centers.min() + min_head)
                    end   = int(centers.max() + max_head + 1)

                    signal = bw.values(chrom, start, end, numpy=True)
                    signal = np.nan_to_num(signal, nan=0.0).astype(np.float32)

                    # Indices absolus -> indices dans "signal"
                    # shape: (n_centers, n_heads)
                    idx = (centers[:, None] + step[None, :] - start).astype(np.int64)

                    # Récupération vectorisée
                    y = signal[idx]   # (n_centers, n_heads)
                    vals.append(y)

            vals = np.vstack(vals)  # (total_windows, n_heads)
            vals = np.round(np.nan_to_num(np.concatenate(vals), nan=0.0), 2)
            # option : ignorer les zéros
            mask = vals != 0
            vals_nz = vals[mask]

            unique, counts = np.unique(vals_nz, return_counts=True)
            N = vals_nz.size
            K = unique.size
            w = N / (K * counts)

            self.weight_map = dict(zip(unique.tolist(), w.astype(np.float32).tolist()))
            self.weight_zero = 0.0

        if self.weights == "auto":
            vals = []
            step = np.asarray(self.step)
            min_head, max_head = step.min(), step.max()

            with pbw.open(self.bw) as bw:
                for chrom, dfc in self.index_epoch.groupby("chrom", sort=False):
                    centers = dfc["center"].to_numpy(np.int64)
                    if centers.size == 0:
                        continue

                    # On lit un bloc unique couvrant tous les centres (+heads)
                    start = int(centers.min() + min_head)
                    end   = int(centers.max() + max_head + 1)

                    signal = bw.values(chrom, start, end, numpy=True)
                    signal = np.nan_to_num(signal, nan=0.0).astype(np.float32)

                    # Indices absolus -> indices dans "signal"
                    # shape: (n_centers, n_heads)
                    idx = (centers[:, None] + step[None, :] - start).astype(np.int64)

                    # Récupération vectorisée
                    y = signal[idx]   # (n_centers, n_heads)
                    vals.append(y)

            vals = np.vstack(vals)  # (total_windows, n_heads)
            vals = np.round(np.nan_to_num(np.concatenate(vals), nan=0.0), 2)
            # option : ignorer les zéros
            mask = vals != 0
            vals_nz = vals[mask]

            unique, counts = np.unique(vals_nz, return_counts=True)
            N = vals_nz.size
            K = unique.size
            w = N / (K * counts)

            self.weight_map = dict(zip(unique.tolist(), w.astype(np.float32).tolist()))
            self.weight_zero = 0.0

    def onehot_with_minus_one(self, idx, n_classes=4):
        idx = np.asarray(idx).astype(np.int8)
        L = idx.shape[0]

        out = np.zeros((L, n_classes), dtype=np.uint8)

        mask = idx >= 0  # ignore -1
        out[np.arange(L)[mask], idx[mask]] = 1

        return out.reshape((-1, self.winsize, n_classes))

    def _get_bw_values(self, path, chrom, min, max, steps):
        with pbw.open(path) as bw:
            return bw.values(chrom, min, max, numpy=True)[steps]

    def __len__(self):
        return int(np.ceil(len(self.index_epoch) / self.batch_size))

    def on_epoch_end(self):
        if not self.training:
            pass
        else:
            self.index_epoch = self.index_epoch.sample(frac=1).reset_index(drop=True)

    def __getitem__(self, index):
        step = np.asarray(self.step, dtype=np.int64)
        low = index * self.batch_size
        high = min(low + self.batch_size, len(self.index_epoch))
        batch = self.index_epoch.iloc[low:high].copy()  # DF columns: chrom, center
        n = len(batch)
        min_head, max_head = step.min(), step.max()

        # ---- SEQUENCES
        with pbw.open(self.one_hot) as handle_seq:
            idx_nuc_list = [
                handle_seq.values(
                    chrom, center - self.half, center + self.half + self.odd, numpy=True
                )
                for (chrom, center) in batch.itertuples(index=False)
            ]
        sequences = self.onehot_with_minus_one(np.concatenate(idx_nuc_list))

        # ---- LABELS
        with pbw.open(self.bw) as lab_handle:
            idx_lab_list = [
                lab_handle.values(
                    chrom,
                    center + min_head,
                    center + max_head+1,
                    numpy=True,
                )[step -min_head]
                for (chrom, center) in batch.itertuples(index=False)
            ]
        labels = np.array(idx_lab_list, dtype=np.float32).reshape((n, len(step)))
        
        
        # ---- WEIGHTS
        if self.weights=="batch":
            weights = np.zeros(labels.shape, dtype=np.float32)
            idx_weights = labels != 0
            if np.any(idx_weights):
                weights[idx_weights] = compute_sample_weight(
                    "balanced", np.round(labels[idx_weights], 2).ravel()
                )
            return sequences, labels, weights
        
        elif self.weights == "auto":
            labs = np.round(labels, 2)
            weights = np.ones_like(labels, dtype=np.float32)

            nz = labs != 0
            weights[~nz] = self.weight_zero  # 0.0

            if np.any(nz):
                flat = labs[nz].ravel()
                # lookup dict
                w_flat = np.fromiter((self.weight_map.get(v, 1.0) for v in flat), dtype=np.float32)
                weights[nz] = w_flat.reshape((-1,))

            return sequences, labels, weights
        
        else:
            return sequences, labels, np.ones(labels.shape, dtype=np.float32)
