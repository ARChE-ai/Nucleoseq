# Deep-learning models and genome-wide nucleosome occupancy predictions

This repository provides trained convolutional neural network (CNN) models and genome-wide
prediction tracks of nucleosome occupancy and in-silico mutagenesis (ISM) in mouse embryonic
stem cells (mESC, genome assembly mm10).

The models were trained on MNase-seq and chemical cleavage datasets to predict nucleosome
occupancy directly from DNA sequence. All files are provided in standard and portable formats
for reproducibility and re-use.

----------------------------------------------------------------------
Repository structure
----------------------------------------------------------------------

data/
 ├── ISM_tracks/
 │    ├── MNaseseq_ism_mm10.bw
 │    └── ChemicalCleavage_ism_mm10.bw
 │      → ISM (in-silico mutagenesis) Δ-score tracks at 1 bp resolution.
 │
 ├── predicted_nucleosome_occupancy/
 │    ├── MNaseseq_prediction_mm10.bw
 │    └── ChemicalCleavage_prediction_mm10.bw
 │      → Genome-wide predicted nucleosome occupancy profiles (1 bp resolution).
 │
 └── modeles/
      ├── MNaseseq.h5
      └── chemical_cleavage.h5
        → Trained Keras models (TensorFlow 2.5) for MNase-seq and chemical-cleavage data.

src/
 ├── train.py          – Training script
 ├── generator_opt.py  – Data generators for sequence windows
 ├── losses.py         – Custom losses and metrics (MAE + 1 − r)
 ├── ism.py            – ISM generation utilities
 ├── utils.py          – Helper functions
 └── modeles.py        – Model's architectures


config_example.yaml    – Example configuration file with input/output paths and parameters
README.txt             – This document

----------------------------------------------------------------------
Methods (summary)
----------------------------------------------------------------------

Two convolutional architectures were implemented:

• **CNN_simple5H** – lightweight 3-layer CNN (32 filters, kernel sizes 3/10/20 bp) for
  general nucleosome occupancy prediction from MNase-seq data.

• **Chemical_5H** – deeper architecture (256 → 64 → 64 filters, kernels 5/11/21 bp)
  optimised for chemical-cleavage data.

Each convolution is followed by batch normalisation and max-pooling; the flattened output
feeds a dense layer (8 units, ReLU) and a final 5-unit sigmoid layer (multi-head prediction).

Training minimised a composite loss:
   L = MAE + (1 − Pearson r),
with reported metrics being MAE and Pearson correlation.  
A bin-wise re-weighting of target values ensured balanced learning over the [0–1] range.

All models were trained on unique-mapping regions from mm10 using only well-covered
genomic windows. Genome-wide predictions were computed with sliding windows of 2001 bp.

----------------------------------------------------------------------
Source datasets
----------------------------------------------------------------------

• MNase-seq: GSE122589 (Pablo Navarro laboratory, Institut Pasteur)  
• Chemical cleavage: GSE82127 (Voong et al., 2017)


----------------------------------------------------------------------
Usage
----------------------------------------------------------------------

The trained models can be loaded with TensorFlow 2.x:

    from tensorflow.keras.models import load_model
    model = load_model("data/modeles/MNaseseq.h5", compile=False)


Prediction tracks (.bigWig) can be visualised in UCSC Genome Browser or processed using
pyBigWig in Python. Configuration examples for reproducing training and inference are
provided in config_example.yaml.


----------------------------------------------------------------------
Technical information
----------------------------------------------------------------------

Genome assembly: mm10 (Mus musculus)
File formats: bigWig (predictions, ISM), HDF5 (.h5 models), YAML (config)
License: CC-BY 4.0
Contact: [maxime.christophe1@gmail.com]
Date: October 2025