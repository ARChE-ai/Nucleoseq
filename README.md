# Nucleoseq 🧬

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow 2.5](https://img.shields.io/badge/TensorFlow-2.5-orange.svg)
![License CC-BY 4.0](https://img.shields.io/badge/License-CC--BY_4.0-green.svg)

**Deep-learning models and genome-wide nucleosome occupancy predictions**

Nucleoseq provides trained convolutional neural network (CNN) models and genome-wide prediction tracks of nucleosome occupancy and in-silico mutagenesis (ISM) in mouse embryonic stem cells (mESC, genome assembly *mm10*).

The models were trained on MNase-seq and chemical cleavage datasets to predict nucleosome occupancy directly from DNA sequences. 

---

## 📖 Table of Contents
- [Features](#-features)
- [Repository Structure](#-repository-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Methods & Architectures](#-methods--architectures)
- [Source Datasets](#-source-datasets)
- [License & Contact](#-license--contact)

---

## ✨ Features

- **Standardised Formats**: All prediction tracks and models are provided in standard, portable formats for reproducibility (bigWig, HDF5).
- **In-Silico Mutagenesis (ISM)**: Evaluate the predicted influence of single base-pair changes on nucleosome occupancy.
- **Multiple Architectures**: Supports flexible multi-head output prediction.

---

## 🗂️ Repository Structure

```text
.
├── config/                  # Configuration files (YAML) for training and inference
├── data/                    # Data storage
│   ├── ISM_tracks/          # ISM Δ-score tracks at 1 bp resolution (.bw)
│   ├── modeles/             # Trained Keras models (.h5)
│   └── predicted_nucleosome_occupancy/ # Genome-wide profiles at 1 bp resolution (.bw)
├── demo/                    # Demonstrations and sample artifacts
├── notebooks/               # Jupyter notebooks for data exploration and figures
├── src/                     # Core deep learning & processing Python scripts
├── environment.base38.yml   # Conda virtual environment definition
└── requirements.txt         # Pip requirements list
```

### Core Modules (`src/`)
- `train.py` : Main training script.
- `modeles.py` : CNN architecture definitions.
- `generator_opt_determinist.py` : Sequence data generators for model training and inference.
- `losses.py` : Custom loss functions (e.g., `MAE + (1 - r)`).
- `ism.py`, `synthetic_kmers.py` : In-silico mutagenesis and sequence analysis utilities.
- `utils.py` : General helper functions.

---

## 🛠️ Installation

### Using Conda (Recommended)
You can recreate the exact training environment using the provided `environment.base38.yml` and `requirements.txt`:

```bash
# 1. Create the conda environment
conda env create -f environment.base38.yml

# 2. Activate the environment
conda activate py38

# 3. Install the remaining pip dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Loading a Trained Model
You can directly load the pre-trained models using TensorFlow/Keras:

```python
import tensorflow as tf

# Load the MNase-seq nucleosome occupancy prediction model
model = tf.keras.models.load_model("data/modeles/MNaseseq.h5", compile=False)

model.summary()
```

### Training a New Model
To train a model from scratch, modify one of the templates in `config/` (e.g. `config_train.yaml`) and run:

```bash
python src/train.py --config config/config_train.yaml
```

### Working with Prediction Tracks
Prediction tracks (`.bw` files format) can be directly visualised in tools like the [UCSC Genome Browser](https://genome.ucsc.edu/) or processed analytically in Python using the `pyBigWig` library:

```python
import pyBigWig
bw = pyBigWig.open("data/predicted_nucleosome_occupancy/MNaseseq_prediction_mm10.bw")
# Retrieve data for a given locus
values = bw.values("chr1", 1000000, 1000500)
```

---

## 🧠 Methods & Architectures

We implemented two primary convolutional neural network architectures:
1. **CNN_simple5H**: A lightweight 3-layer CNN (32 filters, kernel lengths: 3, 10, 20 bp) optimised for general nucleosome occupancy prediction from MNase-seq data.
2. **Chemical_5H**: A deeper architecture (256 → 64 → 64 filters, kernel lengths: 5, 11, 21 bp) tailored for chemical-cleavage data.

**Key details:**
* **Layers**: Each convolution is followed by batch normalisation and max-pooling.
* **Outputs**: Multi-head prediction via a dense layer and final sigmoid layer.
* **Loss Function**: `L = MAE + (1 − Pearson r)` using bin-wise re-weighting of targets for balanced learning.
* **Sequence processing**: Trained on mm10 unique-mapping regions, with genome-wide predictions inferred over sliding windows of 2001 bp.

---

## 📊 Source Datasets

- **MNase-seq**: GSE122589 (Pablo Navarro laboratory, Institut Pasteur)
- **Chemical cleavage**: GSE82127 (Voong et al., 2017)

---

## 📜 License & Contact

- **License**: Creative Commons Attribution 4.0 International (CC-BY 4.0).
- **Contact**: For questions or feedback, please reach out to: maxime.christophe1@gmail.com
- **Maintainer**: Maxime Christophe
