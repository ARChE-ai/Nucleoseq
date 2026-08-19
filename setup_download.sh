#!/usr/bin/env bash
# ==============================================================================
# setup_download.sh — provision the environment + fetch all external data for
# Nucleoseq (distance_svg).
#
# Three stages, each individually re-runnable (skip flags below):
#   1. Conda/pip environment          --skip-env
#   2. Public reference downloads     --skip-public   (not redistributed on Zenodo)
#   3. Zenodo data deposit            --skip-zenodo   (models + genome-wide tracks)
#
# TODO markers below need YOUR values filled in before this script is usable -
# I did not fabricate exact GEO/ENCODE/Zenodo URLs I couldn't verify.
# ==============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${REPO_ROOT}/data"
DEMO_DIR="${REPO_ROOT}/demo"

SKIP_ENV=false
SKIP_PUBLIC=false
SKIP_ZENODO=false
for arg in "$@"; do
  case "$arg" in
    --skip-env) SKIP_ENV=true ;;
    --skip-public) SKIP_PUBLIC=true ;;
    --skip-zenodo) SKIP_ZENODO=true ;;
    *) echo "Unknown flag: $arg" >&2; exit 1 ;;
  esac
done

mkdir -p "$DATA_DIR"/{sequences,rnaseq,atac,dnase,refTSS_3.3,JASPAR_TFBs_2024,modeles,ISM_tracks,predicted_nucleosome_occupancy}
mkdir -p "$DEMO_DIR/test"

# ------------------------------------------------------------------------------
# 1. Environment
# ------------------------------------------------------------------------------
if [ "$SKIP_ENV" = false ]; then
  echo "== [1/3] Creating conda environment from environment.yml =="
  conda env create -f "${REPO_ROOT}/environment.yml"
  echo "Activate with: conda activate nucleoseq"
fi

# ------------------------------------------------------------------------------
# 2. Public reference data (freely available elsewhere - not on Zenodo)
# ------------------------------------------------------------------------------
if [ "$SKIP_PUBLIC" = false ]; then
  echo "== [2/3] Downloading public reference data =="

  # --- mm10 genome sequence + chromosome sizes (UCSC goldenPath) ---
  wget -c -P "${DATA_DIR}/sequences" \
    https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.fa.gz
  gunzip -k "${DATA_DIR}/sequences/mm10.fa.gz"
  wget -c -O "${DATA_DIR}/sequences/mm10.chrom.sizes" \
    https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.chrom.sizes

  # --- RepeatMasker track (UCSC table dump, matches data/sequences/repeat_masker) ---
  wget -c -P "${DATA_DIR}/sequences" \
    https://hgdownload.soe.ucsc.edu/goldenPath/mm10/database/rmsk.txt.gz
  gunzip -k "${DATA_DIR}/sequences/rmsk.txt.gz"
  # NOTE: rmsk.txt.gz's column layout differs from the "repeat_masker" TSV your notebooks
  # read (genoName/genoStart/genoEnd/repName/repClass/repFamily columns) - add the
  # rename/reorder step here once you confirm the exact column mapping you used originally.

  # --- JASPAR2026 CORE PFMs (used to build config/jaspar_clusters2026.tsv derivatives) ---
  wget -c -O "${REPO_ROOT}/JASPAR2026_CORE_non-redundant_pfms_meme.txt" \
    https://jaspar.elixir.no/download/data/2026/CORE/JASPAR2026_CORE_non-redundant_pfms_meme.txt
  # TODO: confirm this exact JASPAR download-API path is still current before relying on it.

  # --- JASPAR TFBS predictions, mm10, CTCF (mencius.uio.no) ---
  # Note 2026 release only ships mm39, not mm10 - see conversation. Keep 2024/mm10 for now:
  wget -c -P "${DATA_DIR}/JASPAR_TFBs_2024" \
    "https://mencius.uio.no/JASPAR/JASPAR_TFBSs/2024/mm10/MA0139.2.tsv.gz"

  # --- refTSS 3.3 (mouse) ---
  # TODO: fill in the exact refTSS_v3.3_mouse download URLs for:
  #   refTSS_v3.3_mouse_coordinate.mm10.bed, TSS.classification.mm10,
  #   refTSS_v3.3_mouse_annotation.txt  (http://reftss.riken.jp/reftss/ downloads page)

  # --- ENCODE RNA-seq / ATAC-seq / DNase-seq bigwigs ---
  # TODO: fill in the exact ENCFF accession download URLs (https://www.encodeproject.org/files/<ENCFF...>/@@download/<ENCFF...>.bigWig)
  #   for ENCFF373ZMQ (RNA-seq minus), ENCFF907JEV (RNA-seq plus), ENCFF672DJH (DNase-seq),
  #   plus the two GSM3719303/GSM3719304 ATAC-seq tracks referenced in your PATHS dict.

  # --- Raw assay data (GEO) — used to train/label the models, not to reproduce figures ---
  # GSE122589 (MNase-seq, Institut Pasteur) and GSE82127 (chemical cleavage, Voong et al. 2017)
  # TODO: fill in specific GSM/supplementary-file URLs for whichever raw tracks your training
  # pipeline needs, if a reviewer must rebuild the labels from scratch.

  echo "Public downloads done. Items marked TODO above still need your exact URLs filled in."
fi

# ------------------------------------------------------------------------------
# 3. Zenodo deposit — trained models + genome-wide prediction/ISM tracks
# ------------------------------------------------------------------------------
if [ "$SKIP_ZENODO" = false ]; then
  echo "== [3/3] Downloading Zenodo data deposit =="
  ZENODO_RECORD_ID="TODO_FILL_IN_AFTER_PUBLISHING"   # e.g. 1234567 from your Zenodo record URL

  if [ "$ZENODO_RECORD_ID" = "TODO_FILL_IN_AFTER_PUBLISHING" ]; then
    echo "Skipping Zenodo download: set ZENODO_RECORD_ID in this script once the deposit" \
         "from GIT_ZENODO_SUBMISSION_PLAN.md step 2B is published." >&2
  else
    pip install --quiet zenodo_get
    zenodo_get "$ZENODO_RECORD_ID" -o "${DATA_DIR}/_zenodo_download"
    # Move each file to its expected location - adjust once the deposit's actual filenames
    # are known (zenodo_get preserves whatever names you uploaded them with):
    # mv "${DATA_DIR}/_zenodo_download/MNaseseq.h5" "${DATA_DIR}/modeles/"
    # mv "${DATA_DIR}/_zenodo_download/chemical_cleavage.h5" "${DATA_DIR}/modeles/"
    # mv "${DATA_DIR}/_zenodo_download/MNaseseq_ism_mm10.bw" "${DATA_DIR}/ISM_tracks/"
    # mv "${DATA_DIR}/_zenodo_download/ChemicalCleavage_ism_mm10.bw" "${DATA_DIR}/ISM_tracks/"
    # mv "${DATA_DIR}/_zenodo_download/MNaseseq_prediction_mm10.bw" "${DATA_DIR}/predicted_nucleosome_occupancy/"
    # mv "${DATA_DIR}/_zenodo_download/ChemicalCleavage_prediction_mm10.bw" "${DATA_DIR}/predicted_nucleosome_occupancy/"
    # mv "${DATA_DIR}/_zenodo_download/crossmut.bw" "${DEMO_DIR}/test/crossmut.bw"
  fi
fi

echo "Done. See GIT_ZENODO_SUBMISSION_PLAN.md for what still needs to happen before all" \
     "TODOs above can be resolved (Zenodo deposit must exist first)."
