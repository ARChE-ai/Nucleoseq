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

mkdir -p "$DATA_DIR"/{sequences,rnaseq,atac,dnase/labels,refTSS_3.3,JASPAR_TFBs_2024,CTCF/all,ZNF384,chemical,modeles,ISM_tracks,predicted_nucleosome_occupancy}
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

  # --- JASPAR TFBS predictions, mm10, CTCF (mencius.uio.no) ---
  # Note 2026 release only ships mm39, not mm10 - see conversation. Keep 2024/mm10 for now:
  wget -c -P "${DATA_DIR}/JASPAR_TFBs_2024" \
    "https://mencius.uio.no/JASPAR/JASPAR_TFBSs/2024/mm10/MA0139.2.tsv.gz"

  # --- refTSS 3.3 (mouse) ---
  REFTSS=https://reftss.riken.jp/datafiles/3.3/mouse
  wget -c -P "${DATA_DIR}/refTSS_3.3" "${REFTSS}/refTSS_v3.3_mouse_coordinate.mm10.bed.gz"
  wget -c -P "${DATA_DIR}/refTSS_3.3" "${REFTSS}/tss_classification/TSS.classification.mm10.gz"
  wget -c -P "${DATA_DIR}/refTSS_3.3" "${REFTSS}/gene_annotation/refTSS_v3.3_mouse_annotation.txt.gz"
  gunzip -kf "${DATA_DIR}"/refTSS_3.3/*.gz

  # --- ENCODE (mESC ES-E14): RNA-seq ENCSR000CWC, DNase-seq ENCSR000CMW, ZNF384 ChIP-seq ENCSR000ERV ---
  ENCODE=https://www.encodeproject.org/files
  wget -c -O "${DATA_DIR}/rnaseq/ENCFF907JEV_plusStrand.bigWig" "${ENCODE}/ENCFF907JEV/@@download/ENCFF907JEV.bigWig"
  wget -c -O "${DATA_DIR}/rnaseq/ENCFF373ZMQ_minusStrand.bigWig" "${ENCODE}/ENCFF373ZMQ/@@download/ENCFF373ZMQ.bigWig"
  wget -c -O "${DATA_DIR}/dnase/labels/DNase_ENCFF672DJH.bw" "${ENCODE}/ENCFF672DJH/@@download/ENCFF672DJH.bigWig"
  wget -c -O "${DATA_DIR}/ZNF384/ENCFF043AHA.bigWig" "${ENCODE}/ENCFF043AHA/@@download/ENCFF043AHA.bigWig"

  # --- ATAC-seq mESC (GEO GSE129673) ---
  wget -c -P "${DATA_DIR}/atac" https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3719nnn/GSM3719303/suppl/GSM3719303_ATAC_1_RPKM.bw
  wget -c -P "${DATA_DIR}/atac" https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3719nnn/GSM3719304/suppl/GSM3719304_ATAC_2_RPKM.bw

  # --- CTCF ChIP-seq (GEO GSE131356, Owens et al. 2019) ---
  # TODO: data/CTCF/all/ctcf.bed (peaks with score) - specify which GSE131356 sample it was built from.

  # --- Raw assay data (GEO) - used to train/label the models, not to reproduce figures ---
  # Chemical cleavage (GSE82127, Voong et al. 2016): NCP scores on mm9 (lift over to mm10, see paper Methods)
  wget -c -P "${DATA_DIR}/chemical" https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM2183nnn/GSM2183909/suppl/GSM2183909_Chemical_NCPscore.txt.gz
  # MNase-seq (GSE122589, Festuccia et al. 2019): no processed files on GEO, raw reads only (SRA).
  # Label sparse_A_16 was built from MNase_A_16_1-3 (GSM3475355, GSM3475356, GSM3475357).

  echo "Public downloads done. Items marked TODO above are not downloaded automatically."
fi

# ------------------------------------------------------------------------------
# 3. Zenodo deposit — trained models + genome-wide prediction/ISM tracks
# ------------------------------------------------------------------------------
if [ "$SKIP_ZENODO" = false ]; then
  echo "== [3/3] Downloading Zenodo data deposit =="
  ZENODO_RECORD_ID="22013958"   # https://doi.org/10.5281/zenodo.22013958

  pip install --quiet zenodo_get
  zenodo_get "$ZENODO_RECORD_ID" -o "${DATA_DIR}/_zenodo_download"
  # Move each file to its expected location (file names as in the Zenodo deposit)
  mv "${DATA_DIR}/_zenodo_download/MNaseseq.h5" "${DATA_DIR}/modeles/"
  mv "${DATA_DIR}/_zenodo_download/chemical_cleavage.h5" "${DATA_DIR}/modeles/"
  mv "${DATA_DIR}/_zenodo_download/MNaseseq_ism_mm10.bw" "${DATA_DIR}/ISM_tracks/"
  mv "${DATA_DIR}/_zenodo_download/ChemicalCleavage_ism_mm10.bw" "${DATA_DIR}/ISM_tracks/"
  mv "${DATA_DIR}/_zenodo_download/MNaseseq_prediction_mm10.bw" "${DATA_DIR}/predicted_nucleosome_occupancy/"
  mv "${DATA_DIR}/_zenodo_download/ChemicalCleavage_prediction_mm10.bw" "${DATA_DIR}/predicted_nucleosome_occupancy/"
  mv "${DATA_DIR}/_zenodo_download/crossmut.bw" "${DEMO_DIR}/test/crossmut.bw"
fi

echo "Done. Items marked TODO above still need their exact URLs filled in."
