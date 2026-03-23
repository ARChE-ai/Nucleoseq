#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# DEFAULTS
###############################################################################
THRESH="5e-2"
MAX_STORED="3000000"
VERBOSITY="1"
NO_QVALUE="--no-qvalue"
SKIP_MATCHED_SEQ=""

###############################################################################
# PARSING ARGUMENTS NOMMÉS
###############################################################################
MEME=""
FA=""
OUT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --meme)
      MEME="$2"
      shift 2
      ;;
    --fa)
      FA="$2"
      shift 2
      ;;
    --out)
      OUT="$2"
      shift 2
      ;;
    --thresh)
      THRESH="$2"
      shift 2
      ;;
    --max-stored)
      MAX_STORED="$2"
      shift 2
      ;;
    --verbosity)
      VERBOSITY="$2"
      shift 2
      ;;
    --skip-matched-sequence)
      SKIP_MATCHED_SEQ="--skip-matched-sequence"
      shift
      ;;
    --qvalue)
      NO_QVALUE=""
      shift
      ;;
    -h|--help)
      echo "Usage: $0 --meme FILE --fa FILE --out DIR [--thresh X] [--max-stored N] [--verbosity N] [--skip-matched-sequence] [--qvalue]"
      exit 0
      ;;
    *)
      echo "[ERROR] Argument inconnu : $1"
      exit 1
      ;;
  esac
done

###############################################################################
# VÉRIFICATION DES ARGUMENTS OBLIGATOIRES
###############################################################################
if [[ -z "$MEME" || -z "$FA" || -z "$OUT" ]]; then
  echo "[ERROR] Les arguments --meme, --fa et --out sont obligatoires."
  echo "Usage: $0 --meme FILE --fa FILE --out DIR [--thresh X] [--max-stored N] [--verbosity N] [--skip-matched-sequence] [--qvalue]"
  exit 1
fi

###############################################################################
# PREP + LOG DE LA COMMANDE
###############################################################################
mkdir -p "$OUT"
LOGDIR="$OUT/_logs"
mkdir -p "$LOGDIR"

CMDLOG="$OUT/command.sh"
printf '%q ' "$0" "$@" > /tmp/_cmd_args.$$  # optionnel si tu veux garder les args initiaux
rm -f /tmp/_cmd_args.$$

# meilleure solution : reconstruire la commande avec les variables résolues
{
  printf '%q ' "$0"
  printf -- '--meme %q ' "$MEME"
  printf -- '--fa %q ' "$FA"
  printf -- '--out %q ' "$OUT"
  printf -- '--thresh %q ' "$THRESH"
  printf -- '--max-stored %q ' "$MAX_STORED"
  printf -- '--verbosity %q ' "$VERBOSITY"
  [[ -n "$SKIP_MATCHED_SEQ" ]] && printf -- '--skip-matched-sequence '
  [[ -z "$NO_QVALUE" ]] || printf -- '--no-qvalue '
  echo
} > "$CMDLOG"

echo "[INFO] Command saved to: $CMDLOG"

MOTIF_LIST="$OUT/motifs.txt"
FAILED_LIST="$OUT/failed.txt"
DONE_LIST="$OUT/done.txt"

# Build motif list (IDs after "MOTIF")
grep "^MOTIF " "$MEME" | awk '{print $2}' > "$MOTIF_LIST"

# Reset status files (keep old ones if you prefer)
: > "$FAILED_LIST"
: > "$DONE_LIST"

echo "[INFO] MEME file : $MEME"
echo "[INFO] FASTA     : $FA"
echo "[INFO] OUT       : $OUT"
echo "[INFO] Motifs    : $(wc -l < "$MOTIF_LIST")"
echo

###############################################################################
# FUNCTIONS
###############################################################################
is_done() {
  # Consider a motif "done" if it has a non-empty fimo.tsv or fimo.txt
  local mdir="$1"
  if [[ -s "$mdir/fimo.tsv" ]] || [[ -s "$mdir/fimo.txt" ]]; then
    return 0
  fi
  return 1
}

run_one_motif() {
  local m="$1"
  local mdir="$OUT/$m"
  local log="$LOGDIR/${m}.log"

  mkdir -p "$mdir"

  # Skip if already done
  if is_done "$mdir"; then
    echo "[SKIP] $m (already has fimo.tsv/fimo.txt)"
    echo "$m" >> "$DONE_LIST"
    return 0
  fi

  echo "[RUN ] $m"
  {
    echo "=== $(date) ==="
    echo "Motif: $m"
    echo "Command:"
    echo "fimo --oc \"$mdir\" --verbosity $VERBOSITY --thresh $THRESH --max-stored-scores $MAX_STORED $NO_QVALUE $SKIP_MATCHED_SEQ --motif \"$m\" \"$MEME\" \"$FA\""
    echo
  } > "$log"

  # Run FIMO (append stdout/stderr to log)
  set +e
  fimo --oc "$mdir" \
       --verbosity "$VERBOSITY" \
       --thresh "$THRESH" \
       --max-stored-scores "$MAX_STORED" \
       $NO_QVALUE \
       $SKIP_MATCHED_SEQ \
       --motif "$m" \
       "$MEME" "$FA" >> "$log" 2>&1
  status=$?
  set -e

  # Check output
  if [[ $status -ne 0 ]]; then
    echo "[FAIL] $m (exit code $status)"
    echo "$m" >> "$FAILED_LIST"
    return 1
  fi

  if is_done "$mdir"; then
    echo "[OK  ] $m"
    echo "$m" >> "$DONE_LIST"
    return 0
  else
    echo "[FAIL] $m (no fimo.tsv/fimo.txt produced)"
    echo "$m" >> "$FAILED_LIST"
    return 1
  fi
}

###############################################################################
# MAIN LOOP
###############################################################################
total=$(wc -l < "$MOTIF_LIST")
i=0

while read -r m; do
  i=$((i+1))
  echo "------------------------------------------------------------"
  echo "[INFO] $i / $total : $m"
  run_one_motif "$m" || true
done < "$MOTIF_LIST"

echo "------------------------------------------------------------"
echo "[DONE] Finished."
echo "[DONE] Successful motifs: $(wc -l < "$DONE_LIST")"
echo "[DONE] Failed motifs    : $(wc -l < "$FAILED_LIST")"
echo
echo "Logs are in: $LOGDIR"
echo "Failed motif IDs (if any): $FAILED_LIST"