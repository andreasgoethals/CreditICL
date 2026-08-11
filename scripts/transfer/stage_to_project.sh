#!/bin/bash
# =============================================================================
#  Copy the big, irreplaceable files to PROJECT STORAGE.
#
#  Run from a Genius LOGIN node (never a compute node — those cannot see all the
#  filesystems), from inside $VSC_DATA/CreditICL:
#
#      bash scripts/transfer/stage_to_project.sh            # show what would happen
#      bash scripts/transfer/stage_to_project.sh --go       # actually copy
#
#  WHAT GOES, AND WHAT DOES NOT
#
#    GOES   data/raw/         the 21 credit datasets. Irreplaceable, not
#                             redistributable, and too big for $VSC_DATA's 75 GiB.
#           checkpoints/      TabICL v2 and TabPFN v3 weights (~670 MB), so the
#                             eval pipeline finds them with no token and no
#                             internet.
#           data/processed/   the parquet cache, if you have already built it.
#                             Optional — scripts/preprocess.py can rebuild it.
#
#    DOES   the synthetic datasets we generate. Those are produced from a seed and
#    NOT   a config, so copying them would waste space on something reproducible.
#           logs, metrics, result CSVs — small, and they belong in $VSC_DATA where
#           they are backed up.
#
#  WHY COPY AND NOT MOVE: `mv` does not count as an "access", and $VSC_SCRATCH
#  deletes anything untouched for 30 days. Project storage has no purge, but the
#  copy-then-touch habit is worth keeping. rsync below preserves nothing that would
#  make a file look old.
# =============================================================================

set -euo pipefail

STAGING="${CREDITICL_STAGING_ROOT:-/lustre1/project/stg_00211/CreditICL}"
REPO="${VSC_DATA:-$(pwd)}/CreditICL"
[[ -d "${REPO}" ]] || REPO="$(pwd)"

DRY="--dry-run"
LABEL="DRY RUN — nothing will be copied. Add --go to do it for real."
if [[ "${1:-}" == "--go" ]]; then
    DRY=""
    LABEL="COPYING FOR REAL"
fi

echo "=============================================================="
echo " ${LABEL}"
echo "=============================================================="
echo " from : ${REPO}"
echo " to   : ${STAGING}"
echo

if [[ ! -d "$(dirname "${STAGING}")" ]]; then
    echo "ERROR: ${STAGING%/*} does not exist." >&2
    echo "       Are you on a login node? Compute nodes cannot see all filesystems." >&2
    echo "       Override the root with \$CREDITICL_STAGING_ROOT if it has moved." >&2
    exit 1
fi

[[ -n "${DRY}" ]] || mkdir -p "${STAGING}"/{data/raw,data/processed,checkpoints}

copy_tree () {
    local src="$1" dst="$2" what="$3"
    if [[ ! -d "${src}" ]]; then
        echo "-- skip ${what}: ${src} not present"
        return
    fi
    local size
    size="$(du -sh "${src}" 2>/dev/null | cut -f1)"
    echo "-- ${what}  (${size})"
    echo "   ${src}  ->  ${dst}"
    # --no-times so the copies look freshly accessed rather than inheriting an old
    # timestamp; -h for readable progress; --info=progress2 for one summary line.
    rsync -a --no-times --human-readable --info=progress2 ${DRY} "${src}/" "${dst}/"
    echo
}

copy_tree "${REPO}/data/raw"        "${STAGING}/data/raw"        "raw datasets (irreplaceable)"
copy_tree "${REPO}/checkpoints"     "${STAGING}/checkpoints"     "model weights"
copy_tree "${REPO}/data/processed"  "${STAGING}/data/processed"  "processed cache (optional, rebuildable)"

echo "=============================================================="
if [[ -n "${DRY}" ]]; then
    echo " That was a dry run. Re-run with --go to copy."
else
    echo " Done. Verify:"
    echo "   ls -la ${STAGING}/data/raw/lgd | head"
    echo "   du -sh ${STAGING}/*"
    echo
    echo " The code finds these automatically — src/utils/paths.py searches the"
    echo " repo first, then project storage. Nothing to configure."
fi
echo "=============================================================="
