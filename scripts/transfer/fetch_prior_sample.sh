#!/bin/bash
# =============================================================================
#  Copy prior pools from the cluster to this machine, for local visualisation.
#
#      bash scripts/transfer/fetch_prior_sample.sh                 # 1 shard per pool  (~1 GB)
#      bash scripts/transfer/fetch_prior_sample.sh --shards 3      # 3 shards per pool (~3 GB)
#      bash scripts/transfer/fetch_prior_sample.sh --full          # everything       (~19 GB)
#
#  WHY THE DEFAULT IS ONE SHARD, NOT EVERYTHING
#
#  A full pool is about 4 GB for LGD and 5.4 GB for PD, per variant. Four pools
#  (2 tasks x 2 variants) is ~19 GB, and it grows with every variant you add.
#
#  The notebooks draw a few hundred datasets. One shard is 2,000 — twenty times
#  more than they use — for ~200-270 MB. `PoolReader` globs whatever shards are
#  present, so a partial copy works with no special handling, and the notebook
#  labels such a pool a SAMPLE so you can never mistake it for the whole thing.
#
#  Download the full pools only if you intend to TRAIN locally. For looking at
#  them, one shard tells you everything a hundred would.
#
#  Set VSC_USER first (or pass --user):
#      export VSC_USER=vsc3xxxxx
# =============================================================================

set -euo pipefail

SHARDS=1
FULL=0
USER_ID="${VSC_USER:-}"
REMOTE_HOST="${VSC_HOST:-login.hpc.kuleuven.be}"
REMOTE_ROOT="${VSC_STAGING:-/lustre1/project/stg_00211/CreditICL/prior_cache}"
LOCAL_ROOT="prior_cache"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --shards) SHARDS="$2"; shift 2 ;;
        --full)   FULL=1; shift ;;
        --user)   USER_ID="$2"; shift 2 ;;
        -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "$USER_ID" ]]; then
    echo "ERROR: set your VSC username first:" >&2
    echo "    export VSC_USER=vsc3xxxxx" >&2
    echo "  or: bash scripts/transfer/fetch_prior_sample.sh --user vsc3xxxxx" >&2
    exit 2
fi

REMOTE="${USER_ID}@${REMOTE_HOST}"
mkdir -p "$LOCAL_ROOT"

echo "remote : ${REMOTE}:${REMOTE_ROOT}"
echo "local  : $(pwd)/${LOCAL_ROOT}"
echo ""

# Ask the cluster which pools exist rather than assuming a list — that way a new
# variant is picked up without editing this script.
POOLS=$(ssh "$REMOTE" "ls -1 '${REMOTE_ROOT}' 2>/dev/null || true")
if [[ -z "$POOLS" ]]; then
    echo "No pools found at ${REMOTE_ROOT}." >&2
    echo "Generate them first:  bash scripts/submit_pipeline.sh both" >&2
    exit 1
fi

echo "pools found on the cluster:"
echo "$POOLS" | sed 's/^/    /'
echo ""

for pool in $POOLS; do
    mkdir -p "${LOCAL_ROOT}/${pool}"
    if [[ "$FULL" == "1" ]]; then
        echo ">>> ${pool}: full pool"
        # Trailing slash on the source: copy the CONTENTS, not the directory.
        rsync -avh --progress \
            "${REMOTE}:${REMOTE_ROOT}/${pool}/" "${LOCAL_ROOT}/${pool}/"
    else
        echo ">>> ${pool}: first ${SHARDS} shard(s)"
        # Both .pt and .json: the payload is what the plots read, the manifest is
        # what tells the notebook how much of the pool this is. Without the JSON a
        # complete-looking sample would report "? (no manifests)".
        files=()
        for ((i = 0; i < SHARDS; i++)); do
            n=$(printf "%05d" "$i")
            files+=("--include=shard_${n}.pt" "--include=shard_${n}.json")
        done
        rsync -avh --progress \
            "${files[@]}" --exclude='*' \
            "${REMOTE}:${REMOTE_ROOT}/${pool}/" "${LOCAL_ROOT}/${pool}/"
    fi
    echo ""
done

cat <<'EOF'
Done. Check what you have:

    python scripts/generate_prior.py --config config/LGD.yaml --status

Then open notebooks/prior_visualisation.ipynb — it discovers whatever is present.
EOF
