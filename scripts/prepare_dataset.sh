#!/usr/bin/env bash
# One-shot CASIA-HWDB 1.1 pipeline for the shared data/prepared dataset:
#   download the two training archives -> unpack writers -> glyphprep -> summary.
#
# The output is read by both DiT-flow and twist. Run it from the repository
# root; nothing else needs to be entered.
#
#   bash scripts/prepare_dataset.sh
#   bash scripts/prepare_dataset.sh --data-root /mnt/netdisk2/pattern/Characters/data
#
# Defaults, all under the repository root:
#   data/gnt/        unpacked writer files, one *.gnt per writer   (~3 GiB)
#   data/downloads/  cached archives                                (~1.8 GiB)
#   data/prepared/   prepared dataset for training                  (~3.4 GiB)
#
# Every stage is idempotent: a complete archive is reused, an unpacked
# directory is skipped, and a prepared directory holding manifest.json is
# skipped. --force rebuilds from scratch; interrupted downloads resume.
#
# The 4090/A100 servers keep home on a full partition, so point --data-root
# at the shared disk, e.g. /mnt/netdisk2/pattern/Characters/data.
#
# Requires curl (or wget) and unzip (or python3). The prepare stage needs an
# interpreter with torch, numpy, Pillow and fontTools; pick it with --python.

set -euo pipefail

readonly BASE_URL="https://nlpr.ia.ac.cn/databases/Download/Offline/CharData"
readonly ARCHIVES=("Gnt1.1TrainPart1.zip" "Gnt1.1TrainPart2.zip")
readonly GNT_MARKER=".gnt-complete"
readonly MIN_FREE_KIB=$((8 * 1024 * 1024))   # prepare needs ~3.4 GiB free

data_root=""
dest=""
cache=""
prepared=""
font=""
python_bin="${PYTHON:-}"
image_size=64
margin=4
val_fraction=0.1
seed=42
keep_zips=1
force=0
skip_download=0
skip_prepare=0

usage() {
    cat <<'USAGE'
Download, unpack and prepare the CASIA-HWDB 1.1 training data shared by
DiT-flow and twist.

Usage: bash scripts/prepare_dataset.sh [options]

Paths (defaults derive from <repo>/data):
  --data-root DIR   parent of gnt/, downloads/ and prepared/
  --dest DIR        unpacked writer files          (default <data-root>/gnt)
  --cache DIR       downloaded archives            (default <data-root>/downloads)
  --prepared DIR    prepared dataset               (default <data-root>/prepared)

Prepare stage:
  --python EXE      interpreter with torch, numpy, Pillow, fontTools
                    (default: $PYTHON, else python3, else python)
  --font FILE       standard-glyph font
                    (default <repo>/fonts/WenJinMinchoP0-Regular.ttf)
  --image-size N    square canvas, default 64
  --margin N        white border, default 4
  --val-fraction F  writer share used for validation, default 0.1
  --seed N          writer split seed, default 42

Behaviour:
  --delete-zips     remove the archives after a successful unpack
  --skip-download   use whatever is already cached, never fetch
  --skip-prepare    stop after unpacking
  --force           re-download, re-unpack and rebuild prepared output
  -h, --help        show this help
USAGE
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

log() {
    printf '%s\n' "$*"
}

have() {
    command -v "$1" >/dev/null 2>&1
}

need_int() {
    case "$2" in
        ''|*[!0-9]*) die "$1 needs a nonnegative integer, got '$2'" ;;
    esac
}

need_fraction() {
    case "$2" in
        ''|*[!0-9.]*|*.*.*) die "$1 needs a number in (0,1), got '$2'" ;;
    esac
}

while [ $# -gt 0 ]; do
    case "$1" in
        --data-root)     [ $# -ge 2 ] || die "--data-root needs a path"; data_root=$2; shift 2 ;;
        --dest)          [ $# -ge 2 ] || die "--dest needs a path"; dest=$2; shift 2 ;;
        --cache)         [ $# -ge 2 ] || die "--cache needs a path"; cache=$2; shift 2 ;;
        --prepared)      [ $# -ge 2 ] || die "--prepared needs a path"; prepared=$2; shift 2 ;;
        --font)          [ $# -ge 2 ] || die "--font needs a path"; font=$2; shift 2 ;;
        --python)        [ $# -ge 2 ] || die "--python needs a path"; python_bin=$2; shift 2 ;;
        --image-size)    [ $# -ge 2 ] || die "--image-size needs a number"; need_int "$1" "$2"; image_size=$2; shift 2 ;;
        --margin)        [ $# -ge 2 ] || die "--margin needs a number"; need_int "$1" "$2"; margin=$2; shift 2 ;;
        --val-fraction)  [ $# -ge 2 ] || die "--val-fraction needs a number"; need_fraction "$1" "$2"; val_fraction=$2; shift 2 ;;
        --seed)          [ $# -ge 2 ] || die "--seed needs a number"; need_int "$1" "$2"; seed=$2; shift 2 ;;
        --delete-zips)   keep_zips=0; shift ;;
        --skip-download) skip_download=1; shift ;;
        --skip-prepare)  skip_prepare=1; shift ;;
        --force)         force=1; shift ;;
        -h|--help)       usage; exit 0 ;;
        *)               die "unknown option '$1' (try --help)" ;;
    esac
done

script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd)
repo_root=$(dirname -- "$script_dir")
data_root=${data_root:-"$repo_root/data"}
dest=${dest:-"$data_root/gnt"}
cache=${cache:-"$data_root/downloads"}
prepared=${prepared:-"$data_root/prepared"}
font=${font:-"$repo_root/fonts/WenJinMinchoP0-Regular.ttf"}

if [ -z "$python_bin" ]; then
    if have python3; then python_bin=python3
    elif have python; then python_bin=python
    else die "need python3 or python for the prepare stage (or use --skip-prepare)"
    fi
fi

if have curl; then
    downloader="curl"
elif have wget; then
    downloader="wget"
else
    die "need curl or wget to download the archives"
fi

if have unzip; then
    unzipper="unzip"
elif have python3; then
    unzipper="python3"
else
    die "need unzip or python3 to unpack the archives"
fi

# --- paths ---------------------------------------------------------------

mkdir -p -- "$dest" "$cache"
dest=$(CDPATH= cd -- "$dest" && pwd)
cache=$(CDPATH= cd -- "$cache" && pwd)
# Unquoted "/*" would glob-expand to every entry of the filesystem root.
case "$font" in
    /*) ;;
    *)  font="$PWD/$font" ;;
esac
if [ "$skip_prepare" -eq 0 ] && [ ! -f "$font" ]; then
    die "font not found: $font (download it into fonts/, see DiT-flow/TRAINING.md)"
fi
prepared_parent=$(dirname -- "$prepared")
mkdir -p -- "$prepared_parent"
prepared_parent=$(CDPATH= cd -- "$prepared_parent" && pwd)
prepared="$prepared_parent/$(basename -- "$prepared")"

log "repository : $repo_root"
log "writers    : $dest"
log "archives   : $cache"
log "prepared   : $prepared"
log "font       : $font"
log ""

# --- download ------------------------------------------------------------

archive_ok() {
    local file=$1
    [ -s "$file" ] || return 1
    if [ "$unzipper" = "unzip" ]; then
        unzip -tqq "$file" >/dev/null 2>&1
    else
        python3 - "$file" <<'PY' >/dev/null 2>&1
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as archive:
    bad = archive.testzip()
if bad is not None:
    raise SystemExit(1)
PY
    fi
}

fetch_archive() {
    local name=$1 file=$2 url="$BASE_URL/$1"
    if [ "$force" -eq 0 ] && archive_ok "$file"; then
        log "  cached   $name ($(du -h "$file" | cut -f1))"
        return 0
    fi
    [ "$skip_download" -eq 0 ] || die "$name is missing or incomplete and --skip-download was given"
    log "  fetching $name"
    log "           $url"
    local attempt=1
    while :; do
        # Write to .part and rename only after the archive passes its test.
        if [ "$downloader" = "curl" ]; then
            curl -fL --retry 5 --retry-delay 3 -C - -o "$file.part" "$url" || true
        else
            wget -c -O "$file.part" "$url" || true
        fi
        if archive_ok "$file.part"; then
            mv -f "$file.part" "$file"
            log "  saved    $name ($(du -h "$file" | cut -f1))"
            return 0
        fi
        attempt=$((attempt + 1))
        [ "$attempt" -le 3 ] || die "could not download a valid $name from $url"
        rm -f -- "$file.part"   # a corrupt partial cannot resume into a valid archive
        log "  retry    $name (attempt $attempt of 3)"
    done
}

stage_download() {
    log "[1/3] download"
    # Writers already on disk: the archives are not needed at all.
    if [ "$force" -eq 0 ] && [ -f "$dest/$GNT_MARKER" ] && [ "$(count_gnt)" -gt 0 ]; then
        log "  skipped  $(count_gnt) writers already unpacked"
        log ""
        return 0
    fi
    if [ "$skip_download" -eq 1 ]; then
        log "  skipped  (--skip-download)"
        log ""
        return 0
    fi
    local name
    for name in "${ARCHIVES[@]}"; do
        fetch_archive "$name" "$cache/$name"
    done
    log ""
}

# --- unpack --------------------------------------------------------------

count_gnt() {
    find "$dest" -type f -name '*.gnt' -print | wc -l | tr -d ' '
}

unpack_archive() {
    local file=$1
    log "  unpacking $(basename -- "$file")"
    if [ "$unzipper" = "unzip" ]; then
        unzip -qq -o "$file" -d "$dest"
    else
        python3 - "$file" "$dest" <<'PY'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as archive:
    archive.extractall(sys.argv[2])
PY
    fi
}

stage_unpack() {
    log "[2/3] unpack"
    if [ "$force" -eq 0 ] && [ -f "$dest/$GNT_MARKER" ] && [ "$(count_gnt)" -gt 0 ]; then
        log "  cached   $(count_gnt) writers already unpacked"
        log ""
        return 0
    fi
    local name
    for name in "${ARCHIVES[@]}"; do
        archive_ok "$cache/$name" \
            || die "$cache/$name is missing or incomplete; rerun without --skip-download"
        unpack_archive "$cache/$name"
    done
    local total
    total=$(count_gnt)
    [ "$total" -gt 0 ] || die "no .gnt files were unpacked into $dest"
    [ "$total" -ge 500 ] || log "  warning: only $total writers; CASIA-HWDB 1.1 train has 717"
    # glyphprep keys writers by file stem, so the two archives must not collide.
    local duplicates
    duplicates=$(find "$dest" -type f -name '*.gnt' -print \
        | sed 's#.*/##' | sed 's/\.gnt$//' | sort | uniq -d | head -5)
    [ -z "$duplicates" ] || die "duplicate writer names: $(printf '%s' "$duplicates" | tr '\n' ' ')"
    {
        printf 'writers=%s\n' "$total"
        printf 'source=%s\n' "${ARCHIVES[*]}"
    } > "$dest/$GNT_MARKER"
    log "  $total writer files, $(du -sh "$dest" | cut -f1)"
    if [ "$keep_zips" -eq 0 ]; then
        for name in "${ARCHIVES[@]}"; do
            rm -f -- "$cache/$name" "$cache/$name.part"
        done
        rmdir -- "$cache" 2>/dev/null || true
        log "  removed cached archives"
    fi
    log ""
}

# --- prepare -------------------------------------------------------------

check_python() {
    "$python_bin" - <<'PY'
import importlib.util, sys
missing = [name for name in ("torch", "numpy", "PIL", "fontTools")
           if importlib.util.find_spec(name) is None]
if missing:
    print("missing: " + " ".join(missing), file=sys.stderr)
    raise SystemExit(3)
import torch, numpy, PIL, fontTools
print("python   :", sys.version.split()[0])
print("torch    :", torch.__version__)
print("numpy    :", numpy.__version__)
print("Pillow   :", PIL.__version__)
PY
}

stage_prepare() {
    log "[3/3] prepare"
    if [ "$force" -eq 0 ] && [ -f "$prepared/manifest.json" ]; then
        log "  cached   $prepared/manifest.json already exists"
        log ""
        return 0
    fi
    if [ -e "$prepared" ] && [ -n "$(ls -A "$prepared" 2>/dev/null)" ]; then
        [ "$force" -eq 1 ] || die "$prepared is not empty; rerun with --force to rebuild it"
        log "  removing $prepared (--force)"
        rm -rf -- "$prepared"
    fi
    check_python || die "$python_bin lacks the packages the prepare stage needs; install with
  $python_bin -m pip install -r requirements.txt
or point --python at a suitable environment"
    free_kib=$(df -Pk "$prepared_parent" | awk 'NR==2 {print $4}')
    if [ -n "$free_kib" ] && [ "$free_kib" -lt "$MIN_FREE_KIB" ]; then
        die "only $((free_kib / 1024)) MiB free on $prepared_parent; prepare needs ~3.4 GiB"
    fi
    log "  running glyphprep.prepare"
    PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}" "$python_bin" -m glyphprep.prepare \
        --gnt-root "$dest" \
        --font "$font" \
        --output "$prepared" \
        --image-size "$image_size" \
        --margin "$margin" \
        --val-fraction "$val_fraction" \
        --seed "$seed"
    [ -f "$prepared/manifest.json" ] || die "prepare finished without $prepared/manifest.json"
    log "  $(du -sh "$prepared" | cut -f1) in $prepared"
    log ""
}

# --- run -----------------------------------------------------------------

# Fail before writing anything: the GPU servers keep home on a full partition
# and the archives alone are 1.8 GiB.
need_kib=0
if [ "$force" -eq 0 ] && [ -f "$dest/$GNT_MARKER" ] && [ "$(count_gnt)" -gt 0 ]; then
    :
else
    need_kib=$((5 * 1024 * 1024))        # archives (~1.8 GiB) + writers (~3 GiB)
fi
if [ "$skip_prepare" -eq 0 ] \
   && { [ "$force" -eq 1 ] || [ ! -f "$prepared/manifest.json" ]; }; then
    need_kib=$((need_kib + 4 * 1024 * 1024))   # prepared (~3.4 GiB)
fi
if [ "$need_kib" -gt 0 ]; then
    free_kib=$(df -Pk "$dest" | awk 'NR==2 {print $4}')
    if [ -n "$free_kib" ] && [ "$free_kib" -lt "$need_kib" ]; then
        die "$dest has $((free_kib / 1024)) MiB free but this run needs ~$((need_kib / 1024)) MiB;
point --data-root at a bigger disk, e.g. /mnt/netdisk2/pattern/Characters/data"
    fi
fi

start=$SECONDS
stage_download
stage_unpack
if [ "$skip_prepare" -eq 1 ]; then
    log "[3/3] prepare"
    log "  skipped (--skip-prepare)"
    log ""
else
    stage_prepare
fi

log "done in $((SECONDS - start)) s"
log ""
log "next, from the repository root:"
log "  PYTHONPATH=DiT-flow $python_bin -m src.train --data $prepared \\"
log "      --output $repo_root/runs/DiT-flow/first --device cuda --stop-after 100"
