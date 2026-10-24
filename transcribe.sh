#!/usr/bin/env bash
# Live transcription + language detection + speaker diarization with Qwen3-ASR.
#
# First run builds antirez/qwen-asr, downloads the model and creates .venv.
#
#   ./transcribe.sh                       # microphone, 0.6B model
#   ./transcribe.sh --language French     # force the ASR language
#   ./transcribe.sh --file meeting.wav    # replay a 16 kHz mono WAV
#   MODEL=large ./transcribe.sh           # 1.7B model (slower, more accurate)
#
# Needs: git, make, a C compiler, Python >= 3.10, PortAudio
#   macOS:  brew install portaudio
#   Ubuntu: sudo apt install build-essential libopenblas-dev libportaudio2 python3-venv
set -euo pipefail

cd "$(dirname "$0")"
ASR_DIR="${QWEN_ASR_DIR:-qwen-asr}"
MODEL="${MODEL:-small}"
case "$MODEL" in
    small) MODEL_DIR="$ASR_DIR/qwen3-asr-0.6b" ;;
    large) MODEL_DIR="$ASR_DIR/qwen3-asr-1.7b" ;;
    *) echo "MODEL must be small or large" >&2; exit 1 ;;
esac

if [ ! -f "$ASR_DIR/Makefile" ]; then
    git submodule update --init qwen-asr
fi
if [ ! -x "$ASR_DIR/qwen_asr" ]; then
    make -C "$ASR_DIR" blas
fi
if [ ! -d "$MODEL_DIR" ] || [ -z "$(ls -A "$MODEL_DIR")" ]; then
    (cd "$ASR_DIR" && bash download_model.sh --model "$MODEL" --dir "$(basename "$MODEL_DIR")")
fi
if [ ! -f .venv/.installed ]; then
    # torch (via resemblyzer) has no wheels for the newest Pythons yet
    PY="${PYTHON:-}"
    for candidate in python3.12 python3.11 python3.13 python3.10; do
        [ -z "$PY" ] && command -v "$candidate" >/dev/null && PY="$candidate"
    done
    [ -z "$PY" ] && { echo "Install Python 3.10-3.13 (e.g. brew install python@3.12)" >&2; exit 1; }
    rm -rf .venv
    "$PY" -m venv .venv
    .venv/bin/python -m pip install -q --upgrade pip
    .venv/bin/python -m pip install -q -r requirements.txt
    touch .venv/.installed
fi

exec .venv/bin/python live_transcribe.py --asr-bin "$ASR_DIR/qwen_asr" --model-dir "$MODEL_DIR" "$@"
