"""Live transcription with language detection and speaker diarization.

Pipeline, one utterance at a time:
  microphone -> energy VAD -> antirez/qwen-asr (Qwen3-ASR, C) -> text
                           -> Resemblyzer embedding -> online speaker clustering
                           -> langid on the text -> detected language

Usage:
  python live_transcribe.py --asr-bin qwen-asr/qwen_asr --model-dir qwen-asr/qwen3-asr-0.6b
  python live_transcribe.py ... --file meeting.wav   # replay a 16 kHz mono WAV instead of the mic
"""

from __future__ import annotations

import argparse
import queue
import subprocess
import sys
import threading
import wave
import warnings

import numpy as np

warnings.filterwarnings("ignore")  # webrtcvad/librosa deprecation noise

SAMPLE_RATE = 16_000
BLOCK = 480  # 30 ms


def rms(block: np.ndarray) -> float:
    return float(np.sqrt(np.mean(block.astype(np.float32) ** 2)))


def utterances(blocks, threshold: float, silence_s: float, max_s: float, min_s: float):
    """Group 30 ms int16 blocks into (start_seconds, utterance) separated by silence."""
    buf: list[np.ndarray] = []
    quiet = start = 0
    for n, block in enumerate(blocks):
        loud = rms(block) > threshold
        if not buf and not loud:
            continue
        if not buf:
            start = n
        buf.append(block)
        quiet = 0 if loud else quiet + 1
        dur = len(buf) * BLOCK / SAMPLE_RATE
        if quiet * BLOCK / SAMPLE_RATE >= silence_s or dur >= max_s:
            if dur >= min_s:
                yield start * BLOCK / SAMPLE_RATE, np.concatenate(buf)
            buf, quiet = [], 0
    if buf and len(buf) * BLOCK / SAMPLE_RATE >= min_s:
        yield start * BLOCK / SAMPLE_RATE, np.concatenate(buf)


def mic_blocks(device):
    import sounddevice as sd

    q: queue.Queue[np.ndarray] = queue.Queue()
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                        blocksize=BLOCK, device=device,
                        callback=lambda data, *_: q.put(data[:, 0].copy())):
        try:
            while True:
                yield q.get()
        except KeyboardInterrupt:  # end the stream so the turn in progress is still flushed
            pass


def file_blocks(path: str):
    with wave.open(path, "rb") as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (SAMPLE_RATE, 1, 2):
            sys.exit("--file must be 16 kHz mono 16-bit PCM (ffmpeg -i in -ar 16000 -ac 1 out.wav)")
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    for i in range(0, len(audio), BLOCK):
        yield audio[i:i + BLOCK]


class Diarizer:
    """Online clustering: a new speaker when no centroid is similar enough."""

    def __init__(self, threshold: float):
        from resemblyzer import VoiceEncoder, preprocess_wav

        self.encoder, self.preprocess = VoiceEncoder(verbose=False), preprocess_wav
        self.threshold = threshold
        self.centroids: list[np.ndarray] = []
        self.counts: list[int] = []

    def identify(self, audio: np.ndarray) -> int:
        """Return a 1-based speaker id, or 0 when the turn is too short to tell."""
        wav = self.preprocess(audio.astype(np.float32) / 32768.0, source_sr=SAMPLE_RATE)
        if len(wav) < SAMPLE_RATE // 2:  # too little voiced audio for a reliable embedding
            return 0
        emb = self.encoder.embed_utterance(wav)
        sims = [float(c @ emb / np.linalg.norm(c)) for c in self.centroids]
        if sims and max(sims) >= self.threshold:
            i = int(np.argmax(sims))
            self.counts[i] += 1
            self.centroids[i] += (emb - self.centroids[i]) / self.counts[i]
            return i + 1
        self.centroids.append(emb.copy())
        self.counts.append(1)
        return len(self.centroids)


def transcribe(audio: np.ndarray, asr_bin: str, model_dir: str, language: str | None) -> str:
    cmd = [asr_bin, "-d", model_dir, "--stdin", "--silent"]
    if language:
        cmd += ["--language", language]
    # Own session: Ctrl+C in the terminal must not kill the turn being transcribed.
    out = subprocess.run(cmd, input=audio.tobytes(), capture_output=True, timeout=120,
                         start_new_session=True)
    if out.returncode != 0:
        print(f"  [asr error] {out.stderr.decode(errors='replace').strip()}", file=sys.stderr)
        return ""
    return out.stdout.decode(errors="replace").strip()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--asr-bin", default="qwen-asr/qwen_asr")
    p.add_argument("--model-dir", default="qwen-asr/qwen3-asr-0.6b")
    p.add_argument("--file", help="replay a 16 kHz mono WAV instead of the microphone")
    p.add_argument("--device", default=None, help="sounddevice input device (name or index)")
    p.add_argument("--language", help="force the ASR language (e.g. French); detection still runs")
    p.add_argument("--speaker-threshold", type=float, default=0.75)
    p.add_argument("--vad-threshold", type=float, default=0.0, help="RMS threshold, 0 = calibrate")
    p.add_argument("--silence", type=float, default=0.7, help="seconds of silence ending a turn")
    p.add_argument("--max-segment", type=float, default=15.0)
    args = p.parse_args()

    import langid

    diarizer = Diarizer(args.speaker_threshold)
    blocks = file_blocks(args.file) if args.file else mic_blocks(args.device)

    threshold = args.vad_threshold
    if not threshold:
        noise = [next(blocks) for _ in range(33)]  # ~1 s of room noise
        threshold = max(3 * float(np.median([rms(b) for b in noise])), 150.0)
    print(f"Listening (VAD threshold {threshold:.0f}). Ctrl+C to stop.", file=sys.stderr)

    # Capture keeps running while the worker transcribes the previous turn.
    turns: queue.Queue[tuple[float, np.ndarray] | None] = queue.Queue()
    offset = 33 * BLOCK / SAMPLE_RATE if not args.vad_threshold else 0.0

    def worker() -> None:
        while (item := turns.get()) is not None:
            t, audio = item
            speaker = diarizer.identify(audio)
            text = transcribe(audio, args.asr_bin, args.model_dir, args.language)
            if text:
                lang = langid.classify(text)[0]
                who = f"Speaker {speaker}" if speaker else "Speaker ?"
                print(f"[{int(t // 60):02d}:{int(t % 60):02d}] {who} ({lang}): {text}",
                      flush=True)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        for t, audio in utterances(blocks, threshold, args.silence, args.max_segment, 0.5):
            turns.put((offset + t, audio))
        turns.put(None)
        thread.join()
    except KeyboardInterrupt:
        pass
    print("\nEnd of the transcription.", file=sys.stderr)


if __name__ == "__main__":
    main()
