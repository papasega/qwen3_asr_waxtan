# qwen3_asr_waxtan

Local, real-time speech transcription with language detection and speaker
diarization, built on [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) through
the pure C engine [antirez/qwen-asr](https://github.com/antirez/qwen-asr).
Runs on a laptop CPU, no cloud.

```
microphone -> energy VAD -> qwen-asr (Qwen3-ASR 0.6B or 1.7B) -> text
                         -> Resemblyzer embedding -> online speaker clustering
                         -> langid on the text    -> language
```

## Quick start

```bash
git clone --recursive https://github.com/papasega/qwen3_asr_waxtan.git
cd qwen3_asr_waxtan
./transcribe.sh
```

The first run builds `qwen-asr`, downloads the model (about 1.8 GB for 0.6B)
and creates `.venv`. Speak, then press Ctrl+C: the turn in progress is still
transcribed before exit.

Requirements: git, make, a C compiler, Python 3.10 to 3.13, PortAudio
(`brew install portaudio` on macOS; `sudo apt install build-essential
libopenblas-dev libportaudio2 python3-venv` on Ubuntu).

## Example

A live session mixing French and English on a MacBook, 0.6B model:

```bash
$ ./transcribe.sh
Listening (VAD threshold 150). Ctrl+C to stop.
[00:04] Speaker 1 (fr): Bonjour, aujourd'hui c'était juste pour tester l'outil que j'ai mis en place. Cet outil va nous permettre de voir
[00:14] Speaker 1 (fr): comment utiliser le modèle Qwen3-ASR 0.6B pour faire de la transcription automatique de la parole.
[00:22] Speaker 1 (fr): L'objectif, c'est de pouvoir le tester : obtenir une transcription de qualité, mais aussi l'utiliser en temps réel et en local.
[00:43] Speaker 1 (en): Hi everybody and welcome to our demo. Today we would like to see how the Qwen3-ASR model works.
[00:58] Speaker 1 (en): The thing I want to test here is to use my YcodY models to run code.
[01:13] Speaker 1 (en): How can we do it? It's really simple. Imagine you want to write code and you don't like
[01:28] Speaker 1 (en): to type every single word for your agent. You can just ask your question with your voice, and the agent executes it.
^C
End of the transcription.
```

The text above was lightly edited for readability (fillers removed, "coin
isr zéro six b" and "Queen Azure" restored to Qwen3-ASR, "Y Cody" to YcodY).
Raw output keeps hesitations and can mishear proper nouns.

## Options

```bash
./transcribe.sh --language French          # force the ASR language
./transcribe.sh --file meeting.wav         # replay a 16 kHz mono WAV instead of the mic
./transcribe.sh --speaker-threshold 0.8    # higher = more speakers
./transcribe.sh --silence 1.0              # longer pause before a turn ends
MODEL=large ./transcribe.sh                # Qwen3-ASR 1.7B, slower and more accurate
QWEN_ASR_DIR=../qwen-asr ./transcribe.sh   # reuse an existing qwen-asr build and models
```

Convert any audio for `--file` with `ffmpeg -i in.mp3 -ar 16000 -ac 1 out.wav`.

## Limits

- Transcription is per turn: text appears after each pause, not word by word.
- `qwen-asr` detects the spoken language internally but does not print it, so
  the language shown is guessed from the transcribed text with `langid`.
  Short turns (one or two words) are often mislabeled.
- Diarization is online and unsupervised; very similar voices can merge and
  short noisy turns can create a spurious speaker. Tune `--speaker-threshold`.

## Credits

Qwen3-ASR by the Qwen team (Alibaba), C inference engine by Salvatore Sanfilippo
(antirez), speaker embeddings by Resemblyzer, language ID by langid.py.
