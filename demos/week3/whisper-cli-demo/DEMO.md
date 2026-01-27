# Whisper.apr CLI Demo

```bash
cd demos/week3/whisper-cli-demo
```

## 1. Basic transcription

```bash
make transcribe
```
> "1.5s audio → text. Tiny model."

## 2. JSON output (machine-readable)

```bash
make json
```
> "Structured output for pipelines."

## 3. Component profiling

```bash
make profile
```
> "Model load, audio load, inference breakdown. RTF measurement."

## 4. Diagnostics

```bash
make diagnose
```
> "Self-check: tokenizer, model config, known issues."

## 5. Clean up

```bash
make clean
```
