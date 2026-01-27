# Whisper.apr CLI Demo Makefile
# Purified with bashrs make lint

.SUFFIXES:
.DELETE_ON_ERROR:
.ONESHELL:
SHELL := /bin/bash

.PHONY: all transcribe benchmark diagnose json srt profile clean

all: transcribe

# Transcribe sample audio
transcribe:
	@echo "=== Whisper.apr Transcription ==="
	@whisper-apr-cli transcribe -f sample.wav -l en

# Benchmark performance (tiny model, 3 iterations)
benchmark:
	@echo "=== Whisper.apr Benchmark ==="
	@whisper-apr-cli benchmark tiny -i 3

# Run diagnostics
diagnose:
	@echo "=== Whisper.apr Diagnostics ==="
	@whisper-apr-cli diagnose

# JSON output mode
json:
	@whisper-apr-cli transcribe -f sample.wav -l en -o json

# SRT subtitle output
srt:
	@whisper-apr-cli transcribe -f sample.wav -l en -o srt

# With profiling
profile:
	@whisper-apr-cli transcribe -f sample.wav -l en --profile

clean:
	@rm -f transcript.txt *.srt
