# Qwen2.5-Coder Showcase Demo Makefile
# Purified with bashrs make lint

.SUFFIXES:
.DELETE_ON_ERROR:
.ONESHELL:
SHELL := /bin/bash

.PHONY: all pull run chat serve serve-stop curl-test bench trace clean help

# Model configuration
MODEL_HF := hf://Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
MODEL_CACHE := $(HOME)/.cache/pacha/models
PORT := 8080

all: help

help:
	@echo "Qwen2.5-Coder Showcase Demo"
	@echo ""
	@echo "Usage:"
	@echo "  make pull       - Download model to cache"
	@echo "  make run        - Single-shot inference"
	@echo "  make chat       - Interactive chat session"
	@echo "  make serve      - Start REST API server"
	@echo "  make curl-test  - Test the server with curl"
	@echo "  make serve-stop - Stop the server"
	@echo "  make clean      - Remove cached model"

# Download model (Ollama-like UX)
pull:
	@echo "=== Pulling Qwen2.5-Coder-1.5B ==="
	@apr pull $(MODEL_HF)

# Single-shot inference
run:
	@echo "=== apr run: Single-shot Inference ==="
	@apr run $(MODEL_HF) \
		--prompt "Write a function to check if a number is prime in Rust" \
		--max-tokens 256

# Interactive chat
chat:
	@echo "=== apr chat: Interactive Mode ==="
	@echo "(Type 'exit' or Ctrl+C to quit)"
	@apr chat $(MODEL_HF) \
		--system "You are a helpful coding assistant. Be concise." \
		--max-tokens 512

# Start server (background)
serve:
	@echo "=== apr serve: REST API Server ==="
	@echo "Starting server on http://127.0.0.1:$(PORT)"
	@echo "Endpoints:"
	@echo "  POST /v1/chat/completions  (OpenAI compatible)"
	@echo "  GET  /health"
	@echo ""
	@apr serve $(MODEL_HF) --port $(PORT)

# Test server with curl
curl-test:
	@echo "=== Testing Server ==="
	@curl -s http://127.0.0.1:$(PORT)/health && echo ""
	@echo ""
	@echo "=== Chat Completion ==="
	@curl -s http://127.0.0.1:$(PORT)/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d '{"model":"qwen","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":32}' \
		| jq -r '.choices[0].message.content // .'

# Benchmark mode
bench:
	@echo "=== apr run: Benchmark Mode ==="
	@apr run $(MODEL_HF) \
		--prompt "Hello" \
		--max-tokens 64 \
		--benchmark

# With tracing
trace:
	@echo "=== apr run: With Tracing ==="
	@apr run $(MODEL_HF) \
		--prompt "Hi" \
		--max-tokens 16 \
		--trace \
		--trace-level basic

clean:
	@echo "=== Cleaning cached models ==="
	-@apr rm "$(MODEL_HF)" 2>/dev/null
