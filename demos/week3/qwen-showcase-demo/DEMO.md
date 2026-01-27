# Qwen2.5-Coder Showcase Demo

```bash
cd demos/week3/qwen-showcase-demo
```

## 1. Pull the model (first time only)

```bash
make pull
```
> "Ollama-like UX. Downloads to ~/.cache/pacha/models/"

## 2. Single-shot inference (`apr run`)

```bash
make run
```
> "One prompt, one response. Auto-downloads if not cached."

## 3. Interactive chat (`apr chat`)

```bash
make chat
```
> "REPL mode. System prompt sets behavior. Ctrl+C to exit."

## 4. REST API server (`apr serve`)

**Terminal 1:**
```bash
make serve
```

**Terminal 2:**
```bash
make curl-test
```
> "OpenAI-compatible API. /v1/chat/completions endpoint."

## 5. Benchmark mode

```bash
make bench
```
> "Performance metrics: tok/s, latency."

## 6. With tracing

```bash
make trace
```
> "Layer-by-layer execution trace."

## Summary

| Command | Mode | Use Case |
|---------|------|----------|
| `apr run` | Single-shot | Scripts, pipelines |
| `apr chat` | Interactive | Development, testing |
| `apr serve` | REST API | Production, integration |
