# Depyler Single-Shot Demo

```bash
cd demos/week3/depyler-demo
```

## 1. Show the Python source

```bash
cat fibonacci.py
```

## 2. Quality gates (ephemeral uv)

```bash
make check
```
> "100% test coverage, ruff lint passing. No global installs - uv handles it."

## 3. Transpile Python → Rust

```bash
make transpile && cat fibonacci.rs
```
> "47ms. Type-safe Rust with proper error handling."

## 4. Compile and run

```bash
make run
```
> "Native binary. No Python runtime."

## 5. Clean up

```bash
make clean
```
