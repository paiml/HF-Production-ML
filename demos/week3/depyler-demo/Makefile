.PHONY: all transpile compile run clean

all: run

transpile:
	@depyler transpile fibonacci.py

compile: transpile
	@rustc fibonacci.rs -o fibonacci

run: compile
	@./fibonacci

clean:
	@rm -f fibonacci fibonacci.rs
