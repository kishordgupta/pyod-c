# pyod-c: build the static library, tests and examples.
#   make            library (build/libpyod.a)
#   make test       build and run the test suite
#   make examples   build the example programs

CC      ?= cc
CFLAGS  ?= -O2
# Required flags live apart from CFLAGS so `make CFLAGS=...` keeps them.
PYOD_CFLAGS := -std=c11 -Wall -Wextra -Wpedantic -Wshadow -Iinclude -Isrc
LDLIBS  ?=
LDLIBS  += -lm

SRC  := $(wildcard src/*.c src/models/*.c src/ts/*.c src/graph/*.c src/deep/*.c)
OBJ  := $(SRC:%.c=build/%.o)
LIB  := build/libpyod.a
TESTS := $(patsubst tests/%.c,build/tests/%,$(wildcard tests/test_*.c))
EXAMPLES := $(patsubst examples/%.c,build/examples/%,$(wildcard examples/*.c))

all: $(LIB)

$(LIB): $(OBJ)
	@mkdir -p $(dir $@)
	$(AR) rcs $@ $^

build/%.o: %.c include/pyod.h src/internal.h $(wildcard src/deep/*.h)
	@mkdir -p $(dir $@)
	$(CC) $(PYOD_CFLAGS) $(CFLAGS) -c $< -o $@

build/tests/%: tests/%.c tests/check.h $(LIB)
	@mkdir -p $(dir $@)
	$(CC) $(PYOD_CFLAGS) $(CFLAGS) $< $(LIB) $(LDFLAGS) $(LDLIBS) -o $@

build/examples/%: examples/%.c $(LIB)
	@mkdir -p $(dir $@)
	$(CC) $(PYOD_CFLAGS) $(CFLAGS) $< $(LIB) $(LDFLAGS) $(LDLIBS) -o $@

test: $(TESTS)
	@set -e; for t in $(TESTS); do echo "== $$t"; ./$$t; done

examples: $(EXAMPLES)

clean:
	rm -rf build

.PHONY: all test examples clean
