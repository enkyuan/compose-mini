CC      = gcc
CFLAGS  = -std=c11 -Wall -Wextra -Werror -Iinclude
SRC     = $(shell find src -name '*.c')
OBJ     = $(patsubst src/%.c, build/%.o, $(SRC))
BIN     = bin/transformer

TEST_SRC = $(wildcard tests/test_*.c)
TEST_BIN = $(patsubst tests/%.c, bin/tests/%, $(TEST_SRC))

.PHONY: all test clean run

all: $(BIN)

$(BIN): $(OBJ) build/main.o | bin
	$(CC) $(CFLAGS) $^ -o $@

build/main.o: main.c | build
	$(CC) $(CFLAGS) -c $< -o $@

build/%.o: src/%.c | build
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

bin/tests/%: tests/%.c $(OBJ) | bin/tests
	$(CC) $(CFLAGS) $^ -o $@

test: $(TEST_BIN)
	@for t in $(TEST_BIN); do echo "Running $$t..."; $$t; done

clean:
	rm -rf build bin

run: all
	./$(BIN)

build:
	mkdir -p build

bin:
	mkdir -p bin

bin/tests:
	mkdir -p bin/tests
