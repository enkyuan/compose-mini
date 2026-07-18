CC      = gcc
CFLAGS  = -std=c11 -Wall -Wextra -Werror -Iinclude
DEPFLAGS = -MMD -MP
LDLIBS  = -lm
SRC     = $(shell find src -name '*.c')
OBJ     = $(patsubst src/%.c, build/%.o, $(SRC))
DEP     = $(OBJ:.o=.d) build/main.d
BIN     = bin/transformer
HEADERS = $(wildcard include/*.h)

TEST_SRC = $(wildcard tests/test_*.c)
BEHAVIOR_TEST_SRC = tests/test_embed.c tests/test_ffn.c \
                    tests/test_norm.c tests/test_utils.c
BEHAVIOR_TEST_BIN = $(patsubst tests/%.c, bin/tests/%, $(BEHAVIOR_TEST_SRC))
STUB_TEST_SRC = $(filter-out $(BEHAVIOR_TEST_SRC), $(TEST_SRC))
STUB_TEST_BIN = $(patsubst tests/%.c, bin/tests/%, $(STUB_TEST_SRC))

.PHONY: all check test compile-stubs clean run

all: $(BIN)

check: all test compile-stubs

$(BIN): $(OBJ) build/main.o | bin
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

build/main.o: main.c | build
	$(CC) $(CFLAGS) $(DEPFLAGS) -c $< -o $@

build/%.o: src/%.c | build
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(DEPFLAGS) -c $< -o $@

bin/tests/%: tests/%.c $(OBJ) $(HEADERS) | bin/tests
	$(CC) $(CFLAGS) $< $(OBJ) -o $@ $(LDLIBS)

test: $(BEHAVIOR_TEST_BIN)
	@set -e; for t in $(BEHAVIOR_TEST_BIN); do echo "Running $$t..."; $$t; done

compile-stubs: $(STUB_TEST_BIN)

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

-include $(DEP)
