CC      = gcc
PYTHON  ?= python3
CFLAGS  = -std=c11 -Wall -Wextra -Werror -Iinclude
DEPFLAGS = -MMD -MP
LDLIBS  = -lm -lz
SRC     = $(shell find src -name '*.c')
OBJ     = $(patsubst src/%.c, build/%.o, $(SRC))
DEP     = $(OBJ:.o=.d) build/main.d build/tests/artifact_fixture.d
BIN     = bin/transformer
HEADERS = $(wildcard include/*.h)

C_TEST_DIR = tests/c
C_TEST_INCLUDE_DIR = tests/include
PYTHON_TEST_DIR = tests/python
TEST_CPPFLAGS = -I$(C_TEST_INCLUDE_DIR)
TEST_SRC = $(wildcard $(C_TEST_DIR)/test_*.c)
TEST_SUPPORT_SRC = $(C_TEST_DIR)/artifact_fixture.c
TEST_SUPPORT_HEADERS = $(C_TEST_INCLUDE_DIR)/artifact_fixture.h
TEST_SUPPORT_OBJ = build/tests/artifact_fixture.o
BEHAVIOR_TEST_SRC = $(addprefix $(C_TEST_DIR)/, test_artifact.c \
                    test_attention.c test_cli.c test_data.c test_embed.c \
                    test_ffn.c test_norm.c test_transformer.c test_utils.c)
BEHAVIOR_TEST_BIN = $(patsubst $(C_TEST_DIR)/%.c, bin/tests/%, \
                    $(BEHAVIOR_TEST_SRC))
STUB_TEST_SRC = $(filter-out $(BEHAVIOR_TEST_SRC), $(TEST_SRC))
STUB_TEST_BIN = $(patsubst $(C_TEST_DIR)/%.c, bin/tests/%, $(STUB_TEST_SRC))
PYTHON_TEST = $(addprefix $(PYTHON_TEST_DIR)/, test_artifact_v1.py \
              test_backtest.py test_data_v1.py test_e2e.py test_massive.py)
PYTORCH_TEST = $(addprefix $(PYTHON_TEST_DIR)/, test_training.py \
               test_experiment.py)

.PHONY: all check check-training test compile-stubs clean run

all: $(BIN)

check: all test compile-stubs

check-training: all
	@set -e; for t in $(PYTORCH_TEST); do echo "Running $$t..."; \
		PYTHONDONTWRITEBYTECODE=1 $(PYTHON) $$t $(BIN); done

$(BIN): $(OBJ) build/main.o | bin
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

build/main.o: main.c | build
	$(CC) $(CFLAGS) $(DEPFLAGS) -c $< -o $@

build/%.o: src/%.c | build
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(DEPFLAGS) -c $< -o $@

$(TEST_SUPPORT_OBJ): $(TEST_SUPPORT_SRC) $(TEST_SUPPORT_HEADERS) $(HEADERS) | build
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(TEST_CPPFLAGS) $(DEPFLAGS) -c $< -o $@

bin/tests/%: $(C_TEST_DIR)/%.c $(TEST_SUPPORT_OBJ) $(OBJ) $(HEADERS) | bin/tests
	$(CC) $(CFLAGS) $(TEST_CPPFLAGS) $< $(TEST_SUPPORT_OBJ) $(OBJ) -o $@ $(LDLIBS)

test: $(BIN) $(BEHAVIOR_TEST_BIN)
	@set -e; for t in $(BEHAVIOR_TEST_BIN); do echo "Running $$t..."; $$t; done
	@set -e; for t in $(PYTHON_TEST); do echo "Running $$t..."; \
		PYTHONDONTWRITEBYTECODE=1 $(PYTHON) $$t; done

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
