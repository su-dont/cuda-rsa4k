SRC=src/main.cu
OUT=bin/main
FLAGS=-g -G --machine 64
NVCC=/usr/local/cuda/bin/nvcc

run: $(OUT)
	./$(OUT)

$(OUT): $(SRC)
	$(NVCC) $(FLAGS) $< -o $@

.PHONY: clean

clean:
	rm -r bin
	mkdir bin
