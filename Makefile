SRC=src/main.cu
OUT=$(SRC:.cu=)
FLAGS=-G --machine 64
NVCC=nvcc

run: $(OUT)
	./$(OUT)

$(OUT): $(SRC)
	$(NVCC) $(CFLAGS) $< -o $@

.PHONY: clean

clean:
	rm -rf $(CSRC:.c=.o) $(COBJ)
	rm -rf $(CSRC1:.c=.o) $(COBJ1)
	rm -rf $(CSRC2:.c=.o) $(COBJ2)
