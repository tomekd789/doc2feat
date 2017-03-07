CC = gcc
#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
CFLAGS = -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result

all: doc2feat

doc2feat : doc2feat.c
	$(CC) doc2feat.c -o doc2feat $(CFLAGS)

clean:
	rm -rf doc2feat
