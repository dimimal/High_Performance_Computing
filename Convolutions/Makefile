# This is a makefile 

# Define compiler
CC=nvcc

# compiler flags
CFLAGS= -O4

all: exec_non_tiled.o exec_tiled.o 

exec_non_tiled.o:
	$(CC) $(CFLAGS) non_tiled_conv.cu -o $@

exec_tiled.o:
	$(CC) $(CFLAGS)  tiled_conv.cu -o $@

clean:
	rm -f *.o

