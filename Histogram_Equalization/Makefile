# Makefile 

# Define compiler
CC=nvcc

# compiler flags
CFLAGS= -O4

all: exec

exec:
	$(CC) $(CFLAGS) final.cu -o $@

clean:
	rm -f exec
