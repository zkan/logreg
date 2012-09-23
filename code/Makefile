#
#	Makefile for compiling all code in this directory.
#	Author: Kan Ouivirach
# 
CFLAGS = -g -Wall
CC = g++ $(CFLAGS)
PROG = logreg
all: $(PROG)

logreg.o: logreg.cc logreg.h
	$(CC) -c logreg.cc

logreg: main.o logreg.o
	$(CC) main.o logreg.o -o logreg

clean:
	rm *.o $(PROG)







