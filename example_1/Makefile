# Makefile for example1
# change the variable to run various c code files
# Authors: Dr Sam Siewert, Rutvij Karkhanis
# Date modified: 07/01/2014
# Last modified by: Rutvij Karkhanis


INCLUDE_DIRS = 
LIB_DIRS = 
CC=gcc
CDEFS= -DPMU_ANALYSIS
CFLAGS= -O3 -g $(INCLUDE_DIRS) $(CDEFS)
LIBS= 

#PRODUCT=perfmon

HFILES=
SRC= perfmon

SRCS= ${HFILES}.h ${SRC}.c
OBJS= ${SRC}.o

all:	perfmon

clean:
	-rm -f *.o *.NEW *~ *.d
	-rm -f ${SRC}

perfmon: ${OBJS}
	$(CC) $(CCFILES) $(LDFLAGS) $(CFLAGS) -o $@ $(OBJS) $(LIBS)

depend:

.c.o:
	$(CC) -MD $(CFLAGS) -c $<
