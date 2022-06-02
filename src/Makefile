CC ?= gcc
OMPFLAG ?= -fopenmp
OPT ?= $(OMPFLAG) -pedantic -Wall -Wextra -O3
LIBDIR ?= ../lib

SOURCEFILES = utils.c
HEADERFILES = utils.h

vpath %.so $(LIBDIR)

%_float.o: %.c $(HEADERFILES)
	$(CC) $(OPT) -DFLOAT32 -fPIC -c $< -o $@

%_double.o: %.c $(HEADERFILES)
	$(CC) $(OPT) -fPIC -c $< -o $@

%.so: %.o
	@mkdir -p $(LIBDIR)
	$(CC) $(OPT) -shared $< -o $(LIBDIR)/$@

all: utils_float.so utils_double.so

clean:
	rm -rf $(LIBDIR)

.PHONY: all clean
