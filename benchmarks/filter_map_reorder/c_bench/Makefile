all: filter.o filter_fixed.o map.o filter_map.o map_filter.o

filter.o: 
	clang filter.c -O3 -o $@

filter_fixed.o:
	clang filter_fixed.c -O3 -o $@

map.o:
	clang map.c -O3 -o $@

filter_map.o:
	clang filter_map.c -O3 -o $@

map_filter.o: map_filter.c
	clang map_filter.c -O3 -o $@

clean:
	rm -f *.o

.phony:
	all
