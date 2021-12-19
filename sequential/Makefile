viola-jones.out: main.c classifier.c bmp_handler.c clean
	gcc main.c classifier.c bmp_handler.c -o viola-jones.out -lm -g

.PHONY: clean run run2

clean:
	rm -f resized/*
	rm -rf viola-jones.out viola-jones.out.dSYM

run: 
	./viola-jones.out ../images/lena.bmp

run2: 
	./viola-jones.out ../images/Ocean-Eleven.bmp

crowd: 
	./viola-jones.out ../images/crowd.bmp
