viola-jones.out: main.c classifier.c bmp_handler.c image.h classifier.h clean
	gcc main.c classifier.c bmp_handler.c -o viola-jones -lm -g

.PHONY: clean run run2

clean:
	rm -f resized/*
	rm -rf viola-jones viola-jones.dSYM

run: 
	./viola-jones ../images/lena.bmp

run2: 
	./viola-jones ../images/Ocean-Eleven.bmp

crowd: 
	./viola-jones ../images/crowd.bmp
