viola-jones.out: main.c classifier.c bmp_handler.c 
	rm -rf viola-jones.out viola-jones.out.dSYM
	gcc main.c classifier.c bmp_handler.c -o viola-jones.out -lm -g

.PHONY: clean run run2

clean:
	rm -f resized/*
	rm -rf viola-jones.out viola-jones.out.dSYM

run: clean
	./viola-jones.out ../images/conv/000001.jpg.bmp

run2: clean
	./viola-jones.out ../images/conv/000002.jpg.bmp
