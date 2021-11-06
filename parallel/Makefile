viola-jones: main.cu bmp_handler.cu
	nvcc -arch sm_37 -Wno-deprecated-gpu-targets main.cu bmp_handler.cu -o viola-jones

run: 
	./viola-jones ../images/lena.bmp

.PHONY: run clean

clean:
	rm -f viola-jones