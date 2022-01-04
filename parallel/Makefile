# all:
# 	$(MAKE) viola-jones || $(MAKE) set_env

viola-jones: main.cu bmp_handler.cu classifier.cu
	nvcc -arch sm_35 -Wno-deprecated-gpu-targets main.cu bmp_handler.cu classifier.cu -o viola-jones

run: clean_resized
	./viola-jones ../images/lena.bmp

# set_env:
# 	./set_env.sh
# 	make viola-jones

run2: clean_resized
	./viola-jones ../images/Ocean-Eleven.bmp

run3: clean_resized
	./viola-jones ../images/obama.bmp

crowd: clean_resized
	./viola-jones ../images/crowd.bmp

.PHONY: run clean debug run2 clean_resized# set_env

debug: main.cu bmp_handler.cu classifier.cu clean
	nvcc -arch sm_37 -Wno-deprecated-gpu-targets main.cu bmp_handler.cu classifier.cu -o viola-jones -g

clean:
	rm -f viola-jones

clean_resized:
	rm -rf resized/*