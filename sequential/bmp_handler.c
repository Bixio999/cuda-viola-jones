#include <stdlib.h>
#include <stdio.h>

#include "image.h"

pel** readBMP_RGB(char* filename) {
	FILE* f = fopen(filename, "rb");
	if (f == NULL) {
		printf("\n\n%s NOT FOUND\n\n", filename);
		exit(1);
	}

	pel HeaderInfo[54];
	fread(HeaderInfo, sizeof(pel), 54, f); // read the 54-byte header

	// extract image height and width from header
	int width = *(int*) &HeaderInfo[18];
	int height = *(int*) &HeaderInfo[22];

	//copy header for re-use
	for (unsigned int i = 0; i < 54; i++)
		im.header[i] = HeaderInfo[i];

	im.height = height;
	im.width = width;
	int RowBytes = (width * 3 + 3) & (~3);
	im.h_offset = RowBytes;
    im.type = "RGB";

	printf("\n   Input BMP File name: %20s  (%u x %u)", filename, im.height,
			im.width);

	pel tmp;
	pel **TheImage = (pel **) malloc(height * sizeof(pel*));
	for (unsigned int i = 0; i < height; i++)
		TheImage[i] = (pel *) malloc(RowBytes * sizeof(pel));

	for (unsigned int i = 0; i < height; i++)
		fread(TheImage[i], sizeof(unsigned char), RowBytes, f);

	fclose(f);
	return TheImage;  // remember to free() it in caller!
}

pel** readBMP_grey(char* filename) {
	FILE* f = fopen(filename, "rb");
	if (f == NULL) {
		printf("\n\n%s NOT FOUND\n\n", filename);
		exit(1);
	}

	pel HeaderInfo[54];
	fread(HeaderInfo, sizeof(pel), 54, f); // read the 54-byte header

	// extract image height and width from header
	int width = *(int*) &HeaderInfo[18];
	int height = *(int*) &HeaderInfo[22];

	//copy header for re-use
	for (unsigned int i = 0; i < 54; i++)
		im.header[i] = HeaderInfo[i];

	im.height = height;
	im.width = width;
	int RowBytes = width;
	im.h_offset = RowBytes;
    im.type = "GREY";

	printf("\n   Input BMP File name: %20s  (%u x %u)", filename, im.height,
			im.width);

	pel tmp;
	pel **TheImage = (pel **) malloc(height * sizeof(pel*));
	for (unsigned int i = 0; i < height; i++)
		TheImage[i] = (pel *) malloc(RowBytes * sizeof(pel));

	for (unsigned int i = 0; i < height; i++)
		fread(TheImage[i], sizeof(unsigned char), RowBytes, f);

	fclose(f);
	return TheImage;  // remember to free() it in caller!
}

/*
 * Store a BMP image
 */
void writeBMP(pel** img, char* filename) {
	FILE* f = fopen(filename, "wb");
	if (f == NULL) {
		printf("\n\nFILE CREATION ERROR: %s\n\n", filename);
		exit(1);
	}

	//write header
	for (unsigned int x = 0; x < 54; x++)
		fputc(im.header[x], f);

	//write data
	for (unsigned int x = 0; x < im.height; x++)
		for (unsigned int y = 0; y < im.width; y++) {
			char temp = img[x][y];
			fputc(temp, f);
		}

	printf("\n  Output BMP File name: %20s  (%u x %u)", filename, im.width,
			im.height);

	fclose(f);
}