#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "image.h"

/*
	This file contails the functions to handle with Bitmap images, in order to 
	read and create the data structure needed for the algorithm, convert the image 
	to greyscale, and write a new bitmap image.
*/

// Read a Bitmap image from the given path
pel** readBMP_RGB(char* filename) {
	// Open the given path
	FILE* f = fopen(filename, "rb");
	// Check if the file exists
	if (f == NULL) {
		printf("\n\n%s NOT FOUND\n\n", filename);
		exit(1);
	}

	// Read the file header
	pel HeaderInfo[54];
	fread(HeaderInfo, sizeof(pel), 54, f); 

	// Extract image height and width from header
	int width = abs(*(int*) &HeaderInfo[18]);
	int height = abs(*(int*) &HeaderInfo[22]);

	unsigned int i;
	// Copy header for re-use
	for (i = 0; i < 54; i++)
		im.header[i] = HeaderInfo[i];

	// Set the image details structure
	im.height = height;
	im.width = width;
	int RowBytes = ROWBYTES(width);
	im.h_offset = RowBytes;
    im.type = "RGB";
	im.bitColor = 24;

	printf("\n   Input BMP File name: %20s  (%u x %u)", filename, im.height,
			im.width);

	// Allocate the memory for the image
	pel **TheImage = (pel **) malloc(height * sizeof(pel*));
	for (i = 0; i < height; i++)
		TheImage[i] = (pel *) malloc(RowBytes * sizeof(pel));

	// Read the image by rows
	for (i = 0; i < height; i++)
		fread(TheImage[i], sizeof(unsigned char), RowBytes, f);

	fclose(f);
	return TheImage; 
}

// Read a greyscale Bitmap image from the given path
pel** readBMP_grey(char* filename) {
	// Open the given path
	FILE* f = fopen(filename, "rb");
	// Check if the file exists
	if (f == NULL) {
		printf("\n\n%s NOT FOUND\n\n", filename);
		exit(1);
	}

	// Read the file header
	pel HeaderInfo[54];
	fread(HeaderInfo, sizeof(pel), 54, f); 

	// Extract image height and width from header
	int width = abs(*(int*) &HeaderInfo[18]);
	int height = abs(*(int*) &HeaderInfo[22]);

	unsigned int i;
	// Copy header for re-use
	for (i = 0; i < 54; i++)
		im.header[i] = HeaderInfo[i];

	// Set the image details structure
	im.height = height;
	im.width = width;
	int RowBytes = width;
	im.h_offset = RowBytes;
    im.type = "GREY";
	im.bitColor = 8;

	printf("\n   Input BMP File name: %20s  (%u x %u)", filename, im.height,
			im.width);

	// Allocate the memory for the image
	pel **TheImage = (pel **) malloc(height * sizeof(pel*));
	for (i = 0; i < height; i++)
		TheImage[i] = (pel *) malloc(RowBytes * sizeof(pel));

	// Read the image by rows
	for (i = 0; i < height; i++)
		fread(TheImage[i], sizeof(unsigned char), RowBytes, f);

	fclose(f);
	return TheImage;  
}

// Store a BMP image
void writeBMP(pel** img, char* filename) {
	// Open the file 
	FILE* f = fopen(filename, "wb");
	if (f == NULL) {
		printf("\n\nFILE CREATION ERROR: %s\n\n", filename);
		exit(1);
	}

	unsigned int x;
	// Write the header
	for (x = 0; x < 54; x++)
		fputc(im.header[x], f);

	unsigned int y;

	// Get the correct row size
	unsigned long rowSize = im.bitColor <= 8 ? im.width : im.h_offset;

	// Write the image data
	for (x = 0; x < im.height; x++)
	{
		for (y = 0; y < rowSize; y++) {
			char temp = img[x][y];
			fputc(temp, f);
		}
	}

	printf("\n  Output BMP File name: %20s  (%u x %u)", filename, im.height,
			im.width);

	fclose(f);
}

// Convert an RGB image to greyscale
pel** rgb2grey(pel** image)
{
	// Allocate memory for the greyscale image
	pel** grey_image = (pel**) malloc( im.height * sizeof(pel *));
	unsigned int i;
	for (i = 0; i < im.height; i++)
		grey_image[i] = (pel*) malloc(im.h_offset * sizeof(pel));

	unsigned int j, k;
	for (j = 0; j < im.height; j ++)
	{
		for (i = 0; i < im.width; i++)
		{
			// Compute the offset for the row
			k = i * 3;

			size_t r,g,b, grey_val;

			// Read the color values from the image
			r = image[j][k+2];
			g = image[j][k+1];
			b = image[j][k];

			// Compute the grey value using the luminance formula
			grey_val = (pel) round(0.3f * r + 0.59f * g + 0.11f * b);

			// Write the grey value 
			grey_image[j][k] = grey_val;
			grey_image[j][k+1] = grey_val;
			grey_image[j][k+2] = grey_val;
		}
	}

	return grey_image;
}

// Write a Bitmap image by recomputing the header info
void write_new_BMP(char* dest_path, pel** image, int h, int w, int bitColor)
{
	FILE *f;
	unsigned char *img = NULL;
	int filesize = 54 + 3 * w * h; // Compute the size of the image 

	// Allocate the memory for the output image with the required format 
	img = (unsigned char *)malloc(3 * w * h);
	memset(img,0,3*w*h);

	int i, j, x, y;
	int r, g ,b;
	for(i=0; i<h; i++)
	{
		for( j=0; j<w; j++)
		{
			// Read the pixel color values 
			if (bitColor == 8)
			{
				r = image[i][j];
				g = image[i][j];
				b = image[i][j];
			}
			else
			{
				r = image[i][j * 3 + 2];
				g = image[i][j * 3 + 1];
				b = image[i][j * 3];
			}
			
			// Scale the values if over the bound
			if (r > 255) r=255;
			if (g > 255) g=255;
			if (b > 255) b=255;

			// Set the values to the output image
			img[( j + i * w )*3+2] = (unsigned char)(r);
			img[( j + i * w )*3+1] = (unsigned char)(g);
			img[( j + i * w )*3+0] = (unsigned char)(b);
		}
	}

	// HEADER CREATION

	unsigned char bmpfileheader[14] = {'B','M', 0,0,0,0, 0,0, 0,0, 54,0,0,0};
	unsigned char bmpinfoheader[40] = {40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0, 24,0};
	unsigned char bmppad[3] = {0,0,0};

	bmpfileheader[ 2] = (unsigned char)(filesize    );
	bmpfileheader[ 3] = (unsigned char)(filesize>> 8);
	bmpfileheader[ 4] = (unsigned char)(filesize>>16);
	bmpfileheader[ 5] = (unsigned char)(filesize>>24);

	bmpinfoheader[ 4] = (unsigned char)(       w    );
	bmpinfoheader[ 5] = (unsigned char)(       w>> 8);
	bmpinfoheader[ 6] = (unsigned char)(       w>>16);
	bmpinfoheader[ 7] = (unsigned char)(       w>>24);
	bmpinfoheader[ 8] = (unsigned char)(       h    );
	bmpinfoheader[ 9] = (unsigned char)(       h>> 8);
	bmpinfoheader[10] = (unsigned char)(       h>>16);
	bmpinfoheader[11] = (unsigned char)(       h>>24);

	// Create the file to the given path and open the stream
	f = fopen(dest_path, "wb");

	// Write the header
	fwrite(bmpfileheader,1,14,f);
	fwrite(bmpinfoheader,1,40,f);

	// Write the pixel data
	for(i=0; i<h; i++)
	{
		fwrite(img+(w*(h-i-1)*3),3,w,f);
		fwrite(bmppad,1,(4-(w*3)%4)%4,f);
	}

	free(img);
	fclose(f);
}