#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "image.h"
#include "utils/common.h"

/*
	This file contails the functions to handle with Bitmap images, in order to 
	read and create the data structure needed for the algorithm, convert the image 
	to greyscale, and write a new bitmap image.

	The function rbg2grey uses a CUDA kernel to speed-up the task's execution.
*/

// Read a Bitmap image from the given path
pel* readBMP_RGB(char* filename) {
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
	pel *TheImage = (pel *) malloc(height * RowBytes * sizeof(pel));
	
	// Read the entire image as mono-dimensional array  
	fread(TheImage, sizeof(unsigned char), height * RowBytes, f);

	fclose(f);
	return TheImage; 
}

// Read a greyscale Bitmap image from the given path
pel* readBMP_grey(char* filename) {
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
	pel *TheImage = (pel *) malloc(height * RowBytes * sizeof(unsigned char));
	
	// Read the entire image as mono-dimensional array  
	fread(TheImage, sizeof(unsigned char), height * RowBytes, f);

	fclose(f);
	return TheImage;
}

// Store a BMP image
void writeBMP(pel* img, char* filename) {
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

	// Get the correct row size
	unsigned long rowSize = im.bitColor <= 8 ? im.width : im.h_offset;

	// Write the image data
	for (x = 0; x < im.height * rowSize; x++)
	{
		char temp = img[x];
		fputc(temp, f);
	}

	printf("\n  Output BMP File name: %20s  (%u x %u)", filename, im.height,
			im.width);

	fclose(f);
}

/*
	CUDA Kernel that converts an RGB image to a greyscale image, by using
	the luminance formula.

	Each thread elaborates a single pixel by computing the greyscale value 
	and overwriting the old RGB values.
*/
__global__ void cuda_rgb2grey(pel* image, int size)
{
	// Compute the image offset for the current CUDA thread 
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	// Check if the current thread should be active or not
	if (id < size)
	{
		// Get the scaled offset for image reading
		int pixId = id * 3;

		// Calculate the grey value
		pel grey_val = (pel) roundf(0.3f * image[pixId + 2] + 0.59f * image[pixId + 1] + 0.11f * image[pixId]);

		// Overwrite the pixel values in the image
		image[pixId] = grey_val;
		image[pixId + 1] = grey_val;
		image[pixId + 2] = grey_val;
	}
}

/*
	Convert an RGB image to greyscale. Handler function that uses a
	CUDA kernel for the task.

	The image argument must be an host pointer, so the RGB image have
	to be in host memory.

	Returns the device pointer of the greyscale image. 
*/
pel* rgb2grey(pel* image)
{
	pel* dev_image;

	// Get the best dimensions of blocks and grid 
	uint dimBlock, dimGrid;
	compute_grid_dimension(im.height * im.width, &dimBlock, &dimGrid);

	printf("\n dimBlock = %d, dimGrid = %d", dimBlock, dimGrid);

	// Get the number of bytes needed to store the image
	int nBytes = sizeof(pel) * im.height * im.h_offset;

	// Copy the image to device
	CHECK(cudaMalloc((void**) &dev_image, nBytes ));
	CHECK(cudaMemcpy(dev_image, image, nBytes, cudaMemcpyHostToDevice));

	// Call the CUDA kernel
	cuda_rgb2grey<<< dimGrid, dimBlock >>>(dev_image, im.height * im.width);
	cudaDeviceSynchronize();

	// Returns the device pointer of the computed greyscale image
	return dev_image;
}

// Write a Bitmap image by recomputing the header info
void write_new_BMP(char* dest_path, pel* image, int h, int w, int bitColor)
{
	FILE *f;
	unsigned char *img = NULL;
	int filesize = 54 + 3 * w * h;  // Compute the size of the image 

	// Allocate the memory for the output image with the required format 
	img = (unsigned char *)malloc(3 * w * h);
	memset(img,0,3*w*h);

	int i, j;
	int r, g ,b;
	for(i=0; i<h; i++)
	{
		for( j=0; j<w; j++)
		{
			// Read the pixel color values 
			if (bitColor == 8)
			{
				r = image[i * h + j];
				g = r;
				b = r;
			}
			else
			{
				r = image[(i * w + j) * 3 + 2];
				g = image[(i * w + j) * 3 + 1];
				b = image[(i * w + j) * 3];
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