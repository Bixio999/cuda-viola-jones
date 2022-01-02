/*

    Definitions of structs for image handling, and function prototypes for 
    read, write and greyscale conversion for BMP images.

*/

// struct that stores the details of the read image
#pragma once
struct Image 
{
    int height;
    int width;
    unsigned char header[54]; // contains the BMP header of the file
    unsigned long int h_offset; // the real size of a row of the image
    char* type; // can be GREY or RGB
    size_t bitColor; // the software only handles 8bit or 24bit images
};

// computation of the real row size
#define ROWBYTES(a) ((a * 3 + 3) & (~3))

// assign the type for pixel values
#pragma once
typedef unsigned char pel;

// FUNCTION PROTOTYPES

pel* readBMP_RGB(char*);
pel* readBMP_grey(char*);
void writeBMP(pel*, char*);
pel* rgb2grey(pel*);
void write_new_BMP(char* dest_path, pel* image, int height, int width, int bitColor);

// allows the access for image details to all the files
extern struct Image im;