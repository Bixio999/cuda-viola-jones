
#pragma once
struct Image 
{
    int height;
    int width;
    unsigned char header[54];
    unsigned long int h_offset;
    char* type;
    size_t bitColor;
};

struct Pixel
{
    unsigned char R;
    unsigned char G;
    unsigned char B;
};

#define ROWBYTES(a) ((a * 3 + 3) & (~3))

#pragma once
typedef unsigned char pel;

pel** readBMP_RGB(char*);
pel** readBMP_grey(char*);
void writeBMP(pel**, char*);
pel** rgb2grey(pel**);
void write_new_BMP(char* dest_path, pel** image, int height, int width, int bitColor);

extern struct Image im;