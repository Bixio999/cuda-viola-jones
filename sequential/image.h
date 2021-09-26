
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

typedef unsigned char pel;

pel** readBMP_RGB(char*);
pel** readBMP_grey(char*);
void writeBMP(pel**, char*);
pel** rgb2grey(pel**);

extern struct Image im;