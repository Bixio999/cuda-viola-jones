#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "image.h"

struct Image im;

int** integral_image(pel** image);

int main(int argc, char const *argv[])
{
    if (argc < 2)
        return 0;

    int size = strlen(argv[1]);

    char path_img[size];
    strcpy(path_img, argv[1]);

    printf("%s", path_img);

    pel** image = readBMP_grey(path_img);
    if (image == NULL)
    {
        printf("\n[ERROR] image not found or unable to correctly read.");
        exit(1);
    }

    
    
}
