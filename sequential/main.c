#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "image.h"

#include <opencv2/objdetect/objdetect_c.h>
// #include <opencv2/core/core_c.h>

struct Image im;

int** integral_image(pel** image);

int main(int argc, char const *argv[])
{
    if (argc != 2 && argc != 3)
    {
        printf("[ERROR] missing arguments.\n");
        return 0;
    }

    int size_path = strlen(argv[1]);

    char path_img[size_path];
    strcpy(path_img, argv[1]);

    printf("%s", path_img);

    pel** image;

    if (argc == 3)
    {
        char grey_opt[strlen(argv[2])];
        strcpy(grey_opt, argv[2]);

        int i;
        for (i = 0; i < strlen(grey_opt); i++)
            grey_opt[i] = tolower(grey_opt[i]);

        if (strcmp(grey_opt, "grey") == 0)
            image = readBMP_grey(path_img);
        else
        {
            printf("[ERROR] option not recognized. insert value: %s. comparison response: $d\n", grey_opt,strcmp(grey_opt, "grey"));
            exit(1);
        }
    }
    else
    {
        image = readBMP_RGB(path_img);
        printf("\nimage correctly read.");
        pel ** oldRGBImage = image;
        image = rgb2grey(oldRGBImage);
        free(oldRGBImage);
    }

    if (image == NULL)
    {
        printf("\n[ERROR] image not found or unable to correctly read.");
        exit(1);
    }
    
    int **int_image = integral_image(image);
    printf("\nintegral image computated.\n");

    CvMemStorage *storage = (CvMemStorage *) malloc(sizeof(CvGraph));

    CvHaarClassifierCascade* face_cascade = cvLoad("../haarcascade_frontalface_default.xml", storage, NULL, NULL);
    if (face_cascade == NULL)
        printf("\n haar classifier not loaded.");
    
}

int** integral_image(pel** image)
{
    if (image == NULL)
        return NULL;

    int** iim = (int**) malloc(im.height * sizeof(int*));
    unsigned int i;
    for (i = 0; i < im.height; i++)
        iim[i] = (int *) malloc(im.width * sizeof(int));

    iim[0][0] = image[0][0];

    unsigned int j;
    for (i = 1; i < im.height; i++)
        iim[i][0] = iim[i-1][0] + image[i][0];

    for (i = 1; i < im.width; i++)
        iim[0][i] = iim[0][i-1] + image[0][i];

    for (i = 1; i < im.height; i++)
    {
        for (j = 1; j < im.width; j++)
            iim[i][j] = iim[i-1][j] + iim[i][j-1] + image[i][j];
    }
    return iim;
}
