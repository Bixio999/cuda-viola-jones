#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <stdbool.h>

#include "image.h"
#include "classifier.h"
#include "utils/common.h"

struct Image im;

double** integral_image(pel** image);
void draw_rectangle(pel** image, Rectangle* face);

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

    pel* dev_image;
    pel* original_image;

    if (argc == 3)
    {
        char grey_opt[strlen(argv[2])];
        strcpy(grey_opt, argv[2]);

        int i;
        for (i = 0; i < strlen(grey_opt); i++)
            grey_opt[i] = tolower(grey_opt[i]);

        if (strcmp(grey_opt, "grey") == 0)
        {
            original_image = readBMP_grey(path_img);
            CHECK(cudaMalloc((void **) &dev_image, sizeof(pel) * im.height * im.width));
            CHECK(cudaMemcpy(dev_image, original_image, sizeof(pel) * im.height * im.width, cudaMemcpyHostToDevice));
        }
        else
        {
            printf("[ERROR] option not recognized. insert value: %s. comparison response: %d\n", grey_opt,strcmp(grey_opt, "grey"));
            exit(1);
        }
    }
    else
    {
        original_image = readBMP_RGB(path_img);
        printf("\nimage correctly read.");
        dev_image = rgb2grey(original_image);
        printf("\nimage correctly converted to greyscale.");
    }

    if (dev_image == NULL)
    {
        printf("\n[ERROR] image not found or unable to correctly read.");
        exit(1);
    }

    const char* classifier_file = "../class.txt";
    const char* config_file = "info.txt";

    if (load_classifier_to_gpu(classifier_file, config_file))
        printf("\nclassifier correctly loaded.");
    
    

    float scaleFactor = 1.2f;
    int minSize = 24;
    int maxSize = 0;

    // List* face = detect_multiple_faces(image, scaleFactor, minSize, maxSize);

    // printf("\nDetected %d faces in image. Starting drawing...", face->size);
    

    // if (face->size > 0)
    // {
    //     while (face->size > 0)
    //         draw_rectangle(original_image, remove_from_head(face));
    //     write_new_BMP("out.bmp", original_image, im.height, im.width, 24);
    //     // writeBMP(original_image, "out_2.bmp");
    // }
    // else
    //     printf("\n no faces detected in given image.\n");
    // writeBMP(original_image, "out_2.bmp");
    exit(0);
}

void draw_rectangle(pel** image, Rectangle* face)
{
    int i,j;

    short color_RGB = 255;
    short color_GREY = 150;

    if (strcmp(im.type, "GREY") != 0)
    {
        for (i = face->y, j = face->x; j < face->x + face->size.width; j++)
        {
            image[i][3 * j + 2] = color_RGB; // r
            image[i][3 * j + 1] = 0;   // g
            image[i][3 * j] = 0;       // b

            image[i + face->size.height - 1] [3 * j + 2] = color_RGB; // r
            image[i + face->size.height - 1] [3 * j + 1] = 0;   // g
            image[i + face->size.height - 1] [3 * j] = 0;       // b
        }

        for (i = face->y, j = face->x; i < face->y + face->size.height; i++)
        {
            image[i][3 * j + 2] = color_RGB; // r
            image[i][3 * j + 1] = 0;   // g
            image[i][3 * j] = 0;       // b

            image[i] [3 * (j + face->size.width - 1) + 2] = color_RGB; // r
            image[i] [3 * (j + face->size.width - 1) + 1] = 0;   // g
            image[i] [3 * (j + face->size.width - 1)] = 0;       // b
        }
        return;
    }

    for (i = face->y, j = face->x; j < face->x + face->size.width; j++)
    {
        image[i][3 * j + 2] = color_GREY; // r
        image[i][3 * j + 1] = color_GREY;   // g
        image[i][3 * j] = color_GREY;       // b

        image[i + face->size.height - 1] [3 * j + 2] = color_GREY; // r
        image[i + face->size.height - 1] [3 * j + 1] = color_GREY;   // g
        image[i + face->size.height - 1] [3 * j] = color_GREY;       // b
    }

    for (i = face->y, j = face->x; i < face->y + face->size.height; i++)
    {
        image[i][3 * j + 2] = color_GREY; // r
        image[i][3 * j + 1] = color_GREY;   // g
        image[i][3 * j] = color_GREY;       // b

        image[i] [3 * (j + face->size.width - 1) + 2] = color_GREY; // r
        image[i] [3 * (j + face->size.width - 1) + 1] = color_GREY;   // g
        image[i] [3 * (j + face->size.width - 1)] = color_GREY;       // b
    }
}

