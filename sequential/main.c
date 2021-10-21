#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <stdbool.h>

#include "image.h"
#include "classifier.h"

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
            printf("[ERROR] option not recognized. insert value: %s. comparison response: %d\n", grey_opt,strcmp(grey_opt, "grey"));
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

    const char* classifier_file = "../class.txt";
    const char* config_file = "info.txt";

    if (load_classifier(classifier_file, config_file))
        printf("\nclassifier correctly loaded.");

    float scaleFactor = 1.2f;
    int minSize = 24;
    int maxSize = 0;

    List* face = detect_multiple_faces(image, scaleFactor, minSize, maxSize);

    printf("\ndrawing detected faces");
    

    if (face->size > 0)
    {
        while (face->size > 0)
            draw_rectangle(image, remove_from_head(face));
        write_new_BMP("out.bmp", image, im.height, im.width, 24);
    }
    else
        printf("\n no faces detected in given image.\n");

    exit(0);
}

void draw_rectangle(pel** image, Rectangle* face)
{
    int i,j;

    for (i = face->y; i < face->y + face->size.height; i++)
    {
        for (j = face->x; j < face->x + face->size.width; i++)
        {
            image[i][3 * j + 2] = 255; // r
            image[i][3 * j + 1] = 0;   // g
            image[i][3 * j] = 0;       // b
        }
    }
}