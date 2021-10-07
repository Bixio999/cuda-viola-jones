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

    Classifier* classifier = load_classifier(classifier_file, config_file);

    if (classifier)
        printf("\nclassifier correctly loaded.");

    float scaleFactor = 1.2f;
    Size minSize = {24, 24};
    Size maxSize = {0,0};

    Rectangle* face = detect_single_face(classifier, image, scaleFactor, minSize, maxSize);

    draw_rectangle(image, face);

    // writeBMP(image, "check_eyes.bmp");
    write_new_BMP("out.bmp", image, im.height, im.width, 24);
    
}



void draw_rectangle(pel** image, Rectangle* face)
{

}

// void checkEyeFeature(double** int_image, pel** image)
// {
//     const float rectSizeW = 0.4f, rectSizeH = 0.05f;

//     int pixSizeW = round(rectSizeW * im.width);
//     int pixSizeH = round(rectSizeH * im.height);

//     unsigned int i, j;

//     unsigned int max_i = 0, max_j = 0;
//     double max = 0;
//     size_t max_color = 0; 
//     for ( i = 0; i < im.height - 2 * pixSizeH; i += 1)
//     {
//         for (j = 0; j < im.width - pixSizeW; j += 1)
//         {
//             double white = int_image[i + pixSizeH][j + pixSizeW] - int_image [i + pixSizeH][j] - int_image[i][j + pixSizeW] + int_image[i][j];

//             double black = int_image[i + 2 * pixSizeH][j + pixSizeW] - int_image [i + 2 * pixSizeH][j] - int_image[i + pixSizeH][j + pixSizeW] + int_image[i + pixSizeH][j];

//             double diff = - black + white;

//             size_t color = 255 - 255 * (1 - diff/ (white > black? white : black) );

//             // printf("\n\trect(%d, %d): white = %f, black = %f, diff = %f", i, j, white, black, diff);

//             if (diff > max)
//             {
//                 max = diff;
//                 max_i = i;
//                 max_j = j;
//                 max_color = color;
//             }
//         }
//     }

//     printf("\n\n\tmax_rect(%d, %d): diff = %f, color = %d", max_i, max_j, max, max_color);

//     unsigned int k;
//     for (k = 0; k< pixSizeW; k ++)
//     {
//         image[max_i][(max_j + k) * 3] = 0;
//         image[max_i][(max_j + k) * 3 + 1] = 0;
//         image[max_i][(max_j + k) * 3 + 2] = max_color;

//         image[max_i + 2 * pixSizeH][(max_j + k) * 3] = 0;
//         image[max_i + 2 * pixSizeH][(max_j + k) * 3 + 1] = 0;
//         image[max_i + 2 * pixSizeH][(max_j + k) * 3 + 2] = max_color;
//     }
//     for (k = 0; k< 2 * pixSizeH; k ++)
//     {
//         image[max_i + k][max_j * 3] = 0;
//         image[max_i + k][max_j * 3 + 1] = 0;
//         image[max_i + k][max_j * 3 + 2] = max_color;

//         image[max_i + k][(max_j + pixSizeW) * 3] = 0;                
//         image[max_i + k][(max_j + pixSizeW) * 3 + 1] = 0;                
//         image[max_i + k][(max_j + pixSizeW) * 3 + 2] = max_color;                
//     }

// }
