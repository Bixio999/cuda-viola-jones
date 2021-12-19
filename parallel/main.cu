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


void compare_grey_images(pel* dev_grey, pel* image);
void draw_rectangle(pel* image, Rectangle* face);

// __global__ void cuda_draw_rectangles(Rectangle** faces, pel* image, unsigned int width, unsigned int height, bool rgb)
// {
//     unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

//     if (id >= width * height)
//         return;

//     Rectangle* face = faces[id];

//     if (face == NULL)
//         return;

//     short color_RGB = 255;
//     short color_GREY = 150;

//     uint i, j;
//     if (rgb)
//     {
        // for (i = face->y, j = face->x; j < face->x + face->size.width; j++)
        // {
        //     image[(i * width + j) * 3 + 2] = color_RGB;
        //     image[(i * width + j) * 3 + 1] = 0;
        //     image[(i * width + j) * 3] = 0;
        //     // image[i * width + 3 * j + 1] = 0;
        //     // image[i * width + 3 * j] = 0;

        //     image[((i + face->size.height - 1) * width + j) * 3 + 2] = color_RGB;
        //     image[((i + face->size.height - 1) * width + j) * 3 + 1] = 0;
        //     image[((i + face->size.height - 1) * width + j) * 3] = 0;
        //     // image[(i + face->size.height - 1) * width + 3 * j + 1] = 0;
        //     // image[(i + face->size.height - 1) * width + 3 * j] = 0;
        // }

        // for (i = face->y, j = face->x; i < face->y + face->size.height; i++)
        // {
        //     image[(i * width + j) * 3 + 2] = color_RGB; // r
        //     image[(i * width + j) * 3 + 1] = 0; // r
        //     image[(i * width + j) * 3] = 0; // r
        //     // image[i * width + 3 * j + 1] = 0;   // g
        //     // image[i * width + 3 * j] = 0;       // b

        //     image[(i * width + j + face->size.width - 1) * 3 + 2] = color_RGB; // r
        //     image[(i * width + j + face->size.width - 1) * 3 + 1] = 0; // r
        //     image[(i * width + j + face->size.width - 1) * 3] = 0; // r
        //     // image[i * width + 3 * (j + face->size.width - 1) + 1] = 0;   // g
        //     // image[i * width + 3 * (j + face->size.width - 1)] = 0;       // b
        // }
//     }
//     else
//     {
        // for (i = face->y, j = face->x; j < face->x + face->size.width; j++)
        // {
        //     image[i * width + 3 * j + 2] = color_GREY; // r
        //     image[i * width + 3 * j + 1] = color_GREY;   // g
        //     image[i * width + 3 * j] = color_GREY;       // b

        //     image[(i + face->size.height - 1) * width + 3 * j + 2] = color_GREY; // r
        //     image[(i + face->size.height - 1) * width + 3 * j + 1] = color_GREY;   // g
        //     image[(i + face->size.height - 1) * width + 3 * j] = color_GREY;       // b
        // }

        // for (i = face->y, j = face->x; i < face->y + face->size.height; i++)
        // {
        //     image[i * width + 3 * j + 2] = color_GREY; // r
        //     image[i * width + 3 * j + 1] = color_GREY;   // g
        //     image[i * width + 3 * j] = color_GREY;       // b

        //     image[i * width + 3 * (j + face->size.width - 1) + 2] = color_GREY; // r
        //     image[i * width + 3 * (j + face->size.width - 1) + 1] = color_GREY;   // g
        //     image[i * width + 3 * (j + face->size.width - 1)] = color_GREY;       // b
        // }
//     }
// }

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

    // write_new_BMP("img.bmp", original_image, im.height, im.width, 24);

    // int numbytes = sizeof(pel) * im.h_offset * im.height;
    // pel* grey_image = (pel*) malloc(numbytes);
    // CHECK(cudaMemcpy(grey_image, dev_image, numbytes, cudaMemcpyDeviceToHost));

    // write_new_BMP("grey.bmp", grey_image, im.height, im.width, 24);
    // compare_grey_images(grey_image, original_image);

    const char* classifier_file = "../class.txt";
    const char* config_file = "info.txt";

    if (load_classifier_to_gpu(classifier_file, config_file))
        printf("\nclassifier correctly loaded.");

    float scaleFactor = 1.2f;
    float groupFactor = 2.0f;
    int minSize = 24;
    int maxSize = 0;

    unsigned int face_counter;

    double intialTime = seconds();

    List* faces = detect_multiple_faces(dev_image, scaleFactor, minSize, maxSize, &face_counter, groupFactor);

    double elapsedTime = seconds() - intialTime;
    printf("\n Elapsed time: %f seconds", elapsedTime);

    if (face_counter > 0)
    {
        printf("\nDetected %u faces, starting drawing...", face_counter);
        // printf("\n faces->size = %d", faces->size);
        
        // unsigned int i = 0;
        while (faces->size > 0)
        {
            Rectangle* r = remove_from_head(faces);
            // if (r)
            //     printf("\n\t\t\trectangle: %d { x = %hu, y = %hu, size.width = %u, size.height = %u }",i, r->x, r->y, r->size.width, r->size.height);
            // else
            //     printf("\n\tNULL pointer returned from list at i = %u", i);
            draw_rectangle(original_image, r);
            // i++;
        }

        write_new_BMP("out.bmp", original_image, im.height, im.width, 24);
    }
    else
        printf("\nNo faces detected in given image.");

    printf("\n\tface detection completed!\n");

    cudaDeviceReset();
    exit(0);
}

void compare_grey_images(pel* dev_grey, pel* image)
{
	unsigned int j, k,i;
	for (j = 0; j < im.height; j ++)
	{
		for (i = 0; i < im.width; i++)
		{
			k = j * im.h_offset + i * 3;

			pel r,g,b, grey_val;
			r = image[k+2];
			g = image[k+1];
			b = image[k];

			grey_val = (pel) round(0.3f * r + 0.59f * g + 0.11f * b); // luminance formula

			if (abs(grey_val - dev_grey[k]) > 1)
            {
                printf("\nDetected value inconsistency at (%u,%u): cpu_grey = %d, gpu_grey = %d", j,i,grey_val, dev_grey[k]);
                exit(1);
            }
		}
	}
}

void draw_rectangle(pel* image, Rectangle* face)
{
    int i,j;

    short color_RGB = 255;
    short color_GREY = 150;

    if (strcmp(im.type, "GREY") != 0)
    {
        for (i = face->y, j = face->x; j < face->x + face->size.width; j++)
        {
            image[(i * im.width + j) * 3 + 2] = color_RGB;
            image[(i * im.width + j) * 3 + 1] = 0;
            image[(i * im.width + j) * 3] = 0;
            // image[i * width + 3 * j + 1] = 0;
            // image[i * width + 3 * j] = 0;

            image[((i + face->size.height - 1) * im.width + j) * 3 + 2] = color_RGB;
            image[((i + face->size.height - 1) * im.width + j) * 3 + 1] = 0;
            image[((i + face->size.height - 1) * im.width + j) * 3] = 0;
            // image[(i + face->size.height - 1) * width + 3 * j + 1] = 0;
            // image[(i + face->size.height - 1) * width + 3 * j] = 0;
        }

        for (i = face->y, j = face->x; i < face->y + face->size.height; i++)
        {
            image[(i * im.width + j) * 3 + 2] = color_RGB; // r
            image[(i * im.width + j) * 3 + 1] = 0; // r
            image[(i * im.width + j) * 3] = 0; // r
            // image[i * width + 3 * j + 1] = 0;   // g
            // image[i * width + 3 * j] = 0;       // b

            image[(i * im.width + j + face->size.width - 1) * 3 + 2] = color_RGB; // r
            image[(i * im.width + j + face->size.width - 1) * 3 + 1] = 0; // r
            image[(i * im.width + j + face->size.width - 1) * 3] = 0; // r
            // image[i * width + 3 * (j + face->size.width - 1) + 1] = 0;   // g
            // image[i * width + 3 * (j + face->size.width - 1)] = 0;       // b
        }
        return;
    }

    for (i = face->y, j = face->x; j < face->x + face->size.width; j++)
    {
        image[i * im.width + 3 * j + 2] = color_GREY; // r
        image[i * im.width + 3 * j + 1] = color_GREY;   // g
        image[i * im.width + 3 * j] = color_GREY;       // b

        image[(i + face->size.height - 1) * im.width + 3 * j + 2] = color_GREY; // r
        image[(i + face->size.height - 1) * im.width + 3 * j + 1] = color_GREY;   // g
        image[(i + face->size.height - 1) * im.width + 3 * j] = color_GREY;       // b
    }

    for (i = face->y, j = face->x; i < face->y + face->size.height; i++)
    {
        image[i * im.width + 3 * j + 2] = color_GREY; // r
        image[i * im.width + 3 * j + 1] = color_GREY;   // g
        image[i * im.width + 3 * j] = color_GREY;       // b

        image[i * im.width + 3 * (j + face->size.width - 1) + 2] = color_GREY; // r
        image[i * im.width + 3 * (j + face->size.width - 1) + 1] = color_GREY;   // g
        image[i * im.width + 3 * (j + face->size.width - 1)] = color_GREY;       // b
    }
}
