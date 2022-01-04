#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <stdbool.h>
#include <sys/time.h>

#include "image.h"
#include "classifier.h"

/*
    VIOLA-JONES - SEQUENTIAL
    MAIN

    The code in this folder is the sequential version of the Viola-Jones' Face Detection,
    which, given a bmp image, analyize, find and show the faces detected inside of it.

    This algorithm uses a pre-trained Cascade Classifier (CC) that outputs if inside the current
    window there is a face. The classification with this model is composed by several
    computation of Weak Classifier, called Filters. Each of them belongs to a stage, and is trained
    to recognize the presence of a specific Haar-like feature. The CC outputs positive
    if the current window passes all the stages' analysis. 
    
    A stage of the CC computes all the Filters it has, and at the end compares a final value
    which is the sum of the filters output with a stage threshold: if the stage output is 
    negative, then the current window doesn't contain a face and interrupt the face detection
    avoiding the computation of the next stages. In short, if a stage rejects the window, the 
    whole face detection in that window gives negative output. 

    A filter basically is a Weak Classifier, which evaluates in the current window the 
    presence of an Haar-like feature and returns its contribute to the stage computation. 
    As its name explains, alone it is not enough to detect a face, but can find some 
    features that may belongs to a face. The real accuracy of the CC is in fact the 
    combination of the detections from the filters. 
    
    An Haar-like feature is a composition of two or three rectangles of pixels which can be 
    distinguished as black or white areas. The combination of them in specific forms and
    positions can be useful to detect the presence of a face. For example, the eyes can be
    detected as a black area with a white area on the top, that is the forehead. 

    To speed-up the computations of the areas of rectangles for the Haar-like feature 
    detection, the Viola-Jones algorithm uses the so called Integral Image (II), which
    is a matrix of values where a cell (i,j) contains the sum of the pixels in the 
    rectangle of the image from (0,0) to (i,j). Its pre-computation allows to 
    significantly reduce the computations for rectangles areas from the manual sum of
    the pixels in the rectangle to a simple sum of four values from the II. That was 
    the key for its usage in real-time face detection. 

    However the computation of the II offers that speed-up respect to the area computation 
    when needed, but is highly dependant to the size of the image to evaluate: with big images, 
    its pre-computation takes a while. That is due to the algorithm complexity of O(n^2). 
    The Viola-Jones algorithm was really innovative for its times (about the 90s) because
    the images to evaluate was small. But at this time, the sizes of high quality images 
    are exponentially bigger than before, making this algorithm too slow. Now, for these 
    kind of tasks, artificial intelligences and neural networks are used rather than Viola-Jones.

*/

// Define a global variable to store the image details 
struct Image im;

// Prototype for the faces rectangle drawings
void draw_rectangle(pel** image, Rectangle* face);

// Function uesd to compute the elapsed time of the face detection
double seconds() {
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

int main(int argc, char const *argv[])
{
    // Check for missing arguments
    if (argc != 2 && argc != 3)
    {
        printf("[ERROR] missing arguments.\n");
        return 0;
    }

    // Get the image path
    int size_path = strlen(argv[1]);
    char path_img[size_path];
    strcpy(path_img, argv[1]);
    printf("%s", path_img);

    pel** image;
    pel** original_image;

    // Check for grey option
    if (argc == 3)
    {
        char grey_opt[strlen(argv[2])];
        strcpy(grey_opt, argv[2]);

        // Lower the argument chars
        int i;
        for (i = 0; i < strlen(grey_opt); i++)
            grey_opt[i] = tolower(grey_opt[i]);

        // Check if the option input is correct
        if (strcmp(grey_opt, "grey") == 0)
        {
            // If the grey option is enabled, read the image in given path as 8 bit image
            image = readBMP_grey(path_img);
            original_image = image;
        }
        else
        {
            printf("[ERROR] option not recognized. insert value: %s. comparison response: %d\n", grey_opt,strcmp(grey_opt, "grey"));
            exit(1);
        }
    }
    else
    {
        // If the image is RGB type, read as 24 bit
        original_image = readBMP_RGB(path_img);
        printf("\nimage correctly read.");
        // Convert the RGB image to greyscale
        image = rgb2grey(original_image);
    }

    // The VJ algorithm is applied on the greyscale image, so its correcly load is mandatory
    if (image == NULL)
    {
        printf("\n[ERROR] image not found or unable to correctly read.");
        exit(1);
    }

    // Path for the classifier file
    const char* classifier_file = "../class.txt";
    // Path for the classifier configuration file
    const char* config_file = "info.txt";

    // Load the classifier
    if (load_classifier(classifier_file, config_file))
        printf("\nclassifier correctly loaded.");
    else
    {
        printf("\nerror during classifier loadng. Aborting...");
        exit(1);
    }

    // DETECTION CONFIGS

    float scaleFactor = 1.2f; // The scaling factor used for the image pyramid 
    float groupFactor = 2.0f; // Distance factor for rectangles merging
    int minSize = 24; // Minimum size that the window must have - cannot be lower than 24 due to classifier training window settings
    int maxSize = 0; // Max size the window must have - with 0 it is set to the image size

    double initialTime = seconds();

    // Start the face detection and get the results list 
    List* face = detect_multiple_faces(image, scaleFactor, minSize, maxSize, groupFactor);

    double elapsedTime = seconds() - initialTime;
    printf("\nElapsed time: %f seconds", elapsedTime);
    
    // Check if any faces was detected
    if (face->size > 0)
    {
        printf("\nDetected %d faces in image. Starting drawing...", face->size);

        // For each result in the list, draw it in the original image
        while (face->size > 0)
            draw_rectangle(original_image, remove_from_head(face));
        // Write the output image
        write_new_BMP("out.bmp", original_image, im.height, im.width, 24);
    }
    else
        printf("\n no faces detected in given image.\n");

    printf("\n\tface detection completed.");
    exit(0);
}

// Function to draw the rectangles given by face detection 
void draw_rectangle(pel** image, Rectangle* face)
{
    int i,j;

    // COLOR SETTINGS

    short color_RGB = 255; // Define the red component for the rectangles in RGB images
    short color_GREY = 150; // Define the grey value for the rectangles in greyscale images

    short r, g, b;

    // Check if the image is RGB
    if (strcmp(im.type, "GREY") != 0)
    {
        r = color_RGB;
        g = 0;
        b = 0;
    }
    else
    {
        r = color_GREY;
        g = color_GREY;
        b = color_GREY;
    }

    // Draw the rectangle by overwriting the pixels values
    
    for (i = face->y, j = face->x; j < face->x + face->size.width; j++)
    {
        image[i][3 * j + 2] = r; // r
        image[i][3 * j + 1] = g;   // g
        image[i][3 * j] = b;       // b

        image[i + face->size.height - 1] [3 * j + 2] = r; // r
        image[i + face->size.height - 1] [3 * j + 1] = g;   // g
        image[i + face->size.height - 1] [3 * j] = b;       // b
    }

    for (i = face->y, j = face->x; i < face->y + face->size.height; i++)
    {
        image[i][3 * j + 2] = r; // r
        image[i][3 * j + 1] = g;   // g
        image[i][3 * j] = b;       // b

        image[i] [3 * (j + face->size.width - 1) + 2] = r; // r
        image[i] [3 * (j + face->size.width - 1) + 1] = g;   // g
        image[i] [3 * (j + face->size.width - 1)] = b;       // b
    }
}