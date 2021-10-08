#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <stdio.h>

#include "classifier.h"
#include "list.h"

const int window_size = 24;

Classifier* load_classifier(const char* classifier_path, const char* config_path)
{
    if (classifier_path && config_path)
    {
        printf("\nnot null parameters: %s, %s", classifier_path, config_path);

        FILE* config_file = fopen(config_path, "r");

        if (!config_file)
        {
            printf("\nerror while reading from config file. aborting...");
            exit(1);
        }

        int stages = 0;
        fscanf(config_file, "%d", &stages);
        if (!stages)
        {
            printf("\nerror while reading from config file. aborting...");
            exit(1);
        }
        
        int* filter_per_stages = (int*) malloc(sizeof(int) * stages);

        int i = 0;
        int n_filters = 0;
        int* temp = filter_per_stages;
        while (i < stages && !feof(config_file))
        {
            fscanf(config_file, "%d", temp);
            i++;
            temp++;
        }

        fclose(config_file);

        FILE* classifier_file = fopen(classifier_path, "r");
        if (!classifier_file)
        {
            printf("\nerror while reading from classifier file. aborting...");
            exit(1);
        }

        Filter*** filters = (Filter***) malloc(sizeof(Filter**) * stages);
        int* stage_thresholds = (int*) malloc(sizeof(int) * stages);

        int j;
        int* st_thr = stage_thresholds;
        for (i = 0; i < stages; i++)
        {
            n_filters = filter_per_stages[i];
            filters[i] = (Filter**) malloc(sizeof(Filter*) * n_filters);

            for (j = 0; j < n_filters && !feof(classifier_file); j++)
            {
                Filter* f = malloc(sizeof(Filter));
                Rectangle rect = f->rect1;

                fscanf(classifier_file, "%d", &(rect.x)); 
                fscanf(classifier_file, "%d", &(rect.y)); 
                fscanf(classifier_file, "%d", &(rect.size.width)); 
                fscanf(classifier_file, "%d", &(rect.size.height));
                
                fscanf(classifier_file, "%d", &(f->weight1));

                rect = f->rect2;

                fscanf(classifier_file, "%d", &(rect.x)); 
                fscanf(classifier_file, "%d", &(rect.y)); 
                fscanf(classifier_file, "%d", &(rect.size.width)); 
                fscanf(classifier_file, "%d", &(rect.size.height));                

                fscanf(classifier_file, "%d", &(f->weight2));

                rect = f->rect3;

                fscanf(classifier_file, "%d", &(rect.x)); 
                fscanf(classifier_file, "%d", &(rect.y)); 
                fscanf(classifier_file, "%d", &(rect.size.width)); 
                fscanf(classifier_file, "%d", &(rect.size.height));                

                fscanf(classifier_file, "%d", &(f->weight3));    

                fscanf(classifier_file, "%d", &(f->threshold));                
                fscanf(classifier_file, "%d", &(f->alpha1));                
                fscanf(classifier_file, "%d", &(f->alpha2));           

                filters[i][j] = f;
            }
            if (!feof(classifier_file))
                fscanf(classifier_file, "%d", st_thr++);
        }
        fclose(classifier_file);

        Classifier* classifier = malloc(sizeof(Classifier));
        classifier->filters = filters;
        classifier->filters_per_stages = filter_per_stages;
        classifier->n_stages = stages;

        return classifier;
    }
    else
        printf("\nnull parameters detected. aborting...");
    return NULL;
}

double** integral_image_24bit(pel** image, Size size)
{
    if (image == NULL)
        return NULL;

    double** iim = (double**) malloc(size.height * sizeof(double*));
    unsigned int i;
    for (i = 0; i < size.height; i++)
        iim[i] = (double *) malloc(size.width * sizeof(double));

    iim[0][0] = image[0][0];

    unsigned int j;
    for (i = 1; i < size.height; i++)
        iim[i][0] = iim[i-1][0] + image[i][0];

    for (i = 1; i < size.width; i++)
        iim[0][i] = iim[0][i-1] + image[0][i * 3];

    for (i = 1; i < size.height; i++)
    {
        for (j = 1; j < size.width; j++)
            iim[i][j] = iim[i-1][j] + iim[i][j-1] - iim[i-1][j-1] + image[i][j * 3];
    }

    return iim;
}

double** integral_image(pel** image, Size size)
{
    if (image == NULL)
        return NULL;

    double** iim = (double**) malloc(size.height * sizeof(double*));
    unsigned int i;
    for (i = 0; i < size.height; i++)
        iim[i] = (double *) malloc(size.width * sizeof(double));

    iim[0][0] = image[0][0];

    unsigned int j;
    for (i = 1; i < size.height; i++)
        iim[i][0] = iim[i-1][0] + image[i][0];

    for (i = 1; i < size.width; i++)
        iim[0][i] = iim[0][i-1] + image[0][i];

    for (i = 1; i < size.height; i++)
    {
        for (j = 1; j < size.width; j++)
            iim[i][j] = iim[i-1][j] + iim[i][j-1] - iim[i-1][j-1] + image[i][j];
    }

    return iim;
}

pel** resize_image(pel** image, Size out_size)
{
    float resize_factor = (float) im.height / out_size.height;
    printf("\nresizing factor: %f", resize_factor);

    pel** output = (pel**) malloc(sizeof(pel*) * out_size.height);
    int i;
    for (i = 0; i < out_size.height; i++)
        output[i] = (pel*) malloc(sizeof(pel) * out_size.width);

    int j;
    for (i = 0; i < out_size.height; i++)
    {
        for (j = 0; j < out_size.width; j++)
        {
            int h = floor(i * resize_factor);
            int w = floor(j * resize_factor);

            output[i][j] = image[h][w * 3];
        }
    }
    return output;
}

void runClassifier(double** iim, Size size, List* faces)
{

}

void free_integral_image(double** iim, Size size)
{
    int i;
    for (i = 0; i < size.height; i++)
        free(iim[i]);
    free(iim);
}

void free_image(pel** image, Size size)
{
    int i;
    for (i = 0; i < size.height; i++)
        free(image[i]);
    free(image);
}

// Rectangle* detect_single_face(Classifier* classifier, pel** image, float scaleFactor, int minWindow, int maxWindow)
// {
//     float currFactor = 1.0f;
//     // do {
        
//     // }while();


//     Size temp_size = { round(im.width / scaleFactor), round(im.height / scaleFactor) };
//     printf("\nresizing image from %d x %d to %d x %d.", im.height, im.width, temp_size.height, temp_size.width);
//     pel** resized = resize_image(image, temp_size);
//     write_new_BMP("resized.bmp", resized, temp_size.height, temp_size.width, 8);
//     return NULL;
// }

List* detect_multiple_faces(Classifier* classifier, pel** image, float scaleFactor, int minWindow, int maxWindow)
{
    List* faces = listInit();

    if ( minWindow <= window_size )
    {
        Size image_size = { im.width, im.height };
        double** iim = integral_image_24bit(image, image_size);

        runClassifier(iim, image_size, faces);
        free_integral_image(iim, image_size);
    }

    if ( maxWindow < window_size )
        maxWindow = min(im.width, im.height);

    float currFactor;
    for (currFactor = scaleFactor; ; currFactor*= scaleFactor)
    {
        Size temp_size = { round(im.width / currFactor), round(im.height / currFactor) };

        if (minWindow > round(window_size * currFactor))
            continue;

        if (maxWindow < min(temp_size.width, temp_size.height))
            break;

        pel** res_im = resize_image(image, temp_size);
        double** iim = integral_image(res_im, temp_size);

        runClassifier(iim, temp_size, faces);

        free_image(res_im, temp_size);
        free_integral_image(iim, temp_size);

        // TODO GROUP RECTANGLES
    }
}