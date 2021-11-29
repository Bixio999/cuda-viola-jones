#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stddef.h>

#define min(a,b) (a < b? a : b)
#define max(a,b) (a > b? a : b)

#include "classifier.h"

typedef struct filter{
    Rectangle rect1;
    int weight1;
    Rectangle rect2;
    int weight2;
    Rectangle rect3;
    int weight3;
    
    int threshold;
    int alpha1, alpha2;
} Filter;

typedef struct classifier
{
    Filter*** filters;
    int* filters_per_stages;
    int* stage_thresholds;
    int n_stages;
} Classifier;

const int window_size = 24;

Classifier* classifier;

bool load_classifier(const char* classifier_path, const char* config_path)
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
                Rectangle* rect = &(f->rect1);

                fscanf(classifier_file, "%d", &(rect->x)); 
                fscanf(classifier_file, "%d", &(rect->y)); 
                fscanf(classifier_file, "%d", &(rect->size.width)); 
                fscanf(classifier_file, "%d", &(rect->size.height));
                
                fscanf(classifier_file, "%d", &(f->weight1));

                rect = &(f->rect2);

                fscanf(classifier_file, "%d", &(rect->x)); 
                fscanf(classifier_file, "%d", &(rect->y)); 
                fscanf(classifier_file, "%d", &(rect->size.width)); 
                fscanf(classifier_file, "%d", &(rect->size.height));              

                fscanf(classifier_file, "%d", &(f->weight2));

                rect = &(f->rect3);

                fscanf(classifier_file, "%d", &(rect->x)); 
                fscanf(classifier_file, "%d", &(rect->y)); 
                fscanf(classifier_file, "%d", &(rect->size.width)); 
                fscanf(classifier_file, "%d", &(rect->size.height));               

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

        Classifier* class = malloc(sizeof(Classifier));
        class->filters = filters;
        class->filters_per_stages = filter_per_stages;
        class->n_stages = stages;
        class->stage_thresholds = stage_thresholds;

        classifier = class;
        return true;
    }
    else
        printf("\nnull parameters detected. aborting...");
    return false;
}

void integral_image_24bit(pel** image, Size size, double*** int_im, double*** squared_int_im)
{
    double** iim = (double**) malloc(size.height * sizeof(double*));
    double** squared_iim = (double**) malloc(size.height * sizeof(double*));
    unsigned int i;
    for (i = 0; i < size.height; i++)
    {
        iim[i] = (double *) malloc(size.width * sizeof(double));
        squared_iim[i] = (double *) malloc(size.width * sizeof(double));
    }

    iim[0][0] = image[0][0];
    squared_iim[0][0] = image[0][0] * image[0][0]; 

    unsigned int j;
    for (i = 1; i < size.height; i++)
    {
        iim[i][0] = iim[i-1][0] + image[i][0];
        squared_iim[i][0] = squared_iim[i-1][0] + image[i][0] * image[i][0];
    }

    for (i = 1; i < size.width; i++)
    {
        iim[0][i] = iim[0][i-1] + image[0][i * 3];
        squared_iim[0][i] = squared_iim[0][i-1] + image[0][i * 3] * image[0][i * 3];
    }

    for (i = 1; i < size.height; i++)
    {
        for (j = 1; j < size.width; j++)
        {
            iim[i][j] = iim[i-1][j] + iim[i][j-1] - iim[i-1][j-1] + image[i][j * 3];
            squared_iim[i][j] = squared_iim[i-1][j] + squared_iim[i][j-1] - squared_iim[i-1][j-1] + image[i][j * 3] * image[i][j * 3];
        }
    }

    *int_im = iim;
    *squared_int_im = squared_iim;
}

void integral_image(pel** image, Size size, double*** int_im, double*** squared_int_im)
{
    double** iim = (double**) malloc(size.height * sizeof(double*));
    double** squared_iim = (double**) malloc(size.height * sizeof(double*));
    unsigned int i;
    for (i = 0; i < size.height; i++)
    {
        iim[i] = (double *) malloc(size.width * sizeof(double));
        squared_iim[i] = (double *) malloc(size.width * sizeof(double));
    }

    iim[0][0] = image[0][0];
    squared_iim[0][0] = image[0][0] * image[0][0]; 

    unsigned int j;
    for (i = 1; i < size.height; i++)
    {
        iim[i][0] = iim[i-1][0] + image[i][0];
        squared_iim[i][0] = squared_iim[i-1][0] + image[i][0] * image[i][0];
    }

    for (i = 1; i < size.width; i++)
    {
        iim[0][i] = iim[0][i-1] + image[0][i];
        squared_iim[0][i] = squared_iim[0][i-1] + image[0][i] * image[0][i];
    }

    for (i = 1; i < size.height; i++)
    {
        for (j = 1; j < size.width; j++)
        {
            iim[i][j] = iim[i-1][j] + iim[i][j-1] - iim[i-1][j-1] + image[i][j];
            squared_iim[i][j] = squared_iim[i-1][j] + squared_iim[i][j-1] - squared_iim[i-1][j-1] + image[i][j] * image[i][j];
        }
    }

    *(int_im) = iim;
    *(squared_int_im) = squared_iim;
}

pel** resize_image(pel** image, Size out_size, float resize_factor)
{
    // float resize_factor = (float) im.height / out_size.height;
    printf("\nresizing factor: %f | height = %d, width = %d", resize_factor, out_size.height, out_size.width);

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

bool runClassifier(double** iim, int y, int x, unsigned int variance)
{
    int stage, i;
    unsigned int temp;
    long int threshold, filter_sum, stage_sum;

    for (stage = 0; stage < classifier->n_stages; stage++)
    {
        Filter** filters = classifier->filters[stage];
        stage_sum = 0;

        for (i = 0; i < classifier->filters_per_stages[stage]; i++)
        {
            Filter* f = filters[i];
            filter_sum = 0;
            
            threshold = (long int) f->threshold * variance;

            Rectangle* r = &(f->rect1);

            double a,b,c,d;

            temp = 0;
            a = iim[y + r->y + r->size.height][x + r->x + r->size.width];
            b = - iim[y + r->y][x + r->x + r->size.width];
            c = - iim[y + r->y + r->size.height][x + r->x];
            d = iim[y + r->y][x + r->x];

            temp = a + b + c + d;

            filter_sum += (long int) temp * f->weight1;

            r = &(f->rect2);
            
            temp = 0;
            a = iim[y + r->y + r->size.height][x + r->x + r->size.width];
            b = - iim[y + r->y][x + r->x + r->size.width];
            c = - iim[y + r->y + r->size.height][x + r->x];
            d = iim[y + r->y][x + r->x];

            temp = a + b + c + d;

            filter_sum += (long int) temp * f->weight2;

            r = &(f->rect3);

            if ( (r->x + r->y + r->size.height + r->size.width) != 0)
            {
                temp = 0;
                a = iim[y + r->y + r->size.height][x + r->x + r->size.width];
                b = - iim[y + r->y][x + r->x + r->size.width];
                c = - iim[y + r->y + r->size.height][x + r->x];
                d = iim[y + r->y][x + r->x];

                temp = a + b + c + d;

                filter_sum += (long int) temp * f->weight3;
            }

            stage_sum += (long int)(filter_sum < threshold ? f->alpha1 : f->alpha2);
        }
        
        if (stage_sum < 0.4f * classifier->stage_thresholds[stage])
            return false;
    }
    // printf("\n face detected in window on (%d,%d)", x, y);
    return true;
}

void evaluate(double** iim, double** sq_iim, Size size, int currWinSize, float factor, List* faces)
{
    int i, j, stage, k;
    unsigned int variance, mean;

    for (i = 0; i < size.height - window_size; i++)
    {
        for (j = 0; j < size.width - window_size; j++)
        {
            /// VARIANCE COMPUTATION

            mean = iim[i + window_size - 1][j + window_size - 1];
            variance = sq_iim[i + window_size - 1][j + window_size - 1];

            if (i > 0)
            {
                mean -= iim[i-1][j + window_size - 1];
                variance -= sq_iim[i-1][j + window_size - 1];
            }
            if (j > 0)
            {
                mean -= iim[i + window_size - 1][j-1];
                variance -= sq_iim[i + window_size - 1][j-1];
            }
            if (i > 0 && j > 0)
            {
                mean += iim[i-1][j-1];
                variance += sq_iim[i-1][j-1];
            }

            variance = (variance * (window_size * window_size)) - mean * mean;
            variance = variance > 0 ? sqrt(variance) : 1;

            // FILTERS EVALUATION

            if (runClassifier(iim, i, j, variance))
            {
                Rectangle* face = malloc(sizeof(Rectangle));
                face->x = floor(j * factor);
                face->y = floor(i * factor);
                face->size.height = currWinSize;
                face->size.width = currWinSize;

                add(faces, face);
            }
        }
    }
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

List* detect_multiple_faces(pel** image, float scaleFactor, int minWindow, int maxWindow)
{
    if (image == NULL)
        return NULL;
        
    List* faces = listInit();
    printf("\n list init completed");

    if ( minWindow <= window_size )
    {
        Size image_size = { im.width, im.height };
        double** iim, **squared_iim;
        if (im.bitColor > 8)
        {
            integral_image_24bit(image, image_size, &iim, &squared_iim);
            printf("\nintegral image 24bit created");
        }
        else
            integral_image(image, image_size, &iim, &squared_iim);
        
        evaluate(iim, squared_iim, image_size, window_size, 1, faces);
        free_integral_image(iim, image_size);
        free_integral_image(squared_iim, image_size);
        printf("\niims deleted and memory free");
    }

    if ( maxWindow < window_size )
        maxWindow = min(im.width, im.height);
    
    printf("\nmaxWindow = %d", maxWindow);

    float currFactor;
    int iteration = 1;
    printf("\ncreating image pyramid...");
    for (currFactor = scaleFactor; ; currFactor*= scaleFactor, iteration++)
    {
        Size temp_size = { round(im.width / currFactor), round(im.height / currFactor) };

        int curr_winSize = round(window_size * currFactor);

        if (minWindow > curr_winSize)
            continue;

        if (maxWindow < curr_winSize)
            break;

        // char file_name[19];
        // snprintf(file_name, 19, "resized/img_%d.bmp", iteration);

        pel** res_im = resize_image(image, temp_size, currFactor);
        
        // write_new_BMP(file_name, res_im, temp_size.height, temp_size.width, 8);

        double** iim, **squared_iim;

        integral_image(res_im, temp_size, &iim, &squared_iim);

        evaluate(iim, squared_iim, temp_size, curr_winSize, currFactor, faces);

        free_image(res_im, temp_size);
        free_integral_image(iim, temp_size);
        free_integral_image(squared_iim, temp_size);

        // TODO GROUP RECTANGLES
    }
    return faces;
}

List* listInit()
{
    List* list = malloc(sizeof(List));
    list->head = NULL;
    list->size = 0;

    return list;
}

void add(List* list, Rectangle* r)
{
    if (list)
    {
        Node* node = (Node*) malloc(sizeof(Node));
        node->elem = r;
        node->next = NULL;

    if (list->size == 0)
        {
            list->head = node;
            list->size++;
            return;
        }

        node->next = list->head;
        list->head = node;
        list->size++;
        return;
    }
    printf("\n Error: received null list parameter.");
}

Rectangle* remove_from_head(List* list)
{
    if (list)
    {
        if (list->size > 0)
        {
            Node* old = list->head;
            Rectangle* r = old->elem;
            // free(old);
            list->head = list->head->next;
            list->size--;
            return r;
        }
        printf("\n Error: trying to remove element with empty list.");
    }
    printf("\n Error: received null list parameter.");
    return NULL;
}