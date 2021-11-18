#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stddef.h>

#include <math_functions.h>

#define min(a,b) (a < b? a : b)
#define max(a,b) (a > b? a : b)
#define atomicAdd(a,b) atomicAdd((int*) a, (int) b)

#include "classifier.h"
#include "utils/common.h"

#define WINDOW_SIZE 24

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
    Filter* filters;
    short* filters_per_stages;
    int* stage_thresholds;
    short n_stages;
} Classifier;

Classifier* classifier;

__global__ void print_classifier(Classifier* classifier)
{
    printf("\n- printing classifier to check correct loading... -\n\n");

    short* filter_per_stages = classifier->filters_per_stages;
    short stages = classifier->n_stages;
    Filter* filters = classifier->filters;
    int* stage_thresholds = classifier->stage_thresholds;

    printf("\n\t\tstages = %hu", stages);

    printf("\n\n\t- printing filter per stages... -");
    int i;
    for (i = 0; i < stages; i++)
        printf("\n\t\tfilter_per_stages[%d] = %hu", i, filter_per_stages[i]);

    printf("\n\n\t- printing stage thresholds... -");
    for (i = 0; i < stages; i++)
        printf("\n\t\tstage_thresholds[%d] = %d", i, stage_thresholds[i]);

    Filter* f = filters;
    int j;
    printf("\n\n\t- printing filters... -");
    for (i = 0; i < stages; i++)
    {
        for (j = 0; j < filter_per_stages[i]; j++)
        {
            printf("\n\n\t\t- printing filter n. %d from stage %d... -", j+1, i+1);

            Rectangle* r = &(f->rect1);

            printf("\n\t\t\trectangle 1: { x = %hu, y = %hu, size.width = %u, size.height = %u }", r->x, r->y, r->size.width, r->size.height);
            printf("\n\t\t\tweight1 = %d", f->weight1);

            r = &(f->rect2);
            printf("\n\t\t\trectangle 2: { x = %hu, y = %hu, size.width = %u, size.height = %u }", r->x, r->y, r->size.width, r->size.height);
            printf("\n\t\t\tweight2 = %d", f->weight2);

            r = &(f->rect3);
            printf("\n\t\t\trectangle 3: { x = %hu, y = %hu, size.width = %u, size.height = %u }", r->x, r->y, r->size.width, r->size.height);
            printf("\n\t\t\tweight3 = %d", f->weight3);

            printf("\n\t\t\tthreshold = %d", f->threshold);
            printf("\n\t\t\talpha1 = %d", f->alpha1);
            printf("\n\t\t\talpha2 = %d", f->alpha2);

            f++;
        }
    }
}

bool load_classifier_to_gpu(const char* classifier_path, const char* config_path)
{
    if (classifier_path && config_path)
    {
        printf("\nnot null parameters: %s, %s", classifier_path, config_path);

        Filter* dev_filters;
        short* dev_filter_per_stages;
        int* dev_stage_thresholds;

        FILE* config_file = fopen(config_path, "r");

        if (!config_file)
        {
            printf("\nerror while opening the config file. aborting...");
            exit(1);
        }

        int stages = 0;
        fscanf(config_file, "%d", &stages);
        if (!stages)
        {
            printf("\nerror while reading from config file. aborting...");
            exit(1);
        }
        
        short* filter_per_stages = (short*) malloc(sizeof(short) * stages);
        

        int i = 0;
        int n_filters = 0;
        int tot_filters = 0;
        int t;
        while (i < stages && !feof(config_file))
        {
            t = 0;
            fscanf(config_file, "%d", &t);
            filter_per_stages[i] = (short) t;
            i++;
            tot_filters += t;
        }

        fclose(config_file);

        FILE* classifier_file = fopen(classifier_path, "r");
        if (!classifier_file)
        {
            printf("\nerror while reading from classifier file. aborting...");
            exit(1);
        }

        Filter* filters = (Filter*) malloc(sizeof(Filter) * tot_filters);

        int* stage_thresholds = (int*) malloc(sizeof(int) * stages);
        

        int j;
        int* st_thr = stage_thresholds;
        Filter* f = filters;
        for (i = 0; i < stages; i++)
        {
            n_filters = filter_per_stages[i];

            for (j = 0; j < n_filters && !feof(classifier_file); j++)
            {
                Rectangle* rect = &(f->rect1);

                fscanf(classifier_file, "%hu", &(rect->x)); 
                fscanf(classifier_file, "%hu", &(rect->y)); 
                fscanf(classifier_file, "%d", &(rect->size.width)); 
                fscanf(classifier_file, "%d", &(rect->size.height));
                
                fscanf(classifier_file, "%d", &(f->weight1));

                rect = &(f->rect2);

                fscanf(classifier_file, "%hu", &(rect->x)); 
                fscanf(classifier_file, "%hu", &(rect->y)); 
                fscanf(classifier_file, "%d", &(rect->size.width)); 
                fscanf(classifier_file, "%d", &(rect->size.height));              

                fscanf(classifier_file, "%d", &(f->weight2));

                rect = &(f->rect3);

                fscanf(classifier_file, "%hu", &(rect->x)); 
                fscanf(classifier_file, "%hu", &(rect->y)); 
                fscanf(classifier_file, "%d", &(rect->size.width)); 
                fscanf(classifier_file, "%d", &(rect->size.height));               

                fscanf(classifier_file, "%d", &(f->weight3));    

                fscanf(classifier_file, "%d", &(f->threshold));                
                fscanf(classifier_file, "%d", &(f->alpha1));                
                fscanf(classifier_file, "%d", &(f->alpha2));   
                
                f++;
            }
            if (!feof(classifier_file))
                fscanf(classifier_file, "%d", st_thr++);
        }
        fclose(classifier_file);

        Classifier* clas = (Classifier*) malloc(sizeof(Classifier));

        CHECK(cudaMalloc((void**) &dev_filters, sizeof(Filter) * tot_filters));
        CHECK(cudaMalloc((void**) &dev_stage_thresholds, sizeof(int) * stages));
        CHECK(cudaMalloc((void**) &dev_filter_per_stages, sizeof(short) * stages));

        CHECK(cudaMemcpy(dev_filters, filters, sizeof(Filter) * tot_filters, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dev_filter_per_stages, filter_per_stages, sizeof(short) * stages, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dev_stage_thresholds, stage_thresholds, sizeof(int) * stages, cudaMemcpyHostToDevice));

        clas->filters = dev_filters;
        clas->filters_per_stages = dev_filter_per_stages;
        clas->n_stages = stages;
        clas->stage_thresholds = dev_stage_thresholds;

        CHECK(cudaMalloc((void**) &classifier, sizeof(Classifier)));
        CHECK(cudaMemcpy(classifier, clas, sizeof(Classifier), cudaMemcpyHostToDevice));

        // print_classifier<<<1,1>>>(classifier);
        // cudaDeviceSynchronize();

        return true;
    }
    else
        printf("\nnull parameters detected. aborting...");
    return false;
}

pel** resize_image(pel** image, Size out_size)
{
    float resize_factor = (float) im.height / out_size.height;
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

// bool runClassifier(double** iim, int y, int x, unsigned int variance)
// {
//     int stage, i;
//     unsigned int temp;
//     long int threshold, filter_sum, stage_sum;

//     for (stage = 0; stage < classifier->n_stages; stage++)
//     {
//         Filter** filters = classifier->filters[stage];
//         stage_sum = 0;

//         for (i = 0; i < classifier->filters_per_stages[stage]; i++)
//         {
//             Filter* f = filters[i];
//             filter_sum = 0;
            
//             threshold = (long int) f->threshold * variance;

//             Rectangle* r = &(f->rect1);

//             temp = 0;
//             temp += iim[y + r->y + r->size.height][x + r->x + r->size.width];
//             temp += - iim[y + r->y][x + r->x + r->size.width];
//             temp += - iim[y + r->y + r->size.height][x + r->x];
//             temp += iim[y + r->y][x + r->x];

//             filter_sum += (long int) temp * f->weight1;

//             r = &(f->rect2);
            
//             temp = 0;
//             temp += iim[y + r->y + r->size.height][x + r->x + r->size.width];
//             temp += - iim[y + r->y][x + r->x + r->size.width];
//             temp += - iim[y + r->y + r->size.height][x + r->x];
//             temp += iim[y + r->y][x + r->x];

//             filter_sum += (long int) temp * f->weight2;

//             r = &(f->rect3);

//             if ( (r->x + r->y + r->size.height + r->size.width) != 0)
//             {
//                 temp = 0;
//                 temp += iim[y + r->y + r->size.height][x + r->x + r->size.width];
//                 temp += - iim[y + r->y][x + r->x + r->size.width];
//                 temp += - iim[y + r->y + r->size.height][x + r->x];
//                 temp += iim[y + r->y][x + r->x];

//                 filter_sum += (long int) temp * f->weight3;
//             }

//             stage_sum += (long int)(filter_sum < threshold ? f->alpha1 : f->alpha2);
//         }
        
//         if (stage_sum < 0.4f * classifier->stage_thresholds[stage])
//             return false;
//     }
//     // printf("\n face detected in window on (%d,%d)", x, y);
//     return true;
// }

__global__ void cuda_evaluate(Classifier* classifier, long unsigned int* iim, long unsigned int* sq_iim, unsigned int width, unsigned int height, int currWinSize, float factor, Rectangle** faces)
{
    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= (height - WINDOW_SIZE) * (width - WINDOW_SIZE))
        return;

    unsigned int variance, mean, y, x;

    y = floor((float) id / width);
    x = id % width;

    mean = iim[(y + WINDOW_SIZE - 1) * width + x + WINDOW_SIZE - 1];
    variance = sq_iim[(y + WINDOW_SIZE - 1) * width + x + WINDOW_SIZE - 1];

    if (y > 0)
    {
        mean -= iim[(y - 1) * width + x];
        variance -= sq_iim[(y - 1) * width + x];
    }
    if (y > 0)
    {
        mean -= iim[y * width + x - 1];
        variance -= sq_iim[y * width + x - 1];
    }
    if (y > 0 && x > 0)
    {
        mean += iim[(y - 1) * width + x - 1];
        variance += sq_iim[(y - 1) * width + x - 1];
    }

    variance = (variance * (WINDOW_SIZE * WINDOW_SIZE)) - mean * mean;
    variance = variance > 0 ? sqrt((double) variance) : 1;

    // RUNNING CLASSIFIER

    int stage, i;
    unsigned int temp;
    long int threshold, filter_sum, stage_sum;
    Filter* f = classifier->filters;

    for (stage = 0; stage < classifier->n_stages; stage++)
    {
        
        stage_sum = 0;

        for (i = 0; i < classifier->filters_per_stages[stage]; i++)
        {
            filter_sum = 0;

            threshold = (long int) f->threshold * variance;

            Rectangle* r = &(f->rect1);

            temp = 0;
            temp += iim[(y + r->y + r->size.height) * width + (x + r->x + r->size.height)];
            temp += - iim[(y + r->y) * width + (x + r->x + r->size.width)];
            temp += - iim[(y + r->y + r->size.height) * width + (x + r->x)];
            temp += iim[(y + r->y) * width + (x + r->x)];

            filter_sum += (long int) temp * f->weight1;

            r = &(f->rect2);

            temp = 0;
            temp += iim[(y + r->y + r->size.height) * width + (x + r->x + r->size.height)];
            temp += - iim[(y + r->y) * width + (x + r->x + r->size.width)];
            temp += - iim[(y + r->y + r->size.height) * width + (x + r->x)];
            temp += iim[(y + r->y) * width + (x + r->x)];

            filter_sum += (long int) temp * f->weight2;

            r = &(f->rect3);

            if ( (r->x + r->y + r->size.height + r->size.width) != 0)
            {
                temp = 0;
                temp += iim[(y + r->y + r->size.height) * width + (x + r->x + r->size.height)];
                temp += - iim[(y + r->y) * width + (x + r->x + r->size.width)];
                temp += - iim[(y + r->y + r->size.height) * width + (x + r->x)];
                temp += iim[(y + r->y) * width + (x + r->x)];

                filter_sum += (long int) temp * f->weight3;
            }

            stage_sum += (long int) (filter_sum < threshold ? f->alpha1 : f->alpha2);

            f++;
        }

        if (stage_sum < 0.4f * classifier->stage_thresholds[stage])
            return;
    }

    Rectangle* face = (Rectangle*) malloc(sizeof(Rectangle));
    face->x = x * factor;
    face->y = y * factor;
    face->size.height = currWinSize;
    face->size.width = currWinSize;

    faces[id] = face;
    printf("\ndetected face at (%u, %u).", x,y);
}

// void evaluate(long unsigned int* iim, long unsigned int* sq_iim, Size size, int currWinSize, float factor, Rectangle** faces)
// {
//     int i, j;
//     unsigned int variance, mean;

//     for (i = 0; i < size.height - WINDOW_SIZE; i++)
//     {
//         for (j = 0; j < size.width - WINDOW_SIZE; j++)
//         {
//             /// VARIANCE COMPUTATION

//             mean = iim[i + WINDOW_SIZE - 1][j + WINDOW_SIZE - 1];
//             variance = sq_iim[i + WINDOW_SIZE - 1][j + WINDOW_SIZE - 1];

//             if (i > 0)
//             {
//                 mean -= iim[i-1][j];
//                 variance -= sq_iim[i-1][j];
//             }
//             if (j > 0)
//             {
//                 mean -= iim[i][j-1];
//                 variance -= sq_iim[i][j-1];
//             }
//             if (i > 0 && j > 0)
//             {
//                 mean += iim[i-1][j-1];
//                 variance += sq_iim[i-1][j-1];
//             }

//             variance = (variance * (WINDOW_SIZE * WINDOW_SIZE)) - mean * mean;
//             variance = variance > 0 ? sqrt(variance) : 1;

//             // FILTERS EVALUATION

//             // if (runClassifier(iim, i, j, variance))
//             // {
//             //     Rectangle* face = (Rectangle *) malloc(sizeof(Rectangle));
//             //     face->x = j * factor;
//             //     face->y = i * factor;
//             //     face->size.height = currWinSize;
//             //     face->size.width = currWinSize;

//             //     add(faces, face);
//             // }
//         }
//     }
// }

__global__ void cuda_integral_image_24bit(pel* image, unsigned int width, unsigned int height, long unsigned int* iim, long unsigned int* squared_iim)
{
    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    long unsigned int sum = 0, sq_sum = 0, offset;

    uint t;

    int i;

    if (id < width)
    {
        for (i = 0; i < height; i++)
        {
            offset = i * width + id;
            t = image[offset * 3];
            sum += t;
            sq_sum += t * t;

            atomicAdd(iim + offset, sum);
            atomicAdd(squared_iim + offset, sq_sum);
        }
    }
    else if (id < width + height)
    {
        for (i = 0; i < width; i++)
        {
            offset = (id - width) * width + i;
            t = image[offset * 3];
            sum += t;
            sq_sum += t * t;

            atomicAdd(iim + offset, sum);
            atomicAdd(squared_iim + offset, sq_sum);
        }
    }

}

__global__ void cuda_integral_image(pel* image, unsigned int width, unsigned int height, long unsigned int* iim, long unsigned int* squared_iim)
{
    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    long unsigned int sum = 0, sq_sum = 0, offset;

    uint t;

    int i;

    if (id < width)
    {
        for (i = 0; i < height; i++)
        {
            offset = i * width + id;
            t = image[offset];
            sum += t;
            sq_sum += t * t;

            atomicAdd(iim + offset, sum);
            atomicAdd(squared_iim + offset, sq_sum);
        }
    }
    else if (id < width + height)
    {
        for (i = 0; i < width; i++)
        {
            offset = (id - width) * width + i;
            t = image[offset];
            sum += t;
            sq_sum += t * t;

            atomicAdd(iim + offset, sum);
            atomicAdd(squared_iim + offset, sq_sum);
        }
    }

}

Rectangle** detect_multiple_faces(pel* image, float scaleFactor, int minWindow, int maxWindow)
{
    if (image == NULL)
        return NULL;

    Rectangle** faces;
    CHECK(cudaMalloc((void **) &faces, sizeof(Rectangle*) * im.width * im.height));

    if ( minWindow <= WINDOW_SIZE )
    {
        Size image_size = { im.width, im.height };

        unsigned long nBytes = sizeof(long unsigned int) * image_size.width * image_size.height;

        long unsigned int* iim, *squared_iim;
        CHECK(cudaMalloc((void **) &iim, nBytes));
        CHECK(cudaMalloc((void **) &squared_iim, nBytes));

        CHECK(cudaMemset(iim, 0, nBytes));

        uint dimBlock = 256, dimGrid;
        dimGrid = ceil((float) (image_size.width + image_size.height) / dimBlock);

        if (im.bitColor > 8)
        {
            printf("\ndimblock = %u, dimgrid = %u | width = %u, height = %u", dimBlock, dimGrid, image_size.width, image_size.height);
            cuda_integral_image_24bit <<< dimGrid, dimBlock >>> (image, image_size.width, image_size.height, iim, squared_iim);
            // cudaDeviceSynchronize();
            printf("\nintegral image 24bit created");         
        }
        else
            cuda_integral_image <<< dimGrid, dimBlock >>> (image, image_size.width, image_size.height, iim, squared_iim);
        
        int rowBlock = (image_size.width + dimBlock - 1) / dimBlock;
        dimGrid = image_size.height * rowBlock;

        cuda_evaluate <<< dimGrid, dimBlock >>>(classifier, iim, squared_iim, image_size.width, image_size.height, WINDOW_SIZE, 1, faces);

        cudaDeviceSynchronize();
        // print_classifier<<<1,1>>>(classifier);
        // cudaDeviceSynchronize();

        CHECK(cudaFree(iim));
        CHECK(cudaFree(squared_iim));
        printf("\niims deleted and memory free");

        // exit(0);
    }

    // if ( maxWindow < WINDOW_SIZE )
    //     maxWindow = min(im.width, im.height);
    
    // printf("\nmaxWindow = %d", maxWindow);

    // float currFactor;
    // int iteration = 1;
    // printf("\ncreating image pyramid...");
    // for (currFactor = scaleFactor; ; currFactor*= scaleFactor, iteration++)
    // {
    //     Size temp_size = { round(im.width / currFactor), round(im.height / currFactor) };

    //     int curr_winSize = round(WINDOW_SIZE * currFactor);

    //     if (minWindow > curr_winSize)
    //         continue;

    //     if (maxWindow < curr_winSize)
    //         break;

    //     char file_name[19];
    //     snprintf(file_name, 19, "resized/img_%d.bmp", iteration);

    //     pel** res_im = resize_image(image, temp_size);
        
    //     // write_new_BMP(file_name, res_im, temp_size.height, temp_size.width, 8);

    //     double** iim, **squared_iim;

    //     integral_image(res_im, temp_size, &iim, &squared_iim);

    //     evaluate(iim, squared_iim, temp_size, curr_winSize, currFactor, faces);

    //     free_image(res_im, temp_size);
    //     free_integral_image(iim, temp_size);
    //     free_integral_image(squared_iim, temp_size);

    //     // TODO GROUP RECTANGLES
    // }
    return faces;
}

List* listInit()
{
    List* list = (List *) malloc(sizeof(List));
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