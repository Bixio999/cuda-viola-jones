#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stddef.h>
#include <assert.h>

#include <math_functions.h>

#define min(a,b) (a < b? a : b)
#define max(a,b) (a > b? a : b)
// #define atomicAdd(a,b) atomicAdd((int*) a, (int) b)

#include "classifier.h"
#include "utils/common.h"

#define N_STREAM 32

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

__global__ void cuda_evaluate(Classifier* classifier, double* iim, double* sq_iim, unsigned int width, unsigned int height, int currWinSize, float factor, Rectangle** faces, unsigned int* face_counter, unsigned int original_width)
{
    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    // if (id >= (height - WINDOW_SIZE) * (width - WINDOW_SIZE))
    //     return;

    int y, x;

    unsigned int variance, mean;

    y = floor((double) id / (width - WINDOW_SIZE));
    x = id % (width - WINDOW_SIZE); 

    if (y >= height - WINDOW_SIZE || x >= width - WINDOW_SIZE)
        return;

    mean = iim[(y + WINDOW_SIZE - 1) * width + x + WINDOW_SIZE - 1];
    variance = sq_iim[(y + WINDOW_SIZE - 1) * width + x + WINDOW_SIZE - 1];

    // if (id == 57587)
    //     printf("\n (%d,%d), initial mean = %u, initial variance = %u, offset of bottom-right corner = %d", y, x, mean, variance,(y + WINDOW_SIZE - 1) * width + x + WINDOW_SIZE - 1);

    if (y > 0)
    {
        mean -= (unsigned int) iim[(y - 1) * width + x + WINDOW_SIZE - 1];
        variance -= (unsigned int) sq_iim[(y - 1) * width + x + WINDOW_SIZE - 1];
        // if (id == 57587)
        //     printf("\n (%d,%d), offset of top-right corner = %d, iim value = %.1f", y, x, (y - 1) * width + x + WINDOW_SIZE - 1, iim[(y - 1) * width + x + WINDOW_SIZE - 1]);
    }
    if (x > 0)
    {
        mean -= (unsigned int) iim[(y + WINDOW_SIZE - 1) * width + x - 1];
        variance -= (unsigned int) sq_iim[(y + WINDOW_SIZE - 1) * width + x - 1];
        // if (id == 57587)
        //     printf("\n (%d,%d), offset of bottom-left corner = %d, iim value = %.1f", y, x, (y + WINDOW_SIZE - 1) * width + x - 1,iim[(y + WINDOW_SIZE - 1) * width + x - 1]);
    }
    if (y > 0 && x > 0)
    {
        mean += (unsigned int) iim[(y - 1) * width + x - 1];
        variance += (unsigned int) sq_iim[(y - 1) * width + x - 1];
        // if (id == 57587)
        //     printf("\n (%d,%d), offset of top-left corner = %d, iim value = %.1f", y, x, (y - 1) * width + x - 1, iim[(y - 1) * width + x - 1]);
    }

    // if (id == 57587)
    //     printf("\n (%d,%d), final mean = %u, final variance = %u", y, x, mean, variance);

    long tmp = (variance * (WINDOW_SIZE * WINDOW_SIZE)) - mean * mean;
    variance = tmp > 0 ? (unsigned int) sqrt((double) tmp) : 1;

    // if (id == 0)
    //     printf("\n\tvariance = %f", variance);

    // RUNNING CLASSIFIER

    int stage, i;
    unsigned int temp;
    long int threshold, filter_sum, stage_sum;
    Filter* f = classifier->filters;
    Rectangle* r;

    for (stage = 0; stage < classifier->n_stages; stage++)
    {
        
        stage_sum = 0;

        for (i = 0; i < classifier->filters_per_stages[stage]; i++)
        {



            // if (id == 57587 && (i ==0 || stage == 3))
            // {
            //     printf("\n\n\t\t- printing filter n. %d from stage %d... -", i+1, stage + 1);

            //     r = &(f->rect1);

            //     printf("\n\t\t\trectangle 1: { x = %hu, y = %hu, size.width = %u, size.height = %u }", r->x, r->y, r->size.width, r->size.height);
            //     printf("\n\t\t\tweight1 = %d", f->weight1);

            //     r = &(f->rect2);
            //     printf("\n\t\t\trectangle 2: { x = %hu, y = %hu, size.width = %u, size.height = %u }", r->x, r->y, r->size.width, r->size.height);
            //     printf("\n\t\t\tweight2 = %d", f->weight2);

            //     r = &(f->rect3);
            //     printf("\n\t\t\trectangle 3: { x = %hu, y = %hu, size.width = %u, size.height = %u }", r->x, r->y, r->size.width, r->size.height);
            //     printf("\n\t\t\tweight3 = %d", f->weight3);

            //     printf("\n\t\t\tthreshold = %d", f->threshold);
            //     printf("\n\t\t\talpha1 = %d", f->alpha1);
            //     printf("\n\t\t\talpha2 = %d", f->alpha2);
            // }




            filter_sum = 0;

            threshold = (long int) f->threshold * variance;

            r = &(f->rect1);

            double a,b,c,d;

            temp = 0;
            a = iim[(y + r->y + r->size.height) * width + (x + r->x + r->size.width)];
            b = - iim[(y + r->y) * width + (x + r->x + r->size.width)];
            c = - iim[(y + r->y + r->size.height) * width + (x + r->x)];
            d = iim[(y + r->y) * width + (x + r->x)];

            temp = a + b + c + d;

            // if (id == 57587 && (i ==0 || stage == 3))
            //     printf("\n\trect1: a = %.1f, b = %.1f, c = %.1f, d = %.1f, temp = %u", a,b,c,d,temp);

            filter_sum += (long int) temp * f->weight1;

            r = &(f->rect2);

            temp = 0;
            a = iim[(y + r->y + r->size.height) * width + (x + r->x + r->size.width)];
            b = - iim[(y + r->y) * width + (x + r->x + r->size.width)];
            c = - iim[(y + r->y + r->size.height) * width + (x + r->x)];
            d = iim[(y + r->y) * width + (x + r->x)];

            temp = a + b + c + d;

            // if (id == 57587 && (i ==0 || stage == 3))
            //     printf("\n\trect2: a = %.1f, b = %.1f, c = %.1f, d = %.1f, temp = %u", a,b,c,d,temp);

            filter_sum += (long int) temp * f->weight2;

            r = &(f->rect3);

            if ( (r->x + r->y + r->size.height + r->size.width) != 0)
            {
                temp = 0;
                a = iim[(y + r->y + r->size.height) * width + (x + r->x + r->size.width)];
                b = - iim[(y + r->y) * width + (x + r->x + r->size.width)];
                c = - iim[(y + r->y + r->size.height) * width + (x + r->x)];
                d = iim[(y + r->y) * width + (x + r->x)];

                temp = a + b + c + d;

                filter_sum += (long int) temp * f->weight3;
            }

            stage_sum += (long int)(filter_sum < threshold ? f->alpha1 : f->alpha2);

            // if (id == 57587 && (i ==0 || stage == 3))
            //     printf("\n\tfilter_sum = %ld, threshold = %ld, f->threshold = %d, variance = %u, temp(rect2) = %u, temp(rect1) * weight1 = %ld", filter_sum, threshold, f->threshold, variance, temp, rect1_val);

            f++;
        }

        // if (id == 57587)
        //     printf("\n\tstage_sum = %ld, 0.4f * stage_threshold = %f", stage_sum, 0.4f * classifier->stage_thresholds[stage]);

        // if (stage > 10)
        //     printf("\n p(%d,%d): currently at stage %d with stage threshold = %d and stage_sum = %ld", y, x, stage, classifier->stage_thresholds[stage], stage_sum);

        if (stage_sum < 0.4f * classifier->stage_thresholds[stage])
        {
            return;
        }    
    }

    Rectangle* face = (Rectangle*) malloc(sizeof(Rectangle));
    face->x = floor(x * factor);
    face->y = floor(y * factor);
    face->size.height = currWinSize;
    face->size.width = currWinSize;

    unsigned long scaledId = face->y * original_width + face->x;

    if (!faces[scaledId])
        atomicAdd(face_counter, 1);

    faces[scaledId] = face;

    // printf("\ndetected face at (%u, %u), id = %lu.", face->y,face->x, scaledId);
}

__global__ void cuda_integral_image_row(pel* image, unsigned int width, unsigned int height, unsigned int original_width, double* iim, double* squared_iim, float factor, short bitColor)
{
    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned long offset, scaled_offset;

    double sum = 0, sq_sum = 0;

    pel t;

    int i;

    // int h,w;

    if (id < height)
    {
        for (i = 0; i < width; i++)
        {
            int h = floor(id * factor);
            int w = floor(i * factor);

            scaled_offset = h * original_width + w;
            offset = id * width + i;
            t = image[scaled_offset * (bitColor > 8 ? 3 : 1)];
            // resized_img[offset] = t;
            sum += (double) t;
            sq_sum += (double) (t * t);
            // printf("\nsum = %f, sq_sum = %f", sum, sq_sum);

            iim[offset] = sum;
            squared_iim[offset] = sq_sum;
            // printf("\n(%lu,%d), sum = %f, sq_sum = %f", id, i, iim[offset], squared_iim[offset]);
        }
    }
}

__global__ void cuda_integral_image_col(unsigned int width, unsigned int height, double* iim, double* squared_iim)
{
    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    int i;

    if (id < width)
    {
        for (i = 1; i < height; i++)
        {
            iim[i * width + id] += iim[(i - 1) * width + id];
            squared_iim[i * width + id] += squared_iim[(i - 1) * width + id];
        }
    }
}

void compare_integral_image(pel* image, Size size, double* dev_iim, double* dev_squared_iim, float factor, short bitColor)
{
    double** iim = (double**) malloc(size.height * sizeof(double*));
    double** squared_iim = (double**) malloc(size.height * sizeof(double*));
    unsigned int i;
    for (i = 0; i < size.height; i++)
    {
        iim[i] = (double *) malloc(size.width * sizeof(double));
        squared_iim[i] = (double *) malloc(size.width * sizeof(double));
    }

    iim[0][0] = image[0];
    squared_iim[0][0] = (double) image[0] * image[0]; 

    unsigned int j, scaled_offset;
    bool error = false;
    for (i = 0; i < size.height; i++)
    {
        for (j = 0; j < size.width; j++)
        {
            scaled_offset = floor(i * factor) * im.width + floor(j * factor);
            pel t = image[scaled_offset * (bitColor > 8 ? 3 : 1)];

            if (i > 0 && j > 0)
            {
                iim[i][j] = iim[i-1][j] + iim[i][j-1] - iim[i-1][j-1] + t;
                squared_iim[i][j] = squared_iim[i-1][j] + squared_iim[i][j-1] - squared_iim[i-1][j-1] + (double)(t * t);
            }
            else if (i > 0)
            {
                iim[i][0] = iim[i-1][0] + t;
                squared_iim[i][0] = squared_iim[i-1][0] + (double)(t * t);
            }
            else if (j > 0)
            {
                iim[0][j] = iim[0][j-1] + t;
                squared_iim[0][j] = squared_iim[0][j-1] + (double)(t * t);
            }

            if (iim[i][j] != dev_iim[i * size.width + j])
            {
                printf("\n\tDetected value inconsistency at (%u,%u): cpu iim = %f | gpu iim = %f", i, j, iim[i][j], dev_iim[i * size.width + j]);

                FILE* f = fopen("log.txt", "w");
                
                int k,z, offset;
                fprintf(f,"\nprinting submatrix of image: \n\n");
                for (k = 0; k < i + 1; k++)
                {
                    for (z = 0; z < j + 1; z++)
                    {
                        offset = floor(k * factor) * im.width + floor(z * factor);
                        fprintf(f,"  %u", image[offset * (bitColor > 8 ? 3 : 1)]);
                    }
                    fprintf(f, "\n");
                }
                fprintf(f,"\n offset = %d", offset);
                fprintf(f,"\nprinting submatrix of iim: \n\n");
                for (k = 0; k < i + 1; k++)
                {
                    for (z = 0; z < j + 1; z++)
                        fprintf(f,"  %.0f", iim[k][z]);
                    fprintf(f,"\n");
                }

                fprintf(f,"\nprinting submatrix of dev_iim: \n\n");
                for (k = 0; k < i + 1; k++)
                {
                    for (z = 0; z < j + 1; z++)
                        fprintf(f,"  %.0f", dev_iim[k * size.width + z]);
                    fprintf(f,"\n");
                }
                fclose(f);
                error = true;
            }
            if (squared_iim[i][j] != dev_squared_iim[i * size.width + j])
            {
                printf("\n\tDetected value inconsistency at (%u,%u): cpu squared_iim = %f | gpu squared_iim = %f", i, j, squared_iim[i][j], dev_squared_iim[i * size.width + j]);
                error = true;
            }

            if (error)
                exit(0);
        }
    }
    printf("\nthe two integral images are equal.");
}

void cuda_integral_image(pel* image, unsigned int width, unsigned int height, unsigned int original_width, double* iim, double* sq_iim, float factor, short bitColor, cudaStream_t stream)
{
    uint dimBlock, dimGrid;

    compute_grid_dimension(height, &dimBlock, &dimGrid);

    cuda_integral_image_row <<< dimGrid, dimBlock, 0, stream >>>(image, width, height, original_width, iim, sq_iim, factor, bitColor);

    compute_grid_dimension(width, &dimBlock, &dimGrid);

    cuda_integral_image_col <<<dimGrid, dimBlock, 0, stream >>>(width, height, iim, sq_iim);
}

__global__ void cuda_collect_results(List** sub_lists, Rectangle** faces, unsigned int height, unsigned int width, unsigned int* face_counter, float merge_threshold)
{
    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= height)
        return;

    List* list = NULL;
    
    int i;
    Rectangle* r, *s;
    Node* node;

    uint diff_x, diff_y;
    float distance;

    int a,b,c,d;

    uint normalize_factor = max(width, height);

    // if (threadIdx.x == 62 && blockIdx.x == 1)
    //     printf("\n hello my id is %lu, my sublist pointer currently is: %p", id, sub_lists[id]);

    for (i = 0; i < width; i++)
    {
        r = faces[id * width + i];
        if (r != NULL)
        {
            // printf("\n found face at (%lu,%d).", id, i);
            // atomicAdd(results_counter, 1);

            if (!list)
            {
                list = (List*) malloc(sizeof(List));
                assert(list != NULL);
                list->size = 0;
                list->head = NULL;
                list->tail = NULL;

                node = (Node*) malloc(sizeof(Node));
                assert(node != NULL);
                node->elem = r;
                node->next = NULL;

                list->head = node;
                list->tail = node;
                list->size++;
            }
            else if (list->size == 0)
            {
                // if (threadIdx.x == 62 && blockIdx.x == 1)
                //     printf("\ncollect_results at row %lu: found first face",id);
                node = (Node*) malloc(sizeof(Node));
                assert(node != NULL);
                node->elem = r;
                node->next = NULL;

                list->head = node;
                list->tail = node;
                list->size++;
            }
            else
            {
                

                node = list->head;
                while(node != NULL)
                {
                    s = node->elem;

                    diff_x = round((abs((r->x + floor( r->size.width / 2.0f)) - (s->x + floor( s->size.width / 2.0f))) / (float) normalize_factor) * 100.0f);
                    diff_y = round((abs((r->y + floor( r->size.height / 2.0f)) - (s->y + floor( s->size.height / 2.0f))) / (float) normalize_factor) * 100.0f);

                    distance = sqrt((double) (diff_x * diff_x + diff_y * diff_y));

                    if (distance <= merge_threshold)
                    {
                        // if (threadIdx.x == 62 && blockIdx.x == 1)
                        //     printf("\ncollect_results at row %lu: found a result equal to another face (distance = %.2f, diff_x = %u, diff_y = %u, r = (%u,%u), s = (%u,%u)), merging...",id, distance, diff_x, diff_y, r->x, r->y, s->x, s->y);
                        a = min(r->x, s->x);
                        b = max(r->x + r->size.width, s->x + s->size.width);
                        c = min(r->y, s->y);
                        d = max(r->y + r->size.height, s->y + s->size.height);

                        s->x = a;
                        s->y = c;
                        s->size.width = b - a;
                        s->size.height = d - c;

                        atomicSub(face_counter, 1);

                        break;
                    }
                    node = node->next;
                }

                if (node == NULL)
                {
                    // if (threadIdx.x == 62 && blockIdx.x == 1)
                    //     printf("\ncollect_results at row %lu: found another face",id);
                    node = (Node*) malloc(sizeof(Node));
                    assert(node != NULL);
                    node->elem = r;
                    node->next = NULL;

                    node->next = list->head;
                    list->head = node;
                    list->size++;
                }   
            }
        }
    }

    if (list && list->size > 0)
    {
        sub_lists[id] = list;
        // printf("\nlist size n. %lu = %d", id, list->size);
    }
}

__global__ void cuda_merge_sublists(List** sub_lists, unsigned int n_lists, unsigned int size, unsigned int offset, Rectangle* results)
{
    uint tid = threadIdx.x;
    unsigned long idx = blockIdx.x * blockDim.x + threadIdx.x;

    // boundary check
    if (idx >= n_lists) 
        return;

    // if (tid < 10)
    //     printf("\n idx = %lu, thisBlock offset = %u", idx, (blockIdx.x * blockDim.x) * offset);

    // convert global data ptr to the local ptr of this block
    int blockOffset = (blockIdx.x * blockDim.x) * offset;
    List **thisBlock = sub_lists + blockOffset;

    // in-place reduction in global memory
    int stride, oldStride, unmerged_range = 0;
    bool odd_lists = n_lists % 2;
    List* a = NULL, *b = NULL;
    for (stride = min(min(n_lists, blockDim.x), size - blockOffset) / 2, oldStride = stride; stride > 0; stride = stride > 1 ? ceil(stride / 2.0f) : 0) 
    { 
        // if (threadIdx.x == 0 && blockIdx.x == 0 && gridDim.x == 1)
        //     printf("\n stride = %d", stride);

        if (oldStride - stride * 2 > 0 && !unmerged_range)
            unmerged_range = oldStride;

        if (tid < stride || (tid <= stride && odd_lists && oldStride == stride))
        {
            a = thisBlock[tid * offset];
            b = thisBlock[(tid + stride) * offset];

            // if (threadIdx.x == 0 && blockIdx.x == 0 && gridDim.x == 1)
            //     printf("\n\t sub_list offset = %u, a = thisBlock[%d] = %p, b = thisBlock[%d] = %p", (blockIdx.x * blockDim.x) * offset, tid * offset, a, (tid + stride) * offset, b);

            if (a && b)
            {
                // if (gridDim.x == 1)
                // // {
                //     printf("\n\t sub_list offset = %u, a = thisBlock[%d] = %p, b = thisBlock[%d] = %p\n\t\t Lists merging\n\t\t list a (%p): { size = %d, head = %p, tail = %p }\n\t\t list b (%p): { size = %d, head = %p, tail = %p }", (blockIdx.x * blockDim.x) * offset, tid * offset, a, (tid + stride) * offset, b, a, a->size, a->head, a->tail, b, b->size, b->head, b->tail);

                // }

                a->tail->next = b->head;
                a->tail = b->tail;
                a->size += b->size;
                thisBlock[(tid + stride) * offset] = NULL;
            }
            else if (b)
            {
                thisBlock[tid * offset] = b;
                thisBlock[(tid + stride) * offset] = NULL;

                // if (gridDim.x == 1)
                // // {
                //     printf("\n\t sub_list offset = %u, a = thisBlock[%d] = %p, b = thisBlock[%d] = %p\n\t\t List b moved to a\n\t\t list: { size = %d, head = %p, tail = %p }", (blockIdx.x * blockDim.x) * offset, tid * offset, a, (tid + stride) * offset, b, b->size, b->head, b->tail);
                // }
            }

            // if (threadIdx.x == 326 && blockIdx.x == 1)
            //     printf("\n\t\t Memory access completed.");

            // a = NULL;
            // b = NULL;
        }

        oldStride = stride;

        // synchronize within threadblock
        __syncthreads();
    }

    if (tid == 0)
    {
        a = thisBlock[0];
        if (a && unmerged_range > 0)
        {
            // printf("\nunmerged lists detected, unmerged_range = %d", unmerged_range);
            int i;
            for (i = 1; i < unmerged_range; i++)
            {
                b = thisBlock[i * offset];
                if (b)
                {
                    a->tail->next = b->head;
                    a->tail = b->tail;
                    a->size += b->size;
                    thisBlock[i * offset] = NULL;
                }
            }
        }
    }

    if (results && tid == 0 && gridDim.x == 1)
    {
        List* list = sub_lists[0];
        int i;
        Node* curr = list->head;
        Rectangle* r, *s;

        // printf("\n list->size after merging = %d", list->size);

        for (i = 0; i < list->size && curr != NULL; i++ )
        {
            r = curr->elem;
            s = &(results[i]);
            // printf("\n\t\t\trectangle %d: { x = %hu, y = %hu, size.width = %u, size.height = %u }", i, r->x, r->y, r->size.width, r->size.height);

            s->size = r->size;
            s->x = r->x;
            s->y = r->y;

            curr = curr->next;
        }
        // if (curr != NULL)
        //     printf("\n other faces weren't computed in merging...");

        // printf("\n searching for unmerged lists:");
        // for (i = 1; i < size; i++ )
        // {
        //     list = sub_lists[i];
        //     if (list)
        //     {
        //         printf("\n\tfound unmerged list at id = %d:", i);
        //         printf("\n\t\t list: { size = %d, head = %p, tail = %p }", list->size, list->head, list->tail);
        //     }
        // }
    }
}

List* gather_results(Rectangle** faces, unsigned int* dev_face_counter, unsigned int* face_counter, float merge_threshold)
{
    List** sub_lists;
    unsigned long size = im.height;
    unsigned long nBytes = size * sizeof(List*);
    CHECK(cudaMalloc((void**) &sub_lists, nBytes));
    CHECK(cudaMemset(sub_lists, 0, nBytes));

    uint dimBlock, dimGrid;
    compute_grid_dimension(size, &dimBlock, &dimGrid);

    // printf("\n face counter before gathering = %u", *face_counter);

    // unsigned int* results_counter, *h_res_counter;
    // CHECK(cudaMalloc((void **) &results_counter, sizeof(unsigned int)));
    // CHECK(cudaMemset(results_counter, 0, sizeof(unsigned int)));

    // tested
    cuda_collect_results <<< dimGrid, dimBlock >>> (sub_lists, faces, im.height, im.width, dev_face_counter, merge_threshold);
    // cuda_collect_results <<< dimGrid, dimBlock >>> (sub_lists, faces, im.height, im.width, dev_face_counter, merge_threshold, results_counter);

    unsigned int n_lists = size;
    unsigned int offset = 1;

    // h_res_counter = (unsigned int*) malloc(sizeof(unsigned int));
    // CHECK(cudaMemcpy(h_res_counter, results_counter, sizeof(unsigned int), cudaMemcpyDeviceToHost));

    // printf("\nactual results counter while collecting = %u", *h_res_counter);

    // printf("\n dimblock = %u, dimGrid = %u, offset = %u", dimBlock, dimGrid, offset);

    while (n_lists > 256 && dimGrid > 1) {
        cuda_merge_sublists <<< dimGrid, dimBlock >>> (sub_lists, n_lists, size, offset, NULL);

        // CHECK(cudaDeviceSynchronize());
        // exit(1);

        n_lists = dimGrid;
        offset *= dimBlock;
        compute_grid_dimension(n_lists, &dimBlock, &dimGrid);
        // printf("\n recomputed: dimblock = %u, dimGrid = %u, offset = %u", dimBlock, dimGrid, offset);
    }

    CHECK(cudaMemcpy(face_counter, dev_face_counter, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    nBytes = sizeof(Rectangle) * (*face_counter);
    Rectangle* dev_results;
    CHECK(cudaMalloc((void **) &dev_results, nBytes));

    // printf("\n face counter after gathering = %d", *face_counter);

    if (dimGrid > 1)
    {
        for (dimBlock = 32; dimBlock < n_lists; dimBlock *= 32);
        dimGrid = 1;
    }

    cuda_merge_sublists <<< dimGrid, dimBlock >>> (sub_lists, n_lists, size, offset, dev_results);

    Rectangle* results = (Rectangle*) malloc(nBytes);
    CHECK(cudaMemcpy(results, dev_results, nBytes, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(sub_lists));
    CHECK(cudaFree(dev_results));

    // GROUP FACES

    List* grouped = listInit();
    int i, arraySize = *face_counter;
    Rectangle* r, *s;

    uint diff_x, diff_y;
    float distance;

    int a,b,c,d;

    uint normalize_factor = min(im.width, im.height);

    bool merged;

    // FILE* f = fopen("log.txt", "w");

    while (arraySize > 0)
    {
        r = &(results[arraySize - 1]);
        merged = false;

        // fprintf(f, "\nevaluating rect %d: { x = %hu, y = %hu, size.width = %u, size.height = %u }", arraySize -1, r->x, r->y, r->size.width, r->size.height);

        for (i = arraySize - 2; i >= 0; i--)
        {
            s = &(results[i]);
            // fprintf(f, "\n\tcomparing rect %d with rect %d = { x = %hu, y = %hu, size.width = %u, size.height = %u }", arraySize -1, i, s->x, s->y, s->size.width, s->size.height);

            diff_x = round((abs((r->x + floor( r->size.width / 2.0f)) - (s->x + floor( s->size.width / 2.0f))) / (float) normalize_factor) * 100.0f);
            diff_y = round((abs((r->y + floor( r->size.height / 2.0f)) - (s->y + floor( s->size.height / 2.0f))) / (float) normalize_factor) * 100.0f);

            distance = sqrt(diff_x * diff_x + diff_y * diff_y);

            // fprintf(f, "\tdiff_x = %u, diff_y = %u, distance = %f", diff_x, diff_y, distance);

            if (distance <= merge_threshold)
            {
                a = min(r->x, s->x);
                b = max(r->x + r->size.width, s->x + s->size.width);
                c = min(r->y, s->y);
                d = max(r->y + r->size.height, s->y + s->size.height);

                s->x = a;
                s->y = c;
                s->size.width = b - a;
                s->size.height = d - c;

                (*face_counter)--;

                merged = true;
                break;
            }
        }

        arraySize--;

        if (!merged)
            add(grouped, r);
        // fprintf(f, "\n\n\tcurrent unique faces detected = %d", grouped->size);
    }
    // fclose(f);
    return grouped;
}

__global__ void cuda_draw_rectangles(Rectangle** faces, pel* image, unsigned int width, unsigned int height, bool rgb)
{
    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= width * height)
        return;

    Rectangle* face = faces[id];

    if (face == NULL)
        return;

    short color_RGB = 255;
    short color_GREY = 150;

    uint i, j;
    if (rgb)
    {
        for (i = face->y, j = face->x; j < face->x + face->size.width; j++)
        {
            image[(i * width + j) * 3 + 2] = color_RGB;
            image[(i * width + j) * 3 + 1] = 0;
            image[(i * width + j) * 3] = 0;
            // image[i * width + 3 * j + 1] = 0;
            // image[i * width + 3 * j] = 0;

            image[((i + face->size.height - 1) * width + j) * 3 + 2] = color_RGB;
            image[((i + face->size.height - 1) * width + j) * 3 + 1] = 0;
            image[((i + face->size.height - 1) * width + j) * 3] = 0;
            // image[(i + face->size.height - 1) * width + 3 * j + 1] = 0;
            // image[(i + face->size.height - 1) * width + 3 * j] = 0;
        }

        for (i = face->y, j = face->x; i < face->y + face->size.height; i++)
        {
            image[(i * width + j) * 3 + 2] = color_RGB; // r
            image[(i * width + j) * 3 + 1] = 0; // r
            image[(i * width + j) * 3] = 0; // r
            // image[i * width + 3 * j + 1] = 0;   // g
            // image[i * width + 3 * j] = 0;       // b

            image[(i * width + j + face->size.width - 1) * 3 + 2] = color_RGB; // r
            image[(i * width + j + face->size.width - 1) * 3 + 1] = 0; // r
            image[(i * width + j + face->size.width - 1) * 3] = 0; // r
            // image[i * width + 3 * (j + face->size.width - 1) + 1] = 0;   // g
            // image[i * width + 3 * (j + face->size.width - 1)] = 0;       // b
        }
    }
    else
    {
        for (i = face->y, j = face->x; j < face->x + face->size.width; j++)
        {
            image[i * width + 3 * j + 2] = color_GREY; // r
            image[i * width + 3 * j + 1] = color_GREY;   // g
            image[i * width + 3 * j] = color_GREY;       // b

            image[(i + face->size.height - 1) * width + 3 * j + 2] = color_GREY; // r
            image[(i + face->size.height - 1) * width + 3 * j + 1] = color_GREY;   // g
            image[(i + face->size.height - 1) * width + 3 * j] = color_GREY;       // b
        }

        for (i = face->y, j = face->x; i < face->y + face->size.height; i++)
        {
            image[i * width + 3 * j + 2] = color_GREY; // r
            image[i * width + 3 * j + 1] = color_GREY;   // g
            image[i * width + 3 * j] = color_GREY;       // b

            image[i * width + 3 * (j + face->size.width - 1) + 2] = color_GREY; // r
            image[i * width + 3 * (j + face->size.width - 1) + 1] = color_GREY;   // g
            image[i * width + 3 * (j + face->size.width - 1)] = color_GREY;       // b
        }
    }
}

List* detect_multiple_faces(pel* image, float scaleFactor, int minWindow, int maxWindow, unsigned int* host_face_counter, float group_factor)
{
    if (image == NULL)
        return NULL;

    Rectangle** faces;
    CHECK(cudaMalloc((void **) &faces, sizeof(Rectangle*) * im.width * im.height));
    CHECK(cudaMemset(faces, 0, sizeof(Rectangle*) * im.width * im.height));

    unsigned int* dev_face_counter;
    CHECK(cudaMalloc((void**) &dev_face_counter, sizeof(unsigned int)));
    CHECK(cudaMemset(dev_face_counter, 0, sizeof(unsigned int)));

    uint dimBlock, dimGrid;
    unsigned long nBytes;

    if ( maxWindow < WINDOW_SIZE )
        maxWindow = min(im.width, im.height);

    int max_iteration = floor(log((double) maxWindow / WINDOW_SIZE) / log(scaleFactor));

    int max_stream = min(max_iteration, N_STREAM);
    cudaStream_t stream[max_stream];
    int k;
    for (k = 0; k < max_stream; k++)
        stream[k] = 0;

    double* iim_vec[max_iteration];
    double* squared_iim_vec[max_iteration];

    double* iim, *squared_iim;
    
    printf("\nmaxWindow = %d", maxWindow);

    // pel* grey_image = (pel*) malloc(sizeof(pel) * im.height * im.h_offset);
    // CHECK(cudaMemcpy(grey_image, image, sizeof(pel) * im.height * im.h_offset, cudaMemcpyDeviceToHost));

    float currFactor;
    int iteration;
    size_t streamId;
    printf("\ncreating image pyramid...");
    for (currFactor = 1, iteration = 0; iteration < max_iteration ; currFactor*= scaleFactor, iteration++)
    {
        int curr_winSize = round(WINDOW_SIZE * currFactor);

        if (minWindow > curr_winSize)
            continue;

        // if (maxWindow < curr_winSize)
        //     break;

        Size temp_size = { round(im.width / currFactor), round(im.height / currFactor) };

        printf("\niteration n: %d | resizing factor: %f | height = %u, width = %u", iteration, currFactor, temp_size.height, temp_size.width);

        streamId = iteration % N_STREAM;

        if (floor((float) streamId / N_STREAM) == 0)
            CHECK(cudaStreamCreate(&(stream[streamId])));

        nBytes = sizeof(double) * temp_size.width * temp_size.height;
        
        CHECK(cudaMalloc((void **) &iim, nBytes));
        CHECK(cudaMalloc((void **) &squared_iim, nBytes));

        // pel* dev_resized_img, *resized_img;
        // CHECK(cudaMalloc((void **) &dev_resized_img, sizeof(pel) * temp_size.width * temp_size.height));

        cuda_integral_image(image, temp_size.width, temp_size.height, im.width, iim, squared_iim, currFactor, im.bitColor, stream[streamId]);

        // resized_img = (pel*) malloc(sizeof(pel) * temp_size.width * temp_size.height);
        // CHECK(cudaMemcpy(resized_img, dev_resized_img, sizeof(pel) * temp_size.height * temp_size.width, cudaMemcpyDeviceToHost));
        // char file_name[19];
        // snprintf(file_name, 19, "resized/img_%d.bmp", iteration);
        // write_new_BMP(file_name, resized_img, temp_size.height, temp_size.width, 8);
        // free(resized_img);
        // CHECK(cudaFree(dev_resized_img));

        
        // double* dev_iim = (double* )malloc(nBytes);
        // double* dev_squared_iim = (double*) malloc(nBytes);
        // CHECK(cudaMemcpy(dev_iim, iim, nBytes, cudaMemcpyDeviceToHost));
        // CHECK(cudaMemcpy(dev_squared_iim, squared_iim, nBytes, cudaMemcpyDeviceToHost));

        // compare_integral_image(grey_image, temp_size, dev_iim, dev_squared_iim, currFactor, im.bitColor);

        // free(dev_iim);
        // free(dev_squared_iim);
        
        compute_grid_dimension(temp_size.height * temp_size.width, &dimBlock, &dimGrid);

        // cudaDeviceSynchronize();
        cuda_evaluate <<< dimGrid, dimBlock, 0, stream[streamId] >>>(classifier, iim, squared_iim, temp_size.width, temp_size.height, curr_winSize, currFactor, faces, dev_face_counter, im.width);

        if (cudaStreamQuery(stream[streamId]) == cudaSuccess)
        {
            CHECK(cudaFree(iim));
            CHECK(cudaFree(squared_iim));
            iim_vec[iteration] = NULL;
            squared_iim_vec[iteration] = NULL;
        }
        else
        {
            iim_vec[iteration] = iim;
            squared_iim_vec[iteration] = squared_iim;
        }
    }

    int i;
    for (i = 0; i < max_iteration; i++)
    {
        if (i < N_STREAM)
        {
            cudaStreamSynchronize(stream[i]);
            CHECK(cudaStreamDestroy(stream[i]));
        }

        iim = iim_vec[i];
        squared_iim = squared_iim_vec[i];

        if (iim != NULL && squared_iim != NULL)
        {
            CHECK(cudaFree(iim));
            CHECK(cudaFree(squared_iim));
        }
    }

    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(host_face_counter, dev_face_counter, sizeof(unsigned int), cudaMemcpyDeviceToHost));

    printf("\ncompleted");

    // compute_grid_dimension(im.height * im.width, &dimBlock, &dimGrid);
    // cuda_draw_rectangles<<< dimGrid, dimBlock >>> (faces, image, im.width, im.height, im.bitColor > 8);
    // pel* host_image = (pel*) malloc(im.height * im.h_offset);
    // CHECK(cudaMemcpy(host_image, image, im.height * im.h_offset, cudaMemcpyDeviceToHost));
    // write_new_BMP("grey_test.bmp", host_image, im.height, im.width, im.bitColor);

    CHECK(cudaFree(image));

    if (*host_face_counter > 0)
        return gather_results(faces, dev_face_counter, host_face_counter, group_factor);
    return NULL;
}

List* listInit()
{
    List* list = (List*) malloc(sizeof(List));
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