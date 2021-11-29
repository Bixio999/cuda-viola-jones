#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stddef.h>

#include <math_functions.h>

#define min(a,b) (a < b? a : b)
#define max(a,b) (a > b? a : b)
// #define atomicAdd(a,b) atomicAdd((int*) a, (int) b)

#include "classifier.h"
#include "utils/common.h"

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

__global__ void cuda_evaluate(Classifier* classifier, double* iim, double* sq_iim, unsigned int width, unsigned int height, int currWinSize, float factor, Rectangle** faces, unsigned int* face_counter)
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
            long int rect1_val = filter_sum;

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

    faces[id] = face;

    atomicAdd(face_counter, 1);
    // printf("\ndetected face at (%u, %u).", face->y,face->x);
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

void cuda_integral_image(pel* image, unsigned int width, unsigned int height, unsigned int original_width, double* iim, double* sq_iim, float factor, short bitColor)
{
    uint dimBlock, dimGrid;

    // dimGrid = ceil((float) height / dimBlock);
    compute_grid_dimension(height, &dimBlock, &dimGrid);

    printf("\n\tiim row - dimblock = %u, dimgrid = %u | width = %u, height = %u", dimBlock, dimGrid, width, height);

    cuda_integral_image_row <<< dimGrid, dimBlock >>>(image, width, height, original_width, iim, sq_iim, factor, bitColor);

    // dimGrid = ceil((float) width / dimBlock);
    compute_grid_dimension(width, &dimBlock, &dimGrid);

    printf("\n\tiim cols - dimblock = %u, dimgrid = %u | width = %u, height = %u", dimBlock, dimGrid, width, height);

    cuda_integral_image_col <<<dimGrid, dimBlock >>>(width, height, iim, sq_iim);
}

Rectangle** detect_multiple_faces(pel* image, float scaleFactor, int minWindow, int maxWindow, unsigned int* dev_face_counter)
{
    if (image == NULL)
        return NULL;

    Rectangle** faces;
    CHECK(cudaMalloc((void **) &faces, sizeof(Rectangle*) * im.width * im.height));
    CHECK(cudaMemset(faces, 0, sizeof(Rectangle*) * im.width * im.height));

    uint dimBlock, dimGrid;
    unsigned long nBytes;
    double* iim, *squared_iim;

    if ( maxWindow < WINDOW_SIZE )
        maxWindow = min(im.width, im.height);
    
    printf("\nmaxWindow = %d", maxWindow);

    // pel* grey_image = (pel*) malloc(sizeof(pel) * im.height * im.h_offset);
    // CHECK(cudaMemcpy(grey_image, image, sizeof(pel) * im.height * im.h_offset, cudaMemcpyDeviceToHost));

    float currFactor;
    int iteration = 0;
    printf("\ncreating image pyramid...");
    for (currFactor = 1; ; currFactor*= scaleFactor, iteration++)
    {

        Size temp_size = { round(im.width / currFactor), round(im.height / currFactor) };

        int curr_winSize = round(WINDOW_SIZE * currFactor);

        if (minWindow > curr_winSize)
            continue;

        if (maxWindow < curr_winSize)
            break;

        printf("\niteration n: %d | resizing factor: %f | height = %u, width = %u", iteration, currFactor, temp_size.height, temp_size.width);

        nBytes = sizeof(double) * temp_size.width * temp_size.height;
        
        CHECK(cudaMalloc((void **) &iim, nBytes));
        CHECK(cudaMalloc((void **) &squared_iim, nBytes));

        // pel* dev_resized_img, *resized_img;
        // CHECK(cudaMalloc((void **) &dev_resized_img, sizeof(pel) * temp_size.width * temp_size.height));

        cuda_integral_image(image, temp_size.width, temp_size.height, im.width, iim, squared_iim, currFactor, im.bitColor);

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

        cudaDeviceSynchronize();
        cuda_evaluate <<< dimGrid, dimBlock >>>(classifier, iim, squared_iim, temp_size.width, temp_size.height, curr_winSize, currFactor, faces, dev_face_counter);

        cudaDeviceSynchronize();
        // print_classifier<<<1,1>>>(classifier);
        // cudaDeviceSynchronize();

        CHECK(cudaFree(iim));
        CHECK(cudaFree(squared_iim));

        // TODO GROUP RECTANGLES
    }
    printf("\ncompleted");
    CHECK(cudaFree(image));
    return faces;
}