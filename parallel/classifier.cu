#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stddef.h>
#include <assert.h>

#include <math_functions.h>

// Aliases for minimum and maximum between two variables as conditional operator 
// due to absence of definitions in C libraries
#define min(a,b) (a < b? a : b)
#define max(a,b) (a > b? a : b) 

// Link the header
#include "classifier.h"
// Link the CUDA utils functions
#include "utils/common.h"

// Max number of CUDA streams that can be set
#define N_STREAM 32

// UTILITY STRUCTURES FOR CLASSIFIER

/* 
- Filter defines the data for the single Weak Classifier. 
    Each of them is composed by three possibile rectangles that represent
    the locations for the Haar-like features in the window. Then each 
    rectangle has its own weight. The sum of the weighted areas is than 
    compared to its threshold. A filter's outputs are two values that 
    represents the positive (alpha1) and negative (alpha2) results, which 
    are its contribution for the stage evaluation.
*/
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

/*
- Classifier defines the data of the Cascade Classifier.
    It contains the references and values for:
    * the array structure of all the filters;
    * an array for the total number of filters for each stage;
    * an array for the stage thresholds;
    * the total number of stages that has to be evaluated.
*/
typedef struct classifier
{
    Filter* filters;
    short* filters_per_stages;
    int* stage_thresholds;
    short n_stages;
} Classifier;

// The classifier variable is defined as global in order to be used by 
// all functions without the direct reference.
Classifier* classifier;

/*
    DEBUG FUNCTION

    Used to print filters info into STDOUT for debug purposes.
*/
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

/*
    Load the classifier in GPU memory by reading from the given classifier's path for
    its data, and from the config's path for the number of stages and the numbers
    of filters per stage. The function outputs the result of the loading operation.

    To successfully load the data into device memory, they have to be read into host
    memory before, and then copied into device's one. That happens due to file reading.
*/
bool load_classifier_to_gpu(const char* classifier_path, const char* config_path)
{
    // Check if the arguments are not null
    if (classifier_path && config_path)
    {
        printf("\nnot null parameters: %s, %s", classifier_path, config_path);

        // Define the device variables
        Filter* dev_filters;
        short* dev_filter_per_stages;
        int* dev_stage_thresholds;

        // Open and read from the configuration's file
        FILE* config_file = fopen(config_path, "r");

        // Check if the file opening successed
        if (!config_file)
        {
            printf("\nerror while opening the config file. aborting...");
            return false;
        }

        // Read the number of stages
        int stages = 0;
        fscanf(config_file, "%d", &stages);
        if (!stages)
        {
            printf("\nerror while reading from config file. aborting...");
            return false;
        }
        
        // Read the number of filters per stage
        short* filter_per_stages = (short*) malloc(sizeof(short) * stages);
        int i = 0;
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

        // Open and read the classifier's data file
        FILE* classifier_file = fopen(classifier_path, "r");
        if (!classifier_file)
        {
            printf("\nerror while reading from classifier file. aborting...");
            return false;
        }

        /* 
            The classifier file contains the filter's data for each stage in sequence, followed
            at the end by the stage threshold. So the reading proceeds by reading n filters 
            (according to the filters number for the current stage, read before from the config
            file) and then read the corresponding stage threshold.

            The filter data structure is a mono dimentional array of size (number of stages * 
            number of filters for each stage = tot_filters). Each stage accesses to its filters
            by an offset computed as the sum of the filters' number in previous stages.  
        */

        Filter* filters = (Filter*) malloc(sizeof(Filter) * tot_filters);
        int* stage_thresholds = (int*) malloc(sizeof(int) * stages);
        
        int j;
        int n_filters = 0;
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

        // Allocate the classifier into host memory
        Classifier* clas = (Classifier*) malloc(sizeof(Classifier));

        // Allocate and copy into device memory the classifier's data 
        CHECK(cudaMalloc((void**) &dev_filters, sizeof(Filter) * tot_filters));
        CHECK(cudaMalloc((void**) &dev_stage_thresholds, sizeof(int) * stages));
        CHECK(cudaMalloc((void**) &dev_filter_per_stages, sizeof(short) * stages));

        CHECK(cudaMemcpy(dev_filters, filters, sizeof(Filter) * tot_filters, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dev_filter_per_stages, filter_per_stages, sizeof(short) * stages, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dev_stage_thresholds, stage_thresholds, sizeof(int) * stages, cudaMemcpyHostToDevice));

        // Assign to classifier the device pointers of its data
        clas->filters = dev_filters;
        clas->filters_per_stages = dev_filter_per_stages;
        clas->n_stages = stages;
        clas->stage_thresholds = dev_stage_thresholds;

        // Copy into device memory the classifier with the correct device pointers
        // for its data
        CHECK(cudaMalloc((void**) &classifier, sizeof(Classifier)));
        CHECK(cudaMemcpy(classifier, clas, sizeof(Classifier), cudaMemcpyHostToDevice));

        return true;
    }
    else
        printf("\nnull parameters detected. aborting...");
    return false;
}

/*
    This CUDA Kernel executes the face detection algorithm on the current image of the
    image pyramid. 

    The parallelization of this task, compared to the sequential version, is to evaluate
    all the windows in the pixels of the image at the same time. This allows to reduce a 
    lot the cost of the computation, and complete the image scanning in a very short time.

    As explained before, each thread of the kernel is assigned to a pixel and its window.
    Then it calculates the variance of pixel values into the window, and runs the 
    classification with the filters computation.  
*/
__global__ void cuda_evaluate(Classifier* classifier, double* iim, double* sq_iim, unsigned int width, unsigned int height, int currWinSize, float factor, Rectangle** faces, unsigned int* face_counter, unsigned int original_width)
{
    // Calculate the thread ID
    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    int y, x;

    unsigned int variance, mean;

    // Calculate the coords of the pixel 
    y = floor((double) id / (width - WINDOW_SIZE));
    x = id % (width - WINDOW_SIZE); 

    // Distinguish the active and inactive threads
    if (y >= height - WINDOW_SIZE || x >= width - WINDOW_SIZE)
        return;

    /// VARIANCE COMPUTATION 

    // Get the variance of the pixels values inside the current window
    // to weight the thresholds of the filters during the classifier
    // evaluation

    mean = iim[(y + WINDOW_SIZE - 1) * width + x + WINDOW_SIZE - 1];
    variance = sq_iim[(y + WINDOW_SIZE - 1) * width + x + WINDOW_SIZE - 1];

    if (y > 0)
    {
        mean -= (unsigned int) iim[(y - 1) * width + x + WINDOW_SIZE - 1];
        variance -= (unsigned int) sq_iim[(y - 1) * width + x + WINDOW_SIZE - 1];
    }
    if (x > 0)
    {
        mean -= (unsigned int) iim[(y + WINDOW_SIZE - 1) * width + x - 1];
        variance -= (unsigned int) sq_iim[(y + WINDOW_SIZE - 1) * width + x - 1];
    }
    if (y > 0 && x > 0)
    {
        mean += (unsigned int) iim[(y - 1) * width + x - 1];
        variance += (unsigned int) sq_iim[(y - 1) * width + x - 1];
    }

    // Get the variance for the current image with the following formula
    long tmp = (variance * (WINDOW_SIZE * WINDOW_SIZE)) - mean * mean;
    // If the squared variance is negative, set the root as 1
    variance = tmp > 0 ? (unsigned int) sqrt((double) tmp) : 1;

    /// RUNNING CLASSIFIER

    /*
        Execute the features evaluation for the current window. The window in these 
        computations is fixed at (24 x 24), and the window is at position
        [(y,x), (y + 24, x + 24)]. 

        The function iterates for each stage, and computes all the filters until a 
        stage rejects the window. If that happens, the function returns, 
        otherwise all the stages give positive output means there is a face 
        detected in the window, so the result is saved into the faces matrix.

        The variance is calculated from the squared integral image, and is used to 
        weight the filter's threshold.
    */

    int stage, i;
    unsigned int temp;
    long int threshold, filter_sum, stage_sum;
    Filter* f = classifier->filters;
    Rectangle* r;

    // Iterate for each stage
    for (stage = 0; stage < classifier->n_stages; stage++)
    {
        // Reset the stage's contributes
        stage_sum = 0;

        // Iterate for each filter in the stage
        for (i = 0; i < classifier->filters_per_stages[stage]; i++)
        {
            filter_sum = 0;

            // Calculate the actual filter's threshold by weighting the
            // the filter's data with the variance of the window
            threshold = (long int) f->threshold * variance;

            // Get the first rectangle 
            r = &(f->rect1);

            double a,b,c,d;

            // Calculate the area for the first rectangle
            temp = 0;
            a = iim[(y + r->y + r->size.height) * width + (x + r->x + r->size.width)];
            b = - iim[(y + r->y) * width + (x + r->x + r->size.width)];
            c = - iim[(y + r->y + r->size.height) * width + (x + r->x)];
            d = iim[(y + r->y) * width + (x + r->x)];

            temp = a + b + c + d;

            // Adjust the rectangle's area with its weight, and sum to the filter's 
            // output
            filter_sum += (long int) temp * f->weight1;

            // Get the second rectangle
            r = &(f->rect2);

            // Calcualate its area
            temp = 0;
            a = iim[(y + r->y + r->size.height) * width + (x + r->x + r->size.width)];
            b = - iim[(y + r->y) * width + (x + r->x + r->size.width)];
            c = - iim[(y + r->y + r->size.height) * width + (x + r->x)];
            d = iim[(y + r->y) * width + (x + r->x)];

            temp = a + b + c + d;

            // Adjust the area with its weight, and sum to the filter's output
            filter_sum += (long int) temp * f->weight2;

            // Get the third rectangle
            r = &(f->rect3);

            // Because only few filters in the classifier have the third rectangle,
            // check if it is defined for the current one
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

            // Compute the filter's contribution in stage evaluation by comparing the filter_sum
            // with the threshold, and then output one of the two values
            stage_sum += (long int)(filter_sum < threshold ? f->alpha1 : f->alpha2);

            // Move the pointer to the next filter
            f++;
        }

        // Check if the stage's filters contributions exeeds the weighted stage's threshold.
        // If the stage gives negative output, interrupt the evaluation in the current window
        // and reject it.

        // The 0.4 as weight is obtained in an sperimental way. 
        if (stage_sum < 0.4f * classifier->stage_thresholds[stage])
            return;
    }

    // If all the stages are passed, then a face was detected

    // If a face was found, allocate a rectangle struct and
    // save the result
    Rectangle* face = (Rectangle*) malloc(sizeof(Rectangle));

    // Get the coords of the box respect to the size of the original image
    face->x = floor(x * factor);
    face->y = floor(y * factor);

    // Store the size for the current window, which is the size for
    // the bounding box of the detected face
    face->size.height = currWinSize;
    face->size.width = currWinSize;

    // Compute the linear offset corresponding to the pixel evaluated
    unsigned long scaledId = face->y * original_width + face->x;

    // Increase the face counter only if the cell at the computed scaled 
    // cell is empty, due to overwriting
    if (!faces[scaledId])
        atomicAdd(face_counter, 1);

    // The overwriting is accepted because near results boxes surely contain
    // the same face 

    faces[scaledId] = face;
}

/*
    INTEGRAL IMAGE 

    CUDA Kernel that computes the cumulative sum and squared sum of the pixel values 
    respect to the rows. Each thread is assigned to a row. 
*/
__global__ void cuda_integral_image_row(pel* image, unsigned int width, unsigned int height, unsigned int original_width, double* iim, double* squared_iim, float factor, short bitColor)
{
    // Calculate the ID of the current thread
    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned long offset, scaled_offset;
    double sum = 0, sq_sum = 0;
    pel t;
    int i;

    // Distinguish the active threads
    if (id < height)
    {
        // For each cell in the assigned row, calculate the cumulative sum
        for (i = 0; i < width; i++)
        {
            // Get the coords of the current pixel respect to the original size
            // of the image 
            int h = floor(id * factor);
            int w = floor(i * factor);

            // Calculate the offset to access the same pixel of the current one
            // in the original image  
            scaled_offset = h * original_width + w;

            // Calculate the offset for the integral and squared integral images
            offset = id * width + i;

            // Read from the original image the pixel value
            t = image[scaled_offset * (bitColor > 8 ? 3 : 1)];

            // Add the read value to the cumulative sums
            sum += (double) t;
            sq_sum += (double) (t * t);

            // Set the value of the matrices's cells
            iim[offset] = sum;
            squared_iim[offset] = sq_sum;
        }
    }
}

/*
    INTEGRAL IMAGE

    CUDA Kernel that computes the cumulative sum and squared sum of the values in 
    the integral and squared integral images along the columns, calculating the 
    final values of them. Each thread is assigned to a column.
*/
__global__ void cuda_integral_image_col(unsigned int width, unsigned int height, double* iim, double* squared_iim)
{
    // Calculate the ID of the current thread
    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    int i;

    // Distinguish the active threads
    if (id < width)
    {
        // For each cell of the matrices, sum the current value with
        // the one in the previous cell of the column
        for (i = 1; i < height; i++)
        {
            iim[i * width + id] += iim[(i - 1) * width + id];
            squared_iim[i * width + id] += squared_iim[(i - 1) * width + id];
        }
    }
}

/*
    INTEGRAL IMAGE

    Handler function that manages the integral and squared integral images creation.

    The parallel version of this algorithm tries to make the most independent possible
    the computation by splitting the task in two phases: a cumulative sum along the rows
    of the pixels value of the image, and then the cumulative sum along the columns of 
    the values obtained in the previous phase. By these, the final matrices contain 
    the correct values of the integral images. The computation cost lowers from 
    O(height * width) of the sequential version, to O(height + width).

    The algorithm is splitted in two kernels because the second phase have to be 
    executed only when the first is complete, so a grid synchronization was needed
    and waiting for the kernel's end was the only way (until CUDA 9).
*/
void cuda_integral_image(pel* image, unsigned int width, unsigned int height, unsigned int original_width, double* iim, double* sq_iim, float factor, short bitColor, cudaStream_t stream)
{
    // Get the best size for the grid and blocks 
    uint dimBlock, dimGrid;
    compute_grid_dimension(height, &dimBlock, &dimGrid);

    // Call the first phase kernel on the assigned stream
    cuda_integral_image_row <<< dimGrid, dimBlock, 0, stream >>>(image, width, height, original_width, iim, sq_iim, factor, bitColor);

    // Get the size of grid and blocks
    compute_grid_dimension(width, &dimBlock, &dimGrid);

    // Call the second phase kernel on the assigned stream
    cuda_integral_image_col <<<dimGrid, dimBlock, 0, stream >>>(width, height, iim, sq_iim);
}

/*
    GATHER RESULTS

    This CUDA kernel is part of the algorithm for getting the results of the
    face detection from the GPU. Due to avoid handling with complex data 
    structures implementation into device, the results of the face detection
    are stored into a matrix of the same size of the original image. To gather
    these results, they have to be converted from sparse pointers of Rectangle
    structure, to a linked list in order to proceed with the grouping phase.

    Considering that the argument faces is a sparse matrix, to take advantages 
    from the parallel execution of the GPU, instead of visit all the cells in a 
    sequential mode, they can be inspected by rows, in a similar way of the first 
    phase of the integral images creation. 

    Whenever a face is found, during the row inspection, it is added into a list.
    The goal of this phase is in fact, to convert the results from a sparse matrix
    to an array of linked lists (of size equal to height). 

    So, each thread is mapped to a row of the faces matrix.

    To facilitate the work of the grouping phase, whenever a face is found, it
    is compared to the others currently in the list, because there is an high 
    probability that more results that contain the same face may be in the same
    row and near to each other. This allows to already reduce the number of 
    results at this first phase, and lower the computation for the CPU (because
    the grouping phase is executed by it).
*/
__global__ void cuda_collect_results(List** sub_lists, Rectangle** faces, unsigned int height, unsigned int width, unsigned int* face_counter, float merge_threshold)
{
    // Get the ID of the current thread
    unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;

    // Distinguish the active threads
    if (id >= height)
        return;

    List* list = NULL;
    
    int i;
    Rectangle* r, *s;
    Node* node;

    uint diff_x, diff_y;
    float distance;

    int a,b,c,d;

    // Compute the normalization factor used for distance evaluation 
    // in results comparison
    uint normalize_factor = max(width, height);

    // Iterate for each cell in the row
    for (i = 0; i < width; i++)
    {
        // Access to the current cell
        r = faces[id * width + i];

        // Check if there is a result in the current cell
        if (r != NULL)
        {
            // Check if the list was initialized, meaning if the first
            // result was already found
            if (!list)
            {
                // If the current result is the first of the list,
                // initialize it and add to it the result

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
            // Else at least one result was already found
            else
            {
                // Visit the results currently in the list and check
                // if some of them is similar to the one just found
                node = list->head;
                while(node != NULL)
                {
                    // Get the element from the list node
                    s = node->elem;

                    // Calculate the differences of pixels along the axes respect 
                    // to the centers of their boxes, and normalize the result
                    diff_x = round((abs((r->x + floor( r->size.width / 2.0f)) - (s->x + floor( s->size.width / 2.0f))) / (float) normalize_factor) * 100.0f);
                    diff_y = round((abs((r->y + floor( r->size.height / 2.0f)) - (s->y + floor( s->size.height / 2.0f))) / (float) normalize_factor) * 100.0f);

                    // Compute the distance between the two boxes
                    distance = sqrt((double) (diff_x * diff_x + diff_y * diff_y));

                    // Check if the distance is enough low to consider them as the 
                    // same face
                    if (distance <= merge_threshold)
                    {
                        // Merge the boxes

                        a = min(r->x, s->x);
                        b = max(r->x + r->size.width, s->x + s->size.width);
                        c = min(r->y, s->y);
                        d = max(r->y + r->size.height, s->y + s->size.height);

                        s->x = a;
                        s->y = c;
                        s->size.width = b - a;
                        s->size.height = d - c;

                        // Decrease the results counter
                        atomicSub(face_counter, 1);

                        // Interrupt the results comparison and proceed to
                        // the next cell of the row
                        break;
                    }
                    // If the distance was higher than the threshold, the two
                    // results were different, so proceed to the next one in 
                    // the list
                    node = node->next;
                }

                // If the node pointer is null, then all the elements in the list
                // were evaluated and the new result is an unique result, so add
                // it into the list 
                if (node == NULL)
                {
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

    // Check if at least a result was found
    if (list && list->size > 0)
    {
        // If there is at least one result, add the list to the array
        sub_lists[id] = list;
    }
}

/*
    GATHER RESULTS

    This CUDA Kernel executes the second phase of the results gathering 
    algorithm. It takes the array of linked lists generated by the first 
    phase, and merge all of them into a single list placed at the first 
    element of the list. 

    The merging strategy is the parallel reduction, that grants to complete
    the merging in a logarithmic time instead of the linear time obtained
    by a sequential visit.

    The original algorithm of the parallel reduction splits the computation
    along the thread blocks, and only merges the elements that belong to them 
    in a single one that is the first element of each block. Considering that 
    the goal in this case is not to reduce the lists number, but merge them all,
    the algorithm is adapted to be called more times, until the number is low
    enough to be computed by a single block that completes the merge in a 
    single unique list with all the results in it.

    Once the lists are all merged in one, the results are copied into an 
    array, in order to allow the memory copy from device to host.

    Arguments explanation:

    - sub_lists is the input lists array that have to be merged;
    - n_lists defines the current number of unmerged lists in the array;
    - size is equal to the original size of the array, used to avoid out 
      of bounds accesses.    
    - offset is equal to the actual offset between the elements, used when the 
      algorithm needs more iterations (at the first it is 1) to correctly move 
      along the array, due to the merging per block;
    - results is the output array that have to be filled with the collected
      results.
*/
__global__ void cuda_merge_sublists(List** sub_lists, unsigned int n_lists, unsigned int size, unsigned int offset, Rectangle* results)
{
    // Get the local thread ID respect to its block
    uint tid = threadIdx.x;
    // Calculate the global thread ID
    unsigned long idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Distinguish the active threads
    if (idx >= n_lists) 
        return;

    // Calculate the local array pointer for the current block
    int blockOffset = (blockIdx.x * blockDim.x) * offset;
    List **thisBlock = sub_lists + blockOffset;

    int stride, oldStride, unmerged_range = 0;
    List* a = NULL, *b = NULL;

    // Define a flag for the odd number of lists in this execution
    bool odd_lists = n_lists % 2;

    // Compute the reduction in global memory. 
    
    // The initial stride is half of the minimum between the number of lists, 
    // the block dimension, and the remaining cells of the array from the 
    // block offset. This is used to avoid out of boundaries when computing
    // the merging. 
    
    // The stride halves its value for each iteration. The actual value is the
    // ceil of the division, due to lost lists issues that happens with odd
    // strides
    for (stride = min(min(n_lists, blockDim.x), size - blockOffset) / 2, oldStride = stride; stride > 0; stride = stride > 1 ? ceil(stride / 2.0f) : 0) 
    { 

        // Due to always present possibilities of lost lists during the 
        // algorithm, caused by odd strides or number of lists, whenever
        // the first odd stride is encounterd, its value is saved to 
        // check later for missing lists. This asserts to always retrieve
        // all the lists without checking again all the cells
        if (oldStride - stride * 2 > 0 && !unmerged_range)
            unmerged_range = oldStride;

        // Distinguish the threads that have to operate for each iteration.
        // In case of odd number of lists, for the first iteration, there 
        // are stride + 1 active threads (of tid = {0, ..., stride}), due to 
        // odd lists handling
        if (tid < stride || (tid <= stride && odd_lists && oldStride == stride))
        {
            // Read from the two cells to evaluate. The index is scaled by the
            // offset in case of repeated execution of the thread
            a = thisBlock[tid * offset];
            b = thisBlock[(tid + stride) * offset];

            // Check if both the pointers have a list 
            if (a && b)
            {
                // Merge the two lists 

                a->tail->next = b->head;
                a->tail = b->tail;
                a->size += b->size;

                // Delete the pointer in the second cell
                thisBlock[(tid + stride) * offset] = NULL;
            }
            else if (b)
            {
                // If only the second cell has a list, shift it to the first
                // cell
                thisBlock[tid * offset] = b;

                // Delete the pointer in the second cell
                thisBlock[(tid + stride) * offset] = NULL;
            }

            // If both cells have no lists, no operation are obviously needed
        }

        // Update the stride memory for the just completed iteration
        oldStride = stride;

        // Synchronize the threads within block
        __syncthreads();
    }

    // If the current thread is the first of the block, check for missing
    // lists if the unmerged_range is set
    if (tid == 0)
    {
        a = thisBlock[0];
        if (a && unmerged_range > 0)
        {
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

    // If the current thread is the first of the grid, and there was only
    // a block in this execution, its list contains all the results of the
    // face detection. So they have to be moved into the output array
    if (results && tid == 0 && gridDim.x == 1)
    {
        List* list = sub_lists[0];
        int i;
        Node* curr = list->head;
        Rectangle* r, *s;

        for (i = 0; i < list->size && curr != NULL; i++ )
        {
            r = curr->elem;
            s = &(results[i]);

            s->size = r->size;
            s->x = r->x;
            s->y = r->y;

            curr = curr->next;
        }
    }
}

/*
    GATHER RESULTS

    This is the main function of the algorithm, which manages the kernels calls
    and handles with the arguments. 
    
    In the end executes the face grouping algorithm that merges all the similar 
    results in unique boxes.
*/
List* gather_results(Rectangle** faces, unsigned int* dev_face_counter, unsigned int* face_counter, float merge_threshold)
{
    // Initialize the lists array for the first phase
    List** sub_lists;
    unsigned long size = im.height;
    unsigned long nBytes = size * sizeof(List*);
    CHECK(cudaMalloc((void**) &sub_lists, nBytes));
    CHECK(cudaMemset(sub_lists, 0, nBytes));

    // Get the optimal grid and blocks sizes
    uint dimBlock, dimGrid;
    compute_grid_dimension(size, &dimBlock, &dimGrid);

    // Execute the first phase 
    cuda_collect_results <<< dimGrid, dimBlock >>> (sub_lists, faces, im.height, im.width, dev_face_counter, merge_threshold);

    // Set the initial values of the arguments for the second phase
    unsigned int n_lists = size;
    unsigned int offset = 1;

    // Repeat the second phase algorithm until the number of the remaining lists
    // is low enough to be handled by a single thread block and complete the phase
    while (n_lists > 256 && dimGrid > 1) {
        // If there are more than one block, the results array isn't used
        cuda_merge_sublists <<< dimGrid, dimBlock >>> (sub_lists, n_lists, size, offset, NULL);

        // Update the arguments
        n_lists = dimGrid;
        offset *= dimBlock;

        // Recompute the grid and blocks sizes for the current number of lists
        compute_grid_dimension(n_lists, &dimBlock, &dimGrid);
    }

    // Get the updated number of results after the first phase execution
    CHECK(cudaMemcpy(face_counter, dev_face_counter, sizeof(unsigned int), cudaMemcpyDeviceToHost));

    // Allocate the device memory for the results array
    nBytes = sizeof(Rectangle) * (*face_counter);
    Rectangle* dev_results;
    CHECK(cudaMalloc((void **) &dev_results, nBytes));

    // If the number of remaining lists is lower than 256, force the CUDA
    // grid to have only one block if it isn't already
    if (dimGrid > 1)
    {
        for (dimBlock = 32; dimBlock < n_lists; dimBlock *= 32);
        dimGrid = 1;
    }

    // Call the second phase kernel to complete the merging and fill the results 
    // array
    cuda_merge_sublists <<< dimGrid, dimBlock >>> (sub_lists, n_lists, size, offset, dev_results);

    // Allocate the host memory for the results
    Rectangle* results = (Rectangle*) malloc(nBytes);

    // Copy the results of the face detection from the device memory
    CHECK(cudaMemcpy(results, dev_results, nBytes, cudaMemcpyDeviceToHost));

    // Free the device memory
    CHECK(cudaFree(sub_lists));
    CHECK(cudaFree(dev_results));

    /// GROUP FACES

    /*
        Execute the results grouping algorithm in order to merge all the 
        results that contain the same face, and obtain an output list with
        only unique results in it. That list will be the final output of the
        Viola-Jones algorithm.

        The grouping algorithm considers the results array as a stack. Until
        the stack is empty, pops an element and compares it with all the 
        others. If another box with near center to the evaluated one is found,
        merge the second one with the first. Otherwise it is considered as 
        unique face, and added to the output list. 
    */

    // Initialize the output list
    List* grouped = listInit();

    int i, arraySize = *face_counter;
    Rectangle* r, *s;

    uint diff_x, diff_y;
    float distance;

    int a,b,c,d;

    // Compute the normalization factor used for distance evaluation 
    // in results comparison
    uint normalize_factor = min(im.width, im.height);

    // Flag to warn if the current evaluated result was merged
    bool merged;

    // Repeat until all items are reviewed
    while (arraySize > 0)
    {
        // Get the current item
        r = &(results[arraySize - 1]);

        // Reset the flag
        merged = false;

        // Compaare the current item with all the others in the array
        for (i = arraySize - 2; i >= 0; i--)
        {
            s = &(results[i]);

            // Calculate the normalized differences of position along the axes 
            // of the centers of the two items
            diff_x = round((abs((r->x + floor( r->size.width / 2.0f)) - (s->x + floor( s->size.width / 2.0f))) / (float) normalize_factor) * 100.0f);
            diff_y = round((abs((r->y + floor( r->size.height / 2.0f)) - (s->y + floor( s->size.height / 2.0f))) / (float) normalize_factor) * 100.0f);

            // Calculate the actual distance
            distance = sqrt(diff_x * diff_x + diff_y * diff_y);

            // If the two items are near enough, merge the evaluated item with
            // the picked one from the array
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

                // Decrease the results counter
                (*face_counter)--;

                // Set the flag
                merged = true;
                break;
            }
        }

        // Update the stack bound
        arraySize--;

        // If the evaluated item wasn't merged, it is an unique result and 
        // can be added to the output list 
        if (!merged)
            add(grouped, r);
    }
    // Return the output list
    return grouped;
}

/*
    This is the main function of the Face Detection algorithm. 
    
    To use the Cascade Classifier, which is trained on (24 x 24) images, and detect
    faces at any distances in the image, the original image is resized at different 
    factors until the virtual size of the window is bigger than the image, that 
    actually is when the resized image has one of the size less than 24 pixels. That
    operation is called image pyramid creation.

    At each iteration, the resizing factor follows the powers of scaleFactor, and
    the size of the current image on which apply the evaluation is computed. Then
    the integral and squared integral images are calculated, and the Cascade Classifier
    applied on them.

    The scaling of the image can be controlled by modifing the minWindow and maxWindow:
    the first one set the minimum size that the window must have to begin with the face
    detection, and the second one set the maximum size that the window must have to end 
    the face detection (the stop condition). 

    If the maxWindow value is less then the default window size (24), it is set to the 
    max size of the image.

    The improvements in parallel version are the possibility to delegate most of the 
    algorithm's computation to the GPU. Also, one of the most effective improvement is 
    the execution of the various iterations of image pyramid at the same time by using
    the Hyper-Q's multi-stream. This significantly reduces the elapsed time, granting 
    an almost immediate execution without any particular dependency of input size. 

    The management of the streams aims to uses all the queues needed by mapping each
    iteration to a stream. When the number of the iterations are greater than 32, they
    are cyclically assigned thanks to the modulo operation, meaning the 33th iteration
    will be assigned to the first stream, that being the first to be launched may be 
    next to its iteration completion.

    The streams's IDs are stored into an array, which dimension is set to the minimum
    between the total number of iterations and 32 (which is the max n of streams that 
    can be created).   

    Once the kernels are successfully called, the CPU waits for the stream's queue to
    ends, and then destroyes them.

    In the end, the faces results matrix is passed to gather_results function which 
    will get the results from the device, and apply the face grouping to them. Its 
    output is then returned as the final output of Viola-Jones's algorithm.
*/
List* detect_multiple_faces(pel* image, float scaleFactor, int minWindow, int maxWindow, unsigned int* host_face_counter, float group_factor)
{
    // Check if image argument is not null
    if (image == NULL)
        return NULL;

    // Allocate the results matrix in device memory
    Rectangle** faces;
    CHECK(cudaMalloc((void **) &faces, sizeof(Rectangle*) * im.width * im.height));
    CHECK(cudaMemset(faces, 0, sizeof(Rectangle*) * im.width * im.height));

    // Allocate the results counter in device memory
    unsigned int* dev_face_counter;
    CHECK(cudaMalloc((void**) &dev_face_counter, sizeof(unsigned int)));
    CHECK(cudaMemset(dev_face_counter, 0, sizeof(unsigned int)));

    uint dimBlock, dimGrid;
    unsigned long nBytes;

    // Set the max size of the window. A value lower than 24 will set
    // the defalut value which executes the detection on the entire image
    if ( maxWindow < WINDOW_SIZE )
        maxWindow = min(im.width, im.height);

    // Calculate the number of iterations 
    int max_iteration = floor(log((double) maxWindow / WINDOW_SIZE) / log(scaleFactor));

    // Define the number of streams needed
    int max_stream = min(max_iteration, N_STREAM);

    // Create the stream array
    cudaStream_t stream[max_stream];
    int k;
    for (k = 0; k < max_stream; k++)
        stream[k] = 0;

    // Define the arrays to store the integral and squared integral
    // images, in order to deallocate them once the face detection
    // is completed
    double* iim_vec[max_iteration];
    double* squared_iim_vec[max_iteration];

    double* iim, *squared_iim;
    
    printf("\nmaxWindow = %d", maxWindow);


    float currFactor;
    int iteration;
    size_t streamId;
    printf("\ncreating image pyramid...");

    // Run the iterations
    for (currFactor = 1, iteration = 0; iteration < max_iteration ; currFactor*= scaleFactor, iteration++)
    {
        // Calculate the size of the window for the current iteration
        int curr_winSize = round(WINDOW_SIZE * currFactor);

        // If the detection have to begin with a greater window, skip
        if (minWindow > curr_winSize)
            continue;

        // Calculate the size of image needed for this iteration
        Size temp_size = { round(im.width / currFactor), round(im.height / currFactor) };

        printf("\niteration n: %d | resizing factor: %f | height = %u, width = %u", iteration, currFactor, temp_size.height, temp_size.width);

        // Calculate the stream ID for this iteration
        streamId = iteration % N_STREAM;

        // Create the stream if not already defined
        if (iteration < N_STREAM)
            CHECK(cudaStreamCreate(&(stream[streamId])));

        // Allocate the memory for integral and squared integral images
        nBytes = sizeof(double) * temp_size.width * temp_size.height;
        CHECK(cudaMalloc((void **) &iim, nBytes));
        CHECK(cudaMalloc((void **) &squared_iim, nBytes));

        // Create the integral and squared integral images
        cuda_integral_image(image, temp_size.width, temp_size.height, im.width, iim, squared_iim, currFactor, im.bitColor, stream[streamId]);
        
        // Get the optimal grid and blocks sizes
        compute_grid_dimension(temp_size.height * temp_size.width, &dimBlock, &dimGrid);

        // Run the classification
        cuda_evaluate <<< dimGrid, dimBlock, 0, stream[streamId] >>>(classifier, iim, squared_iim, temp_size.width, temp_size.height, curr_winSize, currFactor, faces, dev_face_counter, im.width);

        // Free the integral images if the device has already completed 
        // the tasks
        if (cudaStreamQuery(stream[streamId]) == cudaSuccess)
        {
            CHECK(cudaFree(iim));
            CHECK(cudaFree(squared_iim));
            iim_vec[iteration] = NULL;
            squared_iim_vec[iteration] = NULL;
        }
        // Otherwise store the device pointers in the arrays to free them
        // later
        else
        {
            iim_vec[iteration] = iim;
            squared_iim_vec[iteration] = squared_iim;
        }
    }

    // Wait for the completion
    int i;
    for (i = 0; i < max_iteration; i++)
    {
        // Destroy the stream once it completes its tasks
        if (i < N_STREAM)
        {
            cudaStreamSynchronize(stream[i]);
            CHECK(cudaStreamDestroy(stream[i]));
        }

        // Free the integral images of the current iteration
        iim = iim_vec[i];
        squared_iim = squared_iim_vec[i];

        if (iim != NULL && squared_iim != NULL)
        {
            CHECK(cudaFree(iim));
            CHECK(cudaFree(squared_iim));
        }
    }

    CHECK(cudaDeviceSynchronize());

    // Retrive the number of results detected from the device memory
    CHECK(cudaMemcpy(host_face_counter, dev_face_counter, sizeof(unsigned int), cudaMemcpyDeviceToHost));

    printf("\ncompleted");

    // Free the greyscale image from device memory
    CHECK(cudaFree(image));

    // Gather and return the results if at least one result was found
    if (*host_face_counter > 0)
        return gather_results(faces, dev_face_counter, host_face_counter, group_factor);
    return NULL;
}

/// LINKED LIST FUNCTIONS

/*
    Initialize the linked list and return its reference
*/
List* listInit()
{
    List* list = (List*) malloc(sizeof(List));
    list->head = NULL;
    list->size = 0;

    return list;
}

/*
    Add a new element in the given list
*/
void add(List* list, Rectangle* r)
{
    // Check if the list exists
    if (list)
    {
        // Initialize a new node
        Node* node = (Node*) malloc(sizeof(Node));
        node->elem = r;
        node->next = NULL;

        // If the list is empty, assign the new node to the head
        if (list->size == 0)
        {
            list->head = node;
            list->size++;
            return;
        }

        // If there are other nodes in the list, add the new one at the head
        node->next = list->head;
        list->head = node;
        list->size++;
        return;
    }
    printf("\n Error: received null list parameter.");
}

/*
    Remove an element from the head of the list.
*/
Rectangle* remove_from_head(List* list)
{
    // Check if the given list exists
    if (list)
    {
        // Check if the list is not empty
        if (list->size > 0)
        {
            // Retrieve the element in the node at the top 
            Node* old = list->head;
            Rectangle* r = old->elem;

            // Move forward the head pointer and decrease the size
            list->head = list->head->next;
            list->size--;
            
            // Return the element
            return r;
        }
        printf("\n Error: trying to remove element with empty list.");
        return NULL;
    }
    printf("\n Error: received null list parameter.");
    return NULL;
}