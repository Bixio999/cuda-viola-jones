#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stddef.h>

// Aliases for minimum and maximum between two variables as conditional operator 
// due to absence of definitions in C libraries
#define min(a,b) (a < b? a : b)
#define max(a,b) (a > b? a : b)

// Link the header
#include "classifier.h"

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
    * the multi-array structure of all the filters;
    * an array for the total number of filters for each stage;
    * an array for the stage thresholds;
    * the total number of stages that has to be evaluated.
*/
typedef struct classifier
{
    Filter*** filters;
    int* filters_per_stages;
    int* stage_thresholds;
    int n_stages;
} Classifier;

// The classifier variable is defined as global in order to be used by 
// all functions without the direct reference.
Classifier* classifier;

/*
    Load the classifier in memory by reading from the given classifier's path for
    its data, and from the config's path for the number of stages and the numbers
    of filters per stage. The function outputs the result of the loading operation.
*/
bool load_classifier(const char* classifier_path, const char* config_path)
{
    // Check if the arguments are not null
    if (classifier_path && config_path)
    {
        printf("\nClassifier's files paths correctly received: %s, %s", classifier_path, config_path);

        // Open and read from the configuration's file
        FILE* config_file = fopen(config_path, "r");

        // Check if the file opening successed
        if (!config_file)
        {
            printf("\nerror while reading from config file. aborting...");
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
        int* filter_per_stages = (int*) malloc(sizeof(int) * stages);
        int i = 0;
        int* temp = filter_per_stages;
        while (i < stages && !feof(config_file))
        {
            fscanf(config_file, "%d", temp);
            i++;
            temp++;
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

            The filter data structure is a multi dimentional array of size (number of stages * 
            number of filters for each stage), in order to have a mono-dimentional array for each
            stage that contains the exactly number of filters for the current stage. Also the 
            strategy used allows to allocate the memory in a dynamic way. 
        */

        Filter*** filters = (Filter***) malloc(sizeof(Filter**) * stages);
        int* stage_thresholds = (int*) malloc(sizeof(int) * stages);

        int j;
        int n_filters = 0;
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

        // Allocate and assign the classifier's data
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

/*
    Creates the integral image and the squared integral image with the size of the current 
    iteration of image pyramid. The function uses a dynamic programming approach, and a 
    nearest neightborhood strategy for the resize.

    The int_im and squared_int_im must be passed by reference, in order to get the data 
    structure pointer in the memory locations of the variables.

    The argument bitColor is used to adapt the image memory access in case of 24 bit or 8 bit 
    color size.
*/
void integral_image(pel** image, Size size, double*** int_im, double*** squared_int_im, float factor, short bitColor)
{
    // Define the two matrices
    double** iim = (double**) malloc(size.height * sizeof(double*));
    double** squared_iim = (double**) malloc(size.height * sizeof(double*));
    unsigned int i;
    for (i = 0; i < size.height; i++)
    {
        iim[i] = (double *) malloc(size.width * sizeof(double));
        squared_iim[i] = (double *) malloc(size.width * sizeof(double));
    }

    // Assign the first value, that is equal to the pixel value
    iim[0][0] = image[0][0];
    squared_iim[0][0] = image[0][0] * image[0][0]; 

    unsigned int j, scaled_x, scaled_y;

    // Compute the value for each cell of the ii / sq_ii
    for (i = 0; i < size.height; i++)
    {
        for (j = 0; j < size.width; j++)
        {
            // Get the corresponding coords in the image
            scaled_y = floor(i * factor);
            scaled_x = floor(j * factor);

            // Get the pixel value
            pel t = image[scaled_y][scaled_x * (bitColor > 8? 3 : 1)];

            // Compute the value for the current cell as the sum of the left and top value,
            // then subtract the value at top-left cell and add the value of the current 
            // pixel in the image

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
        }
    }

    // Assign the matrices to the argument pointers
    *(int_im) = iim;
    *(squared_int_im) = squared_iim;
}

/*
    Execute the features evaluation for the current window. The window in these 
    computations is fixed at (24 x 24), and the window is at position
    [(y,x), (y + 24, x + 24)]. 

    The function iterates for each stage, and computes all the filters until a 
    stage rejects the window. If that happens, the function returns false, 
    otherwise all the stages give positive output and the function returns true.

    The variance is calculated from the squared integral image, and is used to 
    weight the filter's threshold.
*/
bool runClassifier(double** iim, int y, int x, unsigned int variance)
{
    int stage, i;
    unsigned int temp;
    long int threshold, filter_sum, stage_sum;

    // Iterate for each stage
    for (stage = 0; stage < classifier->n_stages; stage++)
    {
        // Get the stage's filters
        Filter** filters = classifier->filters[stage];
        // Reset the stage's contributes
        stage_sum = 0;

        // Iterate for each filter in the stage
        for (i = 0; i < classifier->filters_per_stages[stage]; i++)
        {
            // Get the current filter
            Filter* f = filters[i];

            filter_sum = 0;
            
            // Calculate the actual filter's threshold by weighting the
            // the filter's data with the variance of the window
            threshold = (long int) f->threshold * variance;

            // Get the first rectangle 
            Rectangle* r = &(f->rect1);

            double a,b,c,d;

            // Calculate the area for the first rectangle
            temp = 0;
            a = iim[y + r->y + r->size.height][x + r->x + r->size.width];
            b = - iim[y + r->y][x + r->x + r->size.width];
            c = - iim[y + r->y + r->size.height][x + r->x];
            d = iim[y + r->y][x + r->x];

            temp = a + b + c + d;

            // Adjust the rectangle's area with its weight, and sum to the filter's 
            // output
            filter_sum += (long int) temp * f->weight1;

            // Get the second rectangle
            r = &(f->rect2);
            
            // Calcualate its area
            temp = 0;
            a = iim[y + r->y + r->size.height][x + r->x + r->size.width];
            b = - iim[y + r->y][x + r->x + r->size.width];
            c = - iim[y + r->y + r->size.height][x + r->x];
            d = iim[y + r->y][x + r->x];

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
                a = iim[y + r->y + r->size.height][x + r->x + r->size.width];
                b = - iim[y + r->y][x + r->x + r->size.width];
                c = - iim[y + r->y + r->size.height][x + r->x];
                d = iim[y + r->y][x + r->x];

                temp = a + b + c + d;

                filter_sum += (long int) temp * f->weight3;
            }

            // Compute the filter's contribution in stage evaluation by comparing the filter_sum
            // with the threshold, and then output one of the two values
            stage_sum += (long int)(filter_sum < threshold ? f->alpha1 : f->alpha2);
        }

        // Check if the stage's filters contributions exeeds the weighted stage's threshold.
        // If the stage gives negative output, interrupt the evaluation in the current window
        // and reject it.

        // The 0.4 as weight is obtained in an sperimental way. 
        if (stage_sum < 0.4f * classifier->stage_thresholds[stage])
            return false;
    }
    // If all the stages are passed, then return a positive output
    return true;
}

/*
    Executes the Face Detection algorithm on the current iteration of the image pyramid.

    Scans the image over all the pixels (excluded the ones with the window out of bounds), 
    computes the variance for each of them and call the runClassifier function to check
    for faces in the positions.

    If the runClassifier function returns true, then a face was found in the current pixel,
    so store the position and the size of the window at that iteration in the faces list.
*/
void evaluate(double** iim, double** sq_iim, Size size, int currWinSize, float factor, List* faces, float merge_threshold)
{
    int i, j, stage, k;
    unsigned int variance, mean;

    // Iterate for each pixel of the resized image, exepted the ones with the window
    // out of bounds
    for (i = 0; i < size.height - WINDOW_SIZE; i++)
    {
        for (j = 0; j < size.width - WINDOW_SIZE; j++)
        {
            /// VARIANCE COMPUTATION

            // Get the variance of the pixels values inside the current window
            // to weight the thresholds of the filters during the classifier
            // evaluation

            mean = iim[i + WINDOW_SIZE - 1][j + WINDOW_SIZE - 1];
            variance = sq_iim[i + WINDOW_SIZE - 1][j + WINDOW_SIZE - 1];

            if (i > 0)
            {
                mean -= iim[i-1][j + WINDOW_SIZE - 1];
                variance -= sq_iim[i-1][j + WINDOW_SIZE - 1];
            }
            if (j > 0)
            {
                mean -= iim[i + WINDOW_SIZE - 1][j-1];
                variance -= sq_iim[i + WINDOW_SIZE - 1][j-1];
            }
            if (i > 0 && j > 0)
            {
                mean += iim[i-1][j-1];
                variance += sq_iim[i-1][j-1];
            }

            // Get the variance for the current image with the following formula
            variance = (variance * (WINDOW_SIZE * WINDOW_SIZE)) - mean * mean;
            // If the squared variance is negative, set the root as 1
            variance = variance > 0 ? sqrt(variance) : 1;

            /// FILTERS EVALUATION

            // Evaluate the features for the current window
            if (runClassifier(iim, i, j, variance))
            {
                // If a face was found, allocate a rectangle struct and
                // save the result
                Rectangle* face = malloc(sizeof(Rectangle));

                // Store the coords that belong to the original image
                face->x = floor(j * factor);
                face->y = floor(i * factor);

                // Store the size for the current window, which is the size for
                // the bounding box of the detected face
                face->size.height = currWinSize;
                face->size.width = currWinSize;

                // Save the result into the list
                add(faces, face);
            }
        }
    }
}

/*
    Deallocate the integral and squared integral images from the memory.
*/
void free_integral_image(double** iim, Size size)
{
    int i;
    for (i = 0; i < size.height; i++)
        free(iim[i]);
    free(iim);
}

/*
    FACE GROUPING

    This function check if the received bounding box (r) can be merged with another result
    because both contain the same face. If r is different from all the others, then is 
    considered as unique result and the function returns true. Otherwise it merges r with
    the found equal box and returns false. 

    The merge_threshold argument set the max distance that two bounding boxes must have
    to be considered as the same face result. The distance is evaluated from the centers
    of the bounding boxes.

    In order to have a size-independent merge_threshold, the difference of the pixels on 
    both axes are normalized to the minimum between width and height of the image.
*/
bool merge_similar_bb(Rectangle* r, List* faces, float merge_threshold)
{
    Node* curr;
    curr = faces->head;

    Rectangle* s;

    uint diff_x, diff_y;
    float distance;

    int a,b,c,d;

    uint normalize_factor = min(im.width, im.height);

    // Iterate until the end of the list is reached
    while(curr != NULL)
    {
        // Get the rectangle from the current list node
        s = curr->elem;

        // Calculate the difference of the position (between r and s) for both axes and normalize them
        diff_x = round((abs((r->x + floor( r->size.width / 2.0f)) - (s->x + floor( s->size.width / 2.0f))) / (float) normalize_factor) * 100);
        diff_y = round((abs((r->y + floor( r->size.height / 2.0f)) - (s->y + floor(  s->size.height / 2.0f))) / (float) normalize_factor) * 100);

        // Get the distance as the Euclidean distance 
        distance = sqrt(diff_x * diff_x + diff_y * diff_y);

        // Check the distance between their centers 
        if (distance <= merge_threshold)
        {
            // Merge the bounding boxes by calculating a new one that contains both

            // Get the verteces of the new box
            a = min(r->x, s->x);
            b = max(r->x + r->size.width, s->x + s->size.width);
            c = min(r->y, s->y);
            d = max(r->y + r->size.height, s->y + s->size.height);

            // Assign the data of the new box to the element inside the list
            s->x = a;
            s->y = c;
            s->size.width = b - a;
            s->size.height = d - c;

            // Notify that the merging was executed, meaning there was at least one box 
            // inside the list that contain the same face of r
            return true;
        }
        // If the two boxes are different, proceed to the next element
        curr = curr->next;
    }
    return false;
}

/*
    FACE GROUPING

    This is the main function for the results grouping algorithm. It have to find and merge 
    the results that contain the same face, in order to have a single box for each face detected.

    The algorithm pops a result from the original list, and compare it with all the others remaining:
    if an equal box (where equal means with near centers) was found, the current one was merged with the
    other so it can be deleted, otherwise it is the only one that contains that face and is added to the
    grouped list. This operation is repeated until the original list isn't empty. 
*/
List* group_rectangles(List* faces, float merge_threshold)
{
    // Check if the argument is null
    if (!faces)
        return NULL;

    // If the received list contains only one element, merging is not needed
    if (faces->size < 2)
        return faces;

    // Initlialize the new list
    List* grouped = listInit();

    Rectangle* r, *s;

    // Repeat until only one element remains in the old list
    while(faces->size > 1)
    {
        // Pop an element from the old list
        r = remove_from_head(faces);
        
        // Find a similar box and merge to it
        if (merge_similar_bb(r, faces, merge_threshold))
            // If r was merged, it can be deleted because a new box that contains r was created
            free(r);
        else
            // Else r is the only box that contains that face, so add to the new list
            add(grouped, r);
    }
    // Add to the new list the remaining element in the old one
    add(grouped, remove_from_head(faces));

    // Return the new list
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
*/
List* detect_multiple_faces(pel** image, float scaleFactor, int minWindow, int maxWindow, float group_factor)
{
    // Check if the image was corretly loaded
    if (image == NULL)
        return NULL;
        
    // Initialize the results list
    List* faces = listInit();
    printf("\n list init completed");

    // Set maxWindow due to unacceptable or default value 
    if ( maxWindow < WINDOW_SIZE )
        maxWindow = min(im.width, im.height);
    
    printf("\nmaxWindow = %d", maxWindow);

    float currFactor;
    int iteration = 0;
    printf("\ncreating image pyramid...");
    for (currFactor = 1; ; currFactor*= scaleFactor, iteration++)
    {
        // Get the virtual size of the window of the current iteration
        int curr_winSize = round(WINDOW_SIZE * currFactor);

        // If the currenet window is smaller than the choosen minimum, continue 
        // to the next iteration
        if (minWindow > curr_winSize)
            continue;

        // If the max window size is reached, interrupt the computation
        if (maxWindow < curr_winSize)
            break;

        // Calculate the size of the resized image in this iteration
        Size temp_size = { round(im.width / currFactor), round(im.height / currFactor) };

        printf("\niteration n: %d | resizing factor: %f | height = %u, width = %u", iteration, currFactor, temp_size.height, temp_size.width);

        double** iim, **squared_iim;

        // Calculate the integral and squared integral images
        integral_image(image, temp_size, &iim, &squared_iim, currFactor, im.bitColor);

        // Run the face detection algorithm on the current image
        evaluate(iim, squared_iim, temp_size, curr_winSize, currFactor, faces, group_factor);

        // Free the current integral and squared integral images
        free_integral_image(iim, temp_size);
        free_integral_image(squared_iim, temp_size);
    }

    // Group the results to merge the ones that contain the same face
    List* grouped = group_rectangles(faces, group_factor);

    // Delete the old list
    free(faces);

    // Return the final results list
    return grouped;
}

/// LINKED LIST FUNCTIONS

/*
    Initialize the linked list and return its reference
*/
List* listInit()
{
    List* list = malloc(sizeof(List));
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