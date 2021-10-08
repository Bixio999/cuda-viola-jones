#include <stdbool.h>

#include "image.h"

#define WINDOW_WIDTH 24
#define WINDOW_HEIGHT 24

typedef struct size {
    int width;
    int height;
} Size;

typedef struct rect
{
    int x;
    int y;
    Size size;
} Rectangle;

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

Classifier* load_classifier(const char* classifier_path, const char* config_path);
// Rectangle* detect_single_face(Classifier* classifier, pel** image, float scaleFactor, Size minWindow, Size maxWindow);
List* detect_multiple_faces(Classifier* classifier, pel** image, float scaleFactor, int minWindow, int maxWindow);

