#include <stdbool.h>

typedef struct rect
{
    int x;
    int y;
    int width;
    int height;
}Rectangle;


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
    int n_stages;
} Classifier;

Classifier* load_classifier(const char* classifier_path, const char* config_path);
Rectangle* detect_single_face(Classifier* classifier, double** integral_image, bool multiscale);
Rectangle** detect_multiple_faces(Classifier* classifier, double** integral_image, bool multiscale);

