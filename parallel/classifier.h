#include <stdbool.h>

#include "image.h"

#define WINDOW_WIDTH 24
#define WINDOW_HEIGHT 24

typedef struct size {
    unsigned int width;
    unsigned int height;
} Size;

typedef struct rect
{
    short x;
    short y;
    Size size;
} Rectangle;

bool load_classifier_to_gpu(const char* classifier_path, const char* config_path);
Rectangle** detect_multiple_faces(pel* image, float scaleFactor, int minWindow, int maxWindow);

