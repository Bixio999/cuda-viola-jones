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

typedef struct node {
    Rectangle* elem;
    struct node* next;
} Node;

typedef struct list {
    Node* head;
    int size;
} List;

List* listInit();
void add(List* list, Rectangle* r);
Rectangle* remove_from_head(List* list);

bool load_classifier_to_gpu(const char* classifier_path, const char* config_path);
List* detect_multiple_faces(pel* image, float scaleFactor, int minWindow, int maxWindow);

