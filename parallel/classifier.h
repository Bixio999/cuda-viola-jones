#include <stdbool.h>

#include "image.h"

#define WINDOW_SIZE 24

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
    Rectangle* elem = NULL;
    struct node* next = NULL;
} Node;

typedef struct list {
    Node* head = NULL;
    Node* tail = NULL;
    int size = 0;
} List;

List* listInit();
void add(List* list, Rectangle* r);
Rectangle* remove_from_head(List* list);

bool load_classifier_to_gpu(const char* classifier_path, const char* config_path);
List* detect_multiple_faces(pel* image, float scaleFactor, int minWindow, int maxWindow, unsigned int* dev_face_counter, float group_factor);

