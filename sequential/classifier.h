#include <stdbool.h>

#include "image.h"

// Define the window size used for classifier's training
#define WINDOW_SIZE 24

// UTILITY STRUCTURES

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

// LINKED LIST STRUCTURES

typedef struct node {
    Rectangle* elem;
    struct node* next;
} Node;

typedef struct list {
    Node* head;
    int size;
} List;

// FUNCTIONS PROTOTYPES

//  - List functions
List* listInit();
void add(List* list, Rectangle* r);
Rectangle* remove_from_head(List* list);

// - Face Detection functions
bool load_classifier(const char* classifier_path, const char* config_path);
List* detect_multiple_faces(pel** image, float scaleFactor, int minWindow, int maxWindow, float group_factor);

