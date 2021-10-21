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

bool load_classifier(const char* classifier_path, const char* config_path);
// Rectangle* detect_single_face(Classifier* classifier, pel** image, float scaleFactor, Size minWindow, Size maxWindow);
List* detect_multiple_faces(pel** image, float scaleFactor, int minWindow, int maxWindow);

