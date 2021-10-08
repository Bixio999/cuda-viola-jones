#include "classifier.h"

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
