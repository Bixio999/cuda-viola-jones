#include "list.h"
#include <stddef.h>

List* listInit()
{
    List* list = malloc(sizeof(List));
    list->head = NULL;
    list->size = 0;

    return list;
}

void add(List* list, Rectangle* r)
{
    if (list)
    {
        Node node = {r, NULL};

        if (list->size == 0)
        {
            list->head = &node;
            list->size++;
            return;
        }

        node.next = list->head;
        list->head = &node;
        list->size++;
        return;
    }
    printf("\n Error: received null list parameter.");
}

Rectangle* remove_from_head(List* list)
{
    if (list)
    {
        if (list->size > 0)
        {
            Node* old = list->head;
            Rectangle* r = old->elem;
            free(&old);
            list->head = list->head->next;
            list->size--;
            return r;
        }
        printf("\n Error: trying to remove element with empty list.");
        return NULL;
    }
    printf("\n Error: received null list parameter.");
}