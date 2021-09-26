#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>

#include "classifier.h"

Classifier* load_classifier(const char* classifier_path, const char* config_path)
{
    if (classifier_path && config_path)
    {
        printf("\nnot null parameters: %s, %s", classifier_path, config_path);

        FILE* config_file = fopen(config_path, "r");

        if (!config_file)
        {
            printf("\nerror while reading from config file. aborting...");
            exit(1);
        }

        int stages = 0;
        fscanf(config_file, "%d", &stages);
        if (!stages)
        {
            printf("\nerror while reading from config file. aborting...");
            exit(1);
        }
        
        int* filter_per_stages = (int*) malloc(sizeof(int) * stages);

        int i = 0;
        int n_filters = 0;
        int* temp = filter_per_stages;
        while (i < stages && !feof(config_file))
        {
            fscanf(config_file, "%d", temp);
            i++;
            temp++;
        }

        fclose(config_file);

        FILE* classifier_file = fopen(classifier_path, "r");
        if (!classifier_file)
        {
            printf("\nerror while reading from classifier file. aborting...");
            exit(1);
        }

        Filter*** filters = (Filter***) malloc(sizeof(Filter**) * stages);

        int j;
        for (i = 0; i < stages; i++)
        {
            n_filters = filter_per_stages[i];
            filters[i] = (Filter**) malloc(sizeof(Filter*) * n_filters);

            for (j = 0; j < n_filters && !feof(classifier_file); j++)
            {
                Filter* f = malloc(sizeof(Filter));
                Rectangle rect = f->rect1;

                fscanf(classifier_file, "%d", &(rect.x)); 
                fscanf(classifier_file, "%d", &(rect.y)); 
                fscanf(classifier_file, "%d", &(rect.width)); 
                fscanf(classifier_file, "%d", &(rect.height));
                
                fscanf(classifier_file, "%d", &(f->weight1));

                rect = f->rect2;

                fscanf(classifier_file, "%d", &(rect.x)); 
                fscanf(classifier_file, "%d", &(rect.y)); 
                fscanf(classifier_file, "%d", &(rect.width)); 
                fscanf(classifier_file, "%d", &(rect.height));                

                fscanf(classifier_file, "%d", &(f->weight2));

                rect = f->rect3;

                fscanf(classifier_file, "%d", &(rect.x)); 
                fscanf(classifier_file, "%d", &(rect.y)); 
                fscanf(classifier_file, "%d", &(rect.width)); 
                fscanf(classifier_file, "%d", &(rect.height));                

                fscanf(classifier_file, "%d", &(f->weight3));    

                fscanf(classifier_file, "%d", &(f->threshold));                
                fscanf(classifier_file, "%d", &(f->alpha1));                
                fscanf(classifier_file, "%d", &(f->alpha2));           

                filters[i][j] = f;
            }
        }
        fclose(classifier_file);

        Classifier* classifier = malloc(sizeof(Classifier));
        classifier->filters = filters;
        classifier->filters_per_stages = filter_per_stages;
        classifier->n_stages = stages;

        return classifier;
    }
    else
        printf("\nnull parameters detected. aborting...");
    return NULL;
}

Rectangle* detect_single_face(Classifier* classifier, double** integral_image, bool multiscale)
{
    return NULL;
}

Rectangle** detect_multiple_faces(Classifier* classifier, double** integral_image, bool multiscale)
{
}