#include <stdio.h>
#include <stdlib.h>

typedef struct Node{
    int data;
    struct Node *next;
} Node;

Node *initList(){
    Node *head = (Node*)malloc(sizeof(Node));
    head->data = 1;
    head->next = NULL;
    return head;
}

int main() {
    Node *head = initList();
    Node *p = head;
    while(p!= NULL){
        printf("%d ", p->data);
        p = p->next;
    }
    printf("\n");
    return 0;
}