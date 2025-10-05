#include <iostream>

#define MAXSIZE 100
using namespace std;
typedef int ElemType;

typedef struct {
    ElemType data[MAXSIZE];
    int length;
} SeqList;

void initList(SeqList *L) {
    new (L) SeqList;
    L->length = 0;
    cout << "Init List success" << endl;
}
int length(SeqList L) {
    return L.length;
}
bool isEmpty(SeqList L) {
    return L.length == 0;
}
bool isFull(SeqList L) {
    return L.length == MAXSIZE;
}
void printList(SeqList L) {
    for (int i = 0; i < L.length; i++) {
        std::cout << L.data[i] << " ";
    }
    std::cout << std::endl;
}
void insert(SeqList *L, int i, ElemType e) {
    if (isFull(*L)) {   // 检查是否已满
        std::cout << "List is full" << std::endl;
        return;
    }
    if (i < 0 || i > L->length) {   // 检查索引是否合法
        std::cout << "Index is invalid" << std::endl;
        return;
    }
    for (int j = L->length - 1; j >= i; j--) {   // 后移元素
        L->data[j + 1] = L->data[j];
    }
    L->data[i] = e;   // 插入元素
    L->length++;
}
void remove(SeqList *L, int i) {
    if (isEmpty(*L)) {   // 检查是否为空
        std::cout << "List is empty" << std::endl;
        return;
    }
    if (i < 0 || i >= L->length) {   // 检查索引是否合法
        std::cout << "Index is invalid" << std::endl;
        return;
    }
    for (int j = i; j < L->length - 1; j++) {   // 前移元素
        L->data[j] = L->data[j + 1];
    }   
    L->length--;
    if (L->length == 0) {   // 若为空，则初始化
        initList(L);
    }
}
void remove_1(SeqList *L, int pos, ElemType *e){
    *e = L->data[pos-1];
    if (pos < L->length ){
        for (int i = pos; i < L->length; i++){
            L->data[i-1] = L->data[i];
        }
        L->length--;
    }

}

void findelem(SeqList L, ElemType e ) {
    int flag = 0;
    for(int i = 0; i < L.length; i++) {
        
        if(L.data[i] == e) {
            std::cout << "Find elem: " << e << " in pos: " << i + 1<< std::endl;
            flag++;
        }
        
    }
    if(flag == 0) {
            std::cout << "Not find elem: " << e << std::endl; 
    }  
}


int main() {
    SeqList L;
    initList(&L);
    insert(&L, 0, 1);
    insert(&L, 1, 2);
    insert(&L, 2, 3);
    insert(&L, 3, 4);
    insert(&L, 4, 5);
    insert(&L, 5, 6);
    printList(L);
    remove(&L, 2);
    printList(L);
    remove_1(&L, 2, &L.data[2]);
    printList(L);
    findelem(L, 3);
    cout <<"***************"<<endl;
    findelem(L, 5);
    int len = length(L);
    std::cout << "Length: " << len << std::endl;
    return 0;   
}