#include <iostream>


using namespace std;
int main() {
    int a = 10;
    int *p = &a;
    printf("a 的值: %d\n", a);
    printf("a 的地址: %p\n", &a);
    printf("p 的值（a 的地址）: %p\n", p);
    printf("p 指向的值: %d\n", *p);

    *p = 20;
    printf("a 的值: %d\n", a);
    printf("p 指向的值: %d\n", *p);

    int arr[5] = {1, 2, 3, 4, 5};
    int *p_arr = arr;
    printf("arr 的值（数组的地址）: %p\n", arr);
    printf("p_arr 的值（数组的地址）: %p\n", p_arr);
    for(int i = 0; i < 5; i++) {
        printf("arr[%d] 的值: %d\n", i, arr[i]);
        printf("p_arr[%d] 的值: %d\n", i, p_arr[i]);
        printf("arr[%d] 的地址: %p\n", i, &arr[i]);
    }

}