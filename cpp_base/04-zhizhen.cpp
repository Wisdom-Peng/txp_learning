#include <iostream>
using namespace std;

int main() {
    // 实际变量
    printf("\n*************实际变量*************\n");
    int a = 10;
    cout << "a 的值: " << a << endl;
    cout << "a 的地址: " << &a << endl;

    // 指针变量
    printf("\n*************指针变量*************\n");
    int *p = &a;
    cout << "p 的值（a 的地址）: " << p << endl;
    cout << "p 指向的值: " << *p << endl;
    cout << "p 的地址: " << &p << endl;

    // 指针的算术运算
    printf("\n*************指针的算术运算*************\n");
    p++;
    cout << "p 的值（a 的地址 + 1）: " << p << endl;
    cout << "p 指向的值: " << *p << endl;

    // 指针的解引用
    printf("\n*************指针的解引用*************\n");
    int c = a;
    int cp = *p;
    cout << "c 的值: " << c << "\t c 的地址" << &c << endl;
    cout << "cp 的值: " << cp << "\t cp 的地址" << &cp << endl;

    // 指针数组
    printf("\n*************指针数组*************\n");
    int *arr[5];
    for (int i = 0; i < 5; i++) {
        a++;
        arr[i] = &a;
        cout << "arr[" << i << "] 的值: " << arr[i] << endl;
        cout << "arr[" << i << "] 指向的值: " << *arr[i] << endl;
    }

    // 数组指针
    printf("\n*************数组指针*************\n");


    // 二维指针
    printf("\n*************二维指针*************\n");
    int **pp = arr;
    cout << "pp 的值（arr 的地址）: " << pp << endl;
    cout << "pp 指向的值（arr 的第一个元素的地址）: " << *pp << endl;
    for (int i = 0; i < 5; i++) {
        cout << "pp[" << i << "] 的值: " << pp[i] << endl;
        cout << "pp[" << i << "] 指向的值: " << *pp[i] << endl;
    }

    // 指针 vs 数组
    printf("\n*************指针 vs 数组*************\n");
    const int MAX = 6;
    int arr1[MAX];
    for(int i = 0; i < MAX; i++) {
        *(arr1+i) = i*100;
        cout << "arr1 的值（数组的地址）: " << arr1 << endl;    // 输出数组的地址
        cout << "arr1[" << i << "] 的值（数组的第一个元素）: " << arr1[i] << endl;  // 输出数组的第一个元素
    }
    cout << "arr1 的值（数组的地址）: " << arr1 << endl;    // 输出数组的地址
    cout << "arr1[0] 的值（数组的第一个元素）: " << arr1[0] << endl;  // 输出数组的第一个元素
    cout << "arr1[1] 的值（数组的第二个元素）: " << arr1[1] << endl;  // 输出数组的第二个元素
    int *p2 = arr1; 
    for(int i = 0; i < MAX; i++) {
        cout << "p2的值: " << p2 << endl;
        cout << "p2指向的值: " << *p2 << endl;
        p2++;
    }  

    return 0;
}
