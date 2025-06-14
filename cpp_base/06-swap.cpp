#include <iostream>

using namespace std;

/*
1.直接main函数里面交换
*/
// int main() {
//     int a = 10;
//     int b = 20;
//     int temp ;

//     cout << "交换前: a = " << a << ", b = " << b << endl;
//     temp = a;
//     a = b;
//     b = temp;
//     cout << "交换后: a = " << a << ", b = " << b << endl;
    
// }

/*
2.swap交换函数，形参为一般变量
*/
// int swap(int a, int b) {
//     int temp = a;
//     a = b;
//     b = temp;
//     cout << "a = "<< a << ", b= "<< b << endl;
//     return 0;
// }
// int main() {
//     int a = 10;
//     int b = 20;
//     cout << "交换前: a = " << a << ", b = " << b << endl;   
//     swap(a, b);
//     cout << "交换后: a = " << a << ", b = " << b << endl;
// }
/*
3.swap交换函数，形参为指针变量
*/
int swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
    cout << "a = "<< a << ", b= "<< b << endl;
    cout << "*a = "<< *a << ", *b= "<< *b << endl;
    return 0;
}
int main() {
    int a = 10;
    int b = 20;
    cout << "交换前: a = " << a << ", b = " << b << endl;
    cout << "交换前: a的地址 = " << &a << ", b的地址 = " << &b << endl;   
    swap(&a, &b);
    cout << "交换后: a = " << a << ", b = " << b << endl;
}


