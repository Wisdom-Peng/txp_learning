#include <iostream>
using namespace std;

int main(){
    int arr[] ={1,2,33,44,45,46};
    int *ptr_arr;

    ptr_arr = arr;
    cout << "arr value :" << arr << endl;
    cout << "ptr_arr value :" << ptr_arr << endl;
    for (int i=0; i < sizeof(arr)/sizeof(arr[0]);i++){
        cout << "arr["<<i<<"] value :" << arr[i] << endl;
        cout << "*ptr_arr value :" << *(ptr_arr+i) << endl;
        cout << "ptr_arr["<<i<<"] value :" << ptr_arr[i] << endl;
        
    };
    
}