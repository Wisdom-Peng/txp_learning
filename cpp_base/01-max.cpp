#include <iostream>
using namespace std;
int max(int a,int b){
    return a>b?a:b;
}   
int main(){
    int a,b;
    cout<<"Please input two numbers:"<<endl;
    cin>>a>>b;
    cout<<"The max number is:"<<endl;
    cout<<max(a,b)<<endl;
}