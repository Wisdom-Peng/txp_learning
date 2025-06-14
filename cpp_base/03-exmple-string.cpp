#include <iostream>
#include <string>
using namespace std;
int main()  
{
    string str1 = "Hello";
    string str2 = "World";
    string str3 = str1 + " " + str2;  
    string str4 = str3.substr(6, 5);
    cout << "str3: " << str3 << "\t str3 length: " << str3.length() << endl;
    cout << "str4: " << str4 << "\t str4 length: " << str4.length() << endl;
    return 0;
}