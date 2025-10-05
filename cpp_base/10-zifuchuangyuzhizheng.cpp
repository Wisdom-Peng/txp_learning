#include <iostream>
using namespace std;

int main() {
    // 使用字符数组表示字符串
    char str[] = "Hello, World!";

    // 使用指针指向字符串
    char *ptr = str;

    // 输出字符串
    cout << "字符串内容: " << ptr << endl;

    // 遍历字符串并输出每个字符
    cout << "逐个字符输出: ";
    while (*ptr != '\0') {
        cout << *ptr << " ";
        ptr++; // 指针移动到下一个字符
    }
    cout << endl;

    // 输出字符串的长度
    int length = 0;
    ptr = str; // 重置指针到字符串开头
    while (*ptr != '\0') {
        length++;
        ptr++;
    }
    cout << "字符串长度: " << length << endl;

    return 0;
}
