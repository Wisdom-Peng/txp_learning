#include <iostream>
using namespace std;

struct Person {
    int age;
    string name;
    float height;
};
struct Person p1 = {25, "Alice", 1.65};

struct point {
    int x;
    int y;
};

struct point createstruct(int x, int y)
{
    struct point p;
    p.x = x;
    p.y = y;
    return p;   
    
}


int main() {
    // /*
    // 1.init
    // */
    // cout << "p1的年龄是：" << p1.age << endl;
    // cout << "p1的名字是：" << p1.name << endl;
    // cout << "p1的身高是：" << p1.height << endl;

    // p1.age = 26;
    // p1.name = "Bob";
    // p1.height = 1.70;

    // cout << "p1的年龄是：" << p1.age << endl;   
    // cout << "p1的名字是：" << p1.name << endl;
    // cout << "p1的身高是：" << p1.height << endl;

    /*
    2.
    */
   point p = createstruct(10, 20);
   cout << "p的x坐标是：" << p.x << endl;
   cout << "p的y坐标是：" << p.y << endl;

   #include <typeinfo>
   cout << "p的数据类型是：" << typeid(p).name() << endl;
   cout << "p.x的数据类型是：" << typeid(p.x).name() << endl;

   

}