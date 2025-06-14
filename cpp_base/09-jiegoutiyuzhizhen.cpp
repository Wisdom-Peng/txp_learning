#include <iostream>
using namespace std;

struct point {
    int x;
    int y;
};

point *pp;

typedef struct {
    int x;
    int y;
    int z;
} point2;



int main() {
    pp = (point*)malloc(sizeof(point));
    // pp = new point;
    (*pp).x = 10;
    (*pp).y = 20;
    cout << "pp = " << pp << endl;
    cout << "x = " << (*pp).x << endl;
    cout << "y = " << (*pp).y << endl;

    cout << "x = " << pp->x << endl;
    cout << "y = " << pp->y << endl;
    

    point2 *pp2;
    point2 p;
    pp2 = &p;
    pp2 -> x = 100;
    pp2 -> y = 200;
    pp2 -> z = 300;
    cout << "pp2 = " << pp2 << endl;
    cout << "x = " << pp2 -> x << endl;
    cout << "y = " << pp2 -> y << endl;
    cout << "z = " << pp2 -> z << endl;

    free(pp);
    // delete pp;


    return 0;
};


