
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Point.h"
#include "Heap.h"
#include "KDTree.h"
#include "Box.h"

#include <iostream>

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
	Point3D p;
	Box<float> b;
	KDTree k;

	std::cout << "Point3D: " << sizeof(p) << std::endl;
	std::cout << "Box: " << sizeof(b) << std::endl;
	std::cout << "KDTree: " << sizeof(k) << std::endl;

	system("PAUSE");
	//Initialize a group of points

	//Build a tree of points

	//Heap all the points


}
