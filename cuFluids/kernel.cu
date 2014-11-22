
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
	KDTree k;

	for (int i = 0; i < 10; i++)
	{
		Point3D p(i, i, i);
		k.insert(&p);
	}



	std::vector <Point3D> a = k.flatten();
	for (auto b : a)
	{
		std::cout << b << std::endl;
	}

	system("PAUSE");

}
