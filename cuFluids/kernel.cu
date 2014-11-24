
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Point.h"
#include "Heap.h"
#include "KDTree.h"

#include <iostream>

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
	KDTree k;

	std::cout << "Inserting Elements" << std::endl;
	for (int i = 1; i < 10; i++)
	{
		Point3D p = Point3D(i, i, i);
		k.insert(&p);
	}

	std::cout << "===============" << std::endl;
	std::cout << "Flattening tree" << std::endl;


	std::vector <Point3D> a = k.flatten();
	for (auto b : a)
	{
		std::cout << b << std::endl;
	}

	system("PAUSE");

}
