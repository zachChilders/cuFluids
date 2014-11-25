
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "KDTree.h"

#include <iostream>

#include <Windows.h>
#include <cuda_gl_interop.h>

__global__ void addKernel(Point3D* x)
{
    int i = threadIdx.x;
	x[i] + 1;
}

void cudaErrorCheck(cudaError_t e)
{
	if (e != cudaSuccess)
	{
		std::cout << "Failed." << std::endl;
	}
}

int main()
{
	cudaGraphicsResource_t cgr;
	
//	KDTree k;

	std::vector<Point3D> v;
	std::cout << "Inserting Elements" << std::endl;
	for (int i = 1; i < 10; i++)
	{
		Point3D p = Point3D(i, 0, 0);
		v.push_back(p);
	}

	for (auto b : v)
	{
		std::cout << b << std::endl;
	}
	std::cout << "==========" << std::endl;

	Point3D *devA;

	cudaError_t cudaStatus;
	cudaStatus = cudaMalloc((void**)&devA, v.size() * sizeof(Point3D));
	cudaErrorCheck(cudaStatus);
	cudaStatus = cudaMemcpy(devA, &v[0], v.size() * sizeof(Point3D), cudaMemcpyHostToDevice);
	cudaErrorCheck(cudaStatus);

	
	
/*
	std::cout << "===============" << std::endl;
	std::cout << "Flattening tree" << std::endl;*/


	//std::vector <Point3D> a = k.flatten();
	for (auto b : v)
	{
		std::cout << b << std::endl;
	}

	system("PAUSE");

}
