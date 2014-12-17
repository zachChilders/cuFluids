#pragma once

#include "Point.h"
#include <vector>

CUDA_CALLABLE_MEMBER
class Box
{
	public:
		//Constructors
		CUDA_CALLABLE_MEMBER
		Box();
		CUDA_CALLABLE_MEMBER
		Box(float minX, float minY, float minZ, float maxX, float maxY, float maxZ);
		CUDA_CALLABLE_MEMBER
		~Box();

		bool isLeaf;

		std::vector<Point3D> points;
		
		Box* parent;
		Box* child1;
		Box* child2;

		float minXbound, minYbound, minZbound, maxXbound, maxYbound, maxZbound;

};
