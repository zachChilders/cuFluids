#include "Box.h"

Box::Box()
{
	isLeaf = false;
	parent = nullptr;
	child1 = nullptr;
	child2 = nullptr;
	
	minXbound, minYbound, minZbound, maxXbound, maxYbound, maxZbound = 0;

}

Box::Box(float minX, float minY, float minZ, float maxX, float maxY, float maxZ)
{
	isLeaf = false;
	parent = nullptr;
	child1 = nullptr;
	child2 = nullptr;
}

Box::~Box(){};