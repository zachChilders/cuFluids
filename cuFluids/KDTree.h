#pragma once

// This is a KDTree implementation.  It's meant to be 
// instantiated and initialized on the CPU and then
// pushed to the GPU after.  

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Point.h"
#include "Box.h"

#include <iostream>
#include <vector>

#define DIMENSIONS 3

class KDTree
{
	public:
		//Constructors
		KDTree();
		KDTree(Point3D* list, int numParticles);
		~KDTree();

		//Methods

		void insert(Point3D point);
		Point3D getPoint();
		Box<float> getNode();
		Point3D getNodeValue(Box<float> node);
		void medianSortNodes();
		void validate();
		Point3D* findKClosestPointIndices(Point3D* p);
		int* findPointIndicesInRegion(Box<float>* b);
		Point3D* queryClosestPoints(Point3D* p);
		Point3D* queryClosestValues(Point3D* p);
		Point3D* queryKPoints(Point3D** p);
		std::vector<Point3D> KDTree::flatten();

		friend std::ostream& operator<<(std::ostream& out, KDTree& kd);

	private:
		Point3D root;

		std::vector<Box<float>> boxes;
		std::vector<Point3D> points;  //Vectors are guaranteed contiguous.
		
		void _insert(Point3D *point, Point3D *root, int dimension);
		void _bfs(Point3D *point, std::vector<Point3D> v);
};

KDTree::KDTree(Point3D* list, int numParticles)
{
	root = Point3D(0, 0, 0);
	for (int i = 0; i < numParticles; i++)
	{
		points.push_back(list[i]);
	}
};

KDTree::~KDTree()
{
	//this needs to cuda free everything too
	points.clear();
};

void KDTree::insert(Point3D point)
{
	int k = 0;

	if (point[k] < root[k])
	{
		_insert(&point, point.left, k++);
	}
	else if (point.position.x > root.position.x)
	{
		_insert(&point, point.right, k++);
	}

}

void KDTree::_insert(Point3D *point, Point3D *root, int dimension)
{
	dimension = dimension % DIMENSIONS;
	if (point[dimension] < root[dimension])
	{
		if (root->left == nullptr)
		{
			root->left = point;
		}
		else
		{
			_insert(point, point->left, dimension++);
		}
	}
	else if (point[dimension] > root[dimension])
	{
		if (root->right == nullptr)
		{
			root->right = point;
		}
		else
		{
			_insert(point, point->right, dimension++);
		}
	}


}

std::vector<Point3D> KDTree::flatten()
{
	std::vector<Point3D> p; //Create an empty vector
	p.push_back(root); // Add our node
	_bfs(&root, p);  //Start bfs
	points = p; //Assign points to the now full vector

}

void KDTree::_bfs(Point3D *point, std::vector<Point3D> v)
{
	//This is depth.  Fix it.
	if (point->left != nullptr)
	{
		v.push_back(*point->left);
		_bfs(point->left, v);
	}
	else if (point->right != nullptr)
	{
		v.push_back(*point->right);
		_bfs(point->right, v);
	}
}