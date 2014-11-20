#pragma once

// This is a KDTree implementation.  It's meant to be 
// instantiated and initialized on the CPU and then
// pushed to the GPU after.  

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Point.h"
#include "Box.h"

#include <iostream>
#include <thread>
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
		
		void _insert(Point3D *point, Point3D *root);

		void _bfs(Point3D *point, std::vector<Point3D> *v);
};

KDTree::KDTree(Point3D* list, int numParticles)
{
	root = Point3D(0, 0, 0);
	for (int i = 0; i < numParticles; i++)
	{
		points.push_back(list[i]);
	}
};

KDTree::KDTree()
{
	root = Point3D(0, 0, 0);
}

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
		_insert(&point, point.left);
	}
	else if (point.position.x > root.position.x)
	{
		_insert(&point, point.right);
	}

}

void KDTree::_insert(Point3D *point, Point3D *root)
{
	root->currentDimension = (root->currentDimension + 1) % 3;
	if (point < root)
	{
		if (root->left == nullptr)
		{
			root->left = point;
		}
		else
		{
			_insert(point, point->left);
		}
	}
	else if (point > root)
	{
		if (root->right == nullptr)
		{
			root->right = point;
		}
		else
		{
			_insert(point, point->right);
		}
	}
}

void KDTree::_bfs(Point3D *point, std::vector<Point3D> *v)
{


	
}

std::vector<Point3D> KDTree::flatten()
{
	std::vector<Point3D> p; //Create an empty vector
	p.push_back(root); // Add our node
	if (root.left != nullptr)
	{
		std::thread t1(&KDTree::_bfs, &root.left, std::ref(p));
		t1.join();
	}
/*
	if (root.right != nullptr)
	{
		std::thread t2(&KDTree::_bfs, &root.right, std::ref(p));
		t2.join();
	}*/
	return p;
}
