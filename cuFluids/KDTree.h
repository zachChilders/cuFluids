#pragma once

// This is a KDTree implementation.  It's meant to be 
// instantiated and initialized on the CPU and then
// pushed to the GPU after.  

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Point.h"

#include <iostream>
#include <thread>
#include <queue>
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
	void insert(Point3D* point);
	Point3D getPoint();
	//Point3D getNodeValue(Box<float> node);
	void medianSortNodes();
	void validate();
	Point3D* findKClosestPointIndices(Point3D* p);
	//int* findPointIndicesInRegion(Box<float>* b);
	Point3D* queryClosestPoints(Point3D* p);
	Point3D* queryClosestValues(Point3D* p);
	Point3D* queryKPoints(Point3D** p);
	std::vector<Point3D> KDTree::flatten();

	friend std::ostream& operator<<(std::ostream& out, KDTree& kd);

	private:
	Point3D root;

	std::vector<Point3D> points;  //Vectors are guaranteed contiguous.

	void _insert(Point3D *point, Point3D *root);
	static void _bfs(Point3D *point, std::vector<Point3D> *v);
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

void KDTree::insert(Point3D *point)
{
	//std::cout << *point << std::endl;

	if (*point < root)
	{
		if (root.left == nullptr)
		{
			root.left = point;
		}
		else
		{	

			std::cout << "A " << *point << " " << *root.left << std::endl;
			_insert(point, root.left);
		}
	}

	else if (*point > root)
	{
		if (root.right == nullptr)
		{
			root.right = point;
		}
		else
		{
			std::cout << *point << " " << *root.right << std::endl;
			_insert(point, root.right);
		}
	}
}

void KDTree::_insert(Point3D *point, Point3D *currNode)
{
	point->currentDimension = (point->currentDimension + 1) % 3;
	if (point < currNode)
	{
		if (currNode->left == nullptr)
		{
			currNode->left = point;
		}
		else
		{
			_insert(point, currNode->left);
		}
	}
	else if (point > currNode)
	{
		if (currNode->right == nullptr)
		{
			currNode->right = point;
		}
		else
		{
			_insert(point, currNode->right);
		}
	}
}

std::vector<Point3D> KDTree::flatten()
{
	std::vector<Point3D> p; //Create an empty vector
	std::queue<Point3D> q; //Helper queue to hold points

	p.push_back(root);
	q.emplace(root);

	//BFS
	while (!q.empty())
	{
		Point3D *temp = &q.front();
		q.pop();

		if (temp->left != nullptr)
		{
			p.push_back(*temp->left);
			q.emplace(*temp->left);
		}
		if (temp->right != nullptr)
		{
			p.push_back(*temp->right);
			q.emplace(*temp->right);
		}
			
	}

	return p;
}
