#pragma once

// This is a KDTree implementation.  It's meant to be 
// instantiated and initialized on the CPU and then
// pushed to the GPU after.  
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "cuda.h"
#include "cuda_runtime_api.h"

#include "Point.h"
#include "Box.h"

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
		~KDTree();

		std::vector<Point3D> points;  //Vectors are guaranteed contiguous.
		std::vector<Box*> boxen;

		//Methods
		void insert(Point3D* point);
		std::vector<Point3D> flatten();
		std::vector<Box*> createBounds(Point3D* currentPoint);
		friend std::ostream& operator<<(std::ostream& out, KDTree& kd);

	private:
		Point3D root;

		void _insert(Point3D *point, Point3D *root);
		static void _bfs(Point3D *point, std::vector<Point3D> *v);
};

KDTree::KDTree()
{
	root = Point3D(0, 0, 0);
}

KDTree::~KDTree()
{
	//this needs to cuda free everything too
	points.clear();
	for (auto b : boxen)
	{
		delete b;
	}
};

//std::vector<Box*> KDTree::createBounds(Point3D* currentPoint)
//{
//	//Set bounds of root box
//	Box* newBox;
//
//	boxen.push_back(newBox);
//
//	//For each point, draw a box
//	
//
//
//
//
//}

void KDTree::insert(Point3D* point)
{
	point->numChildren++;
	if (*point < root)
	{
		if (root.left == nullptr)
		{
			root.left = point;
		}
		else
		{	
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
			_insert(point, root.right);
		}
	}
}

void KDTree::_insert(Point3D *point, Point3D *currNode)
{
	point->numChildren++;
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
	//points = p;
	return p;
}
