#pragma once

#include <math.h>
#include <vector>
#include <deque>
#include <string>

#include "Point.h"
#include "Box.h"

template
<typename T>
class Heap
{
	public:
		typedef struct _closeNode
		{
			unsigned int nodeIndex; //Original Node
			float dist;	//Distance to the next node
		}CloseNode;

		typedef std::vector<CloseNode> closeNodeList;

	protected:
		closeNodeList closeNodes; //Array of close nodes
		int maxNodes; //Max nodes allowed in this set
		int currNodes; //current number of nodes in the set
		float maxDist2;
		float currDist2;

		
		void reset()
		{
			closeNodes.clear();
			maxNodes = 0;
			currNodes = 0;
			maxDist2 = static_cast<float> (FLT_MAX);
			currDist2 = static_cast<float> (FLT_MAX);
		}

		bool copy(const Heap<T> & toCopy);

		void swap(int i, int j);

		void promote(int currIndex);
		void demote(int currIndex);

		void makeHeap();

		void replace(const T newNode, float newDist2);

	public:
		int getMaxNodes() const;

		int getCurrNodes() const;
		
		float cutoffDist2() const;
		void cutoffDist2(float value);

		float currMaxDist2() const;
		void currMaxDist2(float value);

		Heap<T>();
		Heap<T>(int maxNodes);
		Heap<T>(int maxNodes, float maxDist2);

		~Heap<T>();

		void insert(T node, float dis2);

		void clear();

		bool getNodes(closeNodeList &resultes) const;
		
};