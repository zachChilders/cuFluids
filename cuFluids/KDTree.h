#pragma once

#include "kdUtil.h"
#include "Box.h"


template
<typename P, B>
class KDTree
{
	public:
		//Constructors
		KDTree();
		KDTree(P* list);
		~KDTree();

		//Methods
		P getPoint();
		B getNode();
		void swapNodes(B one, B two);
		P getNodeValue(B node);
		void medianSortNodes();
		void validate();
		P* findKClosestPointIndices(P* p);
		int* findPointIndicesInRegion(B* b);
		P* queryClosestPoints(P* p);
		P* queryClosestValues(P* p);
		P* queryKPoints(P** p);

		friend std::ostream& operator<<(ostream& out, KDTree& kd);

	private:
		
		P* points;

};