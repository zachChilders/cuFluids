//-----------------------------------------------------------------------------
//	CS 790-058 GPGPU
//	Final Project (Point Location using CPU)
//	
//	This file contains CPU version of algorithms
//	useful for double checking the GPU versions 
//	of algorithms for correctness.
//	
//	by Shawn Brown (shawndb@cs.unc.edu)
//-----------------------------------------------------------------------------

//-------------------------------------
//
// Includes
//
//-------------------------------------

// Standard Includes
#ifndef _INC_STDLIB
	#include <stdlib.h>
#endif
#ifndef _INC_STDIO
	#include <stdio.h>
#endif
#ifndef _STRING_
	#include <string.h>
#endif
#ifndef _INC_MATH
	#include <math.h>
#endif

// Cuda Includes
#include <cutil.h>
#include <vector_types.h>	

// App Includes
#include "Main.h"
#include "QueryResult.h"
#include "KDTree_CPU.h"
#include "KDTree_GPU.h"


//-------------------------------------
//
// Function Definitions
//
//-------------------------------------

//---------------------------------------------------------
//	Name:	PLQ_CPU_BF_DIST
//	Desc:	Computes distance between each point
//			in point vector and query point
//	Note:	algorithm done on CPU
//			as check on GPU algorithm
//---------------------------------------------------------

void PLQ_CPU_BF_DIST
(
		  float*   dists,		// OUT: 'dists', solution vector, dist from each point to qp
	const float4*  points,		// IN:  'points' vector, n elements in length
	const float4 & queryPoint,	// IN:  'qp' point to locate
	unsigned int   w,			// IN:  'W' number of cols in 2D padded point Vector
	unsigned int   h			// IN:	'H' number of rows in 2D padded point Vector
)
{
	unsigned int n = w * h;
	unsigned int i;
	float d, d2;

	for (i = 0; i < n; i++)	// Iterate over all elements
	{
		// Get Difference Vector between p[i] and queryPoint
		float4 diff;
		diff.x = points[i].x - queryPoint.x;
		diff.y = points[i].y - queryPoint.y;
		diff.z = points[i].z - queryPoint.z;
		diff.w = 0.0f;

		// Compute Distance between p[i] and queryPoint
		d2 = (diff.x * diff.x) +
		     (diff.y * diff.y) +
		     (diff.z * diff.z);
		d = sqrt( d2 );

		// Save Result to dists vector
		dists[i] = d;
	}

	// Success
}


//---------------------------------------------------------
//	Name:	PLQ_CPU_BF_DIST_MIN
//	Desc:	Finds Point with Min distance to query point
//	Note:	
//	1. algorithm done on CPU
//	   as check on GPU algorithm
//---------------------------------------------------------

void PLQ_CPU_BF_DIST_MIN
(
	unsigned int & closestIndex,	// 'closestIndex', index of closest point to query point
		   float & closestDist,		// OUT: 'closestDist', distance between closest point and query point
	const float4*  points,			// IN:  'points' vector, n elements in length
	const float4 & queryPoint,		// IN:  'qp' point to locate
	unsigned int   n				// IN:  'n', numbe of points in solution vector
)
{
	unsigned int i, bestIdx;
	float d, d2, bestDist;
	float4 diff;

	// Check Parameters
	if ((NULL == points) || (n == 0))
	{
		// Error - invalid parameters
		return;
	}

	// Compute Distance to 1st Point
	diff.x = points[0].x - queryPoint.x;
	diff.y = points[0].y - queryPoint.y;
	diff.z = points[0].z - queryPoint.z;
	diff.w = 0.0f;

	d2 = (diff.x * diff.x) +
	     (diff.y * diff.y) +
	     (diff.z * diff.z);
	d = sqrt( d2 );

	// Start off assuming 1st point is closest to query point
	bestDist = d;
	bestIdx  = 0;

	// Brute force compare of all remaining points
	for (i = 1; i < n; i++)	
	{
		// Get Difference Vector between p[i] and queryPoint
		diff.x = points[i].x - queryPoint.x;
		diff.y = points[i].y - queryPoint.y;
		diff.z = points[i].z - queryPoint.z;
		diff.w = 0.0f;

		// Compute Distance between p[i] and queryPoint
		d2 = (diff.x * diff.x) +
		     (diff.y * diff.y) +
		     (diff.z * diff.z);
		d = sqrtf( d2 );

		// Is this point closer than current best point?
		if (d < bestDist)
		{
			// Found a closer point, update
			bestDist = d;
			bestIdx  = i;
		}
	}

	// Success, return closest point (& distance)
	closestIndex  = bestIdx;
	closestDist   = bestDist;
}



//---------------------------------------------------------
//	Name:	PLQ_CPU_KDTREE_FIND_2D
//	Desc:	Computes distance between each point
//			in point vector and query point
//	Note:	algorithm done on CPU
//			as check on GPU algorithm
//---------------------------------------------------------

bool PLQ_CPU_KDTREE_FIND_2D
(
    void            * kdTree,			// IN - KD Tree
	unsigned int      cSearch,			// IN - Count of items in search list
	const float4    * searchList,		// IN - Points to search
	unsigned int      cQuery,			// IN - count of items in query list
	const float4    * queryList,		// IN - Points to Query
	CPU_NN_Result * resultList		// OUT - Result List
)
{
	// Check Parameters
	KDTree_CPU * cpuKDTree = static_cast<KDTree_CPU *>( kdTree );
	if (NULL == cpuKDTree) { return false; }
	if (cSearch <= 0) { return false; }
	if (cQuery <= 0) { return false; }

	bool bResult;
	bResult = cpuKDTree->FindClosestPoint_V2( resultList, cQuery, queryList );
	if (false == bResult)
	{
		// Error - ???
		return false;
	}

	/*
	// Make Sure Results list is large enough to hold results
	unsigned int cResult = cQuery;

	unsigned int idxVal, idVal;
	float distVal;
	bool bResult = true;

	// Search All Query Points for closest points 
	unsigned int i;
	for (i = 0; i < cQuery; i++)
	{
		// Get Current Query Point
		const float4 & currQuery = queryList[i];

		// Search KDTree for closest point
		bResult = cpuKDTree->FindClosestPoint2DAlt( currQuery, idxVal, idVal, distVal );
		if (! bResult)
		{
			// Error
			return false;
		}

		// Store Results in Result List
		resultList[i].id   = idVal;
		resultList[i].dist = distVal;
	}
	*/

	// Success
	return true;
}


//---------------------------------------------------------
//	Name:	PLQ_CPU_KDTREE_NN
//	Desc:	Finds 'k' nearest points to each query point
//	Note:	algorithm done on CPU
//			as check on GPU algorithm
//---------------------------------------------------------

bool PLQ_CPU_KDTREE_KNN
(
    void            * kdTree,			// IN - KD Tree
	unsigned int      cSearch,			// IN - Count of items in search list
	const float4    * searchList,		// IN - Points to search
	unsigned int      cQuery,			// IN - count of items in query list
	unsigned int      cPadQuery,		// IN - padded count ...
	const float4    * queryList,		// IN - Points to Query
	int               kVal,				// IN - kVal
	CPU_NN_Result * resultList		// OUT - Result List
)
{
	// Check Parameters
	KDTree_CPU * cpuKDTree = static_cast<KDTree_CPU *>( kdTree );
	if (NULL == cpuKDTree) { return false; }
	if (cSearch <= 0) { return false; }
	if (cQuery <= 0) { return false; }

	bool bResult;
	bResult = cpuKDTree->Find_K_ClosestPoints_V2( resultList, kVal, cQuery, cPadQuery, queryList );
	if (false == bResult)
	{
		// Error - ???
		return false;
	}

	/*
	// Make Sure Results list is large enough to hold results
	unsigned int cResult = cQuery;

	unsigned int idxVal, idVal;
	float distVal;
	bool bResult = true;

	// Search All Query Points for closest points 
	unsigned int i;
	for (i = 0; i < cQuery; i++)
	{
		// Get Current Query Point
		const float4 & currQuery = queryList[i];

		// Search KDTree for closest point
		bResult = cpuKDTree->FindClosestPoint2DAlt( currQuery, idxVal, idVal, distVal );
		if (! bResult)
		{
			// Error
			return false;
		}

		// Store Results in Result List
		resultList[i].id   = idVal;
		resultList[i].dist = distVal;
	}
	*/

	// Success
	return true;
}


bool BUILD_KD_TREE
(
	void        ** kdTree,			// IN/OUT - KDTree pointer
	unsigned int   nSearch,			// IN - Number of Points
	const float4 * search_CPU		// IN -  CPU Search Point List
)
{
	*kdTree = NULL;
	KDTree_CPU * cpuKDTree = new KDTree_CPU();
	if (NULL == cpuKDTree) { return false; }

	// Build KD Tree
	bool bResult;
	bResult = cpuKDTree->Build2D( nSearch, search_CPU );
	if (false == bResult)
	{
		delete cpuKDTree;
	}

	// Success
	*kdTree = static_cast<void *>( cpuKDTree );
	return true;
}

bool FINI_KD_TREE
(
	void ** kdTree		// IN/OUT - KD Tree pointer
)
{
	KDTree_CPU * cpuKDTree = static_cast<KDTree_CPU *>( *kdTree );
	if (NULL != cpuKDTree)
	{
		delete cpuKDTree;
	}
	*kdTree = NULL;

	// Success
	return true;
}

bool BUILD_GPU_NODES_V1
(
	void *          kdTree,			// IN - KDTree pointer
	unsigned int	nSearch,		// IN - Count of items in search list
	const float4 *  search_CPU,		// IN -  CPU Search Point List
	void *          nodes_GPU		// OUT - GPU Node List
)
{
	KDTree_CPU * cpuKDTree = static_cast<KDTree_CPU *>( kdTree );
	KDTreeNode_GPU_V1 * gpuNodes = static_cast<KDTreeNode_GPU_V1 *>( nodes_GPU );

	if (NULL == cpuKDTree) { return false; }
	if (NULL == gpuNodes)  { return false; }

	// Copy KD Tree Nodes into GPU list
	unsigned int i;
	for (i = 0; i < nSearch; i++)
	{
		KDTreeNode_CPU    * currCPU = cpuKDTree->NODE_PTR( i );
		KDTreeNode_GPU_V1 * currGPU = &(gpuNodes[i]);
		currGPU->pos[0] = currCPU->X();
		currGPU->pos[1] = currCPU->Y();
		currGPU->pos[2] = currCPU->Z();
		currGPU->ID     = currCPU->ID();
		currGPU->Parent = currCPU->Parent();
		currGPU->Left   = currCPU->Left();
		currGPU->Right  = currCPU->Right();
		currGPU->SplitAxis = currCPU->Axis();
	}

	// Success
	return true;
}


bool BUILD_GPU_NODES_V2
(
	void *          kdTree,			// IN - KDTree pointer
	unsigned int	nSearch,		// IN - Count of items in search list
	const float4 *  search_CPU,		// IN -  CPU Search Point List
	void *          nodes_GPU,		// OUT - GPU Node List
	unsigned int *  ids_GPU			// OUT - ID list for GPU nodes
)
{
	KDTree_CPU          * cpuKDTree = static_cast<KDTree_CPU *>( kdTree );
	GPUNode_2D_MED * gpuNodes = static_cast<GPUNode_2D_MED *>( nodes_GPU );

	if (NULL == cpuKDTree) { return false; }
	if (NULL == gpuNodes)  { return false; }
	if (NULL == ids_GPU)   { return false; }

	// Copy KD Tree Nodes into GPU list
	unsigned int i;
	for (i = 0; i < nSearch; i++)
	{
		KDTreeNode_CPU      * currCPU = cpuKDTree->NODE_PTR( i );
		GPUNode_2D_MED * currGPU = &(gpuNodes[i]);

		// Set GPU Node values
		currGPU->pos[0] = currCPU->X();
		currGPU->pos[1] = currCPU->Y();
		currGPU->Left   = currCPU->Left();
		currGPU->Right  = currCPU->Right();

		// Set ID for current Node
		ids_GPU[i] = currCPU->ID();
	}

	// Success
	return true;
}
