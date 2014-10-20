#pragma once
#ifndef _GPUTREE_API_H
#define _GPUTREE_API_H
/*-----------------------------------------------------------------------------
  Name:  GPUTREE_API.h
  Desc:  Simple GPU KDTree structures and
		 GPU API function definitions

  Log:   Created by Shawn D. Brown (4/15/07)
		 Modified by Shawn D. Brown (3/22/10)
-----------------------------------------------------------------------------*/

/*-------------------------------------
  Includes
-------------------------------------*/

// Cuda Includes
#if !defined(__DRIVER_TYPES_H__)
	#include <driver_types.h>
#endif
#if !defined(__VECTOR_TYPES_H__)
	#include <vector_types.h>	
#endif
#ifndef _CUT_
	#include <cutil.h>
#endif

// Project Includes
#include "KD_Flags.h"		// Compiler Flags


/*-------------------------------------
  Structs
-------------------------------------*/

/*-----------------
  GPU Points
-----------------*/

typedef struct __align__(8)
{
	float pos[1];	// Position <x>
} GPU_Point1D;

typedef struct __align__(8)
{
	float pos[2];	// Position <x,y>
} GPU_Point2D;

typedef struct __align__(16)
{
	float pos[3];	// Position <x,y,z>
} GPU_Point3D;

typedef struct __align__(16)
{
	float pos[4];	// Position <x,y,z,w>
} GPU_Point4D;

typedef struct __align__(8)
{
	float pos[5];	// Position <x,y,z,w,s>
} GPU_Point5D;

typedef struct __align__(8)
{
	float pos[6];	// Position <x,y,z,w,s,t>
} GPU_Point6D;

typedef struct __align__(16)
{
	float pos[7];	// Position <x,y,z,w,s,t,u>
} GPU_Point7D;

typedef struct __align__(16)
{
	float pos[8];	// Position <x,y,z,w,s,t,u,v>
} GPU_Point8D;


/*---------------------------
  GPU Build Structures
---------------------------*/

// Phase 1 & 2:  stack (or queue) item for building kd-trees
typedef struct __align__(16)
{
	unsigned int start;			// [start, end] inclusive ...
	unsigned int end;			//              range of points to build
		// Note:  number of elements, n = (end - start) + 1
	unsigned int targetID;		// Target node index
	unsigned int flags;			// Build flags
		// unsigned int half;	// size of half a tree [0..27]	-- 28 bits (2^28 = 268+ million points)
		// unsigned int axis;	// curr axis           [28..30]	--  3 bits (2^3  = 8 dimensions)
		// unsigned int res1;	// reserved            [31]	    --  1 bit (reserved)
} GPU_BUILD_ITEM;

typedef struct __align__(8)
{
	unsigned int cRootReads,  cPivotReads;
	unsigned int cNodeLoops,  cPartLoops;
	unsigned int cCountReads, cCountWrites;
	unsigned int cPartReads,  cPartWrites;
	unsigned int cStoreReads, cStoreWrites;
	unsigned int cD2HReads,   cD2HWrites;
	unsigned int cH2DReads,   cH2DWrites;
} GPU_BUILD_STATS;


// Median pivot counts (per thread)
typedef struct __align__(16)
{
	unsigned int before;	// # of items before pivot value
	unsigned int after;	// # of items after pivot value
	unsigned int equal;	// # of items equal to pivot
} GPU_COUNTS_STARTS;



/*-----------------------------------------------
  GPU KD-Nodes (Left-balanced layout)
-----------------------------------------------*/

// 1D kd-tree node, left balanced tree (minimal)
typedef struct __align__(4)
{
	float		 pos[1];	// Position <x>
} GPUNode_1D_LBT;

// 2D kd-tree node, left balanced tree (minimal)
typedef struct __align__(8)
{
	float		 pos[2];	// Position <x,y>
} GPUNode_2D_LBT;

// 3D kd-tree node, left balanced tree (almost minimal)
typedef struct __align__(16)
{
	float		 pos[4];	// Position <x,y,z>
							// w is wasted space for alignment padding
} GPUNode_3D_LBT;

// 4D kd-tree node, left balanced tree (minimal)
typedef struct __align__(16)
{
	float		 pos[4];	// Position <x,y,z,w>
} GPUNode_4D_LBT;

// 5D kd-tree node, left balanced tree (minimal)
typedef struct __align__(8)
{
	float		 pos[5];	// Position <x,y,z,w,s>
} GPUNode_5D_LBT;

// 6D kd-tree node, left balanced tree (minimal)
typedef struct __align__(8)
{
	float		 pos[6];	// Position <x,y,z,w,s,t>
} GPUNode_6D_LBT;

// 7D kd-tree node, left balanced tree (minimal)
typedef struct __align__(16)
{
	float		 pos[7];	// Position <x,y,z,w,s,t,u>
} GPUNode_7D_LBT;

// 8D kd-tree node, left balanced tree (minimal)
typedef struct __align__(16)
{
	float		 pos[8];	// Position <x,y,z,w,s,t,u,v>
} GPUNode_8D_LBT;


/*-----------------------------------------------
  GPU KD-Nodes (Median layout)
-----------------------------------------------*/

typedef struct __align__(16)
{
	float pos[1];				// 1D Point
	unsigned int m_searchIdx;	// index of point in original search array
	unsigned int m_nodeIdx;		// index of node in kd-node array
} GPUNode_1D_MED;

typedef struct __align__(16)
{
	float pos[2];				// 2D Point
	unsigned int m_searchIdx;	// index of point in original search array
	unsigned int m_nodeIdx;		// index of node in kd-node array
} GPUNode_2D_MED;

typedef struct __align__(8)
{
	float pos[3];				// 3D Point
	unsigned int m_searchIdx;	// index of point in original search array
	unsigned int m_nodeIdx;		// index of node in kd-node array
} GPUNode_3D_MED;

typedef struct __align__(8)
{
	float pos[4];				// 4D Point
	unsigned int m_searchIdx;	// index of point in original search array
	unsigned int m_nodeIdx;		// index of node in kd-node array
} GPUNode_4D_MED;

typedef struct __align__(16)
{
	float pos[5];				// 5D Point
	unsigned int m_searchIdx;	// index of point in original search array
	unsigned int m_nodeIdx;		// index of node in kd-node array
} GPUNode_5D_MED;

typedef struct __align__(16)
{
	float pos[6];				// 6D Point
	unsigned int m_searchIdx;	// index of point in original search array
	unsigned int m_nodeIdx;		// index of node in kd-node array
} GPUNode_6D_MED;

typedef struct __align__(8)
{
	float pos[7];				// 7D Point
	unsigned int m_searchIdx;	// index of point in original search array
	unsigned int m_nodeIdx;		// index of node in kd-node array
} GPUNode_7D_MED;

typedef struct __align__(8)
{
	float pos[8];				// 8D Point
	unsigned int m_searchIdx;	// index of point in original search array
	unsigned int m_nodeIdx;		// index of node in kd-node array
} GPUNode_8D_MED;

/*-----------------
  GPU Search
-----------------*/

// Search Node for Depth First Search (DFS) using a stack
typedef struct __align__(8)
{
	unsigned int nodeFlags;	
		// Node Index	(Bits [0..27])  Limits us to at most 2^28 (268+ million) nodes in search list
		// Split Axis	(Bits [28..30])	{x,y, z,w, s,t, u,v} Up to 8 dimensions (nDim <= 8 = 2^3)
		// On/Off       (Bits [31])		(Onside,offside node tracking for trim optimization)
		// 
		// NOTE: See search node flags in KD_flags.h for bit masks and shifts
	float        splitVal;	// Split Value
} GPU_Search;


/*-----------------
  GPU Result
-----------------*/

// Query Result, improved version (smaller memory footprint)
//typedef struct __align__(16)
typedef struct __align__(8)
{
	unsigned int Id;			// IDx of Closest Point in search List
	float		 Dist;			// Distance to closest point in search list
	//unsigned int cVisited;	// Number of Nodes Visited during processing
	//unsigned int reserved;	// Dummy value for alignment padding
} GPU_NN_Result;



/*-------------------------------------
  GPU interface
-------------------------------------*/


	/*-------------------------------------------
	  Brute Force
	-------------------------------------------*/

	// BF Distance Kernel
__global__ void
ComputeDist3D_GPU
(
  float2* out,	// OUT: Result of 2D distance field calculations
  float4* in,	// IN:  Distance Vector Field
  float4  qp,	// IN:  Query point to compute distance for
  int     w,	// IN:  width of 2D vector field (# of columns)
  int     h		// IN:  height of 2D vector field (# of rows)
);

	// BF Reduction Kernel
__global__ void
Reduce_Min_GPU
(
	float2*  distOut,	// OUT:  Reduced Vector Field
	float2*  distIn		// IN:	 Distance Vector Field
);

	/*-------------------------------------------
	  Build Kernels
	-------------------------------------------*/

	// Pick Pivot (Phase 1)
__global__ void
GPU_2D_NODE_MED_PICK_PIVOT
(
	unsigned int * pivot,		// OUT - pivot result
	GPUNode_2D_MED * currNodes,	// IN - node list
	unsigned int start,			// IN - range [start,end] to median select
	unsigned int end,			
	unsigned int axis			// IN - axis to compare
);

	// Partition Counts (Phase 1)
__global__ void
GPU_2D_NODE_MED_COUNTS
(
	GPU_COUNTS_STARTS * counts,	// OUT: counts
	GPUNode_2D_MED * srcNodes,	// IN:  median node list (source)
	GPUNode_2D_MED * dstNodes,	// OUT: median node list (dest = scratch)
	unsigned int * pivot,		// IN: pivot location
	unsigned int nNodes,		// IN: number of nodes
	unsigned int start,			// IN: start of range to count
	unsigned int end,			// IN: end of range to count
	unsigned int axis			// IN: axis of dimension to work with
);


	// Convert Counts to Starts (Phase 1)
__global__ void 
GPU_2D_COUNTS_TO_STARTS
( 
	GPU_COUNTS_STARTS * starts,	// OUT - start list (store prefix sums here)
	GPU_COUNTS_STARTS * counts,	// IN  - Count list (to total)
	unsigned int nCounts,		// IN  - # of items in count list
	unsigned int currStart,		// IN  - range[start,end]
	unsigned int currEnd		//       ditto
);


	// Actual Partition (Phase 1)
__global__ void
GPU_2D_NODE_MED_PARTITION
(
	GPU_COUNTS_STARTS * starts,	// IN:  starts
	GPUNode_2D_MED * srcNodes,	// IN:  Nodes are read from this array (source = scratch)
	GPUNode_2D_MED * dstNodes,	// OUT: Nodes are partitioned into this array
	unsigned int nNodes,		// IN: number of nodes
	unsigned int start,			// IN: start of range to partition
	unsigned int end,			// IN: end of range to partition
	unsigned int axis,			// IN: axis of dimension to work with
	unsigned int * pivot		// IN: pivot index
);

	// Store result in LBT node list (Phase 1)
__global__ void
GPU_2D_NODE_STORE
(
	GPUNode_2D_MED * medNodes,	// IN:  Median Nodes are read from this array
	GPUNode_2D_LBT * lbtNodes,	// OUT: LBT Nodes are stored in this array
	unsigned int *  pointIDS,	// OUT: point indices are stored in this array
	unsigned int medianIdx,		// IN: left balanced median index
	unsigned int targetIdx		// IN: Target index
);

// Build sub kd-tree in Left balanced Median array layout (Phase 2)
__global__ void
P2_2D_BUILD_LBT
(
	GPUNode_2D_LBT * lbtNodes,	// OUT: lbt node list
	unsigned int   * pointIDs,	// OUT: point indices are stored in this array
	GPUNode_2D_MED * medNodes,	// IN: median node list
	GPUNode_2D_MED * medScratch,// IN: scratch space for temporary copying
	GPU_BUILD_ITEM * buildQ,	// IN: build queue (per thread)
	unsigned int     nPoints	// IN: maximum # of points
);

__global__ void
P2_2D_BUILD_STATS
(
	GPUNode_2D_LBT * lbtNodes,	// OUT: lbt node list
	unsigned int   * pointIDs,	// OUT: point indices are stored in this array
	GPU_BUILD_STATS * statsQ,	// OUT: stats queue (per thread)
	GPUNode_2D_MED * medNodes,	// IN: median node list
	GPUNode_2D_MED * medScratch,// IN: scratch space for temporary copying
	GPU_BUILD_ITEM * buildQ,	// IN: build queue (per thread)
	unsigned int     nPoints	// IN: maximum # of points
);


// Build sub kd-tree in median array layout (Phase 2)
__global__ void
P2_2D_BUILD_MED
(
	GPUNode_2D_MED * medNodes,	// IN/OUT: median node list
	GPUNode_2D_MED * medScratch,// IN/OUT: scratch space for temporary copying
	GPU_BUILD_ITEM * buildQ,   	// IN: build queue (per thread)
	unsigned int     nPoints	// IN: number of points in original range
);

// Convert median kd-tree to Left-balanced median kd-tree (Phase 2)
__global__ void
P2_2D_COPY
(
	GPUNode_2D_LBT * lbmNodes,	// OUT: Left-balanced kd-nodes
	unsigned int   * pointIDs,	// OUT: remapping array
	GPUNode_2D_MED * medNodes,	// IN:  Median kd-nodes
	unsigned int     start,		// IN:  start index
	unsigned int     end		// IN:  end index
);


	/*-------------------------------------------
	  Query Nearest Neighbor (QNN)
	-------------------------------------------*/

	// QNN 2D Left-balanced tree layout
__global__
void GPU_QNN_2D_LBT
(
	GPU_NN_Result		* qrs,		// OUT: Results of kdt-tree Nearest Neighbor Algorithm
	         float2		* qps,		// IN: query points to compute distance for (1D or 2D field)
	GPUNode_2D_LBT	    * kdTree,	// IN: KD Tree (kd-nodes)
		unsigned int	* ids,		// IN: IDs (original point indices)
	    unsigned int      cNodes,	// IN: count of nodes in kd-tree
		 	 int          w			// IN: width of 2D query field (# of columns)
);

	// QNN 3D Left-balanced tree layout
__global__
void GPU_QNN_3D_LBT
(
	GPU_NN_Result		* qrs,		// OUT: Results of kdt-tree Nearest Neighbor Algorithm
	         float4		* qps,		// IN: query points to compute distance for (1D or 2D field)
	GPUNode_3D_LBT	    * kdTree,	// IN: KD Tree (kd-nodes)
		unsigned int	* ids,		// IN: IDs (original point indices)
	    unsigned int      cNodes,	// IN: count of nodes in kd-tree
		 	 int          w			// IN: width of 2D query field (# of columns)
);

	// QNN 4D Left-balanced tree layout
__global__
void GPU_QNN_4D_LBT
(
	GPU_NN_Result		* qrs,		// OUT: Results of kdt-tree Nearest Neighbor Algorithm
	         float4		* qps,		// IN: query points to compute distance for (1D or 2D field)
	GPUNode_4D_LBT	    * kdTree,	// IN: KD Tree (kd-nodes)
		unsigned int	* ids,		// IN: IDs (original point indices)
	    unsigned int      cNodes,	// IN: count of nodes in kd-tree
		 	 int          w			// IN: width of 2D query field (# of columns)
);


	// QNN 6D Left-balanced tree layout
__global__
void GPU_QNN_6D_LBT
(
	GPU_NN_Result		* qrs,		// OUT: Results of kdt-tree Nearest Neighbor Algorithm
	GPU_Point6D			* qps,		// IN: query points to compute distance for (1D or 2D field)
	GPUNode_6D_LBT	    * kdTree,	// IN: KD Tree (kd-nodes)
		unsigned int	* ids,		// IN: IDs (original point indices)
	    unsigned int      cNodes,	// IN: count of nodes in kd-tree
		 	 int          w			// IN: width of 2D query field (# of columns)
);


	/*-------------------------------------------
	  All Nearest Neighbor (All-NN)
	-------------------------------------------*/

	// All-NN 2D Left-balanced tree layout
__global__ 
void GPU_ALL_NN_2D_LBT
(
	GPU_NN_Result		* qrs,		// OUT: Results of KD Nearest Neighbor Algorithm
	GPUNode_2D_LBT	    * kdTree,	// IN: KD Tree (search & query Nodes)
		unsigned int	* ids,		// IN: IDs (from Indexs)
	    unsigned int      cNodes,	// IN: index of root node in KD Tree
		 	 int          w			// IN: width of 2D query field (# of columns)
);

	// All-NN 3D Left-balanced tree layout
__global__ 
void GPU_ALL_NN_3D_LBT
(
	GPU_NN_Result		* qrs,		// OUT: Results of KD Nearest Neighbor Algorithm
	GPUNode_3D_LBT	    * kdTree,	// IN: KD Tree (search & query Nodes)
		unsigned int	* ids,		// IN: IDs (from Indexs)
	    unsigned int      cNodes,	// IN: index of root node in KD Tree
		 	 int          w			// IN: width of 2D query field (# of columns)
);

	// All-NN 4D Left-balanced tree layout
__global__ 
void GPU_ALL_NN_4D_LBT
(
	GPU_NN_Result		* qrs,		// OUT: Results of KD Nearest Neighbor Algorithm
	GPUNode_4D_LBT	    * kdTree,	// IN: KD Tree (search & query Nodes)
		unsigned int	* ids,		// IN: IDs (from Indexs)
	    unsigned int      cNodes,	// IN: index of root node in KD Tree
		 	 int          w			// IN: width of 2D query field (# of columns)
);

	// All-NN 6D Left-balanced tree layout
__global__ 
void GPU_ALL_NN_6D_LBT
(
	GPU_NN_Result		* qrs,		// OUT: Results of KD Nearest Neighbor Algorithm
	GPUNode_6D_LBT	    * kdTree,	// IN: KD Tree (search & query Nodes)
		unsigned int	* ids,		// IN: IDs (from Indexs)
	    unsigned int      cNodes,	// IN: index of root node in KD Tree
		 	 int          w			// IN: width of 2D query field (# of columns)
);


	/*-------------------------------------------
	  'k' Nearest Neighbor (kNN)
	-------------------------------------------*/

	// kNN 2D Left-balanced tree layout
__global__ 
void GPU_KNN_2D_LBT
(
	GPU_NN_Result		* qrs,		// OUT: Results of KD Nearest Neighbor Algorithm
	         float2		* qps,		// IN: query points to compute k nearest neighbors for...
	GPUNode_2D_LBT	    * kdTree,	// IN: KD Tree (Nodes)
		unsigned int	* ids,		// IN: IDs (from Indexs)
	    unsigned int      cNodes,	// IN: 'n' number of nodes in kd-tree
	    unsigned int      k			// IN: number of nearest neighbors to find
);

	// kNN 3D Left-balanced tree layout
__global__ 
void GPU_KNN_3D_LBT
(
	GPU_NN_Result		* qrs,		// OUT: Results of KD Nearest Neighbor Algorithm
	         float4		* qps,		// IN: query points to compute k nearest neighbors for...
	GPUNode_3D_LBT	    * kdTree,	// IN: KD Tree (Nodes)
		unsigned int	* ids,		// IN: IDs (from Indexs)
	    unsigned int      cNodes,	// IN: 'n' number of nodes in kd-tree
	    unsigned int      k			// IN: number of nearest neighbors to find
);

	// kNN 4D Left-balanced tree layout
__global__ 
void GPU_KNN_4D_LBT
(
	GPU_NN_Result		* qrs,		// OUT: Results of KD Nearest Neighbor Algorithm
	         float4		* qps,		// IN: query points to compute k nearest neighbors for...
	GPUNode_4D_LBT	    * kdTree,	// IN: KD Tree (Nodes)
		unsigned int	* ids,		// IN: IDs (from Indexs)
	    unsigned int      cNodes,	// IN: 'n' number of nodes in kd-tree
	    unsigned int      k			// IN: number of nearest neighbors to find
);


	// kNN 6D Left-balanced tree layout
__global__ 
void GPU_KNN_6D_LBT
(
	GPU_NN_Result		* qrs,		// OUT: Results of KD Nearest Neighbor Algorithm
	GPU_Point6D			* qps,		// IN: query points to compute k nearest neighbors for...
	GPUNode_6D_LBT	    * kdTree,	// IN: KD Tree (Nodes)
		unsigned int	* ids,		// IN: IDs (from Indexs)
	    unsigned int      cNodes,	// IN: 'n' number of nodes in kd-tree
	    unsigned int      k			// IN: number of nearest neighbors to find
);


	/*-------------------------------------------
	  All 'k' Nearest Neighbor (All-kNN)
	-------------------------------------------*/

	// All-kNN 2D Left-balanced tree layout
__global__ 
void GPU_ALL_KNN_2D_LBT
(
	GPU_NN_Result		* qrs,		// OUT: Results of KD Nearest Neighbor Algorithm
	GPUNode_2D_LBT	    * kdTree,	// IN: KD Tree (Nodes)
		unsigned int	* ids,		// IN: IDs (from Indexs)
	    unsigned int      cNodes,	// IN: 'n' number of nodes in tree
	    unsigned int      k			// IN: number of nearest neighbors to find
);

	// All-kNN 3D Left-balanced tree layout
__global__ 
void GPU_ALL_KNN_3D_LBT
(
	GPU_NN_Result		* qrs,		// OUT: Results of KD Nearest Neighbor Algorithm
	GPUNode_3D_LBT	    * kdTree,	// IN: KD Tree (Nodes)
		unsigned int	* ids,		// IN: IDs (from Indexs)
	    unsigned int      cNodes,	// IN: 'n' number of nodes in tree
	    unsigned int      k			// IN: number of nearest neighbors to find
);

	// All-kNN 4D Left-balanced tree layout
__global__ 
void GPU_ALL_KNN_4D_LBT
(
	GPU_NN_Result		* qrs,		// OUT: Results of KD Nearest Neighbor Algorithm
	GPUNode_4D_LBT	    * kdTree,	// IN: KD Tree (Nodes)
		unsigned int	* ids,		// IN: IDs (from Indexs)
	    unsigned int      cNodes,	// IN: 'n' number of nodes in tree
	    unsigned int      k			// IN: number of nearest neighbors to find
);

	// All-kNN 6D Left-balanced tree layout
__global__ 
void GPU_ALL_KNN_6D_LBT
(
	GPU_NN_Result		* qrs,		// OUT: Results of KD Nearest Neighbor Algorithm
	GPUNode_6D_LBT	    * kdTree,	// IN: KD Tree (Nodes)
		unsigned int	* ids,		// IN: IDs (from Indexs)
	    unsigned int      cNodes,	// IN: 'n' number of nodes in tree
	    unsigned int      k			// IN: number of nearest neighbors to find
);

#endif // _GPU_API_H


