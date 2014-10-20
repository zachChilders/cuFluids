#pragma once
#ifndef _CPUTREE_API_H_
#define _CPUTREE_API_H_
/*-----------------------------------------------------------------------------
  Name:  CPUTREE_API.h
  Desc:  Simple CPU kd-tree structures and
		 CPU API function definitions

  Log:   Created by Shawn D. Brown (4/15/07)
		 Modified by Shawn D. Brown (3/22/10)
-----------------------------------------------------------------------------*/

/*-------------------------------------
  Includes
-------------------------------------*/

// CUDA includes
#if !defined(__DRIVER_TYPES_H__)
	#include <driver_types.h>
#endif
#ifndef _CUT_
	#include <cutil.h>
#endif
#ifndef __cuda_cuda_h__
	#include <cuda.h>
#endif

// Project Includes
#include "KD_Flags.h"		// Compiler Flags
#include "Random.h"			// Random Number Generator
#include "QueryResult.h"	// CPU_NN_Result definition


/*-------------------------------------
  CPU Structures
-------------------------------------*/

struct BlockGridShape
{
	// Elements
	int nElems;				// nElems - number of original elements
	int nPadded;			// nPadded - number of padded elements to fit grid/block structure

	// Block shape (block of threads)
	int threadsPerRow;		// TPR - Threads Per Row (Columns in Block)
	int rowsPerBlock;		// RPB - Rows Per Block (Rows in Block)
	int threadsPerBlock;	// TPB - Threads Per Block = (TPR * RPB)

	// Grid shape (grid of blocks)
	int blocksPerRow;		// BPR - Blocks Per Row (Columns in Grid)
	int rowsPerGrid;		// RPG - Rows Per Grid (Rows in Grid)
	int blocksPerGrid;		// BPG - Blocks Per Grid = (BPR * RPG)

	int W;					// Width (in elements) of padded 2D Vector = (TPR * BPR)
	int H;					// Height (in elements) of padded 2D Vector = (RPB * RPG)
}; // end BlockGridShape


struct AppGlobals
{
	// Search Vectors
	int nSearch;
	const float4 * searchList;
	int nQuery;
	const float4 * queryList;

	int requestedDevice;			// User requested this GPU card
	int actualDevice;				// We ended up using this GPU card

	// CUDA Device properties
	CUdevprop		rawProps;		// Current CUDA device properties
	cudaDeviceProp	cudaProps;		// Current CUDA device properties

	// Application properties
	unsigned int nopromptOnExit;	// Prompt User before exiting application ?!?
	unsigned int doubleCheckDists;	// Double Check Distances against CPU
	unsigned int doubleCheckMin;	// Double check min results against CPU

	BlockGridShape bgShape;			// Block Grid Shape
	dim3 nnBlock;					// Thread Block
	dim3 nnGrid;					// Thread Grid

	// Performance Profiling
	unsigned int profile;					// Should We profile performance?
	unsigned int profileSkipFirstLast;		// Should we skip first and last loops?
	unsigned int profileRequestedLoops;		// Number of requested Profile Iterations
	unsigned int profileActualLoops;		// Number of actual Profile Iterations

	unsigned int hTimer;					// CUDA Timer
	cudaEvent_t start;						// CUDA event timer (start)
	cudaEvent_t stop;						// CUDA event timer (stop)

	double baseTimerCost;					// Base Timer cost
	double cudaTimerCost;					// CUDA Timer cost

	// Misc
	unsigned int dumpVerbose;
	bool rowByRow;
};	// End struct AppGlobals


struct HostDeviceParams
{
	// Nearest Neighbor Search Type
	unsigned int nnType;		// IN - QNN, All-NN, kNN, or All-kNN search type
	unsigned int nDims;			// IN - number of dimensions (2D, 3D, or 4D)
	unsigned int kVal;			// IN - Original 'k' value for kNN and All-kNN searches
	//unsigned int kPadVal;		// IN - Padded 'k' value

	// Search List
	unsigned int nSearch;		// IN - Original List count
	unsigned int nPadSearch;	//      Padded Search count for alignment + misc.
	const float4 * searchList;	// IN - Original Search Point List

	// Query List
	unsigned int nQuery;		// IN - Original Query List Count
	unsigned int nPadQuery;		//      Padded Query count for alignment + misc.
	const float4 * queryList;	// IN - Original query point list

	// Result List
	unsigned int nOrigResult;   // IN - Count of original results
	unsigned int nPadResult;    //      Padded result count for alignment + misc.
	CPU_NN_Result * resultList; // IN - Original Result list

	// CUDA Device properties
	const CUdevprop *	   rawProps;	// IN - Pointer to Current CUDA device properties
	const cudaDeviceProp * cudaProps;	// IN - Pointer to Current CUDA device properties

	// Block Grid Shape
	BlockGridShape bgShape;			// Used to compute thread block and grid sizes
	dim3 nnBlock;					// Thread Block Size & Shape
	dim3 nnGrid;					// Thread Grid Size & Shape

	// CPU KD-Tree
	void * kdtree;					// kd-tree pointer
	void * reservedPtr1;			// dummy pointer for padding

	// Memory for search points
	//unsigned int mem_size_Search;
	//void * h_Search;		// Host memory for search points
	//void * d_Search;		// Device memory for search points

	// Memory for query points
	unsigned int mem_size_Query;
	void * h_Query;		// Host memory for query points
	void * d_Query;		// Device memory for query points

	// Memory for kd-nodes
	unsigned int mem_size_Nodes;
	void * h_Nodes;		// Host memory for kd-nodes
	void * d_Nodes;		// Device memory for kd-nodes

	// Memory for mapping ID's 
		// used to map node indices to point indices
	unsigned int mem_size_IDs;			
	unsigned int* h_IDs;			// Host memory for ID's
	unsigned int* d_IDs;			// Device memory for ID's

	// Results
	unsigned int mem_size_Results_GPU;
	unsigned int mem_size_Results_CPU;
	//void* h_Results_GPU;			// Host memory for GPU results
	void* d_Results_GPU;			// Device memory for GPU results
	CPU_NN_Result* h_Results_CPU;	// Memory for CPU results

	// Misc
	bool buildGPU;					// Build kd-tree on GPU (true) or CPU (false)
	bool bPinned;					// Use PINNED or PAGEABLE memory
	bool bRowByRow;					// Execute Kernels Row by Row (true) or all at once (false)

};	// End struct HostDeviceParams


/*-------------------------------------
  inline functions
-------------------------------------*/

/*---------------------------------------------------------
  Name:	 KD_Min
  Desc:  returns the minimum of two values
---------------------------------------------------------*/

inline unsigned int KD_Min
( 
	unsigned int a,		// IN: 1st of 2 values to compare
	unsigned int b		// IN: 2nd of 2 values to compare
)
{
	// returns minimum of two values
	return ((a <= b) ? a : b);
}

/*---------------------------------------------------------
  Name:	 KD_Max
  Desc:  returns the maximum of two values
---------------------------------------------------------*/

inline unsigned int KD_Max
( 
	unsigned int a,		// IN: 1st of 2 values to compare
	unsigned int b		// IN: 2nd of 2 values to compare
)
{
	// returns the maximum of two values
	return ((a >= b) ? a : b);
}


/*-------------------------------------
  export C interface
-------------------------------------*/

extern "C" unsigned int g_leftTableCPU[32];
extern "C" unsigned int g_halfTableCPU[32];

extern "C" unsigned int KD_IntLog2_CPU( unsigned int inVal );

extern "C" void KD_LBM_CPU
	( 
		unsigned int n,				// IN:   Number to find left balanced median for
		unsigned int & median,		// OUT:  Left balanced median for 'n'
		unsigned int & half			// OUT:  Size of 1/2 tree including root minus last row
	);

extern "C" bool Host_Build2D( HostDeviceParams & params );
//extern "C" bool Host_Build3D( HostDeviceParams & params );
//extern "C" bool Host_Build4D( HostDeviceParams & params );

extern "C" void FiniHostBuild( HostDeviceParams & params );

	// Allocate Host Memory (pinned or pageable)
extern "C" void * AllocHostMemory( unsigned int memSize, bool bPinned );

	// Free Host Memory (pinned or pageable)
extern "C" void FreeHostMemory( void * origPtr, bool bPinned );

	// Initializes CUDA platform
extern "C" bool InitCUDA( AppGlobals & g );

	// Cleanup CUDA platform
extern "C" bool FiniCUDA();

	// Initialize Global Variables structure
extern "C" bool InitGlobals( AppGlobals & g );

	// Parse Command Line Parameters
extern "C" bool GetCommandLineParams( int argc, const char ** argv, AppGlobals & g );

	// Compute Grid and Block Shapes
	// Various helper methods
extern "C" bool ComputeBlockShapeFromVector( BlockGridShape & bgShape );
extern "C" bool ComputeBlockShapeFromQueryVector( BlockGridShape & bgShape	);
extern "C" void InitShapeDefaults( BlockGridShape & bgShape );
extern "C" void DumpBlockGridShape( const BlockGridShape & bgShape );

	// Initialize Random Search and Query Vectors
extern "C" bool InitSearchQueryVectors( AppGlobals & g, bool bInitSearch, bool bInitQuery, 
								   bool bNonUniformSearch, bool bNonUniformQuery, int scale );
extern "C" void FiniSearchQueryVectors( AppGlobals & g );

	// NN Search API functions (host side)
bool NN_LBT_HOST
( 
	//unsigned int threadsPerRow,	// IN - threads per row in thread block
	//unsigned int rowsPerBlock,	// IN - rows per block in thread block
	unsigned int nnType,		// IN - NN Search type (QNN, All-NN, kNN, All-kNN)
	unsigned int nDims,			// IN - number of dimensions for search
	unsigned int kVal,			// IN - 'k' value for kNN and All-kNN searches
	unsigned int nSearch,		// IN - Count of Search Points
	const float4 * searchList,	// IN - List of Search Points
	unsigned int nQuery,		// IN - Count of Query Points
	const float4 * queryList,	// IN - List of Query Points
	unsigned int nResult,		// IN - Max elements in result list
	CPU_NN_Result * resultList	// OUT - List or results (one for each query point)
);

	// Test NN Search API
extern bool Test_NN_API();

	// Run Various Tests
extern bool BruteForce3DTest();
extern bool CPUTest_2D_MED( AppGlobals & g );
extern bool CPUTest_2D_LBT( AppGlobals & g );
extern bool CPUTest_3D_LBT( AppGlobals & g );
extern bool CPUTest_4D_LBT( AppGlobals & g );

	// Compute Grid and Block Shapes
	// Various helper methods
extern "C" bool ComputeBlockShapeFromVector( BlockGridShape & bgShape );
extern "C" bool ComputeBlockShapeFromQueryVector( BlockGridShape & bgShape	);
extern "C" void InitShapeDefaults( BlockGridShape & bgShape );
extern "C" void DumpBlockGridShape( const BlockGridShape & bgShape );

	// Brute Force NN Search on CPU
extern "C" void ComputeDist3D_CPU( float *, const float4*, const float4 &, int, int );
extern "C" void Reduce_Min_CPU( int &, float &, const float4*, const float4&, int );


	/*-------------------------------------
	  Median Array Layout Functions
	-------------------------------------*/

	// Build KD-Tree
extern "C" bool BUILD_CPU_2D_MEDIAN
(
	void        ** kdTree,			// IN/OUT - KDTree pointer
	unsigned int   nSearch,			// IN - Number of Points
	const float4 * search_CPU		// IN -  CPU Search Point List
);

	// Cleanup KD-Tree
extern "C" bool FINI_CPU_2D_MEDIAN( void ** kdTree );

	// Transfer CPU KD-tree nodes onto GPU tree nodes
extern "C" bool COPY_NODES_2D_MEDIAN
(
	void         * kdTree,			// IN - KD Tree
	unsigned int   nSearch,			// IN - Count of items in search list
	unsigned int   nPadSearch,		// IN - Count of items in padded search list
	void         * nodes_GPU,		// OUT - GPU Node List
	unsigned int * ids_GPU			// OUT - Node IDs List
);

	// Query Nearest Neighbors (QNN) search on CPU for 2D points
extern "C" bool CPU_QNN_2D_MEDIAN
(
	void         *kdTree,			// IN - KD Tree
	unsigned int cSearch,			// IN - Count of items in search list
	const float4 *searchList,		// IN - Points to search
	unsigned int cQuery,			// IN - count of items in query list
	const float4 *queryList,		// IN - Points to Query
	CPU_NN_Result * resultList		// OUT - Result List
);

	// All Nearest Neighbors (All-NN) on CPU for 2D points
extern "C" bool CPU_ALL_NN_2D_MEDIAN
(
	void         *kdTree,			// IN - KD Tree
	unsigned int cSearch,			// IN - Count of items in search list
	const float4 *searchList,		// IN - Points to search
	unsigned int cQuery,			// IN - count of items in query list
	const float4 *queryList,		// IN - Points to Query
	CPU_NN_Result * resultList	// OUT - Result List
);

	// 'k' Nearest neighbors (kNN) on CPU for 2D points
extern "C" bool CPU_KNN_2D_MEDIAN
(
	void            * kdTree,			// IN - KD Tree
	unsigned int      cSearch,			// IN - Count of items in search list
	const float4    * searchList,		// IN - Points to search
	unsigned int      cQuery,			// IN - count of items in query list
	unsigned int      cPadQuery,		// IN - padded count ...
	const float4    * queryList,		// IN - Points to Query
	int               kVal,				// IN - kVal
	CPU_NN_Result * resultList		// OUT - Result List
);

	// All 'k' Nearest Neighbors (All-kNN) on CPU for 2D Points
extern "C" bool CPU_ALL_KNN_2D_MEDIAN
(
	void            * kdTree,			// IN - KD Tree
	unsigned int      cSearch,			// IN - Count of items in search list
	const float4    * searchList,		// IN - Points to search
	unsigned int      cQuery,			// IN - count of items in query list
	unsigned int      cPadSearch,		// IN - padded count ...
	const float4    * queryList,		// IN - Points to Query
	int               kVal,				// IN - kVal
	CPU_NN_Result * resultList		// OUT - Result List
);


	/*-------------------------------------
	  Left-Balanced Tree Layout Functions
	-------------------------------------*/

	// Build 2D KD-Tree
bool BUILD_CPU_2D_LBT
(
	void        ** cpuTree,			// IN/OUT - cpuTree pointer
	unsigned int   nSearch,			// IN - Number of Points
	const float2 * search_CPU		// IN -  CPU Search Point List
);
bool BUILD_CPU_2D_LBT
(
	void        ** cpuTree,			// IN/OUT - cpuTree pointer
	unsigned int   nSearch,			// IN - Number of Points
	const float3 * search_CPU		// IN -  CPU Search Point List
);
bool BUILD_CPU_2D_LBT
(
	void        ** cpuTree,			// IN/OUT - cpuTree pointer
	unsigned int   nSearch,			// IN - Number of Points
	const float4 * search_CPU		// IN -  CPU Search Point List
);

	// Build 3D KD-Tree
bool BUILD_CPU_3D_LBT
(
	void        ** cpuTree,			// IN/OUT - cpuTree pointer
	unsigned int   nSearch,			// IN - Number of Points
	const float3 * search_CPU		// IN -  CPU Search Point List
);
bool BUILD_CPU_3D_LBT
(
	void        ** cpuTree,			// IN/OUT - cpuTree pointer
	unsigned int   nSearch,			// IN - Number of Points
	const float4 * search_CPU		// IN -  CPU Search Point List
);

	// Build 3D KD-Tree
bool BUILD_CPU_4D_LBT
(
	void        ** cpuTree,			// IN/OUT - cpuTree pointer
	unsigned int   nSearch,			// IN - Number of Points
	const float4 * search_CPU		// IN -  CPU Search Point List
);

	// Cleanup KD-Tree (Binary Tree Array Layout)
extern "C" bool FINI_CPU_2D_LBT( void ** kdTree );
extern "C" bool FINI_CPU_3D_LBT( void ** kdTree );
extern "C" bool FINI_CPU_4D_LBT( void ** kdTree );

	// Transfer CPU KD-tree nodes onto GPU tree nodes
bool COPY_NODES_2D_LBT
(
	void         * kdTree,			// IN - KD Tree
	unsigned int   nSearch,			// IN - Count of items in search list
	unsigned int   nPadSearch,		// IN - Padded search count
	void         * nodes_GPU,		// OUT - GPU Node List
	unsigned int * ids_GPU			// OUT - Node IDs List
);

bool COPY_NODES_3D_LBT
(
	void         * kdTree,			// IN - KD Tree
	unsigned int   nSearch,			// IN - Count of items in search list
	unsigned int   nPadSearch,		// IN - Padded search count
	void         * nodes_GPU,		// OUT - GPU Node List
	unsigned int * ids_GPU			// OUT - Node IDs List
);

bool COPY_NODES_4D_LBT
(
	void         * kdTree,			// IN - KD Tree
	unsigned int   nSearch,			// IN - Count of items in search list
	unsigned int   nPadSearch,		// IN - Padded search count
	void         * nodes_GPU,		// OUT - GPU Node List
	unsigned int * ids_GPU			// OUT - Node IDs List
);


	// Query Nearest Neighbors on CPU
bool CPU_QNN_2D_LBT
(
	void            * kdTree,			// IN - KD Tree
	unsigned int      cSearch,			// IN - Count of items in search list
	const float4    * searchList,		// IN - Points to search
	unsigned int      cQuery,			// IN - count of items in query list
	const float4    * queryList,		// IN - Points to Query
	CPU_NN_Result * resultList		// OUT - Result List
);
bool CPU_QNN_3D_LBT
(
	void            * kdTree,			// IN - KD Tree
	unsigned int      cSearch,			// IN - Count of items in search list
	const float4    * searchList,		// IN - Points to search
	unsigned int      cQuery,			// IN - count of items in query list
	const float4    * queryList,		// IN - Points to Query
	CPU_NN_Result * resultList		// OUT - Result List
);
bool CPU_QNN_4D_LBT
(
	void            * kdTree,			// IN - KD Tree
	unsigned int      cSearch,			// IN - Count of items in search list
	const float4    * searchList,		// IN - Points to search
	unsigned int      cQuery,			// IN - count of items in query list
	const float4    * queryList,		// IN - Points to Query
	CPU_NN_Result * resultList		// OUT - Result List
);

	// All Nearest Neighbors on CPU
extern "C" bool CPU_ALL_NN_2D_LBT
(
	void         *kdTree,			// IN - KD Tree
	unsigned int cSearch,			// IN - Count of items in search list
	const float4 *searchList,		// IN - Points to search
	unsigned int cQuery,			// IN - count of items in query list
	const float4 *queryList,		// IN - Points to Query
	CPU_NN_Result * resultList	// OUT - Result List
);
extern "C" bool CPU_ALL_NN_3D_LBT
(
	void         *kdTree,			// IN - KD Tree
	unsigned int cSearch,			// IN - Count of items in search list
	const float4 *searchList,		// IN - Points to search
	unsigned int cQuery,			// IN - count of items in query list
	const float4 *queryList,		// IN - Points to Query
	CPU_NN_Result * resultList	// OUT - Result List
);
extern "C" bool CPU_ALL_NN_4D_LBT
(
	void         *kdTree,			// IN - KD Tree
	unsigned int cSearch,			// IN - Count of items in search list
	const float4 *searchList,		// IN - Points to search
	unsigned int cQuery,			// IN - count of items in query list
	const float4 *queryList,		// IN - Points to Query
	CPU_NN_Result * resultList	// OUT - Result List
);

	// 'k' Nearest Neighbors on CPU
extern "C" bool CPU_KNN_2D_LBT
(
	void            * kdTree,			// IN - KD Tree
	unsigned int      cSearch,			// IN - Count of items in search list
	const float4    * searchList,		// IN - Points to search
	unsigned int      cQuery,			// IN - count of items in query list
	unsigned int      cPadQuery,		// IN - padded count ...
	const float4    * queryList,		// IN - Points to Query
	int               kVal,				// IN - kVal
	CPU_NN_Result * resultList		// OUT - Result List
);
extern "C" bool CPU_KNN_3D_LBT
(
	void            * kdTree,			// IN - KD Tree
	unsigned int      cSearch,			// IN - Count of items in search list
	const float4    * searchList,		// IN - Points to search
	unsigned int      cQuery,			// IN - count of items in query list
	unsigned int      cPadQuery,		// IN - padded count ...
	const float4    * queryList,		// IN - Points to Query
	int               kVal,				// IN - kVal
	CPU_NN_Result * resultList		// OUT - Result List
);
extern "C" bool CPU_KNN_4D_LBT
(
	void            * kdTree,			// IN - KD Tree
	unsigned int      cSearch,			// IN - Count of items in search list
	const float4    * searchList,		// IN - Points to search
	unsigned int      cQuery,			// IN - count of items in query list
	unsigned int      cPadQuery,		// IN - padded count ...
	const float4    * queryList,		// IN - Points to Query
	int               kVal,				// IN - kVal
	CPU_NN_Result * resultList		// OUT - Result List
);

	// All 'k' Nearest Neighbors on CPU
extern "C" bool CPU_ALL_KNN_2D_LBT
(
	void            * kdTree,			// IN - KD Tree
	unsigned int      cSearch,			// IN - Count of items in search list
	const float4    * searchList,		// IN - Points to search
	unsigned int      cQuery,			// IN - count of items in query list
	unsigned int      cPadSearch,		// IN - padded count ...
	const float4    * queryList,		// IN - Points to Query
	int               kVal,				// IN - kVal
	CPU_NN_Result * resultList		// OUT - Result List
);
extern "C" bool CPU_ALL_KNN_3D_LBT
(
	void            * kdTree,			// IN - KD Tree
	unsigned int      cSearch,			// IN - Count of items in search list
	const float4    * searchList,		// IN - Points to search
	unsigned int      cQuery,			// IN - count of items in query list
	unsigned int      cPadSearch,		// IN - padded count ...
	const float4    * queryList,		// IN - Points to Query
	int               kVal,				// IN - kVal
	CPU_NN_Result * resultList		// OUT - Result List
);
extern "C" bool CPU_ALL_KNN_4D_LBT
(
	void            * kdTree,			// IN - KD Tree
	unsigned int      cSearch,			// IN - Count of items in search list
	const float4    * searchList,		// IN - Points to search
	unsigned int      cQuery,			// IN - count of items in query list
	unsigned int      cPadSearch,		// IN - padded count ...
	const float4    * queryList,		// IN - Points to Query
	int               kVal,				// IN - kVal
	CPU_NN_Result * resultList		// OUT - Result List
);


/*---------------------------
  Misc.
---------------------------*/

extern "C" I32 MedianSort_CPU
(
	float* pointValues,	// IN/OUT - 'points' vector
	I32 iLeft,			// IN - left or range to partition
	I32 iRight,			// IN - right of range to partition
	I32 axis			// IN - axis value <x,y,z,...> to work on
);

extern "C" bool KD_TREE_TEST_SELECT
(
	unsigned int   nPoints,			// IN - Number of Points
	unsigned int   kth,				// IN - kth element to select on
	unsigned int   axis				// IN - axis to select on
);



extern "C" int  LBM( int n );

extern "C" bool TestLBM();


#endif // _CPUTREE_API_H_

