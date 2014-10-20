//-----------------------------------------------------------------------------
//	CS 790-058 GPGPU
//	Final Project (Point Location using GPU)
//	
//	This file contains useful defines and includes
//	
//	by Shawn Brown (shawndb@cs.unc.edu)
//-----------------------------------------------------------------------------

#ifndef _MAIN_H_
#define _MAIN_H_


//-------------------------------------
//	Includes
//-------------------------------------

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


// App Includes
#ifndef _QUERY_RESULT_H
	#include "QueryResult.h"
#endif

#define CUDA_UNKNOWN 0
#define CUDA_CUDA    1
#define CUDA_DEVICE  2

#define CUDA_PLATFORM CUDA_DEVICE
//#define CUDA_PLATFORM CUDA_CUDA

#define TEST_KD_QNN      1
#define TEST_KD_KNN      2
#define TEST_KD_ALL_NN   3
#define TEST_KD_ALL_KNN  4

//#define APP_TEST TEST_KD_QNN
#define APP_TEST TEST_KD_KNN
//#define APP_TEST TEST_KD_ALL_NN
//#define APP_TEST TEST_KD_ALL_KNN

//-------------------------------------
//	Defines
//-------------------------------------

// Optimal for Brute Force Distance Threads
#define BFD_THREADS_PER_ROW 64
#define BFD_ROWS_PER_BLOCK   1 
#define BFD_THREADS_PER_BLOCK 64

//#define BFD_THREADS_PER_ROW 512
//#define BFD_ROWS_PER_BLOCK   1 
//#define BFD_THREADS_PER_BLOCK 512

// Optimal ??? for Brute Force Minimum Threads
#define BFMR_THREADS_PER_ROW 64
#define BFMR_THREADS_PER_ROW2 128
#define BFMR_ROWS_PER_BLOCK   1 
#define BFMR_THREADS_PER_BLOCK 64
#define BFMR_THREADS_PER_BLOCK2 128

//#define BFMR_THREADS_PER_ROW 512
//#define BFMR_THREADS_PER_ROW2 1024
//#define BFMR_ROWS_PER_BLOCK   1 
//#define BFMR_THREADS_PER_BLOCK 512
//#define BFMR_THREADS_PER_BLOCK2 1024

//#define BLOCKS_PER_ROW 16
//#define ROWS_PER_GRID  16

#define KD_THREADS_PER_ROW   16
#define KD_ROWS_PER_BLOCK    1
#define KD_THREADS_PER_BLOCK 16

#define KD_STACK_SIZE        32
#define KD_KNN_SIZE          32


//-------------------------------------
//	Structures
//-------------------------------------

struct BlockGridShape
{
	// Elements
	unsigned int nElems;			// nElems - number of original elements
	unsigned int nPadded;			// nPadded - number of padded elements to fit grid/block structure

	// Block (of threads) Shape
	unsigned int threadsPerRow;		// TPR - Threads Per Row (Columns in Block)
	unsigned int rowsPerBlock;		// RPB - Rows Per Block (Rows in Block)
	unsigned int threadsPerBlock;	// TPB - Threads Per Block = (TPR * RPB)

	// Grid (of Blocks) Shape
	unsigned int blocksPerRow;		// BPR - Blocks Per Row (Columns in Grid)
	unsigned int rowsPerGrid;		// RPG - Rows Per Grid (Rows in Grid)
	unsigned int blocksPerGrid;		// BPG - Blocks Per Grid = (BPR * RPG)

	unsigned int W;					// Width (in elements) of padded 2D Vector = (TPR * BPR)
	unsigned int H;					// Height (in elements) of padded 2D Vector = (RPB * RPG)
}; // end BlockGridShape

struct AppGlobals
{
	// Search Vectors
	unsigned int nSearch;
	float4 * searchList;
	unsigned int nQuery;
	float4 * queryList;

	// CUDA Device properties
#if (CUDA_PLATFORM == CUDA_DEVICE)
	CUdevice        currDevice;		// Current CUDA device
#endif
	CUdevprop		rawProps;		// Current CUDA device properties
	cudaDeviceProp	cudaProps;		// Current CUDA device properties

	// Application properties
	unsigned int nopromptOnExit;	// Prompt User before exiting application ?!?
	unsigned int doubleCheckDists;	// Double Check Distances against CPU
	unsigned int doubleCheckMin;	// Double check min results against CPU

	BlockGridShape bgShape;			// Block Grid Shape

	// Performance Profiling
	unsigned int hTimer;					// CUDA Timer
	unsigned int profile;					// Should We profile performance?
	unsigned int profileSkipFirstLast;		// Should we skip first and last loops?
	unsigned int profileRequestedLoops;		// Number of requested Profile Iterations
	unsigned int profileActualLoops;		// Number of actual Profile Iterations
};	// End struct AppGlobals




//-------------------------------------
//
// export C interface
//
//-------------------------------------

extern "C" void PLQ_CPU_BF_DIST( float *, const float4*, const float4 &, unsigned int, unsigned int );
extern "C" void PLQ_CPU_BF_DIST_MIN( unsigned int &, float &, const float4*, const float4&, unsigned int );

extern "C" 
bool CPU_KD_QNN_2D
(
	void         *kdTree,			// IN - KD Tree
	unsigned int cSearch,			// IN - Count of items in search list
	const float4 *searchList,		// IN - Points to search
	unsigned int cQuery,			// IN - count of items in query list
	const float4 *queryList,		// IN - Points to Query
	CPU_NN_Result * resultList	// OUT - Result List
);

extern "C" 
bool CPU_KD_ALL_NN_2D
(
	void         *kdTree,			// IN - KD Tree
	unsigned int cSearch,			// IN - Count of items in search list
	const float4 *searchList,		// IN - Points to search
	unsigned int cQuery,			// IN - count of items in query list
	const float4 *queryList,		// IN - Points to Query
	CPU_NN_Result * resultList	// OUT - Result List
);

extern "C"
bool CPU_KD_KNN_2D
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

extern "C"
bool CPU_KD_ALL_KNN_2D
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

extern "C" bool BUILD_KD_TREE
(
	void        ** kdTree,			// IN/OUT - KDTree pointer
	unsigned int   nSearch,			// IN - Number of Points
	const float4 * search_CPU		// IN -  CPU Search Point List
);

extern "C" bool FINI_KD_TREE( void ** kdTree );

extern "C" bool BUILD_GPU_NODES_V1
(
    void         * kdTree,			// IN - KD Tree
	unsigned int   nSearch,			// IN - Count of items in search list
	const float4 * search_CPU,		// IN -  CPU Search Point List
	void         * nodes_GPU		// OUT - GPU Node List
);

extern "C" bool BUILD_GPU_NODES_V2
(
    void         * kdTree,			// IN - KD Tree
	unsigned int   nSearch,			// IN - Count of items in search list
	const float4 * search_CPU,		// IN -  CPU Search Point List
	void         * nodes_GPU,		// OUT - GPU Node List
	unsigned int * ids_GPU			// OUT - Node IDs List
);



#endif // _MAIN_H_

