/*-----------------------------------------------------------------------------
  Name:  Flags.h
  Desc:  Useful compiler flags

  Log:   Created by Shawn D. Brown (3/15/2010)
-----------------------------------------------------------------------------*/

#ifndef _KD_FLAGS_H_
#define _KD_FLAGS_H_

/*-------------------------------------
  Include Files
-------------------------------------*/

#ifndef _KD_PLATFORM_H
	#include "Platform.h"
#endif
#ifndef _KD_BASEDEFS_H
	#include "BaseDefs.h"
#endif

/*-------------------------------------
  Compiler Flags
-------------------------------------*/

//#define _BUILD_STATS 1


// NN Search Types (QNN, All-NN, kNN, All-kNN)
#define NN_UNKNOWN   0
#define NN_QNN       1
#define NN_ALL_NN    2
#define NN_KNN       3
#define NN_ALL_KNN   4


// NN Test Types (QNN, All-NN, kNN, All-kNN)
#define TEST_KD_QNN       1
#define TEST_KD_ALL_NN    2
#define TEST_KD_KNN       3
#define TEST_KD_ALL_KNN   4

#define APP_TEST TEST_KD_QNN
//#define APP_TEST TEST_KD_ALL_NN
//#define APP_TEST TEST_KD_KNN
//#define APP_TEST TEST_KD_ALL_KNN

// Test Type (BF, kd-tree median, kd-tree left balanced, ...)
#define TEST_BRUTE_FORCE	0
#define TEST_MEDIAN_KDTREE	1
#define TEST_LBT_KDTREE		2
#define TEST_MEDIAN_SELECT	3

//#define TEST_TYPE TEST_BRUTE_FORCE
//#define TEST_TYPE TEST_MEDIAN_KDTREE
#define TEST_TYPE TEST_LBT_KDTREE
//#define TEST_TYPE TEST_MEDIAN_SELECT

// Profile performance with timers
#define QNN_TIMER			1
#define QNN_SKIP_FIRST_LAST	0
#define QNN_REQUESTED_LOOPS	1
#define QNN_CPU_VERIFY		1
#define QNN_DUMP			1
#define QNN_DUMP_VERBOSE	0

#define KNN_TIMER			1
#define KNN_SKIP_FIRST_LAST	0
#define KNN_REQUESTED_LOOPS	1
#define KNN_CPU_VERIFY		1
#define KNN_DUMP_VERBOSE	1

#define ALL_NN_TIMER			1
#define ALL_NN_SKIP_FIRST_LAST	0
#define ALL_NN_REQUESTED_LOOPS	1
#define ALL_NN_CPU_VERIFY		1
#define ALL_NN_DUMP_VERBOSE		1

#define ALL_KNN_TIMER			1
#define ALL_KNN_SKIP_FIRST_LAST	0
#define ALL_KNN_REQUESTED_LOOPS	1
#define ALL_KNN_CPU_VERIFY		1
#define ALL_KNN_DUMP_VERBOSE	1


//---------------------------------
// Search Node Bit Masks & Shifts
//---------------------------------

// Bit Masks for 3 fields compressed into nodeFlags field
	// Onside/Offside Mask  (1 = Offside, 0 = Onside)
#define ON_OFF_MASK      0x80000000u	
	#define ONSIDE_VALUE         0x00000000u
	#define OFFSIDE_VALUE        0x80000000u

	// Split Axis Mask		(nDim <= 8 = 2^3)
#define SPLIT_AXIS_MASK  0x70000000u	
	// Node Index Mask		Up to at most 2^28 (268+ million) nodes in search list
#define NODE_INDEX_MASK  0x0FFFFFFFu	

// Bit Shifts for 3 fields compressed into nodeFlags field
#define ON_OFF_SHIFT      31u
#define SPLIT_AXIS_SHIFT  28u
#define NODE_INDEX_SHIFT   0u

#define FLAGS_ROOT_START   0x00000001u	
	// (1 & NODE_INDEX_MASK) | 
	// ((X_AXIS << SPLIT_AXIS_SHFIT) & SPLIT_AXIS_MASK) |
	// ((ONSIDE_VALUE << ON_OFF_SHIFT) & ON_OFF_MASK)


/*-------------------------------------
  Search, Query Counts
-------------------------------------*/

// Brute Force Search
	// Query assumed to be single point
#define TEST_BF_SEARCH_POINTS  1000000

// kd-tree Search
#define TEST_NUM_SEARCH_POINTS 1000000

// kd-tree query
#define TEST_NUM_QUERY_POINTS  1000000


/*-------------------------------------
  Thread Block Defines
-------------------------------------*/

//-----------------
// GPU Build 
//-----------------

	// Thread Block
#define BUILD_THREADS_PER_ROW     8
#define BUILD_ROWS_PER_BLOCK      1
	// ThreadsPerBlock = ThreadsPerRow * RowsPerBlock
#define BUILD_THREADS_PER_BLOCK   8

	// Grid of Thread Blocks
#define BUILD_BLOCKS_PER_ROW	 30
#define BUILD_ROWS_PER_GRID		  1
	// BlocksPerGrid = BlocksPerRow * RowsPerGrid
#define BUILD_BLOCKS_PER_GRID    30

	// Total Threads = BlocksPerGrid * ThreadsPerBlock
#define BUILD_TOTAL_THREADS     240
//#define BUILD_TOTAL_THREADS     480

	// P1 Scan Counts to Starts
#define BUILD_CS_SCAN_THREADS   128
#define BUILD_CS_SCAN_MAX_ITEMS 256


	// Thread Block
#define P2_BUILD_THREADS_PER_ROW    1
#define P2_BUILD_ROWS_PER_BLOCK     1
#define P2_BUILD_THREADS_PER_BLOCK  1

	// Grid of Thread Blocks
#define P2_BUILD_BLOCKS_PER_ROW	   64
#define P2_BUILD_ROWS_PER_GRID	    1
#define P2_BUILD_BLOCKS_PER_GRID   64

#define P2_BUILD_TOTAL_THREADS     64

#define P2_BUILD_STACK_DEPTH       20


//---------------------------
// Median Selection Kernel
//---------------------------

#define MEDIAN_THREADS_PER_ROW   64
#define MEDIAN_ROWS_PER_BLOCK     1
#define MEDIAN_THREADS_PER_BLOCK 64

//#define MEDIAN_THREADS_PER_ROW 512
//#define MEDIAN_ROWS_PER_BLOCK   1 
//#define MEDIAN_THREADS_PER_BLOCK 512


//--------------------------------
// Brute Force Distance Kernel
//--------------------------------
#define BFD_THREADS_PER_ROW    64
#define BFD_ROWS_PER_BLOCK      1
#define BFD_THREADS_PER_BLOCK  64

//#define BFD_THREADS_PER_ROW 512
//#define BFD_ROWS_PER_BLOCK   1 
//#define BFD_THREADS_PER_BLOCK 512

// Brute Force Reduction Kernel
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


//---------------------------
// kd-tree NN Kernels
//---------------------------

	// Note:  These are used to statically hard-code 
	//        shared memory variable array sizes in the GPU Kernels, 
	//        which makes it hard to make them dynamic

	// QNN and ALL_NN perform about the same so use the same thread parameters
	// On the GTX 285, the optimal thread block is between 6x1 and 16x1
	// For 1 million search and query points 10x1 performs best so thats what we will use
#define QNN_THREADS_PER_ROW    10
#define QNN_ROWS_PER_BLOCK     1
#define QNN_THREADS_PER_BLOCK  10

	// KNN and ALL_KNN perform about the same so use the same thread parameters
	// On the GTX 285, the optimal thread block is between 3x1 and 6x1
	// For 1 million search and query points 4x1 performs best so thats what we will use
#define KNN_THREADS_PER_ROW    4
#define KNN_ROWS_PER_BLOCK     1
#define KNN_THREADS_PER_BLOCK  4

#define ALL_KNN_THREADS_PER_ROW    4
#define ALL_KNN_ROWS_PER_BLOCK     1
#define ALL_KNN_THREADS_PER_BLOCK  4

// Stack Size for NN GPU Kernels
	// Note:  The GTX 285 Display (1 GB RAM) card can only hold 
	//		  up to at most ~36 million 2D points 
	//        before memory resources for NN searches
	//		  in some form or another run out
//#define KD_STACK_SIZE        32 // up to   4 billion points (2^32) // NOT ALLOWED, top 4 bits reserved
//#define KD_STACK_SIZE        30 // up to   1 billion points (2^30) // NOT ALLOWED, top 4 bits reserved
//#define KD_STACK_SIZE        28 // up to 268 million points (2^28) // MAX Allowed, top 4 bits reserved
//#define KD_STACK_SIZE        26 // up to  67 million points (2^26) // Practical maximum, safely fit ~36 million points on GTX 285 card
//#define KD_STACK_SIZE        24 // up to  16 million points (2^24) 
//#define KD_STACK_SIZE        22 // up to   4 million points (2^22)
//#define KD_STACK_SIZE        20 // up to   1 million points (2^20) 
//#define KD_STACK_SIZE        18 // up to 264 thousand points (2^18)
//#define KD_STACK_SIZE        16 // up to  65 thousand points (2^16)

// Note:  The smaller the stack size, the faster the GPU kernels tend to run
//		  The trade-off is that it also limits the maximum # of search points you can handle with the GPU
#define KD_LARGE_STACK		26 // up to 67 million points (2^26) // Practical maximum, ~36 million points is most on GTX 285 card
#define KD_MEDIUM_STACK		20 // up to  1 million points (2^20)
#define KD_SMALL_STACK      16 // up to 65 thousand points (2^16) 

	// Allow fine-tuning of stack size for each type of NN search
#define QNN_STACK_SIZE       KD_MEDIUM_STACK
#define ALL_NN_STACK_SIZE    KD_MEDIUM_STACK
#define KNN_STACK_SIZE       KD_MEDIUM_STACK
#define ALL_KNN_STACK_SIZE   KD_MEDIUM_STACK


// for kNN and All-kNN searches
// Maximum 'k' value for NN GPU Kernels
	// Used to carve out room on GPU shared memory for 'k' closest heaps
	// Note: The smaller the max 'k' value the faster the kNN GPU kernels tend to run
#define KD_KNN_SIZE          32
//#define KD_KNN_SIZE         16


#endif // _KD_FLAGS_H_
