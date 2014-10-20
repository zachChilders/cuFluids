/*-----------------------------------------------------------------------------
  File:  GPU_Util.cpp
  Desc:  Various GPU helper functions

  Log:   Created by Shawn D. Brown (4/15/07)
         Modified by Shawn Brown (5/8/2010)
-----------------------------------------------------------------------------*/

/*-------------------------------------
  Includes
-------------------------------------*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#if (CPU_PLATFORM == CPU_INTEL_X86)
	#include <intrin.h>		// Intrinsics
#elif (CPU_PLATFORM == CPU_INTEL_X64)
	#include <intrin.h>		// Intrinsics
#endif

// includes, CUDA
#include <cutil_inline.h>

// includes, project
#include "CPUTREE_API.h"
#include "GPUTREE_API.h"


/*-------------------------------------
  Global Variables
-------------------------------------*/

extern AppGlobals g_app;


/*-------------------------------------
  Function Declarations
-------------------------------------*/

bool GetCommandLineParams( int argc, const char ** argv, AppGlobals & g );

bool InitGlobals( AppGlobals & g );

bool ComputeBlockShapeFromVector( BlockGridShape & bgShape );

void InitShapeDefaults( BlockGridShape & bgShape );
void DumpBlockGridShape( const BlockGridShape & bgShape );

float RandomFloat( float low, float high );


/*---------------------------------------------------------
  Name:	AllocHostMemory
  Desc:  Allocate host memory
---------------------------------------------------------*/

void * AllocHostMemory( unsigned int memSize, bool bPinned )
{
    unsigned char *memPtr = NULL;

    //allocate host memory
    if( true == bPinned )
    {
        // PINNED memory mode - use cuda function to get OS-pinned memory
		unsigned int cudaFlags = 0;
		cutilSafeCall( cudaHostAlloc( (void**)&memPtr, memSize, cudaFlags ) );
    }
    else
    {
        // PAGEABLE memory mode - use malloc
        memPtr = (unsigned char *)malloc( memSize );
    }

	return (void *)memPtr;
}


/*---------------------------------------------------------
  Name:	FreeHostMemory
  Desc:  Free host memory
---------------------------------------------------------*/

void FreeHostMemory( void * origPtr, bool bPinned )
{
	unsigned char * memPtr = (unsigned char *)origPtr;
	if (NULL != memPtr)
	{
		if (true == bPinned)
		{
			cutilSafeCall( cudaFreeHost(memPtr) );
		}
		else
		{
			free(memPtr);
		}
	}
}



/*---------------------------------------------------------
  Name:	ComputeBlockShapeFromVector
  Desc:	
  Notes:

	0. Assumes following members are initialized properly
	   before this funciton is called
		 shape.nElems = Number of original elements in vector
		 shape.tpr    = threads per row
		 shape.rpb    = rows per block

	1. Block Limits
		 Thread block has at most 512 theads per block

	2. Grid Limits
		Grid has at most 65,535 blocks in any dimension
			So a 1D grid is at most 65,535 x 1
			and a 2D grid is at most 65,535 x 65,535
		We use next smallest even number to these limits
			IE 65,535 - 1
			IE (65,535*65,535 - 1)
		It's useful to have an even number of columns
		in grid structure when doing reductions
---------------------------------------------------------*/

bool ComputeBlockShapeFromVector
( 
	BlockGridShape & bgShape	// IN/OUT - bgShape
)
{
	unsigned int remainder, extra;

	//---------------------------------
	//	1. Compute Threads Per Block
	//---------------------------------

	if ((bgShape.threadsPerRow > 512) || (bgShape.rowsPerBlock > 512))
	{
		// Error - block can't have more than 512 threads
		printf( "block (%d TPR x %d RPB) > 512 TPB\n", 
				bgShape.threadsPerRow, bgShape.rowsPerBlock );
		printf( "Error - can't have more than 512 threads in block" );
	}

	// Compute threads per block
	bgShape.threadsPerBlock = bgShape.threadsPerRow * bgShape.rowsPerBlock;

	// Make sure we don't exceed block limits
	if (bgShape.threadsPerBlock > 512)
	{
		// Error - block can't have more than 512 threads
		printf( "block (%d TPR x %d RPB)  = %d TPB\n", 
				bgShape.threadsPerRow, bgShape.rowsPerBlock, bgShape.threadsPerBlock );
		printf( "Error - can't have more than 512 threads in block" );
		return false;
	}


	//---------------------------------
	//	2. Compute GRID structure
	//---------------------------------

	// Compute number of blocks needed to contain all elements in vector
	bgShape.blocksPerGrid  = bgShape.nElems / bgShape.threadsPerBlock;
	remainder			   = bgShape.nElems % bgShape.threadsPerBlock;
	extra				   = ((0 == remainder) ? 0 : 1);
	bgShape.blocksPerGrid += extra;

	// Check if need a 1D Grid or 2D Grid structure 
	// to contain all blocks in grid
	if (bgShape.blocksPerGrid <= 65534) 					  
	{
		// We can use a simple 1D grid of blocks
		bgShape.blocksPerRow = bgShape.blocksPerGrid;
		bgShape.rowsPerGrid  = 1;
	}
	else if (bgShape.blocksPerGrid <= 4294836224)	// 4294836224 = (65535 * 65535 - 1)
	{
		// We need to use a 2D Grid structure instead...
		// Use square root as an approximation for shape of 2D grid
		float sq_r = sqrtf( (float)( bgShape.blocksPerGrid ) );
		unsigned int uiSqrt = (unsigned int)sq_r;
		bgShape.blocksPerRow = uiSqrt;
		bgShape.rowsPerGrid  = uiSqrt;

		// Increment # of columns until we have enough space
		// in grid layout for all elements in vector
		while ((bgShape.blocksPerRow * bgShape.rowsPerGrid) < bgShape.blocksPerGrid)
		{
			bgShape.blocksPerRow++;
		}
	}
	else
	{
		// Error - Vector is too large for 2D Grid
		printf( "Vector is way too large...\n" );
		return false;
	}

	// Make sure # of columns in 1D or 2D grid is even
		// Which is useful to avoid special cases in reduction kernel
	remainder             = bgShape.blocksPerRow % 2;
	extra	              = ((0 == remainder) ? 0 : 1);
	bgShape.blocksPerRow += extra;

	// Compute # of padded blocks in Grid
	bgShape.blocksPerGrid = bgShape.blocksPerRow * bgShape.rowsPerGrid;

	// Make sure we don't exceed grid limits
	if ((bgShape.blocksPerRow >= 65535) || (bgShape.rowsPerGrid >= 65535))
	{
		// Error - Grid can't have more than 65535 blocks in any dimension
		printf( "Grid (%d BPR x %d RPG)  = %d BPG\n", 
				bgShape.blocksPerRow, bgShape.rowsPerGrid, bgShape.blocksPerGrid );
		printf( "Error - can't have more than 65535 blocks in any dimension\n" );
		return false;
	}

	// Compute Width and Height of 2D vector structure
	bgShape.W = bgShape.threadsPerRow * bgShape.blocksPerRow;	// Width (in elements) of 2D vector
	bgShape.H = bgShape.rowsPerBlock  * bgShape.rowsPerGrid;	// Height (in elements) of 2D vector

	// Compute padded length of 2D vector
	unsigned int sizeWH = bgShape.W * bgShape.H;
	unsigned int sizeBG = bgShape.threadsPerBlock * bgShape.blocksPerGrid;

	if (sizeWH != sizeBG)
	{
		// Programmer error-
		printf( "Error - sizes don't match\n" );
		return false;
	}

	// Compute number of elements in padded block structure
	bgShape.nPadded = sizeWH;

	// Success
	return true;
}


/*---------------------------------------------------------
  Name: ComputeBlockShapeFromQueryVector
  Desc:
  Notes:
    0. Assumes following members are initialized properly
       before this function is called
        shape.nElems = Number of original elements in query vector
        shape.tpr    = threads per row
        shape.rpb    = rows per block

    1. Block Limits
         Thread block has at most 512 theads per block

    2. Grid Limits
         Grid has at most 65,535 blocks in any dimension
           So a 1D grid is at most 65,535 x 1
           and a 2D grid is at most 65,535 x 65,535
         We use next smallest even number to these limits
           IE 65,535 - 1
           IE (65,535*65,535 - 1)
---------------------------------------------------------*/

bool ComputeBlockShapeFromQueryVector
(
	BlockGridShape & bgShape	// IN/OUT - bgShape
)
{
	unsigned int remainder, extra;

	//---------------------------------
	//	1. Compute Threads Per Block
	//---------------------------------

	if ((bgShape.threadsPerRow > 512u) || (bgShape.rowsPerBlock > 512u))
	{
		// Error - block can't have more than 512 threads
		printf( "block (%d TPR x %d RPB) > 512 TPB\n", 
				bgShape.threadsPerRow, bgShape.rowsPerBlock );
		printf( "Error - can't have more than 512 threads in block" );
	}

	// Compute threads per block
	bgShape.threadsPerBlock = bgShape.threadsPerRow * bgShape.rowsPerBlock;

	// Make sure we don't exceed block limits
	if (bgShape.threadsPerBlock > 512u)
	{
		// Error - block can't have more than 512 threads
		printf( "block (%d TPR x %d RPB)  = %d TPB\n", 
				bgShape.threadsPerRow, bgShape.rowsPerBlock, bgShape.threadsPerBlock );
		printf( "Error - can't have more than 512 threads in block" );
		return false;
	}


	//---------------------------------
	//	2. Compute GRID structure
	//---------------------------------

	// Compute number of blocks needed to contain all elements in vector
	bgShape.blocksPerGrid  = bgShape.nElems / bgShape.threadsPerBlock;
	remainder			   = bgShape.nElems % bgShape.threadsPerBlock;
	extra				   = ((0u == remainder) ? 0u : 1u);
	bgShape.blocksPerGrid += extra;

	// Check if need a 1D Grid or 2D Grid structure 
	// to contain all blocks in grid
	if (bgShape.blocksPerGrid <= 65534u)
	{
		// We can use a simple 1D grid of blocks
		bgShape.blocksPerRow = bgShape.blocksPerGrid;
		bgShape.rowsPerGrid  = 1;
	}
	else if (bgShape.blocksPerGrid <= 4294836224u)	// 4294836224 = (65535 * 65535 - 1)
	{
		// We need to use a 2D Grid structure instead...
		// Use square root as an approximation for shape of 2D grid
		float sq_r = sqrtf( (float)( bgShape.blocksPerGrid ) );
		unsigned int uiSqrt = (unsigned int)sq_r;
		bgShape.blocksPerRow = uiSqrt;
		bgShape.rowsPerGrid  = uiSqrt;

		// Increment # of columns until we have enough space
		// in grid layout for all elements in vector
		while ((bgShape.blocksPerRow * bgShape.rowsPerGrid) < bgShape.blocksPerGrid)
		{
			bgShape.blocksPerRow++;
		}
	}
	else
	{
		// Error - Vector is too large for 2D Grid
		printf( "Vector is way too large...\n" );
		return false;
	}

	// Make sure we don't exceed grid limits
	if ((bgShape.blocksPerRow >= 65535u) || (bgShape.rowsPerGrid >= 65535u))
	{
		// Error - Grid can't have more than 65535 blocks in any dimension
		printf( "Grid (%d BPR x %d RPG)  = %d BPG\n", 
				bgShape.blocksPerRow, bgShape.rowsPerGrid, bgShape.blocksPerGrid );
		printf( "Error - can't have more than 65535 blocks in any dimension\n" );
		return false;
	}

	// Compute # of padded blocks in Grid
	bgShape.blocksPerGrid = bgShape.blocksPerRow * bgShape.rowsPerGrid;

	// Compute Width and Height of 2D vector structure
	bgShape.W = bgShape.threadsPerRow * bgShape.blocksPerRow;	// Width (in elements) of 2D vector
	bgShape.H = bgShape.rowsPerBlock  * bgShape.rowsPerGrid;	// Height (in elements) of 2D vector

	// Compute padded length of 2D vector
	unsigned int sizeWH = bgShape.W * bgShape.H;
	unsigned int sizeBG = bgShape.threadsPerBlock * bgShape.blocksPerGrid;

	if (sizeWH != sizeBG)
	{
		// Programmer error-
		printf( "Error - sizes don't match\n" );
		return false;
	}

	// Compute number of elements in padded block structure
	bgShape.nPadded = sizeWH;

	// Success
	return true;
}


/*---------------------------------------------------------
  Name:	InitCUDA
  Desc:	Initialize CUDA system for GPU processing
---------------------------------------------------------*/

// Runtime API version...
bool InitCUDA( AppGlobals & g )
{
	bool bResult = false;
	int nDevices = 0;
	int deviceToUse = 0;
	cudaError_t cudaResult = cudaSuccess;

	// Pick Display Device to perform GPU calculations on...
	cudaResult = cudaGetDeviceCount( &nDevices );
	if (cudaSuccess != cudaResult)
	{
		// Error - cudaGetDeviceCount() failed
		fprintf( stderr, "InitCuda() - cudaGetDeviceCount() failed, error = %x in file '%s' in line %i.\n", 
				 cudaResult, __FILE__, __LINE__ );
		goto lblError;
	}

	if (nDevices <= 0)
	{
		// No Valid Display Device found
		cudaResult = cudaErrorInvalidDevice;
		fprintf( stderr, "InitCuda() - no valid display device found, error = %x in file '%s' in line %i.\n", 
				 cudaResult, __FILE__, __LINE__ );
		goto lblError;
	}
	
	// Did user make a valid request for a particular GPU Card?
	if ((g.requestedDevice >= 0) && (g.requestedDevice < nDevices))
	{
		deviceToUse = g.requestedDevice;
	}
	else if (nDevices >= 2)
	{
		// Note:  Assumes Device 0 = primary display device
		//		  Assumes Device 1 = work horse for CUDA

		// Use secondary GPU card, if there is one
		deviceToUse = 1;
	}
	else
	{
		// Use primary display card
		deviceToUse = 0;
	}

	// Get Display Device Properties
	cudaResult = cudaGetDeviceProperties( &(g.cudaProps) , deviceToUse );
	if (cudaSuccess != cudaResult)
	{
		// Error - cudaDeviceGet() failed
		fprintf( stderr, "InitCuda() - cudaGetDeviceProperties() failed on requested GPU card (%d), error = %x in file '%s' in line %i.\n", 
				 cudaResult, deviceToUse, __FILE__, __LINE__ );
		goto lblError;
	}

	// Setup Display Device
	cudaResult = cudaSetDevice( deviceToUse );
	if (cudaSuccess != cudaResult)
	{
		// Error - cudaDeviceGet() failed
		fprintf( stderr, "InitCuda() - cudaSetDevice() failed on requested GPU card (%d), error = %x in file '%s' in line %i.\n", 
				 cudaResult, __FILE__, __LINE__ );
		goto lblError;
	}

	// Success
	bResult = true;

lblError:
	return bResult;
}


/*---------------------------------------------------------
  Name:	FiniCUDA
  Desc:	Cleanup CUDA system
---------------------------------------------------------*/

bool FiniCUDA()
{
	// Success
	return true;
}


/*---------------------------------------------------------
  Name:	InitGlobals
  Desc:	Initialize Application Globals to Default
---------------------------------------------------------*/

bool InitGlobals( AppGlobals & g )
{
	/*------------------
	  Set Defaults
    ------------------*/

	// Search Vectors
	g.nSearch	 = 100;
	g.searchList = NULL;
	g.nQuery	 = 10;
	g.queryList  = NULL;

	g.requestedDevice = -1;
	g.actualDevice = -1;

	// Cuda Properties
	size_t byteSize;

	// Initialize cuda props to zero
	byteSize = sizeof( g.cudaProps );
	memset( &g.cudaProps, 0, byteSize );

	// Init Block Grid Shape
	InitShapeDefaults( g.bgShape );

	// App Properties
	g.nopromptOnExit    = 0u;
	g.doubleCheckDists	= 1u;

	// Profiling Info
	g.profile				= 1u;
	g.profileSkipFirstLast  = 0u;
	g.profileRequestedLoops	= 1u;
	g.profileActualLoops	= 1u;

	g.hTimer				= 0u;
	g.start                 = 0;
	g.stop                  = 0;

	// Misc
	//g.dumpVerbose = 0;
	g.dumpVerbose = 1;

	return true;
}


/*---------------------------------------------------------
  Name:	GetCommandLineParameters
  Desc:	
  Note: The CUDA command line functions expect command
        line parameters to be in one of the two following
		forms.

		Format:                     Example:
		--------------              -------------
		--<argname>					--noprompt
		--<argname>=<argvalue>		--profile=100

		Where <argname> is the name of the argument
		      <argvalue> is an integer or floating point
			             value associated with argname
---------------------------------------------------------*/

bool GetCommandLineParams
( 
	int argc,			// Count of Command Line Parameters
	const char** argv,	// List of Command Line Parameters
	AppGlobals & g		// Structure to store results in
)
{
	int iVal;

#if 0
	// Dump Command line arguments for debugging
	int argIdx;
	printf( "Command Arguments: \n" );
	for (argIdx = 0; argIdx < argc; argIdx++)
	{
		printf( "%s\n", argv[argIdx] );
	}
#endif

	// Get requested GPU device to use
    if( cutGetCmdLineArgumenti(argc, argv, "device", &iVal) )
    {
		g.requestedDevice = iVal;
	}
	else
	{
		g.requestedDevice = -1;
	}

	// Prompt before exiting application ?!?
	if (cutCheckCmdLineFlag( argc, argv, "noprompt") ) 
	{
		g.nopromptOnExit = true;
	}
	else
	{
		g.nopromptOnExit = false;
	}

	// Double Check Distances
	if (cutCheckCmdLineFlag( argc, argv, "cdist") ) 
	{
		g.doubleCheckDists = true;
	}
	else
	{
		g.doubleCheckDists = false;
	}

	// Double Check Distances
	if (cutCheckCmdLineFlag( argc, argv, "cmin") ) 
	{
		g.doubleCheckMin = true;
	}
	else
	{
		g.doubleCheckMin = false;
	}


	// Get # Threads Per Row (block shape)
	if (cutGetCmdLineArgumenti( argc, argv, "TPR", &iVal )) 
	{
		if (iVal < 1) { iVal = 1; }
		g.bgShape.threadsPerRow = iVal;
	}

	// Get # Rows Per Block
	if (cutGetCmdLineArgumenti( argc, argv, "RPB", &iVal )) 
	{
		if (iVal < 1) { iVal = 1; }
		g.bgShape.rowsPerBlock = iVal;
	}

	// Calculate Threads Per Block
	g.bgShape.threadsPerBlock = g.bgShape.threadsPerRow * g.bgShape.rowsPerBlock;
	if (g.bgShape.threadsPerBlock > 512)
	{	
		// Error - Can't have more than 512 threads per block
		printf( "Max Threads Per Block is 512!!!\n\n" );
		return false;
	}

	// Get search Vector Length
	if (cutGetCmdLineArgumenti( argc, argv, "N", &iVal )) 
	{
		if (iVal < 1) { iVal = 10000; }
		g.nSearch = (int)iVal;
	}

	// Get Query Vector Length
	if (cutGetCmdLineArgumenti( argc, argv, "NQ", &iVal )) 
	{
		if (iVal < 1) { iVal = 100; }
		g.nQuery = (int)iVal;
	}

	// Should we do profiling (performance measurements) on application
	if (cutCheckCmdLineFlag( argc, argv, "profile") ) 
	{
		g.profile = true;
	}
	else
	{
		g.profile = false;
	}

	if (g.profile)
	{
		// Get Skip First Last flag
		if (cutCheckCmdLineFlag( argc, argv, "skip") ) 
		{
			g.profileSkipFirstLast = true;
		}
		else
		{
			g.profileSkipFirstLast = false;
		}

		// Get Number of Iterations for Profiling performance
		if (cutGetCmdLineArgumenti( argc, argv, "profile", &iVal )) 
		{
			if (iVal < 1) { iVal = 100; }
			g.profileRequestedLoops = iVal;
			if (g.profileSkipFirstLast)
			{
				g.profileActualLoops = g.profileRequestedLoops + 2;
			}
			else
			{
				g.profileActualLoops = g.profileRequestedLoops;
			}
		}
	}

	// Should we dump verbose results
	if (cutCheckCmdLineFlag( argc, argv, "verbose") ) 
	{
		// Get Number of Iterations for Profiling performance
		if (cutGetCmdLineArgumenti( argc, argv, "verbose", &iVal )) 
		{
			g.dumpVerbose = (unsigned int)iVal;
		}
		else
		{
			g.dumpVerbose = 1;
		}
	}

	// Should we dump verbose results
	if (cutCheckCmdLineFlag( argc, argv, "rowbyrow") ) 
	{
		g.rowByRow = true;
	}
	else
	{
		g.rowByRow = false;
	}

	// Success
	return true;
}


/*---------------------------------------------------------
  Name:	InitShapeDefaults
---------------------------------------------------------*/

void InitShapeDefaults( BlockGridShape & bgShape )
{
	// Default Thread, Grid, Vector Properties
	bgShape.nElems			= 100;

	bgShape.threadsPerRow	= 1;
	bgShape.rowsPerBlock	= 1;
	bgShape.threadsPerBlock	= bgShape.threadsPerRow * bgShape.rowsPerBlock;

	bgShape.blocksPerRow	= 100;
	bgShape.rowsPerGrid		= 1;
	bgShape.blocksPerGrid	= bgShape.blocksPerRow * bgShape.rowsPerGrid;

	bgShape.W               = bgShape.threadsPerRow * bgShape.blocksPerRow;
	bgShape.H				= bgShape.rowsPerBlock * bgShape.rowsPerGrid;

	bgShape.nPadded			= bgShape.W * bgShape.H;
}


/*---------------------------------------------------------
  Name:	DumpBlockGridShape
---------------------------------------------------------*/

void DumpBlockGridShape( const BlockGridShape & bgShape )
{
	printf( "N = %d, NPadded = %d\n",
		    bgShape.nElems, bgShape.nPadded );
	printf( "Block (%d TPR x %d RPB) = %d TPB\n",
		    bgShape.threadsPerRow, bgShape.rowsPerBlock,
			bgShape.threadsPerBlock );
	printf( "Grid (%d BPR x %d RPG)  = %d BPG\n",
		    bgShape.blocksPerRow, bgShape.rowsPerGrid,
			bgShape.blocksPerGrid );
	printf( "W = %d, H = %d\n",
		    bgShape.W, bgShape.H );
}


/*---------------------------------------------------------
  Name:	 IntLog2
  Desc:  Find the log base 2 for a 32-bit unsigned integer
---------------------------------------------------------*/

// O( log n ) Binary Search (Portable)
	// 15 operations worst case    (n = 0xFFFFFFF)
	// 13 operations expected case (n = 0xFFFF)
	//  7 operations best case     (n = 0)
unsigned int IntLog2( unsigned int inVal )
{
	// Binary search for log2(n)
	unsigned int logVal = 0u;

	// Note: we assume unsigned integers are 32-bit
		// if unsigned integers are actually 64-bit 
		// then uncomment the following line
//  if (inVal >= 1u << 32u) { inVal >>= 32u; logVal |= 32u; }

	if (inVal >= 1u << 16u) { inVal >>= 16u; logVal |= 16u; }
	if (inVal >= 1u <<  8u) { inVal >>=  8u; logVal |=  8u; }
	if (inVal >= 1u <<  4u) { inVal >>=  4u; logVal |=  4u; }
	if (inVal >= 1u <<  2u) { inVal >>=  2u; logVal |=  2u; }
	if (inVal >= 1u <<  1u) { logVal |= 1u; }

	return logVal;
}


/*---------------------------------------------------------
  Name:	 LBM()
  Desc:  Find the left balanced median of 'n' elements
---------------------------------------------------------*/

unsigned int LBM( unsigned int n )
{
	static unsigned int medianTable[32] = 
	{ 
		0u,			// Wasted space (but necessary for 1-based indexing)
		1u,							// Level 1
		2u,2u,						// Level 2
		3u,4u,4u,4u,				// Level 3
		5u,6u,7u,8u,8u,8u,8u,8u,	// Level 4
		9u,10u,11u,12u,13u,14u,15u,16u,16u,16u,16u,16u,16u,16u,16u,16u // Level 5
	};

	// Return answer via lookup table for small 'n'
		// Also solves problem of small trees with height <= 2
	if (n <= 31u) 
	{
		return medianTable[n];
	}

	// Compute height of tree
#if (CPU_PLATFORM == CPU_INTEL_X86)
	// Find position of highest set bit
		// Non-portable solution (Use Intel Intrinsic)
	unsigned long bitPos;
	_BitScanReverse( &bitPos, (unsigned long)(n+1) );
	int h       = (int)(bitPos+1);	
#elif (CPU_PLATFORM == CPU_INTEL_X64)
	// Find position of highest set bit
		// Non-portable solution (Use Intel Intrinsic)
	unsigned long bitPos;
	_BitScanReverse64( &bitPos, (unsigned long long)(n+1) );
	int h       = (int)(bitPos+1);	
#else
	// Binary search for log2(n)
		// Portable solution
	unsigned int height  = IntLog2( n+1 );	// Get height of tree
	unsigned int h = height+1;
#endif

	// Compute Left-balanced median
	unsigned int half    = 1 << (h-2);		// 2^(h-2), Get size of left sub-tree (minus last row)
	unsigned int lastRow = min( half, n-2*half+1 );	// Get # of elements to include from last row
	return (half + lastRow);				// Return left-balanced median
}


/*---------------------------------------------------------
  Name:	 TestLBM()
  Desc:  Test Left-Balanced Median function
---------------------------------------------------------*/

bool TestLBM()
{
	printf( "LBM test: \n" );
	unsigned int i, r;
	for (i = 0; i < 102; i++)
	{
		r = LBM( i );
		printf( "%d = %d\n", i, r );
	}
	printf( "\n" );

	return true;
}

