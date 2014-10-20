/*-----------------------------------------------------------------------------
  File:  Dist_CPU.cpp
  Desc:  Host CPU scaffolding for running and testing GPU Brute Force QNN

  Log:   Created by Shawn D. Brown (4/15/07)
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

bool BruteForce3DTest();

bool GetCommandLineParams( int argc, const char ** argv, AppGlobals & g );

bool InitGlobals( AppGlobals & g );

bool ComputeBlockShapeFromVector( BlockGridShape & bgShape );

void InitShapeDefaults( BlockGridShape & bgShape );
void DumpBlockGridShape( const BlockGridShape & bgShape );

float RandomFloat( float low, float high );


/*---------------------------------------------------------
  Name:	BruteForce3DTest()
  Desc:	Run a simple test of "Point Location" 
        functionality on CUDA GPU framework
---------------------------------------------------------*/

bool BruteForce3DTest()
{
	bool bResult = false;

	/*---------------------------------
	  Step 0.  Initialize Cuda
	---------------------------------*/

	cudaError_t cuda_err = cudaSuccess;

	// set seed for rand()
	RandomInit( 2010 );

	g_app.hTimer = 0;
	cutCreateTimer( &(g_app.hTimer) );


	/*-------------------------------------------
	  Step 1.  Setup Initial parameters
	-------------------------------------------*/

		// Hard Coded for now...
	g_app.nSearch = TEST_BF_SEARCH_POINTS;
	g_app.bgShape.nElems = g_app.nSearch;
	g_app.bgShape.threadsPerRow = BFD_THREADS_PER_ROW;
	g_app.bgShape.rowsPerBlock  = BFD_ROWS_PER_BLOCK;

	bResult = ComputeBlockShapeFromVector( g_app.bgShape );
	if (false == bResult)
	{
		// Error
		return false;
	}

	// Make sure Matrix + vector is not to big to use up all device memory  // 768 Meg on Display Card
	int sizePoints = g_app.bgShape.nPadded * sizeof(float4);
	int sizeDists  = g_app.bgShape.nPadded * sizeof(float2);
	int totalMem   = sizePoints + (2*sizeDists);

	// Make sure memory required to perform this operation doesn't exceed display device memory
	if (totalMem >= g_app.cudaProps.totalGlobalMem)
	{
		// Error - not enough memory to perform operation
		printf( "Matrix + Vector are too large for available device memory, running test will crash..." );
		return false;
	}

	// Setup GPU Kernel execution parameters
		// Brute Force Distance Kernel
	dim3 dimBlock( g_app.bgShape.threadsPerRow, g_app.bgShape.rowsPerBlock, 1 );
	dim3 dimGrid( g_app.bgShape.blocksPerRow, g_app.bgShape.rowsPerGrid, 1 );


	/*-------------------------------------------
	  Step 2.  Allocate Vectors
	-------------------------------------------*/

	int nOrig = g_app.bgShape.nElems;
	int nPad  = g_app.bgShape.nPadded;
	int w     = g_app.bgShape.W;
	int h     = g_app.bgShape.H;

	/*-----------------------
	  Host Memory
	-----------------------*/

    // allocate host memory for points (2D Vector layout)
    int mem_size_Points = nPad * sizeof(float4);
    float4* h_Points = (float4*) malloc( (size_t)mem_size_Points );

	// allocate host memory for GPU Distances vector (initial brute force result)
	unsigned int mem_size_Dists_GPU = nPad * sizeof(float2);
	float2 *h_Dists_GPU = (float2*) malloc( mem_size_Dists_GPU );

	// allocate host memory for CPU Distances vector (double check results)
	unsigned int mem_size_Dists_CPU = nPad * sizeof(float);
	float* h_Dists_CPU = (float*) malloc( mem_size_Dists_CPU );

	// Allocate host memory for singleton result
	unsigned int mem_size_Result = 16 * sizeof(float2);
	float2 *h_result_GPU = (float2*) malloc( mem_size_Result );
	h_result_GPU[0].x =  0.0f;
	h_result_GPU[0].y = -1.0f;


	/*-----------------------
	  Device Memory
	-----------------------*/

	// allocate device memory for points (2D vector layout)
	float4* d_Points;
	cutilSafeCall( cudaMalloc( (void**) &d_Points, mem_size_Points ) );

	// allocate device memory for GPU Distances vector
	float2* d_Dists		= NULL;
	cutilSafeCall( cudaMalloc( (void **) &d_Dists, mem_size_Dists_GPU ) );

	// allocate device memory for Reduction Vector
		// Used for reduction
		// IE Ping Pong between dists vector and reduce vector to get answer
		// to get final answer
	bool bPingPong = true;
	float2* d_Reduce;
	cutilSafeCall( cudaMalloc( (void **) &d_Reduce, mem_size_Dists_GPU ) ); 


	/*-------------------------------------------
	  Step 3.  Initialize Vectors
	-------------------------------------------*/

	// Initialize Input points (to query against)
	int i;
	for (i = 0; i < nOrig; i++)	// Original Points
	{
		// BUGBUG - for now just randomly generate points
		// In future - we should read them in from a file...
		h_Points[i].x = RandomFloat( 0.0, 1.0 );
		h_Points[i].y = RandomFloat( 0.0, 1.0 );
		h_Points[i].z = RandomFloat( 0.0, 1.0 );
		h_Points[i].w = RandomFloat( 0.0, 1.0 );

		// Store point index in this channel
		//h_Points[i].w = (float)i;
	}

	// Initialize padded points (to query against)
	for (i = nOrig; i < nPad; i++)	// Padded points
	{
		// We want padded points to always fail...
		//	1st Approach, 
		//		Use a point that is so far away it is guranteed to never get picked
		//		Cons:  Requires advance knowledge of input point range 
		//				and query point range to pick a point
		//			    so far outside range it doesn't matter
		//	2nd Approach, 
		//		Duplicate the 1st point many times
		//		Cons:  Can fail because of numerical round-off errors
		//			IE what if the 1st point is really the closest to the query point
		//			which point wins (1st point or one of it's duplicates)
		
		//
		// 1st Approach
		//
		h_Points[i].x = 400.0f;	// Note:  Any number much larger than 46,000 and we will overflow on squaring the float
		h_Points[i].y = 400.0f;
		h_Points[i].z = 400.0f;
		h_Points[i].w = (float)-1;		// Store invalid point index in this channel

		//
		// 2nd Approach
		//
		//h_Points[i].x = h_Points[0].x;
		//h_Points[i].y = h_Points[0].y;
		//h_Points[i].z = h_Points[0].z;
		//h_Points[i].w = h_Points[0].w;
	}


	//
	//	Initial Query Point
	//
	// BUGBUG - for now just randomly generate query point
	// In future - we should get from user or file
	float4 queryPoint;
	queryPoint.x = RandomFloat( 0.0, 1.0 );
	queryPoint.y = RandomFloat( 0.0, 1.0 );
	queryPoint.z = RandomFloat( 0.0, 1.0 );
	queryPoint.w = RandomFloat( 0.0, 1.0 );

	//
	// Profile Performance Metric Initialization
	//
	float BF_P_onto_device  = 0.0f;
	float BF_D_from_device  = 0.0f;
	float BF_M_from_device  = 0.0f;
	float BF_GPU_dist		= 0.0f;
	float BF_CPU_dist		= 0.0f;
	float BF_GPU_min		= 0.0f;
	float BF_CPU_min		= 0.0f;
	bool  checkDistResults  = true;
	bool  checkMinResults   = true;

	// Result values
	int gpuMinIdx;				// Index of closest point as computed on GPU
	float gpuMinDist;			// Distance to closest point as computed on GPU

	int cpuMinIdx;				// Index of closest point as computed on CPU
	float cpuMinDist;			// Distance to closest point as computed on CPU

	float gVal, cVal;



	// Profile Measurement Loop
	unsigned int currIter;
	for (currIter = 0; currIter < g_app.profileActualLoops; currIter++)
	{

		//-------------------------------------------------------
		//	Step 3.  Move Points, Indices 
		//			 from main memory to device memory
		//-------------------------------------------------------

		if (g_app.profile)
		{
			// Start Timer
			cutResetTimer( g_app.hTimer );
			cutStartTimer( g_app.hTimer );
		}

		// Copy 'Points' vector from host memory to device memory
		cutilSafeCall( cudaMemcpy( d_Points, h_Points, mem_size_Points, cudaMemcpyHostToDevice ) );

		if (g_app.profile)
		{
			// Stop Timer and save performance measurement
			cutStopTimer( g_app.hTimer );
			
			if (g_app.profileSkipFirstLast)
			{
				if ((1 < currIter) && (currIter <= g_app.profileActualLoops))
				{
					BF_P_onto_device += cutGetTimerValue( g_app.hTimer );
				}
			}
			else
			{
				BF_P_onto_device += cutGetTimerValue( g_app.hTimer );
			}
		}


		//---------------------------------
		//	Step 4.  Call Kernel Function
		//---------------------------------

		if (g_app.profile)
		{
			// Start Timer
			cutResetTimer( g_app.hTimer );
			cutStartTimer( g_app.hTimer );
		}
		
		// Excute the Brute Force Distance Kernel
		ComputeDist3D_GPU<<< dimGrid, dimBlock >>>( d_Dists, d_Points, queryPoint, w, h  );
		
		// Check if GPU kernel execution generated an error
		cuda_err = cudaGetLastError();
		if( cudaSuccess != cuda_err) 
		{
			fprintf( stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",
			"ComputeDist3D_GPU() failed", __FILE__, __LINE__, cudaGetErrorString( cuda_err ) );
			exit( EXIT_FAILURE );
		}

		if (g_app.profile)
		{
			// Stop Timer and save performance measurement
			cutStopTimer( g_app.hTimer );
			if (g_app.profileSkipFirstLast)
			{
				if ((1 < currIter) && (currIter <= g_app.profileActualLoops))
				{
					BF_GPU_dist += cutGetTimerValue( g_app.hTimer );
				}
			}
			else
			{
				BF_GPU_dist += cutGetTimerValue( g_app.hTimer );
			}
		}


		//-------------------------------------------------
		//	Step 5.  Copy result vector (distances)
		//			 from device memory to main memory
		//-------------------------------------------------

		if (g_app.doubleCheckDists)
		{
			// BUGBUG - this is a temporary step to verify brute force distance calculation
			if (g_app.profile)
			{
				// Start Timer
				cutResetTimer( g_app.hTimer );
				cutStartTimer( g_app.hTimer );
			}

			// copy result vector Z from device to host
			cutilSafeCall( cudaMemcpy( (void *) h_Dists_GPU, d_Dists, mem_size_Dists_GPU, cudaMemcpyDeviceToHost ) );

			if (g_app.profile)
			{
				// Stop Timer and save performance measurement
				cutStopTimer( g_app.hTimer );
				if (g_app.profileSkipFirstLast)
				{
					if ((1 < currIter) && (currIter <= g_app.profileActualLoops))
					{
						BF_D_from_device += cutGetTimerValue( g_app.hTimer );
					}
				}
				else
				{
					BF_D_from_device += cutGetTimerValue( g_app.hTimer );
				}
			}
		}


		/*-------------------------------------------------
		  Step 6.  Double check GPU result 
		           against CPU result
		-------------------------------------------------*/

		if (g_app.doubleCheckDists)
		{
			if (g_app.profile)
			{
				// Start Timer
				cutResetTimer( g_app.hTimer );
				cutStartTimer( g_app.hTimer );
			}

			// Compute reference solution (distances) on CPU
			ComputeDist3D_CPU( h_Dists_CPU, h_Points, queryPoint, w, h );

			if (g_app.profile)
			{
				// Stop Timer and save performance measurement
				cutStopTimer( g_app.hTimer );
				if (g_app.profileSkipFirstLast)
				{
					if ((1 < currIter) && (currIter <= g_app.profileActualLoops))
					{
						BF_CPU_dist += cutGetTimerValue( g_app.hTimer );
					}
				}
				else
				{
					BF_CPU_dist += cutGetTimerValue( g_app.hTimer );
				}
			}

			// Double check GPU Result against CPU result (for distances)
			int NCheck = nPad;
			int i;
			for (i = 0; i < NCheck; i++)
			{
				const float eps = 1.0e-2f;
				gVal = h_Dists_GPU[i].x;
				cVal = h_Dists_CPU[i];

				//printf( "[%d] GPU=%f, CPU=%f \n", i, gVal, cVal );

				if ( ((cVal - eps) >= gVal) ||
					 ((cVal + eps) <= gVal) )
				{
					// Error - Out of tolerance check range
					printf( "[%d] GPU %f != CPU %f \n", i, gVal, cVal );
					checkDistResults = false;
				}
			}
		}	// double check distances


		/*-------------------------------------------------
		  Step 7.  GPU Kernel to reduce distances 
		           (& index) vector
		           to single best result
		-------------------------------------------------*/

		if (g_app.profile)
		{
			// Start Timer
			cutResetTimer( g_app.hTimer );
			cutStartTimer( g_app.hTimer );
		}

		// Copy 'Distances' vector to 'Reduction' vector 
			// This is currently necessary to avoid garbage
			// results in output caused by unitialized values
		cutilSafeCall( cudaMemcpy( d_Reduce, d_Dists, mem_size_Dists_GPU, cudaMemcpyDeviceToDevice ) );

		int reduceElems  = nPad;
		dim3 reduceThreads;
		dim3 reduceGrid;
		BlockGridShape reduceShape;

		// Compute Initial Grid Shape
		reduceShape.nElems = reduceElems;
		reduceShape.threadsPerRow = BFMR_THREADS_PER_ROW;
		reduceShape.rowsPerBlock  = BFMR_ROWS_PER_BLOCK;

		ComputeBlockShapeFromVector( reduceShape );

		// Make sure we have an even number of blocks to work on
		if ((reduceShape.blocksPerRow % 2) != 0)
		{
			// Error - not an even number of blocks
			fprintf( stderr, "Error - not an even number of blocks\n" );
			return false;
		}

		reduceThreads.x = reduceShape.threadsPerRow;
		reduceThreads.y = reduceShape.rowsPerBlock;
		reduceThreads.z = 1;

		reduceGrid.x = reduceShape.blocksPerRow / 2;	// Divide by 2 (algorithm works on 2 blocks at a time)
		reduceGrid.y = reduceShape.rowsPerGrid;
		reduceGrid.z = 1;

				
		bool bReduced = false;
		bPingPong = true;
		while (!bReduced)
		{
			// Ping Pong between "Distances" and "Reduce" vectors
			if (bPingPong)
			{
				bPingPong = false;

				// Call GPU Kernel to reduce result vector by THREADS_PER_BLOCK
				Reduce_Min_GPU<<< reduceGrid, reduceThreads >>>( d_Reduce, d_Dists );
			}
			else
			{
				bPingPong = true;

				// Call GPU Kernel to reduce result vector by THREADS_PER_BLOCK
				Reduce_Min_GPU<<< reduceGrid, reduceThreads >>>( d_Dists, d_Reduce );
			}

			// Check if GPU kernel execution generated an error
			cuda_err = cudaGetLastError();
			if( cudaSuccess != cuda_err) 
			{
				fprintf( stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",
				"PLQ_GPU_BF_DIST() failed", __FILE__, __LINE__, cudaGetErrorString( cuda_err ) );
				exit( EXIT_FAILURE );
			}
			
			// Update Number of elements in reduction vector
			reduceElems = reduceShape.blocksPerGrid / 2;	// Divide by 2 - Algorithm works on 2 columns of blocks at a time
			if (reduceElems == 1)
			{
				bReduced = true;
			}
			else
			{
				// Update Shape of Grid
				reduceShape.nElems = reduceElems;
				reduceShape.threadsPerRow = BFMR_THREADS_PER_ROW;
				reduceShape.rowsPerBlock  = BFMR_ROWS_PER_BLOCK;

				ComputeBlockShapeFromVector( reduceShape );

				// Make sure we have an even number of blocks to work on
				if ((reduceShape.blocksPerRow % 2) != 0)
				{
					// Error - not even number of blocks
					fprintf( stderr, "Error - not an even number of blocks" );
					return false;
				}

				reduceThreads.x = reduceShape.threadsPerRow;
				reduceThreads.y = reduceShape.rowsPerBlock;
				reduceThreads.z = 1;
		
				reduceGrid.x = reduceShape.blocksPerRow / 2;	// Divide by 2 (algorithm works on 2 blocks at a time)
				reduceGrid.y = reduceShape.rowsPerGrid;
				reduceGrid.z = 1;
			}
		}

		if (g_app.profile)
		{
			// Stop Timer and save performance measurement
			cutStopTimer( g_app.hTimer );
			if (g_app.profileSkipFirstLast)
			{
				if ((1 < currIter) && (currIter <= g_app.profileActualLoops))
				{
					BF_GPU_min += cutGetTimerValue( g_app.hTimer );
				}
			}
			else
			{
				BF_GPU_min += cutGetTimerValue( g_app.hTimer );
			}
		}


		/*-------------------------------------------------
		  Step 8.  Read Result from GPU
		-------------------------------------------------*/

		if (g_app.profile)
		{
			// Start Timer
			cutResetTimer( g_app.hTimer );
			cutStartTimer( g_app.hTimer );
		}

		// Copy closest point result from device to host memory (singleton distance & index)
		if (!bPingPong)
		{
			cuda_err = cudaMemcpy( h_result_GPU, d_Reduce, mem_size_Result, cudaMemcpyDeviceToHost );
		}
		else
		{
			cuda_err = cudaMemcpy( h_result_GPU, d_Dists, mem_size_Result, cudaMemcpyDeviceToHost );
		}
		if (cudaSuccess != cuda_err)
		{
		    fprintf( stderr, "Cuda error in file '%s' in line %i : %s.\n",
					 __FILE__, __LINE__, cudaGetErrorString( cuda_err ) );
			exit( EXIT_FAILURE );
		}

		if (g_app.profile)
		{
			// Stop Timer and save performance measurement
			cutStopTimer( g_app.hTimer );
			if (g_app.profileSkipFirstLast)
			{
				if ((1 < currIter) && (currIter <= g_app.profileActualLoops))
				{
					BF_M_from_device += cutGetTimerValue( g_app.hTimer );
				}
			}
			else
			{
				BF_M_from_device += cutGetTimerValue( g_app.hTimer );
			}
		}

		// Save Results 
		gpuMinDist = h_result_GPU[0].x;
		gpuMinIdx = (unsigned int)(h_result_GPU[0].y);


		/*-------------------------------------------------
		  Step 9.  Double check GPU result 
		           against CPU result
		-------------------------------------------------*/

		if (g_app.doubleCheckMin)
		{
			// BUGBUG - this is a temporary step to verify brute force distance calculation
			if (g_app.profile)
			{
				// Start Timer
				cutResetTimer( g_app.hTimer );
				cutStartTimer( g_app.hTimer );
			}

			// Compute reference solution (distances) on CPU
			Reduce_Min_CPU( cpuMinIdx, cpuMinDist, h_Points, queryPoint, nOrig );

			if (g_app.profile)
			{
				// Stop Timer and save performance measurement
				cutStopTimer( g_app.hTimer );
				if (g_app.profileSkipFirstLast)
				{
					if ((1 < currIter) && (currIter <= g_app.profileActualLoops))
					{
						BF_CPU_min += cutGetTimerValue( g_app.hTimer );
					}
				}
				else
				{
					BF_CPU_min += cutGetTimerValue( g_app.hTimer );
				}
			}

			//
			// Double check GPU Result against CPU result
			//

				// Index check
			if (gpuMinIdx != cpuMinIdx)
			{
				// Warning - Indices are not the same
				// Note:  This is not truly an error unless 
				//		  the closest distances also don't match
				printf( "WARN - MIN GPU IDX %d != MIN CPU IDX %d \n", gpuMinIdx, cpuMinIdx );
			}

				// Distance Check
			const float minEps = 1.0e-4f;
			gVal = gpuMinDist;		
			cVal = cpuMinDist;
			if ( ((cVal - minEps) >= gVal) ||
				 ((cVal + minEps) <= gVal) )
			{
				// Error - Out of tolerance check range
				printf( "ERR  - MIN GPU DIST %f != MIN CPU DIST %f \n", i, gVal, cVal );
				checkMinResults = false;
			}
		}

	} // Profile Loops


	/*--------------------------------------------------------
	  Step 11. Print out Results
	--------------------------------------------------------*/

	int vectLen = g_app.nSearch;
	printf( "\n" );
	printf( "Search Vector Length = %d\n", vectLen );
	printf( "Query Point:          <%f %f %f>\n", 
			 queryPoint.x, queryPoint.y, queryPoint.z );
	printf( "GPU Closest Distance: %f\n", gpuMinDist );
	printf( "GPU Closest Index:    %d\n", gpuMinIdx );
	printf( "GPU Closest Point:    <%f %f %f>\n",
		    h_Points[gpuMinIdx].x, h_Points[gpuMinIdx].y, h_Points[gpuMinIdx].z );
	if (g_app.doubleCheckMin)
	{
		printf( "CPU Closest Distance: %f\n", cpuMinDist );
		printf( "CPU Closest Index:    %d\n", cpuMinIdx );
		printf( "CPU Closest Point:    <%f %f %f>\n",
		        h_Points[cpuMinIdx].x, h_Points[cpuMinIdx].y, h_Points[cpuMinIdx].z );
	}
	printf( "\n" );
	

	/*--------------------------------------------------------
	  Step 12. Print out Profile Performance Metrics
	--------------------------------------------------------*/

	// Does GPU Distance Kernel match up with CPU ?!?
	if (g_app.doubleCheckDists)
	{
		if (true == checkDistResults)
		{
			printf( "Distance check: CPU and GPU results agree within tolerance.\n" );
		}
		else
		{
			printf( "Distance check: CPU and GPU results don't agree within tolerance !!!\n" );
		}
	}

	// Does GPU Min Distance Kernel match up with CPU ?!?
	if (g_app.doubleCheckMin)
	{
		if (true == checkMinResults)
		{
			printf( "Min Distance check: CPU and GPU results agree within tolerance.\n" );
		}
		else
		{
			printf( "Min Distance check: CPU and GPU results don't agree within tolerance !!!\n" );
		}
	}

	// Dump Profile Info
	if (g_app.profile)
	{
		float loops = (float)g_app.profileActualLoops;
		float o_l = 1.0f / loops;

		float avgP   = BF_P_onto_device * o_l;
		float avgD   = BF_D_from_device * o_l;
		float avgM   = BF_M_from_device * o_l;
		float avgGPUdist = BF_GPU_dist * o_l;
		float avgCPUdist = BF_CPU_dist * o_l;
		float avgGPUmin  = BF_GPU_min * o_l;
		float avgCPUmin  = BF_CPU_min * o_l;

		// Verbose
		printf( "Number of profile loops = %f.\n", loops );
		printf( "BF - Copy 'Point' vector onto GPU,    time: %f msecs.\n", avgP );
		printf( "BF - Copy 'Dists' vector from GPU,    time: %f msecs.\n", avgD );
		printf( "BF - Copy 'Results' from GPU,         time: %f msecs.\n", avgM );
		printf( "BF - GPU Distance computation,        time: %f msecs.\n", avgGPUdist );
		printf( "BF - CPU Distance computation,        time: %f msecs.\n", avgCPUdist );
		printf( "BF - GPU Min Distance computation,    time: %f msecs.\n", avgGPUmin );
		printf( "BF - CPU Min Distance computation,    time: %f msecs.\n\n", avgCPUmin );

		// Terse
		//printf( "BF - P, D, M, G_D, C_D, G_M, C_M\n" );
		//printf( "     %f, %f, %f, %f, %f, %f, %f\n\n", avgP, avgD, avgM, avgGPUdist, avgCPUdist, avgGPUmin, avgCPUmin );

	}
	else
	{
		printf( "BF - Copy 'Point' vector onto GPU,    time: %f msecs.\n", BF_P_onto_device );
		printf( "BF - Copy 'Dists' vector from GPU,    time: %f msecs.\n", BF_D_from_device );
		printf( "BF - Copy 'Results' from GPU,         time: %f msecs.\n", BF_M_from_device );
		printf( "BF - GPU Distance computation,        time: %f msecs.\n", BF_GPU_dist );
		printf( "BF - CPU Distance computation,        time: %f msecs.\n", BF_CPU_dist );
		printf( "BF - GPU Min Distance computation,    time: %f msecs.\n", BF_GPU_min );
		printf( "BF - CPU Min Distance computation,    time: %f msecs.\n\n", BF_CPU_min );
	}


	/*---------------------------------
      Step 13.  Cleanup vector memory
	---------------------------------*/

    printf( "Shutting Down...\n" );

    // clean up allocations
    free( h_Points );
    free( h_Dists_GPU );
    free( h_Dists_CPU );
    free( h_result_GPU );

    cutDeleteTimer( g_app.hTimer );

    cutilSafeCall( cudaFree( d_Points ) );
    cutilSafeCall( cudaFree( d_Dists ) );
    cutilSafeCall( cudaFree( d_Reduce ) );

	printf( "Shutdown done...\n\n" );

	// Success
	return true;
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
#if (CUDA_PLATFORM == CUDA_DEVICE)
	unsigned int cudaContextFlags = CU_CTX_BLOCKING_SYNC | CU_CTX_MAP_HOST;
#elif (CUDA_PLATFORM == CUDA_CUDA)
#endif

	cudaError_t cudaResult = cudaSuccess;

#if (CUDA_PLATFORM == CUDA_DEVICE)
	CUresult cuda_Result = CUDA_SUCCESS;

	// Initialize CUDA
	unsigned int cudaFlags = 0;
	cuda_Result = cuInit( cudaFlags );
	if (CUDA_SUCCESS != cuda_Result)
	{
		// Error - cuInit() failed
		fprintf( stderr, "InitCuda() - cuInit() failed, error = %x in file '%s' in line %i.\n", 
				 cuda_Result, __FILE__, __LINE__ );
		goto lblError;
	}

	// Get count of CUDA Devices
	cuda_Result = cuDeviceGetCount(&nDevices);
	if (CUDA_SUCCESS != cuda_Result)
	{
		// Error - cuGetDeviceCount() failed
		fprintf( stderr, "InitCuda() - cuDeviceGetCount() failed, error = %x in file '%s' in line %i.\n", 
				 cuda_Result, __FILE__, __LINE__ );
		goto lblError;
	}

	if (nDevices <= 0)
	{
		// No Valid Display Device found
		cuda_Result = CUDA_ERROR_INVALID_DEVICE;
		fprintf( stderr, "InitCuda() - no valid display device found, error = %x in file '%s' in line %i.\n", 
				 cuda_Result, __FILE__, __LINE__ );
		goto lblError;
	}
	else if (nDevices >= 2)
	{
		deviceToUse = 1;
	}

	// Get Specified Device
	cuda_Result = cuDeviceGet( &(g.currDevice), deviceToUse );
	if (CUDA_SUCCESS != cudaResult )
	{
		// Error - cudaDeviceGet() failed
		fprintf( stderr, "InitCuda() - cuDeviceGet() failed, error = %x in file '%s' in line %i.\n", 
				 cuda_Result, __FILE__, __LINE__ );
		goto lblError;
	}

	// Get RAW Device Properties
	cuda_Result = cuDeviceGetProperties( &(g.rawProps), g.currDevice );
	if (CUDA_SUCCESS != cuda_Result )
	{
		// Error - cudaDeviceGet() failed
		fprintf( stderr, "InitCuda() - cuDeviceGetProperties() failed, error = %x in file '%s' in line %i.\n", 
				 cuda_Result, __FILE__, __LINE__ );
		goto lblError;
	}

	// Set up the CUDA context
	cuda_Result = cuCtxCreate( &g.currContext, cudaContextFlags, g.currDevice );
	if ( CUDA_SUCCESS != cuda_Result )
	{
		// Error - cudaDeviceGet() failed
	    fprintf( stderr, "InitCuda() - cuCtxCreate() failed, error = %x in file '%s' in line %i.\n", 
			     cuda_Result, __FILE__, __LINE__ );
		goto lblError;
	}

	// Get CUDA Display Device Properties
	cudaResult = cudaGetDeviceProperties( &(g.cudaProps) , deviceToUse );
	if ( cudaSuccess != cudaResult )
	{
		// Error - cudaDeviceGet() failed
		fprintf( stderr, "InitCuda() - cudaGetDeviceProperties() failed, error = %x in file '%s' in line %i.\n", 
				 cudaResult, __FILE__, __LINE__ );
		goto lblError;
	}


#elif (CUDA_PLATFORM == CUDA_CUDA)

	// Pick Display Device to perform GPU calculations on...
	cudaResult = cudaGetDeviceCount( &nDevices );
	if ( cudaSuccess != cudaResult )
	{
		// Error - cudaGetDeviceCount() failed
		fprintf( stderr, "InitCuda() - cudaGetDeviceCount() failed, error = %x in file '%s' in line %i.\n", 
				 cudaResult, __FILE__, __LINE__ );
		goto lblError;
	}

		// Note:  Assumes Device 0 = primary display device
		//		  Assumes Device 1 = work horse for CUDA
	if (nDevices <= 0)
	{
		// No Valid Display Device found
		cudaResult = cudaErrorInvalidDevice;
		fprintf( stderr, "InitCuda() - no valid display device found, error = %x in file '%s' in line %i.\n", 
				 cudaResult, __FILE__, __LINE__ );
		goto lblError;
	}
	else if (nDevices >= 2)
	{
		deviceToUse = 1;
	}

	// Get Display Device Properties
	cudaResult = cudaGetDeviceProperties( &(g.cudaProps) , deviceToUse );
	if ( cudaSuccess != cudaResult )
	{
		// Error - cudaDeviceGet() failed
		fprintf( stderr, "InitCuda() - cudaGetDeviceProperties() failed, error = %x in file '%s' in line %i.\n", 
				 cudaResult, __FILE__, __LINE__ );
		goto lblError;
	}

	// Setup Display Device
	cudaResult = cudaSetDevice( deviceToUse );
	if ( cudaSuccess != cudaResult )
	{
		// Error - cudaDeviceGet() failed
		fprintf( stderr, "InitCuda() - cudaSetDevice() failed, error = %x in file '%s' in line %i.\n", 
				 cudaResult, __FILE__, __LINE__ );
		goto lblError;
	}
#endif // CUDA_CUDA

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
#if (CUDA_PLATFORM == CUDA_DEVICE)
	// Detach CUDA from current thread
	CUresult cuda_Result = CUDA_SUCCESS;
	cuda_Result = cuCtxDetach( g_app.currContext );
	if (CUDA_SUCCESS != cuda_Result)
	{
		// Error - cuCtxDetach() failed
		fprintf( stderr, "FiniCUDA() - cuCtxDetach() failed, error = %x in file '%s' in line %i.\n", 
				 cuda_Result, __FILE__, __LINE__ );
		return false;
	}
#endif

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

	// Cuda Properties
	size_t byteSize;
#if (CUDA_PLATFORM == CUDA_DEVICE)
	g.currDevice        = 0;

	// Initialize cuda device props to zero
	byteSize = sizeof( g.rawProps );
	memset( &g.rawProps, 0, byteSize );

#endif

	// Initialize cuda props to zero
	byteSize = sizeof( g.cudaProps );
	memset( &g.cudaProps, 0, byteSize );

	// Init Block Grid Shape
	InitShapeDefaults( g.bgShape );

	// App Properties
	g.nopromptOnExit    = 0u;
	g.doubleCheckDists	= 1u;

	// Profiling Info
	g.hTimer				= 0u;
	g.profile				= 1u;
	g.profileSkipFirstLast  = 0u;
	g.profileRequestedLoops	= 1u;
	g.profileActualLoops	= 1u;

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

