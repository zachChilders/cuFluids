//-----------------------------------------------------------------------------
//	CS 790-058 GPGPU
//	Final Project (Point Location using GPU)
//	
//	This file is the main entry point for the application
//	
//	by Shawn Brown (shawndb@cs.unc.edu)
//-----------------------------------------------------------------------------


//-------------------------------------
//	
//	Includes
//
//-------------------------------------

// standard includes
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

// CUDA includes
#include <cutil.h>
#include <cuda.h>

// project includes
#include "Main.h"
#include "QueryResult.h"

// GPU kernel includes
#include <FP_GPU.cu>
//#include <KD_GPU1.cu>


#if (APP_TEST == TEST_KD_KNN)
	#include <KD_NN.cu>
#else
	// KD Tree Test
	#include <KD_GPU2.cu>
#endif


//-------------------------------------
//	
//	Local Function Declarations
//
//-------------------------------------

bool InitGlobals( AppGlobals & g );
bool InitSearchVectors( AppGlobals & g, bool bNonUniformSearch, bool bNonUniformQuery, int scale );
void FiniSearchVectors( AppGlobals & g );

bool InitCUDA( AppGlobals & g );	// Initialize CUDA framework
bool FiniCUDA();					// Cleanup CUDA framework

bool GetCommandLineParams( int argc, const char ** argv, AppGlobals & g );
bool RunPLQTest( AppGlobals & g );

float RandomFloat( float low, float high );

bool ComputeBlockShapeFromVector( BlockGridShape & bgShape );
bool ComputeBlockShapeFromQueryVector( BlockGridShape & bgShape	);
void InitShapeDefaults( BlockGridShape & bgShape );
void DumpBlockGridShape( BlockGridShape & bgShape );


bool RunKDTreeTest( AppGlobals & g );



//-------------------------------------
//	
//	Function Definitions
//
//-------------------------------------


//---------------------------------------------------------
//	Name:	main()
//	Desc:	Main entry point to program
//---------------------------------------------------------
int main( int argc, const char** argv )
{
	bool bResult;
	int rc = EXIT_FAILURE;
	AppGlobals g;

	// Initialize Global Variables to Default settings
	bResult = InitGlobals( g );
	if (false == bResult)
	{
		goto lblCleanup;
	}

	// Get Command Line Parameters
		// Which overrides some of the global settings
    printf( "Get Command Line Params...\n" );
	bResult = GetCommandLineParams( argc, argv, g );
	if (false == bResult)
	{
		goto lblCleanup;
	}
    printf( "Done getting Command Line Params...\n\n" );

	// Initialize CUDA display device
    printf( "Initializing Device...\n" );
	bResult = InitCUDA( g );
	if (false == bResult)
	{
		goto lblCleanup;
	}
	printf( "Done Initializing Device...\n\n" );

	// Run Point Location Query Test (BruteForce)
	//bResult = RunPLQTest( g );
	//if (false == bResult)
	//{
	//	goto lblCleanup;
	//}

	// Run Point Location Query Test (KDTree)
	bResult = RunKDTreeTest( g );
	if (false == bResult)
	{
		goto lblCleanup;
	}
	
	// Success
	rc = EXIT_SUCCESS;

lblCleanup:
	// Cleanup
	FiniCUDA();

	// Prompt User before exiting ?!?
	if (!g.nopromptOnExit)
	{
        printf("\nPress ENTER to exit...\n");
        getchar();
	}

	// Exit Application
	exit( rc );
}


//---------------------------------------------------------
//	Name:	RunPLQTest()
//	Desc:	Run a simple test of "Point Location" 
//			functionality on CUDA GPU framework
//---------------------------------------------------------

bool RunPLQTest( AppGlobals & g )
{
	bool bResult = false;

	//---------------------------------
    //	Step 0.  Initialize
	//---------------------------------

    cudaError_t cuda_err = cudaSuccess;

    // set seed for rand()
    srand( 2006 );

	g.hTimer = 0;
    cutCreateTimer( &(g.hTimer) );


	//-------------------------------------------
    //	Step 1.  Setup Initial parameters
	//-------------------------------------------

	//
	// Compute efficient grid and block shape for Brute Force Distance Kernel
	//

		// Hard Coded for now...
		//g.bgShape.nElems = ???;
	g.bgShape.threadsPerRow = BFD_THREADS_PER_ROW;
	g.bgShape.rowsPerBlock  = BFD_ROWS_PER_BLOCK;

	bResult = ComputeBlockShapeFromVector( g.bgShape );
	if (false == bResult)
	{
		// Error
	}

	// Make sure Matrix + vector is not to big to use up all device memory  // 768 Meg on Display Card
	unsigned int sizePoints = g.bgShape.nPadded * sizeof(float4);
	unsigned int sizeDists  = g.bgShape.nPadded * sizeof(float2);
	unsigned int totalMem = sizePoints + (2*sizeDists);

	// Make sure memory required to perform this operation doesn't exceed display device memory
	if (totalMem >= g.cudaProps.bytes)
	{
		printf( "Matrix + Vector are too large for available device memory, running test will crash..." );
		return bResult;
	}

    // Setup GPU Kernel execution parameters
		// Brute Force Distance Kernel
    dim3 bfdThreads( g.bgShape.threadsPerRow, g.bgShape.rowsPerBlock, 1 );
    dim3 bfdGrid( g.bgShape.blocksPerRow, g.bgShape.rowsPerGrid, 1 );

		// Brute Force Min Reduction Kernel
    dim3 bfmrThreads( BFMR_THREADS_PER_ROW, BFMR_ROWS_PER_BLOCK, 1 );
    dim3 bfmrGrid( g.bgShape.blocksPerRow, g.bgShape.rowsPerGrid, 1 );


	//
	// Print out Initialization Parameters
	//
	DumpBlockGridShape( g.bgShape );

	printf( "Skip first Last Loops   = %d\n", g.profileSkipFirstLast );
	printf( "# Requested Loops       = %d\n", g.profileRequestedLoops );
	printf( "# Actual Loops          = %d\n\n", g.profileActualLoops );



	//-------------------------------------------
    //	Step 2.  Allocate Vectors
	//-------------------------------------------

	unsigned int nOrig = g.bgShape.nElems;
	unsigned int nPad  = g.bgShape.nPadded;
	unsigned int w     = g.bgShape.W;
	unsigned int h     = g.bgShape.H;

    // allocate host memory for points (2D Vector layout)
    unsigned int mem_size_Points = nPad * sizeof(float4);
    float4* h_Points = (float4*) malloc( mem_size_Points );

    // allocate device memory for points (2D vector layout)
    float4* d_Points;
    CUDA_SAFE_CALL( cudaMalloc( (void **) &d_Points, mem_size_Points ) );

	// allocate host memory for GPU Distances vector (initial brute force result)
	unsigned int mem_size_Dists_GPU = nPad * sizeof(float2);
	float2 *h_Dists_GPU = (float2*) malloc( mem_size_Dists_GPU );

	// allocate device memory for GPU Distances vector
	float2* d_Dists		= NULL;
	CUDA_SAFE_CALL( cudaMalloc( (void **) &d_Dists, mem_size_Dists_GPU ) );

	// allocate device memory for Reduction Vector
		// Used for reduction
		// IE Ping Pong between dists vector and reduce vector to get answer
		// to get final answer
	bool bPingPong = true;
	float2* d_Reduce;
    CUDA_SAFE_CALL( cudaMalloc( (void **) &d_Reduce, mem_size_Dists_GPU ) ); 

	// allocate host memory for CPU Distances vector (double check results)
	unsigned int mem_size_Dists_CPU = nPad * sizeof(float);
	float* h_Dists_CPU = (float*) malloc( mem_size_Dists_CPU );

	// Allocate host memory for singleton result
	unsigned int mem_size_Result = 16 * sizeof(float2);
	float2 *h_result_GPU = (float2*) malloc( mem_size_Result );
	h_result_GPU[0].x =  0.0f;
	h_result_GPU[0].y = -1.0f;



	//-------------------------------------------
    //	Step 3.  Initialize Vectors
	//-------------------------------------------

	//
	// Initialize Input points (to query against)
	//
	unsigned int i;
	for (i = 0; i < nOrig; i++)	// Original Points
	{
		// BUGBUG - for now just randomly generate points
		// In future - we should read them in from a file...
		h_Points[i].x = RandomFloat( 0.0, 1.0 );
		h_Points[i].y = RandomFloat( 0.0, 1.0 );
		h_Points[i].z = RandomFloat( 0.0, 1.0 );

		// Store point index in this channel
		h_Points[i].w = (float)i;
	}
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
	queryPoint.w = 0.0f;

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
	unsigned int gpuMinIdx;		// Index of closest point as computed on GPU
	float gpuMinDist;			// Distance to closest point as computed on GPU

	unsigned int cpuMinIdx;		// Index of closest point as computed on CPU
	float cpuMinDist;			// Distance to closest point as computed on CPU

	float gVal, cVal;



	// Profile Measurement Loop
	unsigned int currIter;
	for (currIter = 0; currIter < g.profileActualLoops; currIter++)
	{

		//-------------------------------------------------------
		//	Step 3.  Move Points, Indices 
		//			 from main memory to device memory
		//-------------------------------------------------------

		if (g.profile)
		{
			// Start Timer
			cutResetTimer( g.hTimer );
			cutStartTimer( g.hTimer );
		}

		// Copy 'Points' vector from host memory to device memory
		CUDA_SAFE_CALL( cudaMemcpy( d_Points, h_Points, mem_size_Points, cudaMemcpyHostToDevice ) );

		if (g.profile)
		{
			// Stop Timer and save performance measurement
			cutStopTimer( g.hTimer );
			if (g.profileSkipFirstLast)
			{
				if ((1 < currIter) && (currIter <= g.profileActualLoops))
				{
					BF_P_onto_device += cutGetTimerValue( g.hTimer );
				}
			}
			else
			{
				BF_P_onto_device += cutGetTimerValue( g.hTimer );
			}
		}


		//---------------------------------
		//	Step 4.  Call Kernel Function
		//---------------------------------

		if (g.profile)
		{
			// Start Timer
			cutResetTimer( g.hTimer );
			cutStartTimer( g.hTimer );
		}
		
		// Excute the Brute Force Distance Kernel
		PLQ_GPU_BF_DIST<<< bfdGrid, bfdThreads >>>( d_Dists, d_Points, queryPoint, w, h  );
		
		// Check if GPU kernel execution generated an error
		cuda_err = cudaGetLastError();
		if( cudaSuccess != cuda_err) 
		{
			fprintf( stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",
			"PLQ_GPU_BF_DIST() failed", __FILE__, __LINE__, cudaGetErrorString( cuda_err ) );
			exit( EXIT_FAILURE );
		}

		if (g.profile)
		{
			// Stop Timer and save performance measurement
			cutStopTimer( g.hTimer );
			if (g.profileSkipFirstLast)
			{
				if ((1 < currIter) && (currIter <= g.profileActualLoops))
				{
					BF_GPU_dist += cutGetTimerValue( g.hTimer );
				}
			}
			else
			{
				BF_GPU_dist += cutGetTimerValue( g.hTimer );
			}
		}


		//-------------------------------------------------
		//	Step 5.  Copy result vector (distances)
		//			 from device memory to main memory
		//-------------------------------------------------

		if (g.doubleCheckDists)
		{
			// BUGBUG - this is a temporary step to verify brute force distance calculation
			if (g.profile)
			{
				// Start Timer
				cutResetTimer( g.hTimer );
				cutStartTimer( g.hTimer );
			}

			// copy result vector Z from device to host
			CUDA_SAFE_CALL( cudaMemcpy( (void *) h_Dists_GPU, d_Dists, mem_size_Dists_GPU, cudaMemcpyDeviceToHost ) );

			if (g.profile)
			{
				// Stop Timer and save performance measurement
				cutStopTimer( g.hTimer );
				if (g.profileSkipFirstLast)
				{
					if ((1 < currIter) && (currIter <= g.profileActualLoops))
					{
						BF_D_from_device += cutGetTimerValue( g.hTimer );
					}
				}
				else
				{
					BF_D_from_device += cutGetTimerValue( g.hTimer );
				}
			}
		}


		//-------------------------------------------------
		//	Step 6.  Double check GPU result 
		//			 against CPU result
		//-------------------------------------------------

		if (g.doubleCheckDists)
		{
			if (g.profile)
			{
				// Start Timer
				cutResetTimer( g.hTimer );
				cutStartTimer( g.hTimer );
			}

			// Compute reference solution (distances) on CPU
			PLQ_CPU_BF_DIST( h_Dists_CPU, h_Points, queryPoint, w, h );

			if (g.profile)
			{
				// Stop Timer and save performance measurement
				cutStopTimer( g.hTimer );
				if (g.profileSkipFirstLast)
				{
					if ((1 < currIter) && (currIter <= g.profileActualLoops))
					{
						BF_CPU_dist += cutGetTimerValue( g.hTimer );
					}
				}
				else
				{
					BF_CPU_dist += cutGetTimerValue( g.hTimer );
				}
			}

			// Double check GPU Result against CPU result (for distances)
			unsigned int NCheck = nPad;
			unsigned int i;
			for (i = 0; i < NCheck; i++)
			{
				const float eps = 1.0e-2;
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


		//-------------------------------------------------
		//	Step 7.  GPU Kernel to reduce distances 
		//		     (& index) vector
		//		     to single best result
		//-------------------------------------------------

		if (g.profile)
		{
			// Start Timer
			cutResetTimer( g.hTimer );
			cutStartTimer( g.hTimer );
		}

		// Copy 'Distances' vector to 'Reduction' vector 
			// This is currently necessary to avoid garbage
			// results in output caused by unitialized values
		CUDA_SAFE_CALL( cudaMemcpy( d_Reduce, d_Dists, mem_size_Dists_GPU, cudaMemcpyDeviceToDevice ) );


		unsigned int reduceElems  = nPad;
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
				PLQ_GPU_BF_MIN<<< reduceGrid, reduceThreads >>>( d_Reduce, d_Dists );
			}
			else
			{
				bPingPong = true;

				// Call GPU Kernel to reduce result vector by THREADS_PER_BLOCK
				PLQ_GPU_BF_MIN<<< reduceGrid, reduceThreads >>>( d_Dists, d_Reduce );
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

		if (g.profile)
		{
			// Stop Timer and save performance measurement
			cutStopTimer( g.hTimer );
			if (g.profileSkipFirstLast)
			{
				if ((1 < currIter) && (currIter <= g.profileActualLoops))
				{
					BF_GPU_min += cutGetTimerValue( g.hTimer );
				}
			}
			else
			{
				BF_GPU_min += cutGetTimerValue( g.hTimer );
			}
		}


		//-------------------------------------------------
		//	Step 8.  Read Result from GPU
		//-------------------------------------------------

		if (g.profile)
		{
			// Start Timer
			cutResetTimer( g.hTimer );
			cutStartTimer( g.hTimer );
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

		if (g.profile)
		{
			// Stop Timer and save performance measurement
			cutStopTimer( g.hTimer );
			if (g.profileSkipFirstLast)
			{
				if ((1 < currIter) && (currIter <= g.profileActualLoops))
				{
					BF_M_from_device += cutGetTimerValue( g.hTimer );
				}
			}
			else
			{
				BF_M_from_device += cutGetTimerValue( g.hTimer );
			}
		}

		// Save Results 
		gpuMinDist = h_result_GPU[0].x;
		gpuMinIdx = (unsigned int)(h_result_GPU[0].y);


		//-------------------------------------------------
		//	Step 9.  Double check GPU result 
		//			 against CPU result
		//-------------------------------------------------

		if (g.doubleCheckMin)
		{
			// BUGBUG - this is a temporary step to verify brute force distance calculation
			if (g.profile)
			{
				// Start Timer
				cutResetTimer( g.hTimer );
				cutStartTimer( g.hTimer );
			}

			// Compute reference solution (distances) on CPU
			PLQ_CPU_BF_DIST_MIN( cpuMinIdx, cpuMinDist, h_Points, queryPoint, nOrig );

			if (g.profile)
			{
				// Stop Timer and save performance measurement
				cutStopTimer( g.hTimer );
				if (g.profileSkipFirstLast)
				{
					if ((1 < currIter) && (currIter <= g.profileActualLoops))
					{
						BF_CPU_min += cutGetTimerValue( g.hTimer );
					}
				}
				else
				{
					BF_CPU_min += cutGetTimerValue( g.hTimer );
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
			const float minEps = 1.0e-4;
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


	//--------------------------------------------------------
	// Step 11. Print out Results
	//--------------------------------------------------------

	printf( "\n" );
	printf( "Query Point:          <%f %f %f>\n", 
			 queryPoint.x, queryPoint.y, queryPoint.z );
	printf( "GPU Closest Distance: %f\n", gpuMinDist );
	printf( "GPU Closest Index:    %d\n", gpuMinIdx );
	printf( "GPU Closest Point:    <%f %f %f>\n",
		    h_Points[gpuMinIdx].x, h_Points[gpuMinIdx].y, h_Points[gpuMinIdx].z );
	if (g.doubleCheckMin)
	{
		printf( "CPU Closest Distance: %f\n", cpuMinDist );
		printf( "CPU Closest Index:    %d\n", cpuMinIdx );
		printf( "CPU Closest Point:    <%f %f %f>\n",
		        h_Points[cpuMinIdx].x, h_Points[cpuMinIdx].y, h_Points[cpuMinIdx].z );
	}
	printf( "\n" );
	

	//--------------------------------------------------------
	// Step 12. Print out Profile Performance Metrics
	//--------------------------------------------------------

	// Does GPU Distance Kernel match up with CPU ?!?
	if (g.doubleCheckDists)
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
	if (g.doubleCheckMin)
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
	if (g.profile)
	{
		float loops = (float)g.profileActualLoops;
		float o_l = 1.0f / loops;

		float avgP   = BF_P_onto_device * o_l;
		float avgD   = BF_D_from_device * o_l;
		float avgM   = BF_M_from_device * o_l;
		float avgGPUdist = BF_GPU_dist * o_l;
		float avgCPUdist = BF_CPU_dist * o_l;
		float avgGPUmin  = BF_GPU_min * o_l;
		float avgCPUmin  = BF_CPU_min * o_l;

		// Verbose
		//printf( "Number of total iterations = %f.\n", loops );
		//printf( "BF - Copy 'Point' vector onto GPU,    time: %f msecs.\n", BF_P_onto_device );
		//printf( "BF - Copy 'Dists' vector from GPU,    time: %f msecs.\n", BF_D_from_device );
		//printf( "BF - Copy 'Results' from GPU,         time: %f msecs.\n", BF_M_from_device );
		//printf( "BF - GPU Distance computation,        time: %f msecs.\n", BF_GPU_dist );
		//printf( "BF - CPU Distance computation,        time: %f msecs.\n", BF_CPU_dist );
		//printf( "BF - GPU Min Distance computation,    time: %f msecs.\n", BF_GPU_min );
		//printf( "BF - CPU Min Distance computation,    time: %f msecs.\n\n", BF_CPU_min );

		// Terse
		printf( "BF - P, D, M, G_D, C_D, G_M, C_M\n" );
		printf( "     %f, %f, %f, %f, %f, %f, %f\n\n", avgP, avgD, avgM, avgGPUdist, avgCPUdist, avgGPUmin, avgCPUmin );

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


	//---------------------------------
    //	Step 13.  Cleanup vector memory
	//---------------------------------

    printf( "Shutting Down...\n" );

    // clean up allocations
    free( h_Points );
    free( h_Dists_GPU );
    free( h_Dists_CPU );
    free( h_result_GPU );
    cutDeleteTimer( g.hTimer );
    CUDA_SAFE_CALL( cudaFree( d_Points ) );
    CUDA_SAFE_CALL( cudaFree( d_Dists ) );
    CUDA_SAFE_CALL( cudaFree( d_Reduce ) );

	printf( "Shutdown done...\n\n" );

	// Success
	bResult = true;
	return bResult;
}


//---------------------------------------------------------
//	Name:	RandomFloat
//	Desc:	Generates a random float value between 0 and 1
//---------------------------------------------------------

float RandomFloat( float low, float high )
{
    float t = (float)rand() / (float)RAND_MAX;
    return (1.0f - t) * low + t * high;
}


//---------------------------------------------------------
//	Name:	InitCUDA
//	Desc:	Initialize CUDA system for GPU processing
//---------------------------------------------------------

// Runtime API version...
bool InitCUDA( AppGlobals & g )
{
	bool bResult = false;
	int nDevices = 0;
	int deviceToUse = 0;

	cudaError_t cudaResult = cudaSuccess;

#if (CUDA_PLATFORM == CUDA_DEVICE)
	CUresult cuda_Result = CUDA_SUCCESS;

	// Initialize CUDA
    cuda_Result = cuInit();
	if (CUDA_SUCCESS != cuda_Result)
	{
		// Error - cudaGetDeviceCount() failed
	    fprintf( stderr, "InitCuda() - cuInit() failed, error = %x in file '%s' in line %i.\n", 
			     cuda_Result, __FILE__, __LINE__ );
        goto lblError;
	}

	// Get count of CUDA Devices
	cuda_Result = cuDeviceGetCount(&nDevices);
	if (CUDA_SUCCESS != cuda_Result)
	{
		// Error - cudaGetDeviceCount() failed
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
    cuda_Result = cuCtxCreate( g.currDevice );
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


//---------------------------------------------------------
//	Name:	FiniCUDA
//	Desc:	Cleanup CUDA system
//---------------------------------------------------------

bool FiniCUDA()
{
#if (CUDA_PLATFORM == CUDA_DEVICE)
	// Detach CUDA from current thread
	CUresult cuda_Result = CUDA_SUCCESS;
	cuda_Result = cuCtxDetach();
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


//---------------------------------------------------------
//	Name:	GetCommandLineParameters
//	Desc:	
//---------------------------------------------------------

bool GetCommandLineParams
( 
	int argc,			// Count of Command Line Parameters
	const char** argv,	// List of Command Line Parameters
	AppGlobals & g		// Structure to store results in
)
{
	int iVal;

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
    if (cutCheckCmdLineFlag( argc, argv, "chkDist") ) 
	{
		g.doubleCheckDists = true;
    }
	else
	{
		g.doubleCheckDists = false;
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
		g.nSearch = (unsigned int)iVal;
    }

	// Get Query Vector Length
    if (cutGetCmdLineArgumenti( argc, argv, "NQ", &iVal )) 
	{
		if (iVal < 1) { iVal = 100; }
		g.nQuery = (unsigned int)iVal;
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

	// Success
	return true;
}


//---------------------------------------------------------
//	Name:	ComputeBlockShapeFromVector
//	Desc:	
//	Notes:	
//	0. Assumes following members are initialized properly
//	   before this funciton is called
//		shape.nElems = Number of original elements in vector
//		shape.tpr    = threads per row
//		shape.rpb    = rows per block
//  1. Block Limits
//		Thread block has at most 512 theads per block
//
//	2. Grid Limits
//		Grid has at most 65,535 blocks in any dimension
//			So a 1D grid is at most 65,535 x 1
//			and a 2D grid is at most 65,535 x 65,535
//		We use next smallest even number to these limits
//			IE 65,535 - 1
//			IE (65,535*65,535 - 1)
//		It's useful to have an even number of columns
//		in grid structure when doing reductions
//---------------------------------------------------------

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


//---------------------------------------------------------
//	Name:	ComputeBlockShapeFromQueryVector
//	Desc:	
//	Notes:	
//	0. Assumes following members are initialized properly
//	   before this funciton is called
//		shape.nElems = Number of original elements in query vector
//		shape.tpr    = threads per row
//		shape.rpb    = rows per block
//
//  1. Block Limits
//		Thread block has at most 512 theads per block
//
//	2. Grid Limits
//		Grid has at most 65,535 blocks in any dimension
//			So a 1D grid is at most 65,535 x 1
//			and a 2D grid is at most 65,535 x 65,535
//		We use next smallest even number to these limits
//			IE 65,535 - 1
//			IE (65,535*65,535 - 1)
//---------------------------------------------------------

bool ComputeBlockShapeFromQueryVector
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

	// Make sure we don't exceed grid limits
	if ((bgShape.blocksPerRow >= 65535) || (bgShape.rowsPerGrid >= 65535))
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


//---------------------------------------------------------
//	Name:	InitShapeDefaults
//	Desc:	
//---------------------------------------------------------

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


//---------------------------------------------------------
//	Name:	DumpBlockGridShape
//	Desc:	
//---------------------------------------------------------

void DumpBlockGridShape( BlockGridShape & bgShape )
{
	printf( "N = %d, NPadded = %d\n", 
		bgShape.nElems, bgShape.nPadded );
	printf( "Block (%d TPR x %d RPB) = %d TPB\n", 
		bgShape.threadsPerRow, bgShape.rowsPerBlock, bgShape.threadsPerBlock );
	printf( "Grid (%d BPR x %d RPG)  = %d BPG\n", 
		bgShape.blocksPerRow, bgShape.rowsPerGrid, bgShape.blocksPerGrid );
	printf( "W = %d, H = %d\n", 
		bgShape.W, bgShape.H );
}


//---------------------------------------------------------
//	Name:	InitSearchVectors
//	Desc:	Create & Initialize Search Vectors
//---------------------------------------------------------

bool InitSearchVectors( AppGlobals & g, bool bNonUniformSearch, bool bNonUniformQuery, int scale )
{
	//-------------------------------------------
    //	Step 0.  Check Parameters
	//-------------------------------------------

	if (g.nSearch == 0) { return false; }
	if (g.nQuery == 0) { return false; }

	//-------------------------------------------
    //	Step 1.  Allocate Vectors
	//-------------------------------------------

	unsigned int mem_size_Search = g.nSearch * sizeof(float4);
    g.searchList = (float4*) malloc( mem_size_Search );

	unsigned int mem_size_Query = g.nQuery * sizeof(float4);
    g.queryList = (float4*) malloc( mem_size_Query );
	
	//-------------------------------------------
    //	Step 2.  Initialize Vectors
	//-------------------------------------------

	float minS, maxS, minQ, maxQ;
	switch (scale)
	{
	case 1:
		// Search box 2x2 != query box 3x3
		minS = -1.0f;
		maxS =  1.0f;
		minQ = -1.5f;
		maxQ =  1.5f;
		break;

	case 2:
		// Search box 200x200 != Query Box 300x300
		minS = -100.0f;
		maxS =  100.0f;
		minQ = -150.0f;	
		maxQ =  150.0f;
		break;

	case 3:
		// Search box 3x3 != Query Box 2x2
		minS = -1.5f;
		maxS =  1.5f;
		minQ = -1.0f;	
		maxQ =  1.0f;
		break;

	default:
		// Search = query small box (1x1)
		minS = 0.0f;
		maxS = 1.0f;
		minQ = 0.0f;
		maxQ = 1.0f;
		break;
	}

	float minS_small = minS * 0.01f;
	float maxS_small = maxS * 0.01f;

	float minQ_small = minQ * 0.1f;
	float maxQ_small = maxQ * 0.1f;

	//float centerS = (maxS - minS)/2.0f;
	float centerS_small = (maxS_small - minS_small)/2.0f;

	float s_xOffset = minS + (maxS-minS) * 0.83f;
	float s_yOffset = minS + (maxS-minS) * 0.27f;
	float s_zOffset = minS + (maxS-minS) * 0.19f;

	float centerQ_small = (maxQ_small - minQ_small)/2.0f;

	float q_xOffset = minQ + (maxQ-minQ) * 0.83f;
	float q_yOffset = minQ + (maxQ-minQ) * 0.27f;
	float q_zOffset = minQ + (maxQ-minQ) * 0.19f;

	//
	// Initialize Search points (to query against)
	//

	unsigned int i;
	if (true == bNonUniformSearch)
	{
		// Put half the points in unit cube
		unsigned int oneHalf = g.nSearch>>1;
		for (i = 0; i < oneHalf; i++)	// Large Points
		{
			// BUGBUG - for now just randomly generate points
			// In future - we should read them in from a file...
			g.searchList[i].x = RandomFloat( minS, maxS );
			g.searchList[i].y = RandomFloat( minS, maxS );
			g.searchList[i].z = RandomFloat( minS, maxS );

			// Store point index in this channel
			g.searchList[i].w = (float)i;
		}

		// Put other half of points in 1/100th of the cube

		for (i = oneHalf; i < g.nSearch; i++)	// Small Points
		{
			// BUGBUG - for now just randomly generate points
			// In future - we should read them in from a file...
			g.searchList[i].x = RandomFloat( minS_small, maxS_small ) - centerS_small + s_xOffset;
			g.searchList[i].y = RandomFloat( minS_small, maxS_small ) - centerS_small + s_yOffset;
			g.searchList[i].z = RandomFloat( minS_small, maxS_small ) - centerS_small + s_zOffset;

			// Store point index in this channel
			g.searchList[i].w = (float)i;
		}

	}
	else
	{
		for (i = 0; i < g.nSearch; i++)	// Original Points
		{
			// BUGBUG - for now just randomly generate points
			// In future - we should read them in from a file...
			g.searchList[i].x = RandomFloat( minS, maxS );
			g.searchList[i].y = RandomFloat( minS, maxS );
			g.searchList[i].z = RandomFloat( minS, maxS );

			// Store point index in this channel
			g.searchList[i].w = (float)i;
		}
	}


	//float r[3];
	//r[0] = RandomFloat( 0.0, 1.0 );
	//r[1] = RandomFloat( 0.0, 1.0 );
	//r[2] = RandomFloat( 0.0, 1.0 );

	//
	//	Initial Query Points
	//
	if (true == bNonUniformQuery)
	{
		// Put half the points in large cube
		unsigned int oneHalf = g.nSearch>>1;
		for (i = 0; i < oneHalf; i++)	// Large Points
		{
			// BUGBUG - for now just randomly generate points
			// In future - we should read them in from a file...
			g.queryList[i].x = RandomFloat( minQ, maxQ );
			g.queryList[i].y = RandomFloat( minQ, maxQ );
			g.queryList[i].z = RandomFloat( minQ, maxQ );

			// Store point index in this channel
			g.searchList[i].w = (float)i;
		}

		// Put other half of points in smaller cube
		for (i = oneHalf; i < g.nSearch; i++)	// Small Points
		{
			// BUGBUG - for now just randomly generate points
			// In future - we should read them in from a file...
			g.queryList[i].x = RandomFloat( minQ_small, maxQ_small ) - centerQ_small + q_xOffset;
			g.queryList[i].y = RandomFloat( minQ_small, maxQ_small ) - centerQ_small + q_yOffset;
			g.queryList[i].z = RandomFloat( minQ_small, maxQ_small ) - centerQ_small + q_zOffset;

			// Store point index in this channel
			g.queryList[i].w = (float)i;
		}
	}
	else
	{
		for (i = 0; i < g.nQuery; i++)	// Original Points
		{
			// BUGBUG - for now just randomly generate points
			// In future - we should read them in from a file...
			g.queryList[i].x = RandomFloat( minQ, maxQ );
			g.queryList[i].y = RandomFloat( minQ, maxQ );
			g.queryList[i].z = RandomFloat( minQ, maxQ );

			// ALL THE SAME HACK
			//g.queryList[i].x = r[0];
			//g.queryList[i].y = r[1];
			//g.queryList[i].z = r[2];

			// Store point index in this channel
			g.queryList[i].w = (float)i;
		}
	}
	
	// Success
	return true;
}


//---------------------------------------------------------
//	Name:	FiniSearchVectors
//	Desc:	Cleanup Search Vectors
//---------------------------------------------------------

void FiniSearchVectors( AppGlobals & g )
{
	// Cleanup Query List
	if (NULL == g.queryList)
	{
		float4 * tempList = g.queryList;
		g.queryList = NULL;
	    free( tempList );
	}
	g.nQuery = 0;

	// Cleanup Search List
	if (NULL == g.searchList)
	{
		float4 * tempList = g.searchList;
		g.searchList = NULL;
	    free( tempList );
	}
	g.nSearch = 0;
}


//---------------------------------------------------------
//	Name:	InitGlobals
//	Desc:	Initialize Application Globals to Default
//---------------------------------------------------------

bool InitGlobals( AppGlobals & g )
{
	//
	// Set Defaults
	//

	// Search Vectors
	g.nSearch	 = 100;
	g.searchList = NULL;
	g.nQuery	 = 10;
	g.queryList  = NULL;

	// Cuda Properties
#if (CUDA_PLATFORM == CUDA_DEVICE)
	g.currDevice        = 0;
    g.rawProps.maxThreadsPerBlock  = 0;
    g.rawProps.sharedMemPerBlock   = 0;
    g.rawProps.totalConstantMemory = 0;
    g.rawProps.SIMDWidth           = 0;
#endif
	g.cudaProps.name	= (char *)NULL;
	g.cudaProps.bytes	= 0;
	g.cudaProps.major	= 0;
	g.cudaProps.minor	= 0;

	// Init Block Grid Shape
	InitShapeDefaults( g.bgShape );

	// App Properties
	g.nopromptOnExit    = 0;
	g.doubleCheckDists	= 1;

	// Profiling Info
	g.hTimer				= 0;
	g.profile				= 1;
	g.profileSkipFirstLast  = 0;
	g.profileRequestedLoops	= 1;
	g.profileActualLoops	= 1;

	return true;
}


//---------------------------------------------------------
//	Name:	RunKDTreeTest
//	Desc:	Cleanup Search Vectors
//---------------------------------------------------------

bool RunKDTreeTest( AppGlobals & g )
{
	bool bResult = true;

	//---------------------------------
    //	Step 0.  Initialize
	//---------------------------------

    cudaError_t cuda_err = cudaSuccess;

#if (CUDA_PLATFORM == CUDA_DEVICE)
	CUresult cu_err = CUDA_SUCCESS;
#endif

    // set seed for rand()
    //srand( 2006 );

	g.hTimer = 0;
    cutCreateTimer( &(g.hTimer) );


	//-------------------------------------------
    //	Step 1.  Create Search & Query Vectors
	//-------------------------------------------
	//g.nSearch = 100000;
	//g.nQuery  = 100;
	g.nSearch = 1000000;
	g.nQuery  = 1000000;
	bool bNonUniformSearch = false;
	bool bNonUniformQuery  = false;
	int scaleType = 0;
	bResult = InitSearchVectors( g, bNonUniformSearch, bNonUniformQuery, scaleType );
	if (false == bResult)
	{
		// Error
		return false;
	}

#if (APP_TEST == TEST_KD_KNN)
	unsigned int kVal = 64;
	if (kVal >= KD_KNN_SIZE)
	{
		kVal = KD_KNN_SIZE-1;
	}
#endif


	//-------------------------------------------
    //	Step 2.  Setup Initial parameters
	//-------------------------------------------

	// BUGBUG - need to fix up later to support 
	//          2D layouts of KD Tree
	unsigned int nOrigSearch = g.nSearch;
	unsigned int nPadSearch  = nOrigSearch;

	//unsigned int wSearch     = nOrigSearch;
	//unsigned int hSearch     = 1;

	unsigned int nOrigQuery  = g.nQuery;

	BlockGridShape kdShape;
	
	kdShape.threadsPerRow = KD_THREADS_PER_ROW;
	kdShape.rowsPerBlock  = KD_ROWS_PER_BLOCK;
	kdShape.nElems        = nOrigQuery;

	bResult = ComputeBlockShapeFromQueryVector( kdShape );
	if (false == bResult)
	{
		// Error
		return false;
	}

	unsigned int nPadQuery   = kdShape.nPadded;

#if (APP_TEST == TEST_KD_KNN)
#else
	unsigned int wQuery      = kdShape.W;
	//unsigned int hQuery      = kdShape.H;
#endif

	//
	// Print out Initialization Parameters
	//
	DumpBlockGridShape( kdShape );

	printf( "# Requested Search Points  = %d\n", nOrigSearch );
	printf( "# Requested Query Points   = %d\n", nOrigQuery );
	printf( "# Padded Query Points      = %d\n", nPadQuery );
#if (APP_TEST == TEST_KD_KNN)
	printf( "# 'k' for kNN search       = %d\n", kVal );
#endif


	// Make sure Matrix + vector is not to big to use up all device memory  // 768 Meg on Display Card
#if (APP_TEST == TEST_KD_KNN)
	unsigned int sizeResults = kdShape.nPadded * sizeof(GPU_NN_Result) * kVal;
#else
	unsigned int sizeResults = kdShape.nPadded * sizeof(GPU_NN_Result);
#endif
	unsigned int sizeNodes   = nOrigSearch * sizeof(GPUNode_2D_MED);
	unsigned int sizeIDs	 = nOrigSearch * sizeof(unsigned int);
	unsigned int sizeQueries = kdShape.nPadded * sizeof(float4);
	unsigned int totalMem    = sizeNodes + sizeIDs + sizeQueries + sizeResults;


	// Make sure memory required to perform this operation doesn't exceed display device memory
	if (totalMem >= g.cudaProps.bytes)
	{
		printf( "KD Tree Inputs are too large for available device memory, running test will crash..." );
		return bResult;
	}

	printf( "# Onto Memory       = %d\n", sizeNodes + sizeIDs + sizeQueries );
	printf( "# From Memory       = %d\n", sizeResults );
	printf( "# Total Memory      = %d\n", totalMem );

	unsigned int rootIdx = (nOrigSearch-1)>>1;		// (start+end)/2

    // Setup GPU Kernel execution parameters
		// KDTree Distance
    dim3 qryThreads( kdShape.threadsPerRow, kdShape.rowsPerBlock, 1 );
    dim3 qryGrid( kdShape.blocksPerRow, kdShape.rowsPerGrid, 1 );

	float KD_CPU_build        = 0.0f;
	float KD_GPU_onto_device  = 0.0f;
	float KD_GPU_from_device  = 0.0f;
	float KD_GPU_dist		  = 0.0f;
	float KD_CPU_dist		  = 0.0f;
	bool  checkDistResults    = true;
	unsigned int maxNodes     = 0;
	double avgNodes           = 0.0;


	//-------------------------------------------
    //	Step 3.  Allocate GPU Vectors
	//-------------------------------------------

	// allocate host memory for GPU KD Tree Nodes
	unsigned int mem_size_KDNodes = nPadSearch * sizeof(GPUNode_2D_MED);
#if (CUDA_PLATFORM == CUDA_DEVICE)
	GPUNode_2D_MED* h_KDNodes = NULL;	
	cu_err = cuMemAllocSystem( (void**)&h_KDNodes, mem_size_KDNodes );
	if( CUDA_SUCCESS != cu_err) 
	{
		fprintf( stderr, "Cuda error: %x in file '%s' in line %i.\n",
				 cu_err, __FILE__, __LINE__ );
		exit( EXIT_FAILURE );
	}
#elif (CUDA_PLATFORM == CUDA_CUDA)
	GPUNode_2D_MED* h_KDNodes = (GPUNode_2D_MED*) malloc( mem_size_KDNodes );
	if (NULL == h_KDNodes)
	{
		fprintf( stderr, "Out of Memory: in file '%s' in line %i.\n", __FILE__, __LINE__ );
		exit( EXIT_FAILURE );
	}
#endif

	// allocate device memory for GPU KD Tree Nodes
	GPUNode_2D_MED* d_KDNodes = NULL;
	CUDA_SAFE_CALL( cudaMalloc( (void **) &d_KDNodes, mem_size_KDNodes ) );


	// allocate host memory for GPU Node ID's
	unsigned int mem_size_IDs = nPadSearch * sizeof(unsigned int);
#if (CUDA_PLATFORM == CUDA_DEVICE)
	unsigned int* h_IDs = NULL;
	cu_err = cuMemAllocSystem( (void**)&h_IDs, mem_size_IDs );
	if( CUDA_SUCCESS != cu_err) 
	{
		fprintf( stderr, "Cuda error: %x in file '%s' in line %i.\n",
				 cu_err, __FILE__, __LINE__ );
		exit( EXIT_FAILURE );
	}
#elif (CUDA_PLATFORM == CUDA_CUDA)
	unsigned int* h_IDs = (unsigned int*) malloc( mem_size_IDs );
	if (NULL == h_IDs)
	{
		fprintf( stderr, "Out of Memory: in file '%s' in line %i.\n", __FILE__, __LINE__ );
		exit( EXIT_FAILURE );
	}
#endif


	// allocate device memory for GPU Node ID's
	unsigned int* d_IDs = NULL;
	CUDA_SAFE_CALL( cudaMalloc( (void **) &d_IDs, mem_size_IDs ) );


    // allocate host memory for GPU query points 
    unsigned int mem_size_Query = nPadQuery * sizeof(float4);
#if (CUDA_PLATFORM == CUDA_DEVICE)
	float4* h_Queries = NULL;
	cu_err = cuMemAllocSystem( (void**)&h_Queries, mem_size_Query );
	if( CUDA_SUCCESS != cu_err) 
	{
		fprintf( stderr, "Cuda error: %x in file '%s' in line %i.\n",
				 cu_err, __FILE__, __LINE__ );
		exit( EXIT_FAILURE );
	}
#elif (CUDA_PLATFORM == CUDA_CUDA)
    float4* h_Queries = (float4*) malloc( mem_size_Query );
	if (NULL == h_Queries)
	{
		fprintf( stderr, "Out of Memory: in file '%s' in line %i.\n", __FILE__, __LINE__ );
		exit( EXIT_FAILURE );
	}
#endif

    // allocate device memory for GPU query points 
    float4* d_Queries;
    CUDA_SAFE_CALL( cudaMalloc( (void **) &d_Queries, mem_size_Query ) );

	// allocate host memory for GPU Query Results
#if (APP_TEST == TEST_KD_KNN)
    unsigned int mem_size_Results_GPU = nPadQuery * sizeof(GPU_NN_Result) * kVal;
#else
    unsigned int mem_size_Results_GPU = nPadQuery * sizeof(GPU_NN_Result);
#endif

#if (CUDA_PLATFORM == CUDA_DEVICE)
	GPU_NN_Result* h_Results_GPU = NULL;
	cu_err = cuMemAllocSystem( (void**)&h_Results_GPU, mem_size_Results_GPU );
	if( CUDA_SUCCESS != cu_err) 
	{
		fprintf( stderr, "Cuda error: %x in file '%s' in line %i.\n",
				 cu_err, __FILE__, __LINE__ );
		exit( EXIT_FAILURE );
	}
#elif (CUDA_PLATFORM == CUDA_CUDA)
    GPU_NN_Result* h_Results_GPU = (GPU_NN_Result*) malloc( mem_size_Results_GPU );
	if (NULL == h_Results_GPU)
	{
		fprintf( stderr, "Out of Memory: in file '%s' in line %i.\n", __FILE__, __LINE__ );
		exit( EXIT_FAILURE );
	}
#endif

    // allocate device memory for GPU query points 
    GPU_NN_Result* d_Results_GPU;
    CUDA_SAFE_CALL( cudaMalloc( (void **) &d_Results_GPU, mem_size_Results_GPU ) );

	// allocate host memory for CPU Query Results
#if (APP_TEST == TEST_KD_KNN)
    unsigned int mem_size_Results_CPU = nPadQuery * sizeof(CPU_NN_Result) * kVal;
#else
    unsigned int mem_size_Results_CPU = nPadQuery * sizeof(CPU_NN_Result);
#endif
    CPU_NN_Result* h_Results_CPU = (CPU_NN_Result*) malloc( mem_size_Results_CPU );


	//-------------------------------------------
    //	Step 4.  Initialize GPU Vectors
	//-------------------------------------------

	// Copy Query List
	unsigned int i;
	for (i = 0; i < nOrigQuery; i++)
	{
		h_Queries[i] = g.queryList[i];
	}
	for (i = nOrigQuery; i < nPadQuery; i++)
	{
		// Just repeat 1st query a few times
		h_Queries[i] = g.queryList[0];
	}

	if (g.profile)
	{
		// Start Timer
		cutResetTimer( g.hTimer );
		cutStartTimer( g.hTimer );
	}

	// Build KDTree (on CPU)
	void * kdTree = NULL;
	bResult = BUILD_KD_TREE( &kdTree, nOrigSearch, g.searchList );
	if (false == bResult) 
	{
		// Error - goto cleanup
		return false;
	}

	if (g.profile)
	{
		// Stop Timer and save performance measurement
		cutStopTimer( g.hTimer );
		KD_CPU_build += cutGetTimerValue( g.hTimer );
	}

	// Build KD Tree (on GPU)
	//bResult = BUILD_GPU_NODES_V1( kdTree, nOrigSearch, g.searchList, (void*)h_KDNodes );
	bResult = BUILD_GPU_NODES_V2( kdTree, nOrigSearch, g.searchList, (void*)h_KDNodes, h_IDs );
	if (!bResult)
	{
		// Error - goto cleanup
		return false;
	}


// Profile Measurement Loop
unsigned int currIter;
for (currIter = 0; currIter < g.profileActualLoops; currIter++)
{

	//-------------------------------------------------------
	//	Step 5.  Move Input Vectors 
	//			 from main memory to device memory
	//-------------------------------------------------------

	if (g.profile)
	{
		// Start Timer
		cutResetTimer( g.hTimer );
		cutStartTimer( g.hTimer );
	}

	// Copy 'KDNodes' vector from host memory to device memory
	CUDA_SAFE_CALL( cudaMemcpy( d_KDNodes, h_KDNodes, mem_size_KDNodes, cudaMemcpyHostToDevice ) );

	// Copy 'IDs' vector from host memory to device memory
	CUDA_SAFE_CALL( cudaMemcpy( d_IDs, h_IDs, mem_size_IDs, cudaMemcpyHostToDevice ) );

	// Copy 'Query Points' vector from host memory to device memory
	CUDA_SAFE_CALL( cudaMemcpy( d_Queries, h_Queries, mem_size_Query, cudaMemcpyHostToDevice ) );

	if (g.profile)
	{
		// Stop Timer and save performance measurement
		cutStopTimer( g.hTimer );
		if (g.profileSkipFirstLast)
		{
			if ((1 < currIter) && (currIter <= g.profileActualLoops))
			{
				KD_GPU_onto_device += cutGetTimerValue( g.hTimer );
			}
		}
		else
		{
			KD_GPU_onto_device += cutGetTimerValue( g.hTimer );
		}
	}


	//-------------------------------------------------------
	//	Step 6.  Call KDTree GPU Kernel
	//-------------------------------------------------------

	if (g.profile)
	{
		// Start Timer
		cutResetTimer( g.hTimer );
		cutStartTimer( g.hTimer );
	}

	
	// Check if GPU kernel execution generated an error
#if (APP_TEST == TEST_KD_KNN)
	// Call GPU KDTree 'k' Nearest Neighbors Kernel (k results)
	KDTREE_KNN_V2<<< qryGrid, qryThreads >>>( d_Results_GPU, d_Queries, d_KDNodes,
									           d_IDs, rootIdx, kVal  );

#else
	// Call GPU KDTree Nearest Neighbor Kernel (singleton result)
	KDTREE_DIST_V2<<< qryGrid, qryThreads >>>( d_Results_GPU, d_Queries, d_KDNodes,
									           d_IDs, rootIdx, wQuery  );
#endif

	cuda_err = cudaGetLastError();
	if( cudaSuccess != cuda_err) 
	{
		fprintf( stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",
		"PLQ_GPU_BF_DIST() failed", __FILE__, __LINE__, cudaGetErrorString( cuda_err ) );
		exit( EXIT_FAILURE );
	}

	if (g.profile)
	{
		// Stop Timer and save performance measurement
		cutStopTimer( g.hTimer );
		if (g.profileSkipFirstLast)
		{
			if ((1 < currIter) && (currIter <= g.profileActualLoops))
			{
				KD_GPU_dist += cutGetTimerValue( g.hTimer );
			}
		}
		else
		{
			KD_GPU_dist += cutGetTimerValue( g.hTimer );
		}
	}

	//-------------------------------------------------
	//	Step 7.  Copy Outputs
	//			 from device memory to main memory
	//-------------------------------------------------

	if (g.profile)
	{
		// Start Timer
		cutResetTimer( g.hTimer );
		cutStartTimer( g.hTimer );
	}

	// copy result vector Z from device to host
	CUDA_SAFE_CALL( cudaMemcpy( (void *) h_Results_GPU, d_Results_GPU, mem_size_Results_GPU, cudaMemcpyDeviceToHost ) );

	if (g.profile)
	{
		// Stop Timer and save performance measurement
		cutStopTimer( g.hTimer );
		if (g.profileSkipFirstLast)
		{
			if ((1 < currIter) && (currIter <= g.profileActualLoops))
			{
				KD_GPU_from_device += cutGetTimerValue( g.hTimer );
			}
		}
		else
		{
			KD_GPU_from_device += cutGetTimerValue( g.hTimer );
		}
	}

	//-------------------------------------------
	//	Step 8:	Call KDTree CPU Algorithm
	//-------------------------------------------

	if (g.doubleCheckDists)
	{
		if (g.profile)
		{
			// Start Timer
			cutResetTimer( g.hTimer );
			cutStartTimer( g.hTimer );
		}

		// Determine Nearest Neighbors using KDTree
#if (APP_TEST == TEST_KD_KNN)
		bResult = PLQ_CPU_KDTREE_KNN
					(
					kdTree,
					g.nSearch,
					g.searchList,
					g.nQuery,
					nPadQuery,
					g.queryList,
					kVal,
					h_Results_CPU
					);
#else
		bResult = PLQ_CPU_KDTREE_FIND_2D
					(
					kdTree,
					g.nSearch,
					g.searchList,
					g.nQuery,
					g.queryList,
					h_Results_CPU 
					);
#endif
		if (false == bResult)
		{
			// Error
			return false;
		}

		if (g.profile)
		{
			// Stop Timer and save performance measurement
			cutStopTimer( g.hTimer );
			if (g.profileSkipFirstLast)
			{
				if ((1 < currIter) && (currIter <= g.profileActualLoops))
				{
					KD_CPU_dist += cutGetTimerValue( g.hTimer );
				}
			}
			else
			{
				KD_CPU_dist += cutGetTimerValue( g.hTimer );
			}
		}
	}

} // End Profile Loop

	//-------------------------------------------------
	//	Step 9:  Double check GPU result 
	//			 against CPU result
	//-------------------------------------------------

	if (g.doubleCheckDists)
	{
		//double totalNodes = 0.0;
		//maxNodes = 0;
#if (APP_TEST == TEST_KD_KNN)
		// Check each Query Point (GPU vs. CPU)
		/*
		unsigned int j;
		for (i = 0; i < nOrigQuery; i++)
		{
			// Check each Nearest Neighbor
			for (j = 0; j < kVal; j++)
			{
				unsigned int baseIdx = (j * nPadQuery) + i;
				float gpuDist = h_Results_GPU[baseIdx].Dist;
				float cpuDist = h_Results_CPU[baseIdx].dist;
				//float eps = 1.0e-4;
				unsigned int gpuID = h_Results_GPU[baseIdx].Id;
				unsigned int cpuID = h_Results_CPU[baseIdx].id;
				double qX = static_cast<double>( h_Queries[i].x );
				double qY = static_cast<double>( h_Queries[i].y );
				double gpu_D = static_cast<double>( gpuDist );
				double cpu_D = static_cast<double>( cpuDist );
				printf( "[%3d][%3d]=<%.6g, %.6g>: GPU [%5d %.9g], CPU [%5d %.9g]\n",
						 i,j, qX, qY, gpuID, gpu_D, cpuID, cpu_D );
			}
		}
		*/
		checkDistResults = true;
#else
		for (i = 0; i < nOrigQuery; i++)
		{
			// Need to use a fuzzy compare on distance
			float gpuDist = h_Results_GPU[i].Dist;
			float cpuDist = h_Results_CPU[i].dist;
			float eps = 1.0e-4;
			unsigned int gpuID = h_Results_GPU[i].Id;
			unsigned int cpuID = h_Results_CPU[i].id;

			//unsigned int gpuCount = h_Results_GPU[i].cNodes;
			//totalNodes += (double)gpuCount;

			// Get Maximum Nodes visited of any query
			//if (gpuCount > maxNodes)
			//{
			//	maxNodes = gpuCount;
			//}

			if ( ((gpuDist < (cpuDist-eps)) || (gpuDist > (cpuDist+eps))) ||
				  (gpuID != cpuID) )
			{
				double qX = static_cast<double>( h_Queries[i].x );
				double qY = static_cast<double>( h_Queries[i].y );
				double gpu_D = static_cast<double>( gpuDist );
				double cpu_D = static_cast<double>( cpuDist );
				printf( "Warning - [%3d]=<%.6g, %.6g> - GPU KD[%5d %.9g] != CPU KD[%5d %.9g] !!!\n",
						 i, qX, qY, gpuID, gpu_D, cpuID, cpu_D );
				checkDistResults = false;
			}
		}
#endif

		// Get Average Nodes Visited Per Query
		//avgNodes = totalNodes/(double)nOrigQuery;
	}


	//--------------------------------------------------------
	// Step 10: Print out Results
	//--------------------------------------------------------
	
	/*
	for (i = 0; i < nOrigQuery; i++)
	{
		double gpuDist = static_cast<double>( h_Results_GPU[i].Dist );
		unsigned int gpuID = h_Results_GPU[i].Id;
		//unsigned int gpuCount = h_Results_GPU[i].cNodes;
		//printf( "QR[%3d]: <ID=%5d, Dist=%.6g, cNodes=%d>\n", i, gpuID, gpuDist, gpuCount );
		printf( "QR[%3d]: <ID=%5d, Dist=%.6g>\n", i, gpuID, gpuDist );
	}
	*/

	if (g.doubleCheckDists)
	{
		printf ("Max GPU Nodes = %d, Avg GPU Nodes = %g\n", maxNodes, avgNodes );
		if (true == checkDistResults)
		{
			printf( "Distance check: CPU and GPU results agree within tolerance.\n" );
		}
		else
		{
			printf( "Distance check: CPU and GPU results don't agree within tolerance !!!\n" );
		}
	}


	//--------------------------------------------------------
	// Step 11: Print out Profile Statistics
	//--------------------------------------------------------

	if (g.profile)
	{
		// Dump Profile Statistics
		if (g.profile > 1)
		{
			float loops = (float)g.profileActualLoops;
			float o_l = 1.0f / loops;

			float avgOnto    = KD_GPU_onto_device * o_l;
			float avgFrom    = KD_GPU_from_device * o_l;
			float avgGPUdist = KD_GPU_dist * o_l;
			float avgCPUdist = KD_CPU_dist * o_l;
			float avgBuild   = KD_CPU_build;

			// Verbose
			//printf( "Number of total iterations = %f.\n", loops );
			//printf( "KD - KD Tree build on CPU,            time: %f msecs.\n", KD_CPU_build );
			//printf( "KD - Copy 'Input' vectors onto GPU,   time: %f msecs.\n", KD_GPU_onto_device );
			//printf( "KD - Copy 'Results' vector from GPU,  time: %f msecs.\n", KD_GPU_from_device );
			//printf( "KD - GPU Kernel computation,        time: %f msecs.\n", KD_GPU_dist );
			//printf( "KD - CPU Kernel computation,        time: %f msecs.\n", KD_CPU_dist );

			// Terse
			printf( "KD - In, Out, G_D, C_D, C_B\n" );
			printf( "     %f, %f, %f, %f, %f\n\n", avgOnto, avgFrom, avgGPUdist, avgCPUdist, avgBuild );
		}
		else
		{
			printf( "KD - KD Tree build on CPU,            time: %f msecs.\n", KD_CPU_build );
			printf( "KD - Copy 'Input' vectors onto GPU,   time: %f msecs.\n", KD_GPU_onto_device );
			printf( "KD - Copy 'Results' vector from GPU,  time: %f msecs.\n", KD_GPU_from_device );
			printf( "KD - GPU Kernel computation,        time: %f msecs.\n", KD_GPU_dist );
			printf( "KD - CPU Kernel computation,        time: %f msecs.\n", KD_CPU_dist );
		}
	}


	//--------------------------------------------------------
	// Step 13: Cleanup Resources
	//--------------------------------------------------------

    printf( "Shutting Down...\n" );

	// cleanup CUDA Timer
	cutDeleteTimer( g.hTimer );

    // clean up allocations
    //free( h_KDNodes );
	//free( h_IDs );
    //free( h_Queries );
    //free( h_Results_GPU );
    CU_SAFE_CALL( cuMemFreeSystem( h_KDNodes ) );
    CU_SAFE_CALL( cuMemFreeSystem( h_IDs ) );
    CU_SAFE_CALL( cuMemFreeSystem( h_Queries ) );
    CU_SAFE_CALL( cuMemFreeSystem( h_Results_GPU ) );

	free( h_Results_CPU );

	CUDA_SAFE_CALL( cudaFree( d_KDNodes ) );
	CUDA_SAFE_CALL( cudaFree( d_IDs ) );
    CUDA_SAFE_CALL( cudaFree( d_Queries ) );
    CUDA_SAFE_CALL( cudaFree( d_Results_GPU ) );

	FINI_KD_TREE( &kdTree );

	FiniSearchVectors( g );

	printf( "Shutdown done...\n\n" );

	// Success
	return true;
}



