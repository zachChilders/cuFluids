/*-----------------------------------------------------------------------------
  Name:	Random.cpp
  Desc:	Code for Generating Random Numbers
  Log:	Created by Shawn D. Brown (3/18/10)
-----------------------------------------------------------------------------*/

/*-------------------------------------
  Includes
-------------------------------------*/

// App Includes
#ifndef _KD_RANDOM_H
	#include "Random.h"
#endif

// STL includes
#include <ios>
#include <iomanip>
#include <iostream>


/*-------------------------------------
  Local Variables
-------------------------------------*/

static U32 well512_state[16];	// 16*32 = 512 bits of state
static U32 well512_index = 0;


/*-------------------------------------
  Local Function Declarations
-------------------------------------*/

struct TestEnv_Random 
{
public:
	// Fields
	U32 m_unused;
};

bool InitTestEnv_Random( TestEnv_Random & te );
bool FiniTestEnv_Random( TestEnv_Random & te );

bool Test_RandomU32( TestEnv_Random & te );
bool Test_RandomI32( TestEnv_Random & te );

bool Test_RandomU64( TestEnv_Random & te );
bool Test_RandomI64( TestEnv_Random & te );

bool Test_RandomF32( TestEnv_Random & te );
bool Test_RandomF64( TestEnv_Random & te );


/*-------------------------------------
  Function Definitions
-------------------------------------*/

/*---------------------------------------------------------
  Name:	RandomInit()
  Desc:	Initializes to random seed state 
        from single seed value
---------------------------------------------------------*/

void RandomInit( U32 seedVal )
{
	well512_index = 0;

    // New seeding algorithm from 
    // http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/MT2002/emt19937ar.html
    // In the previous versions, MSBs of the seed affected only MSBs of the state[].
    
	const U32 mask = ~0UL;
	U32 w = 32;
	well512_state[0] = seedVal & mask;

	for (U32 i = 1; i < 16; i++) 
	{
		// See Knuth "The Art of Computer Programming" Vol. 2, 3rd ed., page 106
		well512_state[i] = (1812433253UL * (well512_state[i-1] ^ (well512_state[i-1] >> (w-2))) + i) & mask;
    }
}


void RandomInit( U32 * stateBuff, int stateSize /*=16*/ )
{
	well512_index = 0;

	// Check Parameters
	if ((NULL == stateBuff) || (stateSize < 1))
	{
		// Pick a Known good default to seed state from
		RandomInit( 5489 );
		return;
	}
	else if (stateSize < 16)
	{
		// Use first value of state as seed
		RandomInit( stateBuff[0] );
		return;
	}

	// Set initial State to specified 512 bit buffer
	for (U32 i = 0; i < 16; i++) 
	{
		well512_state[i] = stateBuff[i];
	}
}


/*---------------------------------------------------------
  Name:	RandomU32()
  Desc:	returns pseudo-random unsigned 32 bit number
---------------------------------------------------------*/

U32 RandomU32()
{
	// Algorithm based on WELL 512 C/C++ algorithm described in
	//
	// [Panneton06] Panneton F., L'Ecuyer P., & Matsusmoto M.
	// "Improved Long Period Generators Based on Linear
	//  Recurrences Modulo 2", ACM Transactions on Mathematical 
	//	Software, 32, 1 (2006), pp. 1-16
	//  http://www.iro.umontreal.ca/~lecuyer/papers.html
	//
	// Code for algorithm derived from article on 
	//
	// [Lomont07] Lomont Chris., article "Random Number Generation",
	// "Game Programming Gems 7", Charles River Media, (2007), pp. 120-121

	// WELL Random Number Generator 512 for C/C++
	U32 a, b, c, d;
	a  = well512_state[well512_index];
	c  = well512_state[(well512_index+13)&15];
	b  = a^c^(a<<16)^(c<<15);
	c  = well512_state[(well512_index+9)&15];
	c ^= (c>>11);
	a  = well512_state[well512_index] = b^c;
	d  = a^((a<<5) & static_cast<U32>(0xDA442D20UL));
	well512_index = (well512_index + 15) & 15;
	a  = well512_state[well512_index];
	well512_state[well512_index] = a^b^d^(a<<2)^(b<<18)^(c<<28);

	return well512_state[well512_index];
}


/*---------------------------------------------------------
  Name:	UnitTest_Random()
  Desc:	Unit Test on Random Functionality
---------------------------------------------------------*/

bool UnitTest_Random()
{
	bool bTest = true;
	bool bResult;

	// Setup Test Environment
	TestEnv_Random myTE;
	bResult = InitTestEnv_Random( myTE );
	if (!bResult) { return false; }

	// Run Tests
	try 
	{
		bResult = Test_RandomU32( myTE );
		if (!bResult) { bTest = false; }

		bResult = Test_RandomI32( myTE );
		if (!bResult) { bTest = false; }

		bResult = Test_RandomU64( myTE );
		if (!bResult) { bTest = false; }

		bResult = Test_RandomI64( myTE );
		if (!bResult) { bTest = false; }

		bResult = Test_RandomF32( myTE );
		if (!bResult) { bTest = false; }

		bResult = Test_RandomF64( myTE );
		if (!bResult) { bTest = false; }
	}
	catch (...)
	{
		// Catch exceptions in testing here
		std::cerr << "UnitTest_Random() - Error, Unexpected Exception!!!" << std::endl;
		bTest = false;
	}

	// Cleanup Test Environment
	FiniTestEnv_Random( myTE );

	// Return results of all our tests
	return bTest;
}


/*---------------------------------------------------------
  Name:	InitTestEnv_Random()
  Desc:	Setup Test Environment
---------------------------------------------------------*/

bool InitTestEnv_Random( TestEnv_Random & te )
{
	bool bInit = true;
	return bInit;
}


/*---------------------------------------------------------
  Name:	FiniTestEnv_Random()
  Desc:	Setup Test Environment
---------------------------------------------------------*/

bool FiniTestEnv_Random( TestEnv_Random & te )
{
	bool bFini = true;
	return bFini;
}


/*---------------------------------------------------------
  Name:	Test_RandomU32()
  Desc:	Test RandomU32() functionality
---------------------------------------------------------*/

bool Test_RandomU32( TestEnv_Random & te )
{
	// BUGBUG -- need to write
	bool bTest = true;
	return bTest;
}


/*---------------------------------------------------------
  Name:	Test_RandomI32()
  Desc:	Test RandomI32() functionality
---------------------------------------------------------*/

bool Test_RandomI32( TestEnv_Random & te )
{
	// BUGBUG -- need to write
	bool bTest = true;
	return bTest;
}


/*---------------------------------------------------------
  Name:	Test_RandomU64()
  Desc:	Test RandomU64() functionality
---------------------------------------------------------*/

bool Test_RandomU64( TestEnv_Random & te )
{
	// BUGBUG -- need to write
	bool bTest = true;
	return bTest;
}


/*---------------------------------------------------------
  Name:	Test_RandomI64()
  Desc:	Test RandomI64() functionality
---------------------------------------------------------*/

bool Test_RandomI64( TestEnv_Random & te )
{
	// BUGBUG -- need to write
	bool bTest = true;
	return bTest;
}


/*---------------------------------------------------------
  Name:	Test_RandomF32()
  Desc:	Test RandomF32() functionality
---------------------------------------------------------*/

bool Test_RandomF32( TestEnv_Random & te )
{
	// BUGBUG -- need to write
	bool bTest = true;
	return bTest;
}


/*--------------------------------------------------------
  Name:   Test RandomF64() 
  Desc:	Test RandomF64() functionality
  Notes:  Double that random function appears to be
		  uniform using Pearson's Chi Square Test
--------------------------------------------------------*/

bool Test_RandomF64( TestEnv_Random & te )
{
	// Assume success to begin with
	bool bTest = true;	

	// Compute Storage Bins
	const U32 c_bins = 100;						 // Number of bins
	const F64 c_start = static_cast<F64>( 0.0 );
	const F64 c_end   = static_cast<F64>( 1.0 );
	const F64 c_eps   = static_cast<F64>( 1e-14 );
	const F64 o_bi    = static_cast<F64>( 0.01 ); // 1.0 / c_bins

	F64 binStarts[c_bins];
	F64 binEnds[c_bins];
	int binCounts[c_bins];

	F64 intervalStart = c_start;
	F64 intervalStop  = c_end;
	F64 intervalLen   = intervalStop - intervalStart;
	if ((-c_eps < intervalLen) && (intervalLen < c_eps))
	{
		// Error - interval is too small to use
		std::cerr << "UnitTest_Random(): RandomF64 test, interval [" << intervalStart << "," << intervalStop << "] is too small to use to use" << std::endl;
		return false;
	}
	F64 o_il = 1.0 / intervalLen;
	F64 intervalStep = intervalLen * o_bi;

	// Initialize Random Number Generator
	U32 seedVal  = 5489;
	U32 numRuns  = 100;
	U32 numRVals = 10000;
	RandomInit( seedVal );

	// Set First Interval start point 
	binStarts[0] = intervalStart;
	binCounts[0] = 0;

	// Set interior intervals
	U32 i;
	for (i = 1; i < 100; i++)
	{
		binCounts[i] = 0;
		binStarts[i] = intervalStart + ((F64)i*0.01)*intervalLen;
		binEnds[i-1] = binStarts[i];
	}

	// Set Last Interval endpoint
	binEnds[99] = intervalStop;

	F64 chiTot = 0.0;
	F64 chiSum = 0.0;
	
	U32 j;
	for (j = 0; j < numRuns; j++)
	{

		// Reset Bin Counts
		for (i = 0; i < 100; i++)
		{
			binCounts[i] = 0;
		}

		// Start Another Run - Bin each random number in run
		for (i = 0; i < numRVals; i++) 
		{
			F64 rVal = RandomF64();
			F64 rangeVal = intervalStart + (rVal * intervalLen);

			// Bin value
			//if (! FloatHack::IsValid( rVal ))
			//{
				// Error - Invalid Floats (Infinity, NaN, etc.)
				//std::cerr << "UnitTest_Random(): RandomF64() test, invalid float value<" << rVal << "> created by Random Number Generator." << std::endl;
				//bTest = false;
			//}
			//else
			if ((rangeVal < intervalStart) || (rangeVal > intervalStop))
			{
				// Error - random value outside [0.0,1.0] range
				std::cerr << "UnitTest_Random(): RandomF64() test, random value<" << rangeVal << "> is outside expected range[" << intervalStart << "," << intervalStop << "]." << std::endl;
				bTest = false;
			}
			else
			{
				I32	binNum = (I32)(rVal * 100.0);
				if (binNum < 0)
				{
					binNum = 0;
				}
				else if (binNum >= 100)
				{
					binNum = 99;
				}
					
				// Increment Bin Count
				binCounts[binNum] = binCounts[binNum] + 1;
			}
		}

		//	Pearson's Chi Square Test
		//		given 'n' values in random value run
		//		given 'k' bins in bin vector
		//		Ebc = expected bin count (IE 10,000random values/100 bins = 100)
		//		Abc = actual bin count 
		//	Formula:
		//		X^2 = sum[0 to k-1] of [(Abc - Ebc)^2 / Ebc]
		// Which can be rewritten as
		//		X^2 = -n + (1/Ebc)*<sum[0 to k-1] of [Abc^2]>
		F64 n = static_cast<F64>( numRVals );
		F64 k = static_cast<F64>( c_bins );
		F64 Ebc = n/k;	
		F64 o_Ebc = 1.0/Ebc;
		F64 runSum = 0.0;
		for (i = 0; i < 100; i++)
		{
			F64 Abc  = static_cast<F64>( binCounts[i] );
			F64 Abc2 = Abc*Abc;
			runSum += Abc2;		
		}
		F64 X2 = -n + o_Ebc*runSum;

		//std::cerr << "Run[" << j+1 << "], X2 = " << X2 << std::endl;

		F64 runVal = X2 - 100.0;
		F64 runVal2 = runVal*runVal;
		
		chiTot += X2;
		chiSum += runVal2;
	} // end Runs

	F64 chiAvg = chiTot/100.0;
	F64 chiTest = -100.0 + 1.0/100 * chiSum;

	// BUGBUG - need more statistical rigour here...
	if ((chiTest < 50.0) || (chiTest > 150.0))
	{
		std::cerr << "ChiAvg  = " << chiAvg  << std::endl;
		std::cerr << "ChiTest = " << chiTest << std::endl;
		bTest = false;
	}

	return bTest;
}
