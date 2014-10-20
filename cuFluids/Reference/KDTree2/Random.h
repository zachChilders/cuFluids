#pragma once
#ifndef _KD_RANDOM_H
#define _KD_RANDOM_H
/*-----------------------------------------------------------------------------
  Name:	Random.h
  Desc:	Code for Generating Random Numbers
  Log:	Created by Shawn D. Brown (3/18/10)

  Based on WELLS 512 C/C++ algorithm found in 
  "Game Programing Gems 7" on pp. 120-121
-----------------------------------------------------------------------------*/

/*-------------------------------------
  Includes
-------------------------------------*/

#include "BaseDefs.h"


/*-------------------------------------
  Function Declarations
-------------------------------------*/

	// Initialize random state from single random seed
void RandomInit( U32 seedVal );						

	// Initialize random state from 512 bit buffer
void RandomInit( U32 * stateBuff, I32 stateSize = 16 );

	// Random 32 bit unsigned (using Well 512 C/C++ algorithm)
unsigned int RandomU32();

	// Random 32 bit signed
inline I32 RandomI32()	
{
	return static_cast<I32>( RandomU32() );
}

	// Random 64 bit unsigned
inline U64 RandomU64()	
{
	U64 aVal = static_cast<U64>( RandomU32() );
	U64 bVal = static_cast<U64>( RandomU32() );
	return ((aVal << 32ul) | bVal);
}

	// Random 64 bit signed
inline I64 RandomI64()	
{
	return static_cast<I64>( RandomU64() );
}

	// Random 32 bit (float)	-- in range [0,1]
inline F32 RandomF32()
{
	double rVal = static_cast<double>( RandomU32() ) / static_cast<double>( 0xFFFFFFFFul );
	return static_cast<F32>( rVal );
}

	// Random 64 bit (double)	-- in range [0,1]
inline F64 RandomF64()
{
	double rVal = static_cast<double>( RandomU32() ) / static_cast<double>( 0xFFFFFFFFul );
	return static_cast<F64>( rVal );
}

	// Compute a Random 32-bit float number in specified range
inline float RandomFloat( float low, float high )
{
	float t = (float)RandomF32();
	return ((1.0f - t) * low + t * high);
}

	// Compute a Random 64-bit double number in specified range
inline double RandomDouble( double low, double high )
{
	double t = (double)RandomF64();
	return ((1.0 - t) * low + t * high);
}



/*-----------------------------------------------
  Unit Test Random Functionality
-----------------------------------------------*/

bool UnitTest_Random();


#endif // _KD_RANDOM_H

