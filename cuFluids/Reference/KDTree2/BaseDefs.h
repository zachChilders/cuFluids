#pragma once
#ifndef _KD_BASEDEFS_H
#define _KD_BASEDEFS_H

/*-----------------------------------------------------------------------------
  Name:	BaseDefs.h
  Desc:	Common Type definitions go here
  Notes:	

  History:
    Mar 18, 2010 -- Created (Shawn Brown)
-----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
  Includes
-----------------------------------------------------------------------------*/

#ifndef _KD_PLATFORM_H
	#include "Platform.h"
#endif

#if (KD_PLATFORM == KD_PLATFORM_WIN_VS2008)
	
#elif (KD_PLATFORM == KD_UNIX)
	#include <stdint.h>
	#include <iostream>
#else
	// Unknown platform
#endif


/*-----------------------------------------------------------------------------
  Platform picking
-----------------------------------------------------------------------------*/

#define KD_UNKNOWN_DEFS 0
#define KD_WIN_DEFS     1
#define KD_UNIX_DEFS    2

#if (KD_PLATFORM == KD_PLATFORM_WIN_VS2008)
	#define KD_DEFS KD_WIN_DEFS
#elif (KD_PLATFORM == KD_PLATFORM_UNIX)
	#define KD_DEFS KD_UNIX_DEFS
#else
	#define KD_DEFS KD_UNKNOWN_DEFS
#endif


/*-------------------------------------
  Base Type Definitions 
-------------------------------------*/

// Misc. Types
typedef int					BOOL;
typedef char				CHAR;

#if (KD_DEFS == KD_WIN_DEFS)

	/*-------------------------------------
	  Microsoft Windows Platforms
	-------------------------------------*/

// Integers
typedef __int8				I8;		// 8-Bit
typedef __int16				I16;	// 16-bit
typedef __int32				I32;	// 32-bit
typedef __int64				I64;	// 64-bit
//typedef __int128            I128;	// 128-bit
//typedef __int256            I256;	// 256-bit

// Unsigned Integers
typedef unsigned __int8		U8;
typedef unsigned __int16	U16;
typedef unsigned __int32	U32;
typedef unsigned __int64	U64;
//typedef unsigned __int128            U128;	// 128-bit
//typedef unsigned __int256            U256;	// 256-bit

// Real Types (floating point types)
//typedef ???				F8;		// 8-bit   (quarter precision)
//typedef ???				F16;	// 16-bit  (half precision)
typedef float				F32;	// 32-bit  (single precision)
typedef double				F64;	// 64-bit  (double precision)
//typedef long double       F80;	// 80-bit  (???)
//typedef ???				F128;	// 128-bit (quad precision)
//typedef ???				F256;	// 256-bit (octal precision)


// Characters
typedef char			CH8;	// 8-bit
typedef wchar_t			CH16;	// 16-bit
typedef U32             CH32;	// 32-bit
typedef U64             CH64;	// 64-bit

#else
	/*-------------------------------------
	  Unix Platform
	-------------------------------------*/

// 8-Bit Integers
typedef int8_t		I8;
typedef uint8_t		U8;

// 16-Bit Integers
typedef int16_t		I16;
typedef uint16_t	U16;

// 32-bit Integers
typedef int32_t		I32;
typedef uint32_t	U32;

// 64 bit Integers
typedef int64_t		I64;
typedef uint64_t	U64;

// Real types (floating point types)
typedef float			F32;
typedef double			F64;

#endif


/*-------------------------------------
  Base Value Definitions
-------------------------------------*/

#ifndef NULL
#define NULL	0
#endif

#ifndef FALSE
#define FALSE	0
#endif

#ifndef TRUE
#define TRUE	1
#endif


#if (KD_DEFS == KD_WIN_DEFS)

	/*-------------------------------------
	  Microsoft Windows Platforms
	-------------------------------------*/

#define I8_MIN		(-127i8 - 1)
#define I8_MAX		127i8

#define U8_MIN		0x00ui8
#define U8_MAX		0xFFui8

#define I16_MIN		(-32767i16-1)
#define I16_MAX		32767i16

#define U16_MIN		0x0000ui16
#define U16_MAX		0xFFFFui16

#define I32_MIN		(-2147483647i32 - 1)
#define I32_MAX		2147483647i32

#define U32_MIN		0x00000000ui32
#define U32_MAX		0xFFFFFFFFui32

#define I64_MIN		(-9223372036854775807i64 - 1)
#define I64_MAX		9223372036854775807i64

#define U64_MIN		0x0000000000000000ui64
#define U64_MAX		0xFFFFFFFFFFFFFFFFui64

#define F32_MIN     -3.0e+38f        /* reasonable min value */
#define F32_MAX     +3.0e+38f        /* reasonable max value */
#define F32_EPS     +1.0e-06f		 /* reasonable epsilon */

#define F64_MIN     -1.0e+300        /* min value */
#define F64_MAX     +1.0e+300        /* max value */
#define F64_EPS     +1.0e-12		 /* reasonable epsilon */

// Other useful values (F32) (see float.h)
#define F32_SIGN_MASK       0x80000000ui32		/* Mask off just the sign bit */
#define F32_EXP_MASK        0x7F800000ui32		/* Mask off just the exponent bits */
#define F32_MANT_MASK       0x007FFFFFui32		/* Mask off just the mantissa bits */

#define F32_SMALL_POS		1.175494351e-38f	/* smallest representable positive value */
#define F32_LARGE_POS       3.402823466e+38f    /* largest representable positive value */
#define F32_DIGITS			6					/* # of decimal digits of precision */
#define F32_EPSILON			1.192092896e-07f	/* smallest value, such that 1.0+F32_EPSILON != 1.0 */
#define F32_MANTISSA_BITS   24                  /* # of bits in mantissa, 23 + implied leading 1 */
#define F32_MIN_10_EXP      (-37)               /* min decimal exponent */
#define F32_MAX_10_EXP      38                  /* max decimal exponent */
#define F32_MIN_2_EXP       (-125)              /* min binary exponent */
#define F32_MAX_2_EXP       128                 /* max binary exponent */

// Other useful values (F64) (see float.h)
#define F64_SIGN_MASK       0x8000000000000000ui64	/* Mask off just the sign bit */
#define F64_EXP_MASK        0x7FF0000000000000ui64	/* Mask off just the exponent bits */
#define F64_MANT_MASK       0x000FFFFFFFFFFFFFui64	/* Mask off just the mantissa bits */

#define F64_SMALL_POS   	2.2250738585072014e-308	/* smallest representable positive value */
#define F64_LARGE_POS		1.7976931348623158e+308 /* largest representable positive value */
#define F64_DIGITS			16						/* # of decimal digits of precision */ 
#define F64_EPSILON			2.2204460492503131e-016 /* smallest value such that 1.0+F64_EPSILON != 1.0 */
#define F64_MANTISSA_BITS   53                      /* # of bits in mantissa, 52 + implied leading 1 */
#define F64_MIN_10_EXP      (-307)                  /* min decimal exponent */
#define F64_MAX_10_EXP		308                     /* max decimal exponent */
#define F64_MIN_2_EXP       (-1021)                 /* min binary exponent */
#define F64_MAX_2_EX        1024                    /* max binary exponent */

#else

	/*-------------------------------------
	  Unix Platforms
	-------------------------------------*/

#define I8_MIN		-128
#define I8_MAX		+127

#define U8_MIN		0x0u
#define U8_MAX		0xFFu

#define I16_MIN		-32768l
#define I16_MAX		+32767l

#define U16_MIN		0x0u
#define U16_MAX		0xFFFFu

#define I32_MIN		-2147483648l
#define I32_MAX		+2147483647l

#define U32_MIN		0x0ul
#define U32_MAX		0xFFFFFFFFul

#define I64_MIN		-9223372036854775808ll
#define I64_MAX		+9223372036854775807ll

#define U64_MIN		0x0ull
#define U64_MAX		0xFFFFFFFFFFFFFFFFull

#define F32_MIN		-3.0e+38f
#define F32_MAX		+3.0e+38f
#define F32_EPS     +1.0e-06f

#define F64_MIN		-1.0e+300
#define F64_MAX		+1.0e+300
#define F64_EPS     +1.0e-12

#endif


/*-------------------------------------
  Global type definitions
-------------------------------------*/

typedef U64	SIZE_TYPE;	
typedef F32	VALUE_TYPE; 


/*-------------------------------------
  Enumerations
-------------------------------------*/

enum AXIS
{
	X_AXIS = 0,
	Y_AXIS = 1,
	Z_AXIS = 2,
	W_AXIS = 3,
	S_AXIS = 4,
	T_AXIS = 5,
	U_AXIS = 6,
	V_AXIS = 7,
	INVALID_AXIS = 128
}; // End Axis

enum POSITION
{
	POS_UNKNOWN = 0,
	POS_INSIDE	= 1,
	POS_ON		= 2,
	POS_OUTSIDE = 3,
	INVALID_POS = 128
}; // End Position

enum KD_LR
{
	KD_LEFT  = 0,
	KD_RIGHT = 1,
	KD_NEITHER = 128
}; // End LeftRight

enum KD_IO
{
	KD_IN  = 0,
	KD_OUT = 1,
	KD_UNKNOWN = 128
}; // End InOut


/*-----------------
  CPU Points
-----------------*/

typedef struct
{
	float pos[1];	// Position <x>
} CPU_Point1D;

typedef struct
{
	float pos[2];	// Position <x,y>
} CPU_Point2D;

typedef struct
{
	float pos[3];	// Position <x,y,z>
} CPU_Point3D;

typedef struct
{
	float pos[4];	// Position <x,y,z,w>
} CPU_Point4D;

typedef struct
{
	float pos[5];	// Position <x,y,z,w,s>
} CPU_Point5D;

typedef struct
{
	float pos[6];	// Position <x,y,z,w,s,t>
} CPU_Point6D;

typedef struct
{
	float pos[7];	// Position <x,y,z,w,s,t,u>
} CPU_Point7D;

typedef struct
{
	float pos[8];	// Position <x,y,z,w,s,t,u,v>
} CPU_Point8D;


#endif	// _KD_BASEDEFS_H
