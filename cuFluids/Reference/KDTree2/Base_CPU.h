#pragma once
#ifndef _BASE_CPU_H
#define _BASE_CPU_H
//-----------------------------------------------------------------------------
//	Name:	Base_CPU.h
//	Desc:	Simple Defines, Macros, and Utility Functions
//	Log:	Created by Shawn D. Brown (4/15/07)
//	
//	Copyright (C) by UNC-Chapel Hill, All Rights Reserved
//-----------------------------------------------------------------------------

//-------------------------------------
//	Includes
//-------------------------------------

#include "VS_Platform.h"

#if (VS_PLATFORM == PLATFORM_WIN_VS2003)
	#include <windows.h>
	#include <tchar.h>
#elif (VS_PLATFORM == PLATFORM_WIN_VS2005)
	#include <windows.h>
	#include <tchar.h>
#elif (VS_PLATFORM == PLATFORM_UNIX)
	// ... What to include here? ...
#endif


//-------------------------------------
//	
//	Macros
//
//-------------------------------------

//
// <TODO> - Note to self, my old style ASSERT macros strangely quit working with VS 2005 
//			and Unicode further complicates things
//			The approach used below works...
//			but there is probably is a better way than double wrapping the macros
//

// Convert _Value into a string, IE "_Value"
#ifndef _STRINGIZE
	#define __STRINGIZE(_Value) #_Value
	#define _STRINGIZE(_Value) __STRINGIZE(_Value)
#endif

// Convert _Value into a Char, IE '_Value' 
#if (VS_PLATFORM == PLATFORM_WIN_VS2003)
#ifndef _CHARIZE
	#define __CHARIZE(_Value) #@_Value
	#define _CHARIZE(_Value) __CHARIZE(_Value)
#endif
#elif (VS_PLATFORM == PLATFORM_WIN_VS2005)
#ifndef _CHARIZE
	#define __CHARIZE(_Value) #@_Value
	#define _CHARIZE(_Value) __CHARIZE(_Value)
#endif
#endif

// Concatenate 2 Tokens into a single larger token 
#ifndef _TOKEN_PASTE
	#define __TOKEN_PASTE(_Token1, _Token2) _Token1 ## _Token2
	#define _TOKEN_PASTE(_Token1, _Token2) __TOKEN_PASTE(_Token1, _Token2)
#endif

// Convert a regular (ANSI) string into a wide (UNICODE) String
#ifndef _WIDE_STRING
	#define __WIDE_STRING(_String) L ## _String
	#define _WIDE_STRING(_String) __WIDE_STRING(_String)
#endif

// Macro to convert a String to ANSI or UNICODE as appropriate
#if (VS_PLATFORM == PLATFORM_UNIX)
	#ifdef UNICODE
		#define __T(_String) L ## _String
	#else
		#define __T(_String) _String
	#endif
	#define _T(_String) __T(_String)
	#define TEXT(_String) __T(_String)

	#ifndef _TCHAR_DEFINED
		typedef unsigned short  TCHAR;
		#define _TCHAR_DEFINED
	#endif
#endif


//-------------------------------------
//	Defines
//-------------------------------------

// BOOLEAN Definitions
#ifndef TRUE
	#define TRUE 1
#endif

#ifndef FALSE
	#define FALSE 0
#endif

// NULL definition
#ifndef NULL
	#define NULL ((void*)0)
#endif

// Misc Max Sizes
#ifndef MAX_STRING
	#define MAX_STRING 256
#endif

#ifndef MAX_FILE
	#define MAX_FILE 256
#endif

#ifndef MAX_PATH
	#define MAX_PATH 256
#endif

#ifndef MAX_BUFFER
	#define	MAX_BUFFER 4096
#endif


// End of String (EOS) Definition
#ifndef A_EOS
	#define A_EOS '\0'
#endif
#ifndef W_EOS
	#define W_EOS	L'\0'
#endif
#ifndef EOS
	#ifdef UNICODE
		#define EOS W_EOS
	#else
		#define EOS A_EOS
	#endif
#endif

// End of Line (EOL) definiiton
#ifndef A_EOL
	#define A_EOL "\r\n"
#endif
#ifndef W_EOL
	#define W_EOL	L"\r\n"
#endif
#ifndef EOL
	#ifdef UNICODE
		#define EOL W_EOL
	#else
		#define EOL A_EOL
	#endif
#endif

#define ARRAY_SIZE(a) (sizeof((a)) / sizeof((a)[0]))

//-------------------------------------
//	StringHelpers
//-------------------------------------

unsigned int StringPrintfA( char * pszMsgBuff, unsigned int maxBuff, const char * pszFormat, ... );
#if (VS_PLATFORM == PLATFORM_WIN_VS2003)
	unsigned int StringPrintfW( wchar_t * pszMsgBuff, unsigned int maxBuff, const wchar_t * pszFormat, ... );
	#ifdef UNICODE
		#define StringPrintf StringPrintfW
	#else
		#define StringPrintf StringPrintfA
	#endif
#elif (VS_PLATFORM == PLATFORM_WIN_VS2005)
	unsigned int StringPrintfW( wchar_t * pszMsgBuff, unsigned int maxBuff, const wchar_t * pszFormat, ... );
	#ifdef UNICODE
		#define StringPrintf StringPrintfW
	#else
		#define StringPrintf StringPrintfA
	#endif
#elif (VS_PLATFORM == PLATFORM_UNIX)
	#define StringPrintf StringPrintfA
#endif


//-------------------------------------
//	Tolerances
//-------------------------------------

	// Look at #include <float.h> for actual values

	//	Reasonable small tolerance for float values 
	//  Note:  FLT_EPSILON = 1.192092896e-07f is smallest
	//	value s.t. 1.0f+FLT__EPSILON != 1.0f
const float  c_fEPSILON = 1.0e-6f;		// float tolerance 

	// Reasonable max value for float values
	//	Note:  FLT_MAX = 3.402823466e+38F is actual maximum value
const float  c_fHUGE    = 1.0e+37f;		// Huge float value
	
	//	Reasonable small tolerance for double values 
	//  Note:  DBL_EPSILON = 2.2204460492503131e-016 is smallest
	//  value s.t.  1.0+DBL__EPSILON != 1.0
const double c_dEPSILON = 1.0e-14;		// double tolerance

	// Reasonable max value for double values
	//	Note:  DBL_MAX = 1.7976931348623158e+308 is actual maximum value
const double c_dHUGE = 1.0e+307;


//-------------------------------------
//	Real Number Tests
//-------------------------------------

bool IsBadNumber( float val );
bool IsBadNumber( double val );

bool IsNAN( float val );
bool IsNAN( double val );

bool IsInfinite( float val );
bool IsInfinite( double val );

bool IsZero( float val );
bool IsZero( double val );

bool IsEqual( float a, float b );
bool IsEqual( double a, double b );

float absf( float val );
double absd( double val );

#endif // _VS_BASE_H

