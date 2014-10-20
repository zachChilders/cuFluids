#pragma once
#ifndef _BOX_CPU_H
#define _BOX_CPU_H
//-----------------------------------------------------------------------------
//	Name:	Box_CPU.h
//	Desc:	Defines Simple 2D, 3D Box classes
//	Log:	Created by Shawn D. Brown (4/15/07)
//	
//	Copyright (C) by UNC-Chapel Hill, All Rights Reserved
//-----------------------------------------------------------------------------

//-------------------------------------
//	Includes
//-------------------------------------

#include <math.h>
#include <vector>
#include <string>

#include "Base_CPU.h"
#include "Point_CPU.h"


//-------------------------------------
//	Classes
//-------------------------------------

//---------------------------------------------------------
//	Name:	Interval1D
//	Desc:	Simple 1D Interval class
//---------------------------------------------------------

template <typename T> 
class Interval1D_T
{
public:
	// Internal Type Definitions
	typedef unsigned int    SIZE_TYPE;		// Underlying Size Type
	typedef T				VALUE_TYPE;		// Underlying Value Type

protected:
	//	Fields
	VALUE_TYPE m_minX, m_maxX;

public:
	//	Properties
		// Center
	inline const T & CENTER() const 
		{
			return ((m_minX+m_maxX)/static_cast<T>( 2.0 )); 
		}
	inline const T CX() const
		{
			return ((m_minX+m_maxX)/static_cast<T>( 2.0 )); 
		}

		// Half-width
	inline const T & HALF_WIDTH() const { return (CX() - m_minX); }
	inline const T HW() const { return CX() - m_minX; }

		// Min/Max Extents of Interval
	inline T MINX() const { return m_minX; }
	inline void MINX( T value ) { m_minX = value; }
	
	inline T MAXX() const { return m_maxX; }
	inline void MAXX( T value ) { m_maxX = value; }

	// Constructors
	Interval1D_T()
		{
		}
	Interval1D_T( T minVal, T maxVal ) : 
		m_minX( minVal ), 
		m_maxX( maxVal ) 
		{
		}
	Interval1D_T( const Interval1D_T & toCopy ) : 
		m_minX( toCopy.m_minX ), 
		m_maxX( toCopy.m_maxX ) 
		{
		}
	~Interval1D_T() 
		{
		}

	// Operators
	Interval1D_T & operator = ( const Interval1D_T & toCopy ) 
		{ 
			if (this == &toCopy)
				return (*this);

			m_minX = toCopy.m_minX; 
			m_maxX = toCopy.m_maxX; 

			return (*this); 
		}

	// Methods 
	void SetFromCenterHW( T c, T hw )
	{
		if (hw < static_cast<T>( 0.0 )) { hw = -hw; }
		m_minX = c - hw;
		m_maxX = c + hw;
	}

	void SetFromMinMax( T minX, T maxX )
	{
		// Ensure valid ordering of input parameters
		if (minX > maxX) { std::swap( minX, maxX ); }

		// Set new min and max
		m_minX = minX;
		m_maxX = maxX;
	}

	void ToString( std::string & value ) const
	{
		char szBuff[256];
		
		double mx = static_cast< double >( m_minX );
		double MX = static_cast< double >( m_maxX );

		StringPrintfA( szBuff, 256, "INTERVAL_1D{m[%3.6f, %3.6f], M[%3.6f,%3.6f]}", mx, MX );

		value = szBuff;
	}

	void Dump() const
	{
		std::string intervalValue;
		ToString( intervalValue );
		DumpfA( "%s", intervalValue.c_str() );
	}



	// Friends
	friend bool InInterval( const Interval1D_T<float> & interval, float value );
	friend bool InInterval( const Interval1D_T<double> & interval, double value );

	friend POSITION PositionInterval( const Interval1D_T<float> & interval, float value );
	friend POSITION PositionInterval( const Interval1D_T<double> & interval, double value );

	template <typename U> friend bool IntersectIntervals( const Interval1D_T<U> & A, const Interval1D_T<U> & B, Interval1D_T<U> & result );

	template <typename U> friend bool TestIntersects( const Interval1D_T<U> & A, const Interval1D_T<U> & B );
	template <typename U> friend bool TestContains( const Interval1D_T<U> & A, const Interval1D_T<U> & B );
	template <typename U> friend bool ComputeIntersection( const Interval1D_T<U> & A, const Interval1D_T<U> & B, Interval1D_T<U> & result );

	template <typename U> friend bool SplitInterval( const Interval1D_T<U> & origInterval, T splitValue, 
													 Interval1D_T<U> & leftInterval, Interval1D_T<U> & rightInterval );

}; // End Interval1D_T

typedef Interval1D_T<float> Interval;			// 1D Interval (float)
//typedef Interval1D_T<double> Interval_D;		// 1D Interval (Double)

typedef std::vector<Interval> IntervalList;		// Vector (Dynamic Array) of 1D Intervals (Float)
//typedef std::vector<Box2D_D> Interval_D_List;	// Vector (Dynamic Array) of 1D Intervals (Double)


// Intersection of 2 1D intervals
template <typename U>
bool Intersect1DIntervals( U & a, U & b, U & c, U & d, U & l, U & r )
{
	if (a > b) { std::swap( a, b ); }
	if (c > d) { std::swap( c, d ); }

	if (a < c)
	{
		if (b < c)
		{
			// [a,b] precedes [c,d] - NO INTERSECTION
			return false;	
		}
		else if (b < d)
		{
			// [a,b] overlaps [c,d] (IE ...a,c,b,d...)
			// Intersection = [c,b]
			l = c;			
			r = b;
			return true;
		}
		else
		{
			// [a,b] contains [c,d] (IE a,c,d,b)
			//Intersection = [c,d]
			l = c;
			r = d;
			return true;
		}
	}
	else
	{
		// c <= a
		if (d < a)
		{
			// [c,d] precedes [a,b] - NO INTERSECTION
			return false;
		}
		else if (d < b)
		{
			// [c,d] overlaps [a,b] (IE ...,c,a,d,b,...
			// Intersection = [a,d]
			l = a;
			r = d;
			return true;
		}
		else
		{
			// [c,d] contains [a,b] (IE ...,c,a,b,d,...)
			// Intersection = [a,b]
			l = a;
			r = b;
			return true;
		}
	}
}

// Intersection of 2 1D intervals
template <typename U>
bool Intersect1DIntervals
( 
	const Interval1D_T<U> & A,		// Interval #1 [a,b]
	const Interval1D_T<U> & B,		// Interval #2 [c,d]
	Interval1D_T<U> & result		// Resulting Intersection, if there is one
)
{
	U a, b, c, d;

	a = A.MIN();
	b = A.MAX();
	c = B.MIN();
	d = B.MAX();

	if (a > b) { std::swap( a, b ); }
	if (c > d) { std::swap( c, d ); }

	if (a < c)
	{
		if (b < c)
		{
			// [a,b] precedes [c,d] - NO INTERSECTION
			return false;	
		}
		else if (b < d)
		{
			// [a,b] overlaps [c,d] (IE ...a,c,b,d...)
			// Intersection = [c,b]
			result.SetFromMinMax( c, b );
			return true;
		}
		else
		{
			// [a,b] contains [c,d] (IE a,c,d,b)
			//Intersection = [c,d]
			result.SetFromMinMax( c, d );
			return true;
		}
	}
	else
	{
		// c <= a
		if (d < a)
		{
			// [c,d] precedes [a,b] - NO INTERSECTION
			return false;
		}
		else if (d < b)
		{
			// [c,d] overlaps [a,b] (IE ...,c,a,d,b,...
			// Intersection = [a,d]
			result.SetFromMinMax( a, d );
			return true;
		}
		else
		{
			// [c,d] contains [a,b] (IE ...,c,a,b,d,...)
			// Intersection = [a,b]
			result.SetFromMinMax( a, b );
			return true;
		}
	}
}

// Test if 2 Intervals Intersect (IE Overlap)
template <typename U>
bool TestIntersects( const Interval1D_T<U> & a, const Interval1D_T<U> & b )
{
	// Check Extents
	if (a.MAX() < b.MIN()) { return false; }
	if (b.MAX() < a.MIN()) { return false; }

	// Success - They overlap so they must intersect
	return true;
}


// Test if interval 'a' contains interval 'b'
	// true iff
	//    a.minx <= b.minx <= b.maxx <= a.maxx
template <typename U>
bool TestContains( const Interval1D_T<U> & a, const Interval1D_T<U> & b )
{
	// Check Extents
	if (b.MINX() < a.MINX()) { return false; }
	if (a.MAXX() < b.MAXX()) { return false; }

	// Must be contained by process of elimination
	return true;
}


// Computes Intersection of 2 intervals (if they do intersect)
template <typename U>
bool ComputeIntersection
( 
	const Interval1D_T<U> & a,		// IN - 1st interval to intersect
	const Interval1D_T<U> & b,		// IN - 2nd interval to intersect
	      Interval1D_T<U> & result	// OUT - resulting intersection box
)
{
	return ( Intersect1DIntervals( a, b, result ) );
}


template <typename U>
bool SplitInterval
(
	const Interval1D_T<U> & origInterval,	// IN - Interval to split
		  U					splitValue,		// IN - value to split on
	      Interval1D_T<U> & leftInterval,	// OUT - left split box
		  Interval1D_T<U> & rightInterval	// OUT - right split box
)
{
	U mX, MX;

	mX = origInterval.MIN();
	MX = origInterval.MAX();

	if ((splitValue < mX) || (MX < splitValue)) { return false; }

	leftInterval.SetFromMinMax( mX, splitValue );
	rightInterval.SetFromMinMax( splitValue, MX );

	return true;
}




//---------------------------------------------------------
//	Name:	Box2D
//	Desc:	Simple 2D Box class
//---------------------------------------------------------

// TODO:  Consider adding transform functions from
	// Double points and box to float points

template <typename T> 
class Box2D_T
{
public:
	// Internal Type Definitions
	typedef unsigned int    SIZE_TYPE;		// Underlying Size Type
	typedef T				VALUE_TYPE;		// Underlying Value Type
	typedef Point2D_T<T>	POINT_TYPE;		// Underlying Point Type

protected:
	//	Fields
	POINT_TYPE m_min;		// min(x,y) of box
	POINT_TYPE m_max;		// max(x,y) of box

public:
	//	Properties
		// Center
	inline const T CX() const 
		{ 
			return ( (m_min.X() + m_max.X()) / static_cast<T>( 2.0 ) ); 
		}
	inline const T CY() const
		{ 
			return ( (m_min.Y() + m_max.Y()) / static_cast<T>( 2.0 ) ); 
		}
	inline POINT_TYPE CENTER() const 
		{ 
			POINT_TYPE pntCenter( CX(), CY() ) ;
			return pntCenter; 
		}

		// Half-widths
	inline const T HX() const 
		{ 
			return ( CX() - m_min.X() ); 
		}
	inline const T HY() const 
		{ 
			return ( CY() - m_min.Y() ); 
		}
	inline POINT_TYPE HALF_WIDTHS() const 
		{ 
			POINT_TYPE pntHW( HX(), HY() );
			return pntHW;
		}

		// Min/Max Extents of Box
	inline T MINX() const { return m_min.X(); }
	inline void MINX( T value ) { m_min.X( value ); }

	inline T MAXX() const { return m_max.X(); }
	inline void MAXX( T value ) { m_max.X( value ); }

	inline T MINY() const { return m_min.Y(); }
	inline void MINY( T value ) { m_min.Y( value ); }

	inline T MAXY() const { return m_max.Y(); }
	inline void MAXY( T value ) { m_max.Y( value ); }

	inline T MIN( SIZE_TYPE index ) const
		{
			// ASSERT( index < 2 );
			return (m_min[index]);
		}

	inline T MAX( SIZE_TYPE index ) const
		{
			// ASSERT( index < 2 );
			return (m_max[index]);
		}	


		// 4 Corners of Box
	inline POINT_TYPE BOTTOM_LEFT() const 
		{
			POINT_TYPE result( MINX(), MINY() );
			return result;
		}
	inline POINT_TYPE BOTTOM_RIGHT() const 
		{
			POINT_TYPE result( MAXX(), MINY() );
			return result;
		}
	inline POINT_TYPE TOP_LEFT() const 
		{
			POINT_TYPE result( MINX(), MAXY() );
			return result;
		}
	inline POINT_TYPE TOP_RIGHT() const 
		{
			POINT_TYPE result( MAXX(), MAXY() );
			return result;
		}

	// Constructors
	Box2D_T()
		{
		}
	Box2D_T( T xMin, T yMin, T xMax, T yMax ) : 
		m_min( xMin, yMin ), 
		m_max( xMax, yMax ) 
		{
		}
	Box2D_T( const POINT_TYPE & minVal, const POINT_TYPE & maxVal ) : 
		m_min( minVal ), 
		m_max( maxVal ) 
		{
		}
	Box2D_T( const Box2D_T & toCopy ) : 
		m_min( toCopy.m_min ), 
		m_max( toCopy.m_max ) 
		{
		}
	~Box2D_T() 
		{
		}

	// Operators
	Box2D_T & operator = ( const Box2D_T & toCopy ) 
		{ 
			if (this == &toCopy)
				return (*this);

			m_min = toCopy.m_min; 
			m_max = toCopy.m_max; 

			return (*this); 
		}

	// Methods 
	void SetFromCenterHW( T cX, T cY,
						  T hX, T hY )
	{
		if (hX < static_cast<T>( 0.0 )) { hX = -hX; }
		if (hY < static_cast<T>( 0.0 )) { hY = -hY; }

		m_min.X( cX - hX );
		m_min.Y( cY - hY );
		m_max.X( cX + hX );
		m_max.Y( cY + hY );
	}

	void SetFromCenterHW( const POINT_TYPE & center, const POINT_TYPE & hws )
	{
		SetFromCenterHW( center.X(), center.Y(), hws.X(), hws.Y() );
	}

	void SetFromMinMax( T minX, T minY,
						T maxX, T maxY )
	{
		// Ensure valid ordering of input parameters
		if (minX > maxX) { std::swap( minX, maxX ); }
		if (minY > maxY) { std::swap( minY, maxY ); }

		m_min.X( minX );
		m_min.Y( minY );

		m_max.X( maxX );
		m_max.Y( maxY );
	}

	void SetFromMinMax( const POINT_TYPE & minValues, const POINT_TYPE & maxValues )
	{
		SetFromMinMax( minValues.X(), minValues.Y(),
					   maxValues.X(), maxValues.Y() );
	}

	void ToString( std::string & value ) const
	{
		char szBuff[256];
		
		double mx = static_cast< double >( MINX() );
		double my = static_cast< double >( MINY() );
		double MX = static_cast< double >( MAXX() );
		double MY = static_cast< double >( MAXY() );

		StringPrintfA( szBuff, 256, "BOX2D{m[%3.6f, %3.6f], M[%3.6f,%3.6f]}",
				       mx, my, MX, MY );

		value = szBuff;
	}

	void Dump() const
	{
		std::string boxValue;
		ToString( boxValue );
		DumpfA( "%s", boxValue.c_str() );
	}


	// Friends
	friend bool InBox( const Box2D_T<float> & box, const Point2D_T<float> & pnt );
	friend bool InBox( const Box2D_T<double> & box, const Point2D_T<double> & pnt );

	friend bool InBox( const Box2D_T<float> & box, const Point3D_T<float> & pnt );
	friend bool InBox( const Box2D_T<double> & box, const Point3D_T<double> & pnt );


	friend POSITION PositionBox( const Box2D_T<float> & box, const Point2D_T<float> & pnt );
	friend POSITION PositionBox( const Box2D_T<double> & box, const Point2D_T<double> & pnt );

	friend POSITION PositionBox( const Box2D_T<float> & box, const Point3D_T<float> & pnt );
	friend POSITION PositionBox( const Box2D_T<double> & box, const Point3D_T<double> & pnt );

	template <typename U> friend bool Intersect1DIntervals( U & a, U & b, U & c, U & d, U & l, U & r );

	template <typename U> friend bool TestIntersects( const Box2D_T<U> & a, const Box2D_T<U> & b );
	template <typename U> friend bool TestContains( const Box2D_T<U> & a, const Box2D_T<U> & b );
	template <typename U> friend bool ComputeIntersection( const Box2D_T<U> & a, const Box2D_T<U> & b, Box2D_T<U> & result );

	template <typename U> friend bool SplitBox( const Box2D_T<U> & origBox, unsigned int splitAxis, 
										        U splitValue, Box2D_T<U> & leftBox, Box2D_T<U> & rightBox );

}; // End Box2D_T


typedef Box2D_T<float> Box2D;						// 2D Box (float)
//typedef Box2D_T<double> Box2D_D;				// 2D Box (Double)

typedef std::vector<Box2D> Box2DList;			// Vector (Dynamic Array) of 2D Boxes (Float)
//typedef std::vector<Box2D_D> Box2D_D_List;	// Vector (Dynamic Array) of 2D Boxes (Double)


// Test if 2 boxes Intersect (IE Overlap)
template <typename U>
bool TestIntersects( const Box2D_T<U> & a, const Box2D_T<U> & b )
{
	// Check X Extents
	if (a.MAXX() < b.MINX()) { return false; }
	if (b.MAXX() < a.MINX()) { return false; }

	// Check Y extents
	if (a.MAXY() < b.MINY()) { return false; }
	if (b.MAXY() < a.MINY()) { return false; }

	// Success - They overlap in X and Y so they must intersect
	return true;
}


// Test if box 'a' contains box 'b'
	// true iff
	//    a.minx <= b.minx <= b.maxx <= a.maxx
	//    a.miny <= b.miny <= b.maxy <= a.maxy
template <typename U>
bool TestContains( const Box2D_T<U> & a, const Box2D_T<U> & b )
{
	if ( (a.MINX() <= b.MINX()) &&
		 (b.MAXX() <= a.MAXX()) &&
		 (a.MINY() <= b.MINY()) &&
		 (b.MAXY() <= a.MAXY()) )
	{
		return true;
	}
	return false;
}


// Computes Intersection of 2 boxes (if they do intersect)
template <typename U>
bool ComputeIntersection
( 
	const Box2D_T<U> & a,		// IN - 1st box to intersect
	const Box2D_T<U> & b,		// IN - 2nd box to intersect
	      Box2D_T<U> & result	// OUT - resulting intersection box
)
{
	bool bResult; 

	float axm = a.MINX();
	float axM = a.MAXX();
	float bxm = b.MINX();
	float bxM = b.MAXX();

	float lx, rx;

	bResult = Intersect1DIntervals( axm, axM, bxm, bxM, lx, rx );
	if (bResult)
	{
		float aym = a.MINY();
		float ayM = a.MAXY();
		float bym = b.MINY();
		float byM = b.MAXY();

		float ly, ry;
		bResult = Intersect1DIntervals( aym, ayM, bym, byM, ly, ry );
		if (bResult)
		{
			result.SetFromMinMax( lx, ly, rx, ry );
		}
	}
	return bResult;
}

template <typename U>
bool SplitBox
(
	const Box2D_T<U> & origBox,		// IN - Box to split
		  unsigned int splitAxis,	// IN - axis to split on
		  U			   splitValue,	// IN - value to split on
	      Box2D_T<U> & leftBox,		// OUT - left split box
		  Box2D_T<U> & rightBox		// OUT - right split box
)
{
	U mX, MX, mY, MY;

	mX = origBox.MINX();
	MX = origBox.MAXX();
	mY = origBox.MINY();
	MY = origBox.MAXY();

	switch (splitAxis)
	{
	case X_AXIS:
		leftBox.SetFromMinMax( mX, mY, splitValue, MY );
		rightBox.SetFromMinMax( splitValue, mY, MX, MY );
		break;

	case Y_AXIS:
		leftBox.SetFromMinMax( mX, mY, MX, splitValue );
		rightBox.SetFromMinMax( mX, splitValue, MX, MY );
		break;

	default:
		// Error
		return false;
	}

	return true;
}


//---------------------------------------------------------
//	Name:	Box3D
//	Desc:	Simple 3D Box class
//---------------------------------------------------------

template <typename T> 
class Box3D_T 
{
public:
	// Internal Type Definitions
	typedef unsigned int		SIZE_TYPE;			// Underlying Size Type
	typedef T					VALUE_TYPE;			// Underlying Value Type
	typedef Point3D_T<T>		POINT_TYPE;			// Underlying Point Type

protected:
	//	Fields
	POINT_TYPE m_min;	// Minimum x,y,z
	POINT_TYPE m_max;	// Maximum x,y,z

public:
	//	Properties
		// Center
	inline const T CX() const 
		{ 
			return static_cast<T>( ( (m_min.X() + m_max.X()) / static_cast<T>( 2.0 ) ) ); 
		}
	inline const T CY() const
		{ 
			return static_cast<T>( ( (m_min.Y() + m_max.Y()) / static_cast<T>( 2.0 ) ) ); 
		}
	inline const T CZ() const
		{ 
			return static_cast<T>( ( (m_min.Z() + m_max.Z()) / static_cast<T>( 2.0 ) ) ); 
		}
	inline POINT_TYPE CENTER() const 
		{ 
			POINT_TYPE pntCenter( CX(), CY(), CZ() ) ;
			return pntCenter; 
		}

		// Half-widths
	inline const T HX() const 
		{ 
			return ( CX() - m_min.X() ); 
		}
	inline const T HY() const 
		{ 
			return ( CY() - m_min.Y() ); 
		}
	inline const T HZ() const 
		{ 
			return ( CZ() - m_min.Z() ); 
		}
	inline POINT_TYPE HALF_WIDTHS() const 
		{ 
			POINT_TYPE pntHW( HX(), HY(), HZ() );
			return pntHW;
		}

		// Min/Max Extents of Box
	inline T MINX() const { return m_min.X(); }
	inline void MINX( T value ) { m_min.X( value ); }

	inline T MAXX() const { return m_max.X(); }
	inline void MAXX( T value ) { m_max.X( value ); }

	inline T MINY() const { return m_min.Y(); }
	inline void MINY( T value ) { m_min.Y( value ); }

	inline T MAXY() const { return m_max.Y(); }
	inline void MAXY( T value ) { m_max.Y( value ); }

	inline T MINZ() const { return m_min.Z(); }
	inline void MINZ( T value ) { m_min.Z( value ); }

	inline T MAXZ() const { return m_max.Z(); }
	inline void MAXZ( T value ) { m_max.Z( value ); }

	inline T MIN( SIZE_TYPE index ) const
		{
			// ASSERT( index < 3 );
			return m_min[index];
		}

	inline T MAX( SIZE_TYPE index ) const
		{
			// ASSERT( index < 3 );
			return m_max[index];
		}	

		// 8 Corners of Box
	inline POINT_TYPE BOTTOM_LEFT_BACK() const 
		{
			POINT_TYPE result( MINX(), MINY(), MINZ() );
			return result;
		}
	inline POINT_TYPE BOTTOM_RIGHT_BACK() const 
		{
			POINT_TYPE result( MAXX(), MINY(), MINZ() );
			return result;
		}
	inline POINT_TYPE TOP_LEFT_BACK() const 
		{
			POINT_TYPE result( MINX(), MAXY(), MINZ() );
			return result;
		}
	inline POINT_TYPE TOP_RIGHT_BACK() const 
		{
			POINT_TYPE result( MAXX(), MAXY(), MINZ() );
			return result;
		}
	inline POINT_TYPE BOTTOM_LEFT_FRONT() const 
		{
			POINT_TYPE result( MINX(), MINY(), MAXZ() );
			return result;
		}
	inline POINT_TYPE BOTTOM_RIGHT_FRONT() const 
		{
			POINT_TYPE result( MAXX(), MINY(), MAXZ() );
			return result;
		}
	inline POINT_TYPE TOP_LEFT_FRONT() const 
		{
			POINT_TYPE result( MINX(), MAXY(), MAXZ() );
			return result;
		}
	inline POINT_TYPE TOP_RIGHT_FRONT() const 
		{
			POINT_TYPE result( MAXX(), MAXY(), MAXZ() );
			return result;
		}

	// Constructors
	Box3D_T()
		{
		}
	Box3D_T( T xMin, T yMin, T zMin, T xMax, T yMax, T zMax ) : 
		m_min( xMin, yMin, zMin ), 
		m_max( xMax, yMax, zMax ) 
		{
		}
	Box3D_T( const POINT_TYPE & minVal, const POINT_TYPE & maxVal ) : 
		m_min( minVal ), 
		m_max( maxVal ) 
		{
		}
	Box3D_T( const Box3D_T & toCopy ) : 
		m_min( toCopy.m_min ), 
		m_max( toCopy.m_max ) 
		{
		}
	~Box3D_T() 
		{
		}

	// Operators
	Box3D_T & operator = ( const Box3D_T & toCopy ) 
		{ 
			if (this == &toCopy)
				return (*this);

			m_min = toCopy.m_min; 
			m_max = toCopy.m_max; 

			return (*this); 
		}

	// Methods 
	void SetFromCenterHW( T cx, T cy, T cz,
		                  T hx, T hy, T hz )
	{
		if (hx < static_cast<T>( 0.0 )) { hx = -hx; }
		if (hy < static_cast<T>( 0.0 )) { hy = -hy; }
		if (hz < static_cast<T>( 0.0 )) { hz = -hz; }

		MINX( cx - hx );
		MAXX( cx + hx );

		MINY( cy - hy );
		MAXY( cy + hy );

		MINZ( cz - hz );
		MAXZ( cz + hz );
	}

	void SetFromCenterHW( const Point3D & center, const Point3D & hws )
	{
		SetFromCenterHW( center.X(), center.Y(), center.Z(),
					     hws.X(), hws.Y(), hws.Z() );
	}

	void SetFromMinMax( float xMin, float yMin, float zMin,
						float xMax, float yMax, float zMax )
	{
		// Ensure valid ordering of input parameters
		if (xMin > xMax) { std::swap( xMin, xMax ); }
		if (yMin > yMax) { std::swap( yMin, yMax ); }
		if (zMin > zMax) { std::swap( zMin, zMax ); }

		// Update Box Coordinates 
		MINX( xMin );
		MAXX( xMax );
		MINY( yMin );
		MAXY( yMax );
		MINZ( zMin );
		MAXZ( zMax );
	}

	void SetFromMinMax( const Point3D & minValues, const Point3D & maxValues )
	{
		SetFromMinMax( minValues.X(), minValues.Y(), minValues.Z(),
					   maxValues.X(), maxValues.Y(), maxValues.Z() );
	}

	void ToString( std::string & value ) const
	{
		char szBuff[256];
		
		double mx = static_cast< double >( MINX() );
		double my = static_cast< double >( MINY() );
		double mz = static_cast< double >( MINZ() );
		double MX = static_cast< double >( MAXX() );
		double MY = static_cast< double >( MAXY() );
		double MZ = static_cast< double >( MAXZ() );

		StringPrintfA( szBuff, 256, "BOX3D{m[%3.6f, %3.6f, %3.6f], M[%3.6f,%3.6f, %3.6f]}",
					   mx, my, mz, MX, MY, MZ );

		value = szBuff;
	}

	void Dump() const
	{
		std::string boxValue;
		ToString( boxValue );
		DumpfA( "%s", boxValue.c_str() );
	}

	// Friends
	friend bool InBox( const Box3D_T<float> & box, const Point2D_T<float> & pnt );
	friend bool InBox( const Box3D_T<double> & box, const Point2D_T<double> & pnt );

	friend bool InBox( const Box3D_T<float> & box, const Point3D_T<float> & pnt );
	friend bool InBox( const Box3D_T<double> & box, const Point3D_T<double> & pnt );

	friend POSITION PositionBox( const Box3D_T<float> & box, const Point2D_T<float> & pnt );
	friend POSITION PositionBox( const Box3D_T<double> & box, const Point2D_T<double> & pnt );

	friend POSITION PositionBox( const Box3D_T<float> & box, const Point3D_T<float> & pnt );
	friend POSITION PositionBox( const Box3D_T<double> & box, const Point3D_T<double> & pnt );

	template <typename U> friend bool Intersect1DIntervals( T & a, T & b, T & c, T & d, T & l, T & r );

	template <typename U> friend bool TestIntersects( const Box3D_T<U> & a, const Box3D_T<U> & b );
	template <typename U> friend bool TestContains( const Box3D_T<U> & a, const Box3D_T<U> & b );
	template <typename U> friend bool ComputeIntersection( const Box3D_T<U> & a, const Box3D_T<U> & b, Box3D_T<U> & result );

	template <typename U> friend bool SplitBox( const Box3D_T<U> & origBox, unsigned int splitAxis, 
										        T splitValue, Box3D_T<U> & leftBox, Box3D_T<U> & rightBox );

}; // End Box3D


typedef Box3D_T<float> Box3D;					// 3D Box (float)
//typedef Box3D_T<double> Box3D_D;				// 3D Box (Double)

typedef std::vector<Box3D> Box3DList;			// Vector (Dynamic Array) of 3D Boxes (Float)
//typedef std::vector<Box3D_D> Box3D_D_List;	// Vector (Dynamic Array) of 3D Boxes (Double)


// Test if 2 boxes Intersect (IE Overlap)
template <typename U>
bool TestIntersects( const Box3D_T<U> & a, const Box3D_T<U> & b )
{
	// Check X Extents
	if (a.MAXX() < b.MINX()) { return false; }
	if (b.MAXX() < a.MINX()) { return false; }

	// Check Y extents
	if (a.MAXY() < b.MINY()) { return false; }
	if (b.MAXY() < a.MINY()) { return false; }

	// Check Z extents
	if (a.MAXZ() < b.MINZ()) { return false; }
	if (b.MAXZ() < a.MINZ()) { return false; }

	// Success - They overlap in X, Y, and Z so they must intersect
	return true;

}


// Test if box 'a' contains box 'b'
	// true iff
	//    a.minx <= b.minx <= b.maxx <= a.maxx
	//    a.miny <= b.miny <= b.maxy <= a.maxy
template <typename U>
bool TestContains( const Box3D_T<U> & a, const Box3D_T<U> & b )
{
	// Check X Extents
	if (b.MINX() < a.MINX()) { return false; }
	if (a.MAXX() < b.MAXX()) { return false; }

	// Check Y extents
	if (b.MINY() < a.MINY()) { return false; }
	if (a.MAXY() < b.MINY()) { return false; }

	// Check Z extents
	if (b.MINZ() < a.MINZ()) { return false; }
	if (a.MAXZ() < b.MINZ()) { return false; }

	// Must be contained by process of elimination
	return true;
}


// Computes Intersection of 2 boxes (if they do intersect)
template <typename U>
bool ComputeIntersection
( 
	const Box3D_T<U> & a,		// IN - 1st box to intersect
	const Box3D_T<U> & b,		// IN - 2nd box to intersect
	      Box3D_T<U> & result	// OUT - resulting intersection box
)
{
	bool bResult; 

	float axm = a.MINX();
	float axM = a.MAXX();
	float bxm = b.MINX();
	float bxM = b.MAXX();

	float lx, rx;

	bResult = Intersect1DIntervals( axm, axM, bxm, bxM, lx, rx );
	if (bResult)
	{
		float aym = a.MINY();
		float ayM = a.MAXY();
		float bym = b.MINY();
		float byM = b.MAXY();

		float ly, ry;
		bResult = Intersect1DIntervals( aym, ayM, bym, byM, ly, ry );
		if (bResult)
		{
			float azm = a.MINY();
			float azM = a.MAXY();
			float bzm = b.MINY();
			float bzM = b.MAXY();

			float lz, rz;
			bResult = Intersect1DIntervals( azm, azM, bzm, bzM, lz, rz );
			if (bResult)
			{
				result.SetFromMinMax( lx, ly, lz, rx, ry, rz );
			}
		}
	}
	return bResult;
}


template <typename U>
bool SplitBox
(
	const Box3D_T<U> &  origBox,		// IN - Box to split
		  unsigned int  splitAxis,		// IN - axis to split on
		  U				splitValue,		// IN - value to split on
	      Box3D_T<U> &  leftBox,		// OUT - left split box
		  Box3D_T<U> &  rightBox		// OUT - right split box
)
{
	U mX, MX, mY, MY, mZ, MZ;

	mX = origBox.MINX();
	MX = origBox.MAXX();
	mY = origBox.MINY();
	MY = origBox.MAXY();
	mZ = origBox.MINZ();
	MZ = origBox.MAXZ();

	switch (splitAxis)
	{
	case X_AXIS:
		leftBox.SetFromMinMax( mX, mY, mZ, splitValue, MY, MZ );
		rightBox.SetFromMinMax( splitValue, mY, mZ, MX, MY, MZ );
		break;

	case Y_AXIS:
		leftBox.SetFromMinMax( mX, mY, mZ, MX, splitValue, MZ );
		rightBox.SetFromMinMax( mX, splitValue, mZ, MX, MY, MZ );
		break;

	case Z_AXIS:
		leftBox.SetFromMinMax( mX, mY, mZ, MX, MY, splitValue );
		rightBox.SetFromMinMax( mX, mY, splitValue, MX, MY, MZ );
		break;

	default:
		// Error
		return false;
	}

	return true;
}



#endif // _BOX_CPU_H

