#pragma once
#ifndef _POINT_CPU_H
#define _POINT_CPU_H
//-----------------------------------------------------------------------------
//	Name:	VS_Point.h
//	Desc:	Defines Simple 2D, 3D Point classes
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


//-------------------------------------
//	Enumerations
//-------------------------------------

enum Axis
{
	X_AXIS = 0,
	Y_AXIS = 1,
	Z_AXIS = 2
}; // End Axis

enum POSITION
{
	POS_UNKNOWN = 0,
	POS_INSIDE	= 1,
	POS_ON		= 2,
	POS_OUTSIDE = 3
}; // End Position


//-------------------------------------
//	Classes
//-------------------------------------

//---------------------------------------------------------
//	Name:	Point2D
//	Desc:	Simple 2D Point class
//---------------------------------------------------------

class Point2D
{
public:
	//------------------------------------
	//	Type definitions
	//------------------------------------
	typedef struct _val2 { 
		float x; 
		float y; 
	} VAL2;

protected:
	//------------------------------------
	//	Fields
	//------------------------------------
	union {
		float m_v[2];
		VAL2  m_p;
	};

	//------------------------------------
	//	Helper Methods
	//------------------------------------


public:
	//------------------------------------
	//	Properties
	//------------------------------------

	static unsigned int MaxDimension() { return static_cast<unsigned int>( 2 ); }

	inline float X() const { return m_p.x; }
	inline void X( float val ) { m_p.x = val; }

	inline float Y() const { return m_p.y; }
	inline void Y( float val ) { m_p.y = val; }

	inline float length2() const { return ( X()*X() + Y()*Y() ); }
	inline float length() const { return static_cast<float>( sqrt( length2() ) ); }
	inline void MakePerp() { float temp = m_p.x; m_p.x = -m_p.y; m_p.y = temp; }

	//------------------------------------
	//	Constructors
	//------------------------------------
	Point2D()
		{
		}
	Point2D( float x, float y )
		{
			m_p.x = x;
			m_p.y = y;
		}
	Point2D( const Point2D & toCopy )
		{
			m_p.x = toCopy.m_p.x;
			m_p.y = toCopy.m_p.y;
		}
	~Point2D() 
		{
		}

	// Operators

		// Copy Operator
	Point2D & operator = ( const Point2D & toCopy ) 
		{ 
			if (this == &toCopy)
				return (*this);
			m_p.x = toCopy.m_p.x; 
			m_p.y = toCopy.m_p.y; 
			return (*this); 
		}

		// Comparision Operators
	bool operator == ( const Point2D & toCompare ) const 
	{		
		return ( IsEqual( m_p.x, toCompare.m_p.x ) && 
			     IsEqual( m_p.y, toCompare.m_p.y ) );
	}

	bool operator != ( const Point2D & toCompare ) const 
	{		
		return (! ( (*this) == toCompare ) );
	}


		// Index Operators
	float & operator[]( unsigned int nIndex )
		{
			// ASSERT( nIndex < 2 );
			return m_v[nIndex];
		}	
	const float & operator[] ( unsigned int nIndex ) const 
		{
			// ASSERT( nIndex < 2 );
			return m_v[nIndex];
		}

	// Addition
	Point2D & operator += ( const float & value )
		{
			m_p.x += value;
			m_p.y += value;
			return (*this);
		}

	Point2D & operator += ( const Point2D & value )
		{
			m_p.x += value.m_p.x;
			m_p.y += value.m_p.y;
			return (*this);
		}

	// Subtraction
	Point2D & operator -= ( const float & value )
		{
			m_p.x -= value;
			m_p.y -= value;
			return (*this);
		}

	Point2D & operator -= ( const Point2D & value )
		{
			m_p.x -= value.m_p.x;
			m_p.y -= value.m_p.y;
			return (*this);
		}

	// Multiplication
	Point2D & operator *= ( const float & value )
		{
			m_p.x *= value;
			m_p.y *= value;
			return (*this);
		}

	// Division
	Point2D & operator /= ( const float & value )
		{
			T o_t = (T)(1.0 / (double)value);
			m_p.x *= o_t;
			m_p.y *= o_t;
			return (*this);
		}

	// Comparision Methods

	// Methods 
	float length2() { return ((m_p.x*m_p.x) + (m_p.y*m_p.y)); }
	float length() { return static_cast<float>( sqrt( static_cast<double>( length2() ) ) ); }
	void Normalize() { (*this) /= length(); }

	void ToString( std::string & value ) const
	{
		char szBuff[256];
		unsigned int cchBuff = ARRAY_SIZE(szBuff);

		
		double x = static_cast< double >( X() );
		double y = static_cast< double >( Y() );

		StringPrintfA( szBuff, cchBuff, "[%3.6f, %3.6f]", x, y );
		value = szBuff;
	}

	void Dump() const
	{
		std::string pointValue;
		ToString( pointValue );
		Dumpf( "%s", pointValue.c_str() );
	}

	// Comparision Methods
	bool IsLessThan( const Point2D & toCompare ) const;
	bool IsLessEqual( const Point2D & toCompare ) const;
	bool IsGreaterThan ( const Point2D & toCompare ) const;
	bool IsGreaterEqual( const Point2D & toCompare ) const;
	bool IsEqual( const Point2D & toCompare ) const;
	bool IsNotEqual( const Point2D & toCompare ) const;

	// Line\Ray Tests
	int  Orient2D( const Point2D & p, const Point2D & q ) const;
	bool OnLine( const Point2D & p, const Point2D & q ) const;
	bool LeftOf( const Point2D & p, const Point2D & q ) const;
	bool LeftOn( const Point2D & p, const Point2D & q ) const;
	bool RightOf( const Point2D & p, const Point2D & q ) const;
	bool RightOn( const Point2D & p, const Point2D & q ) const;
	bool OnRay( const Point2D & p, const Point2D & q ) const;
	bool RightAhead( const Point2D & p, const Point2D & q ) const;
	bool LeftBehind( const Point2D & p, const Point2D & q ) const;


	// Friends
	friend Point2D operator + ( const Point2D & a, const Point2D & b );

	friend Point2D operator + ( const Point2D & a, const float & b );
	friend Point2D operator + ( const float & a, const Point2D & b );

	friend Point2D operator - ( const Point2D & a, const Point2D & b );
	friend Point2D operator - ( const Point2D & a, const float & b );
	friend Point2D operator - ( const float & a, const Point2D & b );

	friend Point2D operator * ( const Point2D & a, const float & b );
	friend Point2D operator * ( const float & a, const Point2D & b );

	friend Point2D operator / ( const Point2D & a, const float & b );
	friend Point2D operator / ( const float & a, const Point2D & b );

	friend Point2D Normalize( const Point2D & a );
	friend float   Dot( const Point2D & a, const Point2D & b );
	friend Point2D Perpendicular( const Point2D & a );

	// Comparision Friends
	template <typename U> friend bool IsLessThan    ( const Point2D & p, const Point2D & q );
	template <typename U> friend bool IsLessEqual   ( const Point2D & p, const Point2D & q );
	template <typename U> friend bool IsGreaterThan ( const Point2D & p, const Point2D & q );
	template <typename U> friend bool IsGreaterEqual( const Point2D & p, const Point2D & q );
	template <typename U> friend bool IsEqual( const Point2D & p, const Point2D & q );
	template <typename U> friend bool IsNotEqual( const Point2D & p, const Point2D & q );

	// Line\Ray Friends
	template <typename U> friend int  Orient2D( const Point2D & r, const Point2D & p, const Point2D & q );
	template <typename U> friend bool OnLine( const Point2D & r, const Point2D & p, const Point2D & q );
	template <typename U> friend bool LeftOf( const Point2D & r, const Point2D & p, const Point2D & q );
	template <typename U> friend bool LeftOn( const Point2D & r, const Point2D & p, const Point2D & q );
	template <typename U> friend bool RightOf( const Point2D & r, const Point2D & p, const Point2D & q );
	template <typename U> friend bool RightOn( const Point2D & r, const Point2D & p, const Point2D & q );
	template <typename U> friend bool OnRay( const Point2D & r, const Point2D & p, const Point2D & q );
	template <typename U> friend bool RightAhead( const Point2D & r, const Point2D & p, const Point2D & q );
	template <typename U> friend bool LeftBehind( const Point2D & r, const Point2D & p, const Point2D & q );

}; // End Point2D_T


// Simplify Messy Template Semantics
typedef Point2D_T<float>	Point2D;					// 2D Point (Float)
//typedef Point2D_T<double>	Point2D_D;					// 2D Point (Double)

typedef Point2D		Vector2D;							// 2D Vector (Float)
//typedef Point2D_D Vector2D_D;							// 2D Vector (Double)

typedef Point2D Normal2D;								// 2D Normal (Float)
//typedef Point2D_D Normal2D_D;							// 2D Normal (Double)

// Define Vectors of Point Types
typedef std::vector<Point2D>	Point2DList;			// Vector (Dynamic Array) of 2D Points (Float)
//typedef std::vector<Point2D_D>	Point2D_D_List;		// Vector (Dynamic Array) of 2D Points (Double)

typedef std::vector<Vector2D>	Vector2DList;			// Vector (Dynamic Array) of 2D Vectors (Float)
//typedef std::vector<Vector2D_D> Vector2D_D_List;		// Vector (Dynamic Array) of 2D Vectors (Double)

typedef std::vector<Normal2D> Normal2DList;				// Vector (Dynamic Array) of 2D Normals (Float)
//typedef std::vector<Normal2D_D> Normal2D_D_List;		// Vector (Dynamic Array) of 2D Normals (Double)


// Addition of 2 points
template <typename U>
Point2D operator + ( const Point2D & a, const Point2D & b )
{
	Point2D c( a.X() + b.X(), a.Y() + b.Y());
	return c;
}

// Add scalar to a point
template <typename U>
Point2D operator + ( const Point2D & a, const U & b )
{
	Point2D c( a.X() + b, a.Y() + b );
	return c;
}

// Add scalar to a point
template <typename U>
Point2D operator + ( const U & a, const Point2D & b )
{
	Point2D c( a + b.X(), a + b.Y() );
	return c;
}


// Subtract 2 points
template <typename U>
Point2D operator - ( const Point2D & a, const Point2D & b )
{
	Point2D c( a.X() - b.X(), a.Y() - b.Y() );
	return c;
}

// Subtract scalar from a point
template <typename U>
Point2D operator - ( const Point2D & a, const U & b )
{
	Point2D c( a.X() - b, a.Y() - b );
	return c;
}

// Subtract point from a scalar
template <typename U>
Point2D operator - ( const U & a, const Point2D & b )
{
	Point2D c( a - b.X(), a - b.Y() );
	return c;
}


// Multiply point by a scalar
template <typename U>
Point2D operator * ( const Point2D & a, const U & b )
{
	Point2D c( a.X() * b, a.Y() * b );
	return c;
}

// Multiply scalar by a point
template <typename U>
Point2D operator * ( const U & a, const Point2D & b )
{
	Point2D c( a * b.X(), a * b.Y() );
	return c;
}

// Divide point by a scalar
template <typename U>
Point2D operator / ( const Point2D & a, const U & b )
{
	U o_b = (U)(1.0 / (double)b);
	Point2D c( a.X() * o_b, a.Y() * o_b );
	return c;
}

// Divide scalar by a point
template <typename U>
Point2D operator / ( const U & a, const Point2D & b )
{
	Point2D c( a / b.X(), a / b.Y() );
	return c;
}

// Dot Product
template <typename U>
U Dot( const Point2D & a, const Point2D & b )
{
	U result = a.X()*b.X() + a.Y()*b.Y();
	return result;
}

template <typename U>
Point2D Normalize( const Point2D & value )
{
	Point2D c( value );
	c.Normalize();
	return c;
}

template <typename U>
Point2D Perpendicular( const Point2D & value )
{
	Point2D c( -value.Y(), value.X() );
	return c;
}

//---------------------------------------------------------
//	Name:	Point2D::Orient2D
//	Desc:	is the point R to the left, on, or to the right 
//			of the line	defined by directed edge PQ
//	Notes:
//	From Jonathan Shewchuk
//	Orientation (left, right, or on) of point r with respect 
//	to line pq can be determined by
//	
//	   |px py 1|
//	det|qx qy 1| = det|px-rx py-ry| = (px-rx)*(qy-ry)-(py-ry)*(qx-rx)
//     |rx ry 1|      |qx-rx qy-ry|
//
//	a positive '+' result means R is to left (CCW) of PQ
//  a negative '-' result means R is to right (CW) of PQ
//  a zero result means R is on line PQ
//---------------------------------------------------------

template <typename T>
int Point2D::Orient2D
( 
	const Point2D & p,		// IN - Point defining line PQ
	const Point2D & q		// IN - Point defining line PQ
) const
{
	// BUGBUG:
	//	Jonathans original Fast and inaccurate technique, 
	//  can be replaced by a more exact adaptive method
	//  +: eliminates the need for the tolerance check below...
	//  -: the adaptive approach takes a lot more operations...
	//	
	double prX = p.X() - this->X();
	double qrX = q.X() - this->X();
	double prY = p.Y() - this->Y();
	double qrY = q.Y() - this->Y();

	// Tolerance based check
	double pos = prX*qrY;
	double neg = prY*qrX;
	if (pos > (neg+c_dEPSILON))			{ return  1; }	// Left of Edge
	else if (pos < (neg-c_dEPSILON))	{ return -1; }	// Right of Edge
	else								{ return  0; }	// On Edge
}

template <typename U>
int Orient2D
( 
	const Point2D & r,		// IN - Point to test
	const Point2D & p,		// IN - Point defining line PQ
	const Point2D & q		// IN - Point defining line PQ
)
{
	return r.Orient2D( p, q );
}


template <typename T>
bool Point2D::IsLessThan( const Point2D & toCompare ) const
{
	if (X() < toCompare.X())
	{
		// p < q
		return true;
	}
	else if (toCompare.X() < X())
	{
		// q < p
		return false;
	}
	else /* p.X == q.X */
	{
		// Tie-Break on Y
		if (Y() < toCompare.Y())
		{
			// p < q
			return true;
		}
		else /* p.Y <= q.Y */
		{
			// q <= p
			return false;
		}
	}
}

template <typename T>
bool Point2D::IsGreaterThan( const Point2D & toCompare ) const
{
	return toCompare.IsLessThan( *this );
}

template <typename T>
bool Point2D::IsGreaterEqual( const Point2D & toCompare ) const
{
	return (! IsLessThan( toCompare ));
}

template <typename T>
bool Point2D::IsLessEqual( const Point2D & toCompare ) const
{
	return (toCompare.IsGreaterEqual( p ));
}

template <typename T>
bool Point2D::IsEqual( const Point2D & toCompare ) const
{
	return (((X() == toCompare.X()) && (Y() == toCompare.Y())) ? true : false);
}

template <typename T>
bool Point2D::IsNotEqual( const Point2D & toCompare ) const
{
	return (! IsEqual( toCompare ));
}



// IsLessThan (check if p < q)
template <typename U>
bool IsLessThan( const Point2D & p, const Point2D & q )
{
	if (p.X() < q.X()) // Compare p,q on X-Axis first
	{ 
		// p < q
		return true; 
	}	
	else if (q.X() < p.X())
	{
		// q < p
		return false;
	}
	else /* p.X == q.X */
	{
		// Tie-Break on Y
		if (p.Y() < q.Y()) 
		{
			// p < q
			return true; 
		}
		else /* q <= p */
		{
			// q <= p
			return false;
		}
	}
}

// IsGreaterThan (check if p > q)
template <typename U>
bool IsGreaterThan( const Point2D & p, const Point2D & q )
{
	return IsLessThan( q, p );
}

// IsGreaterEqual (check if p >= q)
template <typename U>
bool IsGreaterEqual( const Point2D & p, const Point2D & q )
{
	return (! IsLessThan( p, q ));
}

// IsLessEqual (check if p <= q)
template <typename U>
bool IsLessEqual( const Point2D & p, const Point2D & q )
{
	return IsGreaterEqual( q, p );
}

// IsEqual (check if p == q)
template <typename U>
bool IsEqual( const Point2D & p, const Point2D & q )
{
	return ((p.X() == q.X()) && (p.Y() == q.Y()));
}

// IsNotEqual (check if p != q)
template <typename U>
bool IsNotEqual( const Point2D & p, const Point2D & q )
{
	return (! IsEqual( p, q ));
}


template <typename T>
bool Point2D::LeftOf( const Point2D & p, const Point2D & q ) const
{
	int iResult = Orient2D( p, q );
	return (iResult > 0) ? true : false;
}

// Is 'this' = 'r' point strictly to the right of the line defined by points 'p' & 'q' ?
template <typename T>
bool Point2D::RightOf( const Point2D & p, const Point2D & q ) const
{
	int iResult = Orient2D( p, q );
	return (iResult < 0) ? true : false;
}

// Is 'this' = 'r' point to the right of or on the line defined by points 'p' & 'q' ?
template <typename T>
bool Point2D::RightOn( const Point2D & p, const Point2D & q ) const
{
	return (!LeftOf( p, q ));
}

// Is 'this' = 'r' point to the left of or on the line defined by points 'p' & 'q' ?
template <typename T>
bool Point2D::LeftOn( const Point2D & p, const Point2D & q ) const
{
	return RightOn( q, p );
}

// Is 'this' = 'r' point on the line defined by points 'p' & 'q' ?
template <typename T>
bool Point2D::OnLine( const Point2D & p, const Point2D & q ) const
{
	int iResult = Orient2D( p, q );
	return (iResult == 0) ? true : false;
}


// Point2D::OnRay
// Check if this point is on ray from point 'p' thru point 'q'
template <typename T>
bool Point2D::OnRay( const Point2D & p, const Point2D & q ) const
{
	if (OnLine( p, q ))
	{
		Point2D rp = r - p;
		Point2D qp = q - p;
		T val = Dot( rp, qp );
		if (val >= static_cast<T>( 0.0 ))
		{
			return true;
		}
	}
	return false;
}

// Point2D::RightAhead
template <typename T>
bool Point2D::RightAhead( const Point2D & p, const Point2D & q ) const
{
	int iResult = Orient2D( p, q );
	if (iResult < 0) { return true; }		// Right of PQ
	else if (iResult > 0) { return false;}	// left of PQ
	else									// Must be on PQ
	{
		// Additional Ray check required
		Point2D rq( m_point.X() - q.m_point.X(), m_point.Y() - q.m_point.Y() );
		Point2D qp( q.m_point.X() - p.m_point.X(), q.m_point.Y() - p.m_point.Y() );
		float val = Dot( rq, qp );
		if (val >= static_cast<float>( 0.0 ))
		{
			return true;
		}
		return false;
	}
}

// Point2D::LeftBehind
template <typename T>
bool Point2D::LeftBehind( const Point2D & p, const Point2D & q ) const
{
	return (! RightAhead( p, q ));
}


template <typename U>
bool LeftOf( const Point2D & r, const Point2D & p, const Point2D & q )
{
	return r.LeftOf( p, q );
}

// Is 'this' = 'r' point strictly to the right of the line defined by points 'p' & 'q' ?
template <typename U>
bool RightOf( const Point2D & r, const Point2D & p, const Point2D & q )
{
	return r.RightOf( p, q );
}

// Is 'this' = 'r' point to the right of or on the line defined by points 'p' & 'q' ?
template <typename U>
bool RightOn( const Point2D & r, const Point2D & p, const Point2D & q )
{
	return r.RightOn( p, q );
}

// Is 'this' = 'r' point to the left of or on the line defined by points 'p' & 'q' ?
template <typename U>
bool LeftOn( const Point2D & r, const Point2D & p, const Point2D & q )
{
	return r.LeftOn( p, q );
}

// Is 'this' = 'r' point on the line defined by points 'p' & 'q' ?
template <typename U>
bool OnLine( const Point2D & r, const Point2D & p, const Point2D & q )
{
	return r.OnLine( p, q );
}

// Point2D::OnRay
// Check if this point is on ray from point 'p' thru point 'q'
template <typename U>
bool OnRay( const Point2D & r, const Point2D & p, const Point2D & q )
{
	return r.OnRay( p, q );
}

// Point2D::RightAhead
template <typename U>
bool RightAhead( const Point2D & r, const Point2D & p, const Point2D & q )
{
	return r.RightAhead( p, q );
}

// Point2D::LeftBehind
template <typename U>
bool LeftBehind( const Point2D & r, const Point2D & p, const Point2D & q )
{
	return r.LeftBehind( p, q );
}



//---------------------------------------------------------
//	Name:	Point3D
//	Desc:	Simple 3D Point class
//---------------------------------------------------------

template <typename T> 
class Point3D_T
{
public:
	//------------------------------------
	//	Type definitions
	//------------------------------------
	typedef unsigned int	unsigned int;		// Underying Size Type
	typedef T				float;		// Underlying Value Type
	typedef struct _val3 { 
		float x; 
		float y; 
		float z;
	} VAL3;

protected:
	//------------------------------------
	//	Fields
	//------------------------------------
	union {
		T	 m_v[3];
		VAL3 m_p;
	};

	//------------------------------------
	//	Helper Methods
	//------------------------------------

public:
	//------------------------------------
	//	Properties
	//------------------------------------
	static unsigned int MaxDimension() { return static_cast<unsigned int>( 3 ); }

	inline float X() const { return m_p.x; }
	inline void X( float val ) { m_p.x = val; }

	inline float Y() const { return m_p.y; }
	inline void Y( float val ) { m_p.y = val; }

	inline float Z() const { return m_p.z; }
	inline void Z( float val ) { m_p.z = val; }

	inline float length2() const { return ( m_p.x*m_p.x + m_p.y*m_p.y + m_p.z*m_p.z ); }
	inline float length() const { return static_cast<T>( sqrt( (m_p.x*m_p.x + m_p.y*m_p.y + m_p.z*m_p.z ) ) ) ; }

	//------------------------------------
	//	Constructors
	//------------------------------------
	Point3D_T()
		{
		}
	Point3D_T( float x, float y, float z )
		{
			m_p.x = x;
			m_p.y = y;
			m_p.z = z;
		}
	Point3D_T( const Point3D_T & toCopy )
		{
			m_p.x = toCopy.m_p.x;
			m_p.y = toCopy.m_p.y;
			m_p.z = toCopy.m_p.z;
		}
	~Point3D_T() 
		{
		}

	//------------------------------------
	//	Operators
	//------------------------------------
		// Copy Operator
	Point3D_T & operator = ( const Point3D_T & toCopy ) 
		{ 
			if (this == &toCopy)
				return (*this);
			m_p.x = toCopy.m_p.x; 
			m_p.y = toCopy.m_p.y; 
			m_p.z = toCopy.m_p.z;
			return (*this); 
		}

		// Comparision Operators
	bool operator == ( const Point3D_T<T> & toCompare ) const 
	{		
		return ( IsEqual( m_p.x, toCompare.m_p.x ) && 
			     IsEqual( m_p.y, toCompare.m_p.y ) &&
				 IsEqual( m_p.z, toCompare.m_p.z ) );
	}

	bool operator != ( const Point3D_T<T> & toCompare ) const 
	{		
		return (! ( (*this) == toCompare ) );
	}

		// Index Operators
	float & operator[]( unsigned int nIndex )
		{
			// ASSERT( nIndex < 3 );
			return m_v[nIndex];
		}	
	const float & operator[] ( unsigned int nIndex ) const 
		{
			// ASSERT( nIndex < 3 );
			return m_v[nIndex];
		}

	// Addition
	Point3D_T & operator += ( const float & value )
		{
			m_p.x += value;
			m_p.y += value;
			m_p.z += value;
			return (*this);
		}

	Point3D_T & operator += ( const Point3D_T & value )
		{
			m_p.x += value.m_p.x;
			m_p.y += value.m_p.y;
			m_p.z += value.m_p.z;
			return (*this);
		}

	// Subtraction
	Point3D_T & operator -= ( const float & value )
		{
			m_p.x -= value;
			m_p.y -= value;
			m_p.z -= value;
			return (*this);
		}

	Point3D_T & operator -= ( const Point3D_T & value )
		{
			m_p.x -= value.m_p.x;
			m_p.y -= value.m_p.y;
			m_p.z -= value.m_p.z;
			return (*this);
		}

	// Multiplication
	Point3D_T & operator *= ( const float & value )
		{
			m_p.x *= value;
			m_p.y *= value;
			m_p.z *= value;
			return (*this);
		}

	// Division
	Point3D_T & operator /= ( const float & value )
		{
			T o_t = (T)(1.0 / (double)value);
			m_p.x *= o_t;
			m_p.y *= o_t;
			m_p.z *= o_t;
			return (*this);
		}

	// Methods 
	T length2() { return ( X()*X() + Y()*Y() + Z()*Z() ); }
	T length() { return static_cast<T>( sqrt( static_cast<double>( length2() ) ) ); }
	void Normalize() { (*this) /= length(); }

	void ToString( std::string & value ) const
	{
		char szBuff[256];
		
		double x = static_cast< double >( X() );
		double y = static_cast< double >( Y() );
		double z = static_cast< double >( Z() );

		StringPrintfA( szBuff, 256, "[%3.6f, %3.6f, %3.6f]", x, y, z );
		value = szBuff;
	}

	void Dump() const
	{
		std::string pointValue;
		ToString( pointValue );
		DumpfA( "%s", pointValue.c_str() );
	}


	// Friends
	template <typename U> friend Point3D_T<U> operator + ( const Point3D_T<U> & a, const Point3D_T<U> & b );
	template <typename U> friend Point3D_T<U> operator + ( const Point3D_T<U> & a, const U & b );
	template <typename U> friend Point3D_T<U> operator + ( const U & a, const Point3D_T<U> & b );

	template <typename U> friend Point3D_T<U> operator - ( const Point3D_T<U> & a, const Point3D_T<U> & b );
	template <typename U> friend Point3D_T<U> operator - ( const Point3D_T<U> & a, const U & b );
	template <typename U> friend Point3D_T<U> operator - ( const U & a, const Point3D_T<U> & b );

	template <typename U> friend Point3D_T<U> operator * ( const Point3D_T<U> & a, const U & b );
	template <typename U> friend Point3D_T<U> operator * ( const U & a, const Point3D_T<U> & b );

	template <typename U> friend Point3D_T<U> operator / ( const Point3D_T<U> & a, const U & b );
	template <typename U> friend Point3D_T<U> operator / ( const U & a, const Point3D_T<U> & b );

	template <typename U> friend Point3D_T<U> Normalize( const Point3D_T<U> & a );
	template <typename U> friend U			  Dot( const Point3D_T<U> & a, const Point3D_T<U> & b );
	template <typename U> friend Point3D_T<U> CrossProduct( const Point3D_T<U> & a, const Point3D_T<U> & b );

	// Comparision Friends
	template <typename U> friend bool IsLessThan    ( const Point3D_T<U> & p, const Point3D_T<U> & q );
	template <typename U> friend bool IsLessEqual   ( const Point3D_T<U> & p, const Point3D_T<U> & q );
	template <typename U> friend bool IsGreaterThan ( const Point3D_T<U> & p, const Point3D_T<U> & q );
	template <typename U> friend bool IsGreaterEqual( const Point3D_T<U> & p, const Point3D_T<U> & q );
	template <typename U> friend bool IsEqual( const Point3D_T<U> & p, const Point3D_T<U> & q );
	template <typename U> friend bool IsNotEqual( const Point3D_T<U> & p, const Point3D_T<U> & q );

}; // End Point3D_T

// Addition of 2 points
template <typename U>
Point3D_T<U> operator + ( const Point3D_T<U> & a, const Point3D_T<U> & b )
{
	Point3D_T<U> c( a.X() + b.X(), a.Y() + b.Y(), a.Z() + b.Z() );
	return c;
}

// Add scalar to a point
template <typename U>
Point3D_T<U> operator + ( const Point3D_T<U> & a, const U & b )
{
	Point3D_T<U> c( a.X() + b, a.Y() + b, a.Z() + b );
	return c;
}

// Add scalar to a point
template <typename U>
Point3D_T<U> operator + ( const U & a, const Point3D_T<U> & b )
{
	Point3D_T<U> c( a + b.X(), a + b.Y(), a + b.Z() );
	return c;
}


// Subtract 2 points
template <typename U>
Point3D_T<U> operator - ( const Point3D_T<U> & a, const Point3D_T<U> & b )
{
	Point3D_T<U> c( a.X() - b.X(), a.Y() - b.Y(), a.Z() - b.Z() );
	return c;
}

// Subtract scalar from a point
template <typename U>
Point3D_T<U> operator - ( const Point3D_T<U> & a, const U & b )
{
	Point3D_T<U> c( a.X() - b, a.Y() - b, a.Z() - b );
	return c;
}

// Subtract point from a scalar
template <typename U>
Point3D_T<U> operator - ( const U & a, const Point3D_T<U> & b )
{
	Point3D_T<U> c( a - b.X(), a - b.Y(), a - b.Z() );
	return c;
}


// Multiply point by a scalar
template <typename U>
Point3D_T<U> operator * ( const Point3D_T<U> & a, const U & b )
{
	Point3D_T<U> c( a.X() * b, a.Y() * b, a.Z() * b );
	return c;
}

// Multiply scalar by a point
template <typename U>
Point3D_T<U> operator * ( const U & a, const Point3D_T<U> & b )
{
	Point3D_T<U> c( a * b.X(), a * b.Y(), a * b.Z() );
	return c;
}

// Divide point by a scalar
template <typename U>
Point3D_T<U> operator / ( const Point3D_T<U> & a, const U & b )
{
	U o_b = (U)(1.0 / (double)b);
	Point3D_T<U> c( a.X() * o_b, a.Y() * o_b, a.Z() * o_b );
	return c;
}

// Divide scalar by a point
template <typename U>
Point3D_T<U> operator / ( const U & a, const Point3D_T<U> & b )
{
	Point3D_T<U> c( a / b.X(), a / b.Y(), a / b.Z() );
	return c;
}

// Dot Product
template <typename U>
U Dot( const Point3D_T<U> & a, const Point3D_T<U> & b )
{
	U result = a.X()*b.X() + a.Y()*b.Y() + a.Z()*b.Z();
	return result;
}

template <typename U>
Point3D_T<U> Normalize( const Point3D_T<U> & value )
{
	Point3D_T<U> c( value );
	c.Normalize();
	return c;
}

template <typename U>
Point3D_T<U> CrossProduct( const Point3D_T<U> & a, const Point3D_T<U> & b )
{
	// Assumes Right Hand rule (ccw)
	Point3D_T<U> c(
					(a.Y()*b.Z()) - (a.Z()*b.Y()),
					(a.Z()*b.X()) - (a.X()*b.Z()),		// - [a.x()*b.z() - a.z()*b.x()]
					(a.X()*b.Y()) - (a.Y()*b.X())
				  );
	return c;
}


// IsLessThan (check if p < q)
template <typename U>
bool IsLessThan( const Point3D_T<U> & p, const Point3D_T<U> & q )
{
	if (p.X() < q.X()) // Compare p,q on X-Axis first
	{ 
		// p < q
		return true; 
	}	
	else if (q.X() < p.X())
	{
		// q < p
		return false;
	}
	else /* p.X == q.X */
	{
		// Tie Break on Y
		if (p.Y() < q.Y()) 
		{
			// p < q
			return true; 
		}
		else if (q.Y() < p.Y())
		{
			// q < p
			return false;
		}
		else /* p.Y == q.Y */
		{
			// Tie Break on Z
			if (p.Z() < q.Z())
			{
				// p < q
				return true;
			}
			else /* q.Z <= p.Z */
			{
				// q <= p
				return false;
			}
		}
	}
}

// IsGreaterThan (check if p > q)
template <typename U>
bool IsGreaterThan( const Point3D_T<U> & p, const Point3D_T<U> & q )
{
	return IsLessThan( q, p );
}

// IsGreaterEqual (check if p >= q)
template <typename U>
bool IsGreaterEqual( const Point3D_T<U> & p, const Point3D_T<U> & q )
{
	return (! IsLessThan( p, q ));
}

// IsLessEqual (check if p <= q)
template <typename U>
bool IsLessEqual( const Point3D_T<U> & p, const Point3D_T<U> & q )
{
	return IsGreaterEqual( q, p );
}

// IsEqual (check if p == q)
template <typename U>
bool IsEqual( const Point3D_T<U> & p, const Point3D_T<U> & q )
{
	return ((p.X() == q.X()) && (p.Y() == q.Y()));
}

// IsNotEqual (check if p != q)
template <typename U>
bool IsNotEqual( const Point3D_T<U> & p, const Point3D_T<U> & q )
{
	return (! IsEqual( p, q ));
}


// Simplify Messy Template Semantics
typedef Point3D_T<float> Point3D;					// 3D Point (Float)
//typedef Point3D_T<double> Point3D_D;					// 3D Point (Double)

typedef Point3D Vector3D;							// 3D Vector (Float)
//typedef Point3D_D Vector3D_D;						// 3D Vector (Double)

typedef Point3D Normal3D;							// 3D Normal (Float)
//typedef Point3D_D Normal3D_D;						// 3D Normal (Double)

// Define Vectors of Point types
typedef std::vector<Point3D> Point3DList;			// Vector (Dynamic Array) of 3D Points (Float)
//typedef std::vector<Point3D_D> Point3D_D_List;	// Vector (Dynamic Array) of 3D Points (Double)

typedef std::vector<Vector3D> Vector3DList;			// Vector (Dynamic Array) of 3D Vectors (Float)
//typedef std::vector<Vector3D_D> Vector3D_D_List;	// Vector (Dynamic Array) of 3D Vectors (Double)

typedef std::vector<Normal3D> Normal3DList;			// Vector (Dynamic Array) of 3D Normals (Float)
//typedef std::vector<Normal3D_D> Normal3D_D_List;	// Vector (Dynamic Array) of 3D Normals (Double)



//---------------------------------------------------------
//	Name:	Point4D
//	Desc:	Simple 4D Point class
//---------------------------------------------------------

template <typename T> 
class Point4D_T
{
public:
	//------------------------------------
	//	Type definitions
	//------------------------------------
	
	typedef unsigned int	unsigned int;		// Underying Size Type
	typedef T				float;		// Underlying Value Type
	typedef struct _val4 { 
		float x; 
		float y; 
		float z;
		float w;
	} VAL4;


protected:
	//------------------------------------
	//	Fields
	//------------------------------------
	
	union {
		T	 m_v[4];
		VAL4 m_p;
	};


	//------------------------------------
	//	Helper Methods
	//------------------------------------


public:
	//------------------------------------
	//	Properties
	//------------------------------------
	
	static unsigned int MaxDimension() { return static_cast<unsigned int>( 4 ); }

	inline float X() const { return m_p.x; }
	inline void X( float val ) { m_p.x = val; }

	inline float Y() const { return m_p.y; }
	inline void Y( float val ) { m_p.y = val; }

	inline float Z() const { return m_p.z; }
	inline void Z( float val ) { m_p.z = val; }

	inline float W() const { return m_p.w; }
	inline void W( float val ) { m_p.w = val; }

	inline float length2() const { return ( m_p.x*m_p.x + m_p.y*m_p.y + m_p.z*m_p.z + m_p.w*m_p.w ); }
	inline float length() const { return static_cast<T>( sqrt( (m_p.x*m_p.x + m_p.y*m_p.y + m_p.z*m_p.z + m_p.w*m_p.w ) ) ) ; }


	//------------------------------------
	//	Constructors
	//------------------------------------

	Point4D_T()
		{
		}

	Point4D_T( float x, float y, float z, float w )
		{
			m_p.x = x;
			m_p.y = y;
			m_p.z = z;
			m_p.w = w;
		}

	Point4D_T( const Point4D_T & toCopy )
		{
			m_p.x = toCopy.m_p.x;
			m_p.y = toCopy.m_p.y;
			m_p.z = toCopy.m_p.z;
			m_p.w = toCopy.m_p.w;
		}

	~Point4D_T() 
		{
		}

	void SetFrom3DPoint( const Point3D_T<T> & value )
		{
			m_p.x = value.X();
			m_p.y = value.Y();
			m_p.z = value.Z();
			m_p.w = (T)1.0;
		}

	void SetFrom3DVector( const Point3D_T<T> & value )
		{
			m_p.x = value.X();
			m_p.y = value.Y();
			m_p.z = value.Z();
			m_p.w = (T)0.0;
		}

	void Get3DPoint( Point3D_T<T> & value )
		{
			T o_w = (T)(1.0/(double)m_p.w);
			value.X( m_p.x * o_w );
			value.Y( m_p.y * o_w );
			value.Z( m_p.z * o_w );
		}

	void Get3DVector( Point3D_T<T> & value )
		{
			value.X( m_p.x );
			value.Y( m_p.y );
			value.Z( m_p.z );
		}

	//------------------------------------
	//	Operators
	//------------------------------------
		// Copy Operator
	Point4D_T & operator = ( const Point4D_T & toCopy ) 
		{ 
			if (this == &toCopy) { return (*this); }
			m_p.x = toCopy.m_p.x; 
			m_p.y = toCopy.m_p.y; 
			m_p.z = toCopy.m_p.z;
			m_p.w = toCopy.m_p.w;
			return (*this); 
		}

		// Comparision Operators
	bool operator == ( const Point4D_T<T> & toCompare ) const 
	{		
		return ( IsEqual( m_p.x, toCompare.m_p.x ) && 
			     IsEqual( m_p.y, toCompare.m_p.y ) &&
				 IsEqual( m_p.z, toCompare.m_p.z ) &&
				 IsEqual( m_p.w, toCompare.m_p.w ) );
	}

	bool operator != ( const Point4D_T<T> & toCompare ) const 
	{		
		return (! ( (*this) == toCompare ) );
	}

		// Index Operators
	float & operator[]( unsigned int nIndex )
		{
			// ASSERT( nIndex < 4 );
			return m_v[nIndex];
		}	

	const float & operator[] ( unsigned int nIndex ) const 
		{
			// ASSERT( nIndex < 4 );
			return m_v[nIndex];
		}

	// Addition
	Point4D_T & operator += ( const float & value )
		{
			m_p.x += value;
			m_p.y += value;
			m_p.z += value;
			m_w.z += value;
			return (*this);
		}

	Point4D_T & operator += ( const Point4D_T & value )
		{
			m_p.x += value.m_p.x;
			m_p.y += value.m_p.y;
			m_p.z += value.m_p.z;
			m_p.w += value.m_p.w;
			return (*this);
		}

	// Subtraction
	Point4D_T & operator -= ( const float & value )
		{
			m_p.x -= value;
			m_p.y -= value;
			m_p.z -= value;
			m_p.w -= value;
			return (*this);
		}

	Point4D_T & operator -= ( const Point4D_T & value )
		{
			m_p.x -= value.m_p.x;
			m_p.y -= value.m_p.y;
			m_p.z -= value.m_p.z;
			m_p.w -= value.m_p.w;
			return (*this);
		}

	// Multiplication
	Point4D_T & operator *= ( const float & value )
		{
			m_p.x *= value;
			m_p.y *= value;
			m_p.z *= value;
			m_p.w *= value;
			return (*this);
		}

	// Division
	Point4D_T & operator /= ( const float & value )
		{
			T o_t = (T)(1.0 / (double)value);
			m_p.x *= o_t;
			m_p.y *= o_t;
			m_p.z *= o_t;
			m_p.w *= o_t;
			return (*this);
		}

	// Methods 
	T length2() { return ( X()*X() + Y()*Y() + Z()*Z() + W()*W() ); }
	T length() { return static_cast<T>( sqrt( static_cast<double>( length2() ) ) ); }
	void Normalize() { (*this) /= length(); }

	void ToString( std::string & value ) const
	{
		char szBuff[256];
		
		double x = static_cast< double >( X() );
		double y = static_cast< double >( Y() );
		double z = static_cast< double >( Z() );
		double w = static_cast< double >( W() );

		StringPrintfA( szBuff, 256, "[%3.6f, %3.6f, %3.6f %3.6f]", x, y, z, w );
		value = szBuff;
	}

	void Dump() const
	{
		std::string pointValue;
		ToString( pointValue );
		DumpfA( "%s", pointValue.c_str() );
	}


	// Friends
	template <typename U> friend Point4D_T<U> operator + ( const Point4D_T<U> & a, const Point4D_T<U> & b );
	template <typename U> friend Point4D_T<U> operator + ( const Point4D_T<U> & a, const U & b );
	template <typename U> friend Point4D_T<U> operator + ( const U & a, const Point4D_T<U> & b );

	template <typename U> friend Point4D_T<U> operator - ( const Point4D_T<U> & a, const Point4D_T<U> & b );
	template <typename U> friend Point4D_T<U> operator - ( const Point4D_T<U> & a, const U & b );
	template <typename U> friend Point4D_T<U> operator - ( const U & a, const Point4D_T<U> & b );

	template <typename U> friend Point4D_T<U> operator * ( const Point4D_T<U> & a, const U & b );
	template <typename U> friend Point4D_T<U> operator * ( const U & a, const Point4D_T<U> & b );

	template <typename U> friend Point4D_T<U> operator / ( const Point4D_T<U> & a, const U & b );
	template <typename U> friend Point4D_T<U> operator / ( const U & a, const Point4D_T<U> & b );

	template <typename U> friend Point4D_T<U> Normalize( const Point4D_T<U> & a );
	template <typename U> friend U			  Dot( const Point4D_T<U> & a, const Point4D_T<U> & b );
	//template <typename U> friend Point4D_T<U> CrossProduct( const Point4D_T<U> & a, const Point4D_T<U> & b );

}; // End Point4D_T

// Addition of 2 points
template <typename U>
Point4D_T<U> operator + ( const Point4D_T<U> & a, const Point4D_T<U> & b )
{
	Point4D_T<U> c( a.X() + b.X(), a.Y() + b.Y(), 
					a.Z() + b.Z(), a.W() + b.W() );
	return c;
}

// Add scalar to a point
template <typename U>
Point4D_T<U> operator + ( const Point4D_T<U> & a, const U & b )
{
	Point4D_T<U> c( a.X() + b, a.Y() + b, 
					a.Z() + b, a.W() + b );
	return c;
}

// Add scalar to a point
template <typename U>
Point4D_T<U> operator + ( const U & a, const Point4D_T<U> & b )
{
	Point4D_T<U> c( a + b.X(), a + b.Y(), 
					a + b.Z(), a + b.W() );
	return c;
}


// Subtract 2 points
template <typename U>
Point4D_T<U> operator - ( const Point4D_T<U> & a, const Point4D_T<U> & b )
{
	Point4D_T<U> c( a.X() - b.X(), a.Y() - b.Y(),
					a.Z() - b.Z(), a.W() - b.W() );
	return c;
}

// Subtract scalar from a point
template <typename U>
Point4D_T<U> operator - ( const Point4D_T<U> & a, const U & b )
{
	Point4D_T<U> c( a.X() - b, a.Y() - b, 
					a.Z() - b, a.W() - b );
	return c;
}

// Subtract point from a scalar
template <typename U>
Point4D_T<U> operator - ( const U & a, const Point4D_T<U> & b )
{
	Point4D_T<U> c( a - b.X(), a - b.Y(), 
					a - b.Z(), a - b.W() );
	return c;
}


// Multiply point by a scalar
template <typename U>
Point4D_T<U> operator * ( const Point4D_T<U> & a, const U & b )
{
	Point4D_T<U> c( a.X() * b, a.Y() * b, 
					a.Z() * b, a.W() * b );
	return c;
}

// Multiply scalar by a point
template <typename U>
Point4D_T<U> operator * ( const U & a, const Point4D_T<U> & b )
{
	Point4D_T<U> c( a * b.X(), a * b.Y(), 
					a * b.Z(), a * b.W() );
	return c;
}

// Divide point by a scalar
template <typename U>
Point4D_T<U> operator / ( const Point4D_T<U> & a, const U & b )
{
	U o_b = (U)(1.0 / (double)b);
	Point4D_T<U> c( a.X() * o_b, a.Y() * o_b,
					a.Z() * o_b, a.W() * o_b );
	return c;
}

// Divide scalar by a point
template <typename U>
Point4D_T<U> operator / ( const U & a, const Point4D_T<U> & b )
{
	Point4D_T<U> c( a / b.X(), a / b.Y(), 
					a / b.Z(), a / b.W() );
	return c;
}

// Dot Product
template <typename U>
U Dot( const Point4D_T<U> & a, const Point4D_T<U> & b )
{
	U result = a.X()*b.X() + a.Y()*b.Y() + a.Z()*b.Z() + a.W()*b.W();
	return result;
}

// Normalize 4D Vector
template <typename U>
Point4D_T<U> Normalize( const Point4D_T<U> & value )
{
	Point4D_T<U> c( value );
	c.Normalize();
	return c;
}

/*
template <typename U>
Point4D_T<U> CrossProduct( const Point4D_T<U> & a, const Point4D_T<U> & b )
{
	// Assumes Right Hand rule (ccw)
	Point4D_T<U> c(
					(a.Y()*b.Z()) - (a.Z()*b.Y()),
					(a.Z()*b.X()) - (a.X()*b.Z()),		// - [a.x()*b.z() - a.z()*b.x()]
					(a.X()*b.Y()) - (a.Y()*b.X())
				  );
	return c;
}
*/

// Simplify Messy Template Semantics
typedef Point4D_T<float> Point4D;					// 4D Point (Float)
//typedef Point4D_T<double> Point4D_D;				// 4D Point (Double)

typedef Point4D Vector4D;							// 4D Vector (Float)
//typedef Point4D_D Vector4D_D;						// 4D Vector (Double)

//typedef Point4D Normal4D;							// 4D Normal (Float)
//typedef Point4D_D Normal4D_D;						// 4D Normal (Double)

// Define Vectors of Point types
typedef std::vector<Point4D> Point4DList;			// Vector (Dynamic Array) of 4D Points (Float)
//typedef std::vector<Point4D_D> Point4D_D_List;	// Vector (Dynamic Array) of 4D Points (Double)

typedef std::vector<Vector4D> Vector4DList;			// Vector (Dynamic Array) of 4D Vectors (Float)
//typedef std::vector<Vector4D_D> Vector4D_D_List;	// Vector (Dynamic Array) of 4D Vectors (Double)

//typedef std::vector<Normal4D> Normal4DList;			// Vector (Dynamic Array) of 4D Normals (Float)
//typedef std::vector<Normal4D_D> Normal4D_D_List;	// Vector (Dynamic Array) of 4D Normals (Double)



#endif // _VS_POINT_H

