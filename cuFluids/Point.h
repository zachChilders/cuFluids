#pragma once

#include "kdUtil.h"

#include <iostream>

class Point3D
{

	public:

		//Constructors
		Point3D();
		Point3D(float x, float y, float z);
		Point3D(const Point3D &p);
		~Point3D(){};

		//Methods
		void update(float x, float y, float z);
		float length();
		float dot(Point3D &point);
		void set(float x, float y, float z);

		//Operators
		friend std::ostream& operator<<(std::ostream &out, Point3D &point);

		Point3D operator +(Point3D &point);
		Point3D operator +(float scalar);
		Point3D operator +=(Point3D &point);
		Point3D operator +=(float scalar);

		Point3D operator -(Point3D &point);
		Point3D operator -(float scalar);
		Point3D operator -=(Point3D &point);
		Point3D operator -=(float scalar);

		Point3D operator *(Point3D &point);
		Point3D operator *(float scalar);
		Point3D operator *=(Point3D &point);
		Point3D operator *=(float scalar);

		Point3D operator /(Point3D &point);
		Point3D operator /(float scalar);
		Point3D operator /=(Point3D &point);
		Point3D operator /=(float scalar);

		bool operator <(Point3D &point);
		bool operator <(float p);
		bool operator >(Point3D &point);
		bool operator >(float p);
		bool operator ==(Point3D &point);
		bool operator !=(Point3D &point);

		float& operator [](int d);
		
		//KD specific
		Point3D *left;
		Point3D *right;
		int currentDimension;
		
		vec3 position;
		vec3 velocity;
		vec3 angle;

};