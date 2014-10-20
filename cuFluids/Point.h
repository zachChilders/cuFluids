#pragma once

#include "kdUtil.h"

#include <iostream>

class Point3D
{

	public:

		//Constructors
		Point3D();
		Point3D(float x, float y, float z);
		~Point3D();

		//Methods
		void update(float x, float y, float z);
		float length();
		float dot(Point3D &point);

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
		bool operator >(Point3D &point);
		bool operator ==(Point3D &point);
		bool operator !=(Point3D &point);

		float operator [](DIMENSION d);
		
	private:
		vec3 position;
		vec3 velocity;
		vec3 angle;

};