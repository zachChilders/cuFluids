#pragma once

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

#include "cuda_runtime.h"
#include <iostream>
#include <glm/glm.hpp>


CUDA_CALLABLE_MEMBER
class Point3D
{
	public:
		//Constructors
		CUDA_CALLABLE_MEMBER 
		Point3D();
		CUDA_CALLABLE_MEMBER
		Point3D(float x, float y, float z);
		CUDA_CALLABLE_MEMBER
		~Point3D(){};

		//Methods
		void update(float x, float y, float z);
		float length();
		float dot(Point3D &point);
		void set(float x, float y, float z);

		//Operators
		friend std::ostream& operator<<(std::ostream &out, Point3D &point);

		Point3D operator +(Point3D &point);
		CUDA_CALLABLE_MEMBER 
		void operator +(float scalar);
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

		bool operator <(const Point3D& point);
		bool operator <(float p);
		bool operator >(const Point3D& point);
		bool operator >(float p);
		bool operator ==(Point3D &point);
		bool operator !=(Point3D &point);

		float* toFloats();

		float& operator [](int d);

		glm::vec3 position;
		glm::vec3 velocity;
		glm::vec3 angle;

		glm::vec3 viscosity;
		glm::vec3 pressure;

		float size;

		bool grounded;

		//KD specific
		Point3D *left;
		Point3D *right;
		int currentDimension;
		int numChildren;
		
};