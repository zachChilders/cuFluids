#include "Point.h"

CUDA_CALLABLE_MEMBER
Point3D::Point3D()
{
	left = nullptr;
	right = nullptr;
	currentDimension = 0;

	position.x = 0; position.y = 0; position.z = 0;
	velocity.x = 0; velocity.y = 0; velocity.z = 0;
	angle.x = 0;    angle.y = 0;    angle.z = 0;

	viscosity.x = 0; viscosity.y = 0; viscosity.z = 0;
	pressure.x = 0; pressure.y = 0; pressure.z = 0;

	numChildren = 0;
	grounded = false;

};

CUDA_CALLABLE_MEMBER
Point3D::Point3D(float x, float y, float z)
{
	left = nullptr;
	right = nullptr;
	currentDimension = 0;

	position.x = x; position.y = y; position.z = z;
	velocity.x = 0; velocity.y = 0; velocity.z = 0;
	angle.x = 0;    angle.y = 0;    angle.z = 0;

	viscosity.x = 0; viscosity.y = 0; viscosity.z = 0;
	pressure.x = 0; pressure.y = 0; pressure.z = 0;

	numChildren = 0;
	grounded = false;
};

void Point3D::update(float x, float y, float z)
{
	velocity.x = x; velocity.y = y; velocity.z = z;
	position += velocity;
};

float Point3D::length()
{
	return sqrt((position.x * position.x) * (position.y * position.y) * (position.z * position.z));
};

float& Point3D::operator[](int dimension)
{
	switch (dimension)
	{
		case 0:
			return position.x;
		case 1:
			return position.y;
		case 2:
			return position.z;
		default:
			return position.x; //This probably shouldn't be a thing.
	}
};

bool Point3D::operator<(const Point3D& other)
{
	switch (currentDimension)
	{
		case 0:
			return (position.x < other.position.x);
		case 1:
			return (position.y < other.position.y);
		case 2:
			return (position.z < other.position.z);
	}
}

bool Point3D::operator>(const Point3D& other)
{
	switch (currentDimension)
	{
		case 0:
			return (position.x > other.position.x);
		case 1:
			return (position.y > other.position.y);
		case 2:
			return (position.z > other.position.z);
	}
}

std::ostream& operator<<(std::ostream &out, Point3D &point)
{
	out << "(" << point.position.x << ", " << point.position.y << ", " << point.position.z << ")";
	return out;
}

CUDA_CALLABLE_MEMBER
void Point3D::operator+(float scalar)
{
	position.y += scalar;
}

float* Point3D::toFloats()
{
	float* floats = new float[3];
	floats[0] = position.x;
	floats[1] = position.y;
	floats[2] = position.z;

	return floats;
}