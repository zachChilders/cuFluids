#include "Point.h"

Point3D::Point3D()
{
	position.x = 0; position.y = 0; position.z = 0;
	velocity.x = 0; velocity.y = 0; velocity.z = 0;
	angle.x = 0;    angle.y = 0;    angle.z = 0;
};

Point3D::Point3D(float x, float y, float z)
{
	position.x = x; position.y = y; position.z = z;
	velocity.x = 0; velocity.y = 0; velocity.z = 0;
	angle.x = 0;    angle.y = 0;    angle.z = 0;
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
			return ;
	}
};