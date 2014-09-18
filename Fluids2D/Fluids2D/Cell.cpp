#include "Cell.h"

Cell::Cell()
{
	getParticles();
}


Cell::~Cell()
{
	delete [] particles;
}

//Navier-Stokes Here

//Weight kernel
void Cell::
	weight()
{

};

void Cell::
	pressure()
{

};

void Cell::
	viscosity()
{

};

void Cell::
	external()
{

};

void Cell::
	gravity()
{
	yVel -= 9.8f;
};
