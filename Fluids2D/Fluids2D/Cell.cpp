#include "Cell.h"

Cell::Cell()
{
	xPressure = 1;
	yPressure = 1;
	totalPressure = 1;

	xViscosity = 1;
	yViscosity = 1;
	totalViscosity = 1;

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
{	//		This pressure should change somehow				Density of water at room temp
	totalPressure = (-1 * (xPressure / 0.9982) + (yPressure / 0.9982)); //* gradient of weight

};

void Cell::
	viscosity()
{

					//Viscosity of water	
	totalViscosity = 0.894 * 1  * ((xViscosity - yViscosity) / (xPressure * yPressure)) ;

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
