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
	external(float eX, float eY)
{
	externalForce = std::sqrt(eX * eX + eY * eY);
};

void Cell::
	gravity()
{
	yVel -= 9.8f;
};

//Loads particles into the local buffer
void Cell::
	getParticles()
{
	//Check every particle in the simulation
	for (std::vector<fluidParticle>::const_iterator it = grid->master.begin(); it != grid->master.end(); ++it)
	{
		//If it's in our bounds, move it to our cell storage for processing.
		if (( it->xPos > xPos) && ( it->xPos < xPos + 10) && (it->yPos > yPos) && (it->yPos < yPos + 10))
		{
			particles.push_back(*it);
		}
	}
}