// Fluids2D.cpp : Defines the entry point for the console application.
//

#include "particle.h"

#include <iostream>
#include <Windows.h>

using namespace std;

#define SIDE_LENGTH 16

//Creates a 2D particle box
fluidParticle* init2D()
{
	fluidParticle *fluidBox = new fluidParticle[SIDE_LENGTH * SIDE_LENGTH];
	for(int u = 0; u < SIDE_LENGTH; u++)
	{
		for (int v = 0; v < SIDE_LENGTH; v++)
		{
			fluidBox[u + v * SIDE_LENGTH] = fluidParticle(u, v, 1);
		}
	}
	return fluidBox;

};

//Outputs box to CLI
void print2D(fluidParticle *fluidBox)
{
	//Draw some particulates
	for(int u = 0; u < SIDE_LENGTH; u++)
	{
		for (int v = 0; v < SIDE_LENGTH; v++)
		{
			float tmp = fluidBox[u + v * SIDE_LENGTH].getMagnitude();

			//Beautiful Formatting.
			if (tmp < 10){
				cout << " " << tmp;
			}
			else if (tmp < 100)
			{
				cout << " " << tmp;
			}
			else
			{
				cout <<" " << tmp;
			}
		}
		cout << endl;
	}
}

int main(void)
{
	//Our Particle Box
	fluidParticle *fluidBox = init2D();

	//Event loops.  
	while(true)
	{
		//Apply Navier-Stokes
		for (int i = 0; i < SIDE_LENGTH; i++)
		{
			for (int j = 0; j < SIDE_LENGTH; j++)
			{
				fluidBox[i + j * SIDE_LENGTH].update(
								fluidBox[i + (j - 1) * SIDE_LENGTH],  //up
								fluidBox[i + (j + 1) * SIDE_LENGTH],  //down
								fluidBox[(i - 1) + j * SIDE_LENGTH],  //left
								fluidBox[(i + 1) + j * SIDE_LENGTH]); //right
													
			}
		}
		print2D(fluidBox);
		Sleep(200);
		system( "CLS" );
	}

	return 0;
}

