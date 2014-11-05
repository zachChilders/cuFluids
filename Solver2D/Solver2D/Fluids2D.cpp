// Fluids2D.cpp : Defines the entry point for the console application.
//

#include "Solver.h"

#include <iostream>
#include <Windows.h>

using namespace std;

#define SIDE_LENGTH 5


int main(void)
{
	//Our Particle Box
	Solver fluidBox;

	float x = 0.0;
	float y = 0.0;

	//Event loops.  
	while(x < 20)
	{
		x += 0.1;
		fluidBox.solve(x, y);
		fluidBox.print();
		fluidBox.log();
		Sleep(200);
		system( "CLS" );
	}

	return 0;
}

