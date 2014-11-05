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

	//Event loops.  
	while(true)
	{
		float x = 0;
		float y = 0;

		if (GetAsyncKeyState(VK_UP))
		{
			x = 0;
			y = 5;
		}
		else if (GetAsyncKeyState(VK_DOWN))
		{
			x = 0;
			y = -5;
		}
		else if (GetAsyncKeyState(VK_LEFT))
		{
			x = -5;
			y = 0;
		}
		else if (GetAsyncKeyState(VK_RIGHT))
		{
			x = 5;
			y = 0;
		}

		fluidBox.solve(x, y);
		fluidBox.print();
		Sleep(200);
		system( "CLS" );
	}

	return 0;
}

