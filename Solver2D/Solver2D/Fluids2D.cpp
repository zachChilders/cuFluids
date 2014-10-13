// Fluids2D.cpp : Defines the entry point for the console application.
//

#include "Solver.h"

#include <iostream>
#include <Windows.h>

using namespace std;

#define SIDE_LENGTH 16

//Outputs box to CLI
void print2D(Solver s)
{
	//Draw some particulates
	for(int u = 0; u < SIDE_LENGTH; u++)
	{
		for (int v = 0; v < SIDE_LENGTH; v++)
		{
			float tmp = s[u + v * SIDE_LENGTH].getMagnitude();

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
	static Solver fluidBox;

	//Event loops.  
	while(true)
	{
		fluidBox.solve();
		print2D(fluidBox);
		Sleep(200);
		system( "CLS" );
	}

	return 0;
}

