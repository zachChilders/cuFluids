#include "Solver.h"

Solver::Solver()
{
	width = 0;
	height = 0;

	cells = new Cell[width * height];
}

Solver::Solver(int xLength, int yLength)
{
	width = xLength;
	height = yLength;

	cells = new Cell[width * height];
}

Solver::~Solver()
{
	delete [] cells;
}

void Solver::solve()
{
	for (int i = 0; i < width * height; i++)
	{
		cells[i].solve();
	}
}

Cell Solver::operator[](int index)
{
	return cells[index];
}
