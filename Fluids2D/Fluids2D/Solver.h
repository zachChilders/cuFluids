#include "Cell.h"

class Solver
{
	public:
		Solver();
		Solver(int xLength, int yLength);
		~Solver();
		void solve();

	private:
		int width, height;
		Cell *cells;
};
