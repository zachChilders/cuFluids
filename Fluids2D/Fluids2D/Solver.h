#include "Cell.h"

class Solver
{
	public:
		Solver();
		Solver(int xLength, int yLength);
		~Solver();
		void solve();
		Cell operator[](int index);

	private:
		int width, height;
		Cell *cells;
};
