class fluidParticle
{
	private:
		float xPos, yPos, zPos;
		int xVelocity, yVelocity, zVelocity;

	public:
		fluidParticle();
		fluidParticle(float x, float y);
		fluidParticle(float x, float y, float z);
		~fluidParticle();
		void reset();

		//These should be applied in order functionally
		void update(float xVelocity, float yVelocity, float zVelocity);
		void pressure();
		void external();
		void viscosity();
		void gravity();
		void weight();
		float* distances;

		void getPositions(float *x, float *y, float *z);
		float getMagnitude();

		//Overload * for matrix multiplication.
};