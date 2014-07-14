class fluidParticle
{
	private:
		float xPos, yPos, zPos;
		int xVelocity, yVelocity, zVelocity;
		float viscosity;

	public:
		fluidParticle();
		fluidParticle(float x, float y, float viscosity);
		fluidParticle(float x, float y, float z, float viscosity);
		~fluidParticle();
		void reset();

		//These should be applied in order funcctionally
		void update(fluidParticle up, fluidParticle down, fluidParticle left, fluidParticle right);
		void advection(fluidParticle up, fluidParticle down, fluidParticle left, fluidParticle right);
		void pressure();
		void external();

		void getPositions(float *x, float *y, float *z);
		float getMagnitude();

};