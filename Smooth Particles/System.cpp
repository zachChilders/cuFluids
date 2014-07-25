/******************************************************
 * Georg Albrecht                                     *
 * Final Project for CMPS 161, Winter 2009            *
 *                                                    *
 * System.cpp                                         *
 *    This is the source file for System.h. It        *
 *    contains the code necessary to initalize, update*
 *    and manage and array of particles               *
 ******************************************************/

#include "System.h"

System::System(void)
{
}

System::~System(void)
{
}

/*
 * initalizes a single particle according to its type
 */
void System::createParticle(Particle *p)
{

	p->lifespan = (((rand()%10+1)))/10.0f;
	p->type = 0;

	p->age = 0.0f;
	p->scale = 0.25f;
	p->direction = 0;   
	p->position[X] = ((rand()%2)-(rand()%2));

	p->position[Y] = -30;

	p->position[Z] = 0; 

	p->movement[X] = (((((((2) * rand()%11) + 1)) * rand()%11) + 1) * 0.0035) - (((((((2) * rand()%11) + 1)) * rand()%11) + 1) * 0.0035);
	p->movement[Y] = ((((((5) * rand()%11) + 3)) * rand()%11) + 7) * 0.015; 
	p->movement[Z] = (((((((2) * rand()%11) + 1)) * rand()%11) + 1) * 0.0015) - (((((((2) * rand()%11) + 1)) * rand()%11) + 1) * 0.0015);
   
	p->color[X] = 0.0f;
	p->color[Y] = 0.0f;
	p->color[Z] = 1.0f;

	p->pull[X] = 0.0f;
	p->pull[Y] = 0.0f;
	p->pull[Z] = 0.0f;
}

/*
 * initalizes a particle system according to its type
 * calls create particle() to initalize individual particles
 */
void System::createParticles(void)
{

	systemPull[Y] = 0.005;
	systemPull[X] = systemPull[Z] = 0.0f; 

	
	for(int i = 0; i < MAX_PARTICLES; i++)
	{
		createParticle(&particles[i]);
	}
}

/*
 * updates required particle attributes for all particles in a system
 * also responsible for killing and respawning (via createparticle()) individual particles
 */
void System::updateParticles(void)
{
   for(int i = 0; i < MAX_PARTICLES; i++)
   {
      particles[i].age = particles[i].age + 0.02;
      
      if(systemType == Smoke || particles[i].type == 1)
         particles[i].scale = particles[i].scale + 0.001; //increasing scale makes textures bigger over lifetime

      particles[i].direction = particles[i].direction + ((((((int)(0.5) * rand()%11) + 1)) * rand()%11) + 1);

      particles[i].position[X] = particles[i].position[X] + particles[i].movement[X] + particles[i].pull[X];
      particles[i].position[Y] = particles[i].position[Y] + particles[i].movement[Y] + particles[i].pull[Y];
      particles[i].position[Z] = particles[i].position[Z] + particles[i].movement[Z] + particles[i].pull[Z];
      
      particles[i].pull[X] = particles[i].pull[X] + systemPull[X];
      particles[i].pull[Y] = particles[i].pull[Y] + systemPull[Y]; // acleration due to gravity
      particles[i].pull[Z] = particles[i].pull[Z] + systemPull[Z];

      // color changing for fire particles light yellow -> red with age
      if(systemType == Fire || particles[i].type == 0)
      {
         float temp = particles[i].lifespan/particles[i].age;
         if((temp) < 1.75)
         {//red
            particles[i].color[X] = 1.0f;
            particles[i].color[Y] = 0.25f;
            particles[i].color[Z] = 0.0f;
         }
         else if((temp) < 3.0)
         {//gold
            particles[i].color[X] = 1.0f;
            particles[i].color[Y] = 0.9f;
            particles[i].color[Z] = 0.0f;
         }
         else if((temp) < 10.0)
         {//yellow
            particles[i].color[X] = 1.0f;
            particles[i].color[Y] = 1.0f;
            particles[i].color[Z] = 0.0f;
         }
         else
         {// initial light yellow
            particles[i].color[X] = 1.0f;
            particles[i].color[Y] = 0.95f;
            particles[i].color[Z] = 0.8f;
         }
      }

      
        if (particles[i].age > particles[i].lifespan || particles[i].position[Y] > 45 || particles[i].position[Y] < -35 || particles[i].position[X] > 80 || particles[i].position[X] < -80)
			createParticle(&particles[i]);
     
   }
}

void System::setSystemType(int type)
{
   systemType = type;
}

int System::getNumOfParticles(void)
{
   return MAX_PARTICLES;
}

float System::getXPos(int i)
{
   return particles[i].position[X];
}

float System::getYPos(int i)
{
   return particles[i].position[Y];
}
float System::getZPos(int i)
{
   return particles[i].position[Z];
}

float System::getR(int i)
{
   return particles[i].color[X];
}

float System::getG(int i)
{
   return particles[i].color[Y];
}
float System::getB(int i)
{
   return particles[i].color[Z];
}

float System::getScale(int i)
{
   return particles[i].scale;
}

float System::getDirection(int i)
{
   return particles[i].direction;
}

float System::getAlpha(int i)
{
   return (1 - particles[i].age/particles[i].lifespan);
}

void System::modifySystemPull(float x, float y, float z)
{
   systemPull[X] += x;
   systemPull[Y] += y;
   systemPull[Z] += z;
}