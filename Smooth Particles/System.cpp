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
 * initalizes a particle system according to its type
 * calls create particle() to initalize individual particles
 */
void System::createParticles(void)
{

	systemPull.y = 0.005;
	systemPull.x = systemPull.z = 0.0f; 

	
	for(int i = 0; i < MAX_PARTICLES; i++)
	{
		//createParticle(&particles[i]);
		particles[i] = Particle();
	}
}
 
//We're going to replace this with a shader.
void System::updateParticles(void)
{
   for(int i = 0; i < MAX_PARTICLES; i++)
   {
      particles[i].age = particles[i].age + 0.02;
      
      
      particles[i].direction = particles[i].direction + ((((((int)(0.5) * rand()%11) + 1)) * rand()%11) + 1);

      particles[i].position.x = particles[i].position.x + particles[i].movement.x + particles[i].pull.x;
      particles[i].position.y = particles[i].position.y + particles[i].movement.y + particles[i].pull.y;
      particles[i].position.z = particles[i].position.z + particles[i].movement.z + particles[i].pull.z;
      
      particles[i].pull.x = particles[i].pull.x + systemPull.x;
      particles[i].pull.y = particles[i].pull.y + systemPull.y; // acleration due to gravity
      particles[i].pull.z = particles[i].pull.z + systemPull.z;

         float temp = particles[i].lifespan/particles[i].age;
         if((temp) < 1.75)
         {//red
            particles[i].color.x = 1.0f;
            particles[i].color.y = 0.25f;
            particles[i].color.z = 0.0f;
         }
         else if((temp) < 3.0)
         {//gold
            particles[i].color.x = 1.0f;
            particles[i].color.y = 0.9f;
            particles[i].color.z = 0.0f;
         }
         else if((temp) < 10.0)
         {//yellow
            particles[i].color.x = 1.0f;
            particles[i].color.y = 1.0f;
            particles[i].color.z = 0.0f;
         }
         else
         {// initial light yellow
            particles[i].color.x = 1.0f;
            particles[i].color.y = 0.95f;
            particles[i].color.z = 0.8f;
         }
      

      
        if (particles[i].age > particles[i].lifespan || particles[i].position.y > 45 || particles[i].position.y < -35 || particles[i].position.x > 80 || particles[i].position.x < -80)
			particles[i] = Particle();
     
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
   return particles[i].position.x;
}

float System::getYPos(int i)
{
   return particles[i].position.y;
}
float System::getZPos(int i)
{
   return particles[i].position.z;
}

float System::getR(int i)
{
   return particles[i].color.x;
}

float System::getG(int i)
{
   return particles[i].color.y;
}
float System::getB(int i)
{
   return particles[i].color.z;
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
   systemPull.x += x;
   systemPull.y += y;
   systemPull.z += z;
}