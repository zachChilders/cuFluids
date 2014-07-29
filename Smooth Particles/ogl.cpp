#include <stdio.h>
#include <stdlib.h>

#include <GL/glew.h>
#include <GL/glut.h>
#include <GLFW/glfw3.h>

#include "System.h"

int windowInit();

GLFWwindow* window;

float zoom;
System particleSystem;

GLfloat texture[10];
GLuint LoadTextureRAW(const char * filename, int width, int height);
void FreeTexture( GLuint textures );

void DrawParticles (void)
{
	//This should be a vertex shader (geometry??)
	for (int i = 1; i < particleSystem.getNumOfParticles(); i++)
	{
		glPushMatrix();
		// set color and alpha of current particle
		glColor4f(particleSystem.getR(i), particleSystem.getG(i), 
			particleSystem.getB(i), particleSystem.getAlpha(i));
		//move the current particle to its new position
		glTranslatef(particleSystem.getXPos(i), particleSystem.getYPos(i),
			particleSystem.getZPos(i) + zoom);

		glScalef(particleSystem.getScale(i), particleSystem.getScale(i),
			particleSystem.getScale(i));

		glDisable(GL_DEPTH_TEST);
		glEnable (GL_BLEND);

		glBlendFunc ( GL_DST_COLOR, GL_ZERO);
		glBindTexture(GL_TEXTURE_2D, texture[0]);

		glBegin (GL_QUADS);
		glTexCoord2d (0, 0);
		glVertex3f (-1, -1, 0);
		glTexCoord2d (1, 0);
		glVertex3f (1, -1, 0);
		glTexCoord2d (1, 1);
		glVertex3f (1, 1, 0);
		glTexCoord2d (0, 1);
		glVertex3f (-1, 1, 0);
		glEnd();

		glBlendFunc( GL_ONE, GL_ONE);
		glBindTexture( GL_TEXTURE_2D, texture[1]);

		glBegin (GL_QUADS);
		glTexCoord2d (0, 0);
		glVertex3f (-1, -1, 0);
		glTexCoord2d (1, 0);
		glVertex3f (1, -1, 0);
		glTexCoord2d (1, 1);
		glVertex3f (1, 1, 0);
		glTexCoord2d (0, 1);
		glVertex3f (-1, 1, 0);
		glEnd();

		glEnable(GL_DEPTH_TEST);

		glPopMatrix();
	}
}

void display (void)
{
	glClearDepth(1);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();
	glTranslatef( 0, 0, -10);

	particleSystem.updateParticles();
	DrawParticles();

	glutSwapBuffers();
}

void init (void)
{
	glEnable (GL_TEXTURE_2D);
	glEnable (GL_DEPTH_TEST);

	zoom = -80.0f;
	particleSystem.createParticles();

	//Soil would handle this much better
	texture [0] = LoadTextureRAW( "particle_mask.raw", 256, 256);
	texture [1] = LoadTextureRAW( "particle.raw", 256, 256);

}

//Called when a key is pressed
void handleKeypress(unsigned char key, int x, int y)
{
	switch (key)
   {
      case 49: //1 key: smoke
         zoom = -80.0f;
         particleSystem.createParticles();
         break;
      case 50: //2 key: fountain high
         zoom = -40.0f;
         particleSystem.createParticles();
         break;
      case 51: //3 key: fire
         zoom = -40.0f;
         particleSystem.createParticles();
         break;
      case 52: //4 key: fire with smoke
         zoom = -60.0f;
         particleSystem.createParticles();
         break;
      case 61: //+ key: change x pull for more wind to right
         particleSystem.modifySystemPull(0.0005f, 0.0f, 0.0f);
         break;
      case 45: //- key: change x pull for wind wind to left
         particleSystem.modifySystemPull(-0.0005f, 0.0f, 0.0f);
         break;
      case 91: //[ key: change y pull for more gravity
         particleSystem.modifySystemPull(0.0f, 0.0005f, 0.0f);
         break;
      case 93: //] key; change y pull for less gravity
         particleSystem.modifySystemPull(0.0f, -0.0005f, 0.0f);
         break;
		case 27: //Escape key
			exit(0);
	}
}

void reshape(int w, int h)
{
	glViewport(0, 0, (GLsizei)w, (GLsizei)h);
	glMatrixMode (GL_PROJECTION);
	glLoadIdentity();
	gluPerspective ( 60, (GLfloat) w / (GLfloat)h, 1.0, 100.0);
	glMatrixMode(GL_MODELVIEW);
}

int main(int argc, char **argv)
{
	srand((unsigned int) time(0));
	glutInit ( &argc, argv);
	glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
	glutInitWindowSize(500, 500);
	glutInitWindowPosition( 100, 100);
	glutCreateWindow( "Particle System");
	init();
	glutDisplayFunc(display);
	glutIdleFunc(display);
	glutKeyboardFunc(handleKeypress);
	glutReshapeFunc(reshape);
	glutMainLoop();
	return 0;


	/*
	windowInit();

	do{

		glClearColor(0.0f, 0.0f, 1.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glfwSwapBuffers(window);
		glfwPollEvents();

	}while(glfwGetKey(window, GLFW_KEY_ENTER ) != GLFW_PRESS &&
		   glfwWindowShouldClose(window) == 0 );*/
}

// Functions to load RAW files
// I did not write the following functions.
// They are form the OpenGL tutorials at http://www.swiftless.com
GLuint LoadTextureRAW( const char * filename, int width, int height )
{
  GLuint texture;
  unsigned char * data;
  FILE * file;
  file = fopen( filename, "rb" );
  if ( file == NULL ) return 0;
  data = (unsigned char *)malloc( width * height * 3 );
  fread( data, width * height * 3, 1, file );
  fclose( file );
  glGenTextures(1, &texture );
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
  gluBuild2DMipmaps(GL_TEXTURE_2D, 3, width, height, GL_RGB, GL_UNSIGNED_BYTE, data);
  free( data );
  return texture;
}

void FreeTexture( GLuint texture )
{
  glDeleteTextures( 1, &texture );
}

int windowInit()
{
	
	if (!glfwInit())
	{
		printf("Failed to init GLFW.\n");
		fprintf(stderr, "Failed to initialize GLFW\n");
		return -1;
	}

	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	window = glfwCreateWindow(1024, 768, "PARTICLES!!", NULL, NULL);

	if (window == NULL)
	{
		printf("Failed to open GLFW window.\n");
		fprintf(stderr, "Failed to open GLFW window.");
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(window);
	return 0;
}