/*****************************************************
Smooth Particle System

Zach Childers

Summer 2014

/*****************************************************/


#include <stdio.h>
#include <stdlib.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <GL/glut.h>
#include <SOIL\SOIL.h>
#include "GLutils.h"

#include "System.h"

int windowInit();

GLFWwindow* window;

float zoom;
System particleSystem;

GLuint shaderProgram;

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
		glColor4f(particleSystem.getRGBA(i).r, particleSystem.getRGBA(i).g, particleSystem.getRGBA(i).b, particleSystem.getRGBA(i).a);
		//move the current particle to its new position
		glTranslatef(particleSystem.getPosition(i).x, particleSystem.getPosition(i).y, particleSystem.getPosition(i).z + zoom);

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
	glutInitWindowSize(800, 600);
	glutInitWindowPosition( 100, 100);
	glutCreateWindow( "Particle System");
	init();
	glutDisplayFunc(display);
	glutIdleFunc(display);
	glutKeyboardFunc(handleKeypress);
	glutReshapeFunc(reshape);
	glutMainLoop();

	//windowInit();

	////GL config
	//glEnable(GL_DEPTH_TEST);
	//glEnable(GL_BLEND);
	//glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	//glDepthFunc(GL_LESS);

	////ShaderFile and lineBuffer used by every shader.
	//std::ifstream shaderFile;
	//std::string lineBuffer;

	////Load vertex shader
	//std::string vertexSource;
	//shaderFile.open("Shaders/VertexShader.shader");
	//while(getline(shaderFile, lineBuffer))
	//{
	//	vertexSource += "\n" + lineBuffer;
	//}
	//shaderFile.close();

	////Load fragment shader
	//std::string fragmentSource;
	//shaderFile.open("Shaders/FragmentShader.shader");
	//while(getline(shaderFile, lineBuffer))
	//{
	//	fragmentSource += "\n" + lineBuffer;
	//}
	//shaderFile.close();
	//
	////Bind shaders into GLchars
	//const GLchar* vertexShaderSource = vertexSource.c_str();
	//const GLchar* fragmentShaderSource = fragmentSource.c_str();

	//GLuint vbo; //Create a handle for our vertex buffer object
	//glGenBuffers(1, &vbo); //Generate 1 buffer

	//GLuint vertShader = glCreateShader(GL_VERTEX_SHADER);
	//glShaderSource(vertShader, 1, &vertexShaderSource, NULL);
	//glCompileShader(vertShader);

	//GLuint fragShader = glCreateShader(GL_FRAGMENT_SHADER);
	//glShaderSource(fragShader, 1, &fragmentShaderSource, NULL);
	//glCompileShader(fragShader);

	//GLuint shaderProgram = glCreateProgram();
	//glAttachShader(shaderProgram, vertShader);
	//glAttachShader(shaderProgram, fragShader);
	//glLinkProgram(shaderProgram);
	//glUseProgram(shaderProgram);
	//
	//do{

	//	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	//	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//	glm::mat4 model;

	//	glfwSwapBuffers(window);
	//	glfwPollEvents();

	//}while(glfwGetKey(window, GLFW_KEY_ENTER ) != GLFW_PRESS &&
	//	   glfwWindowShouldClose(window) == 0 );

	//
	return 0;
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

	window = glfwCreateWindow(1024, 768, "Particle System", NULL, NULL);

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