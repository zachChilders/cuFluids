/*****************************************************
Smooth Particle System

Zach Childers

Summer 2014

/*****************************************************/

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <SOIL\SOIL.h>

#include "System.h"

// Shader macro
#define GLSL(src) "#version 330 core\n" #src

// Vertex shader
const GLchar* vertexShaderSrc = GLSL(
    in vec2 pos;
	in vec3 color;
	in float sides;

	out vec3 vColor; //output to geometry shader
	out float vSides;

    void main() {
        gl_Position = vec4(pos, 0.0, 1.0);
		vColor = color;
		vSides = sides;
    }
);

// Fragment shader
const GLchar* fragmentShaderSrc = GLSL(
	in vec3 fColor;

    out vec4 outColor;
    void main() {
        outColor = vec4(fColor, 1.0);
    }
);

const GLchar* geomShaderSrc = GLSL(
	layout(points) in;
	layout(triangle_strip, max_vertices = 64) out;

	in vec3 vColor[]; //Output from vertex shader
	in float vSides[];
	out vec3 fColor;

	const float PI = 3.1415926;

	void main(){
		fColor = vColor[0];

		for (int i = 0; i <= vSides[0]; i++)
		{
			//Angle between each side in radians
			float ang = PI * 2.0 / vSides[0] * i;

			//Offset from center of point (0.3 to accomodate for aspect ratio)
			vec4 offset = vec4(cos(ang) * 0.3, -sin(ang) * 0.4, 0.0, 0.0);
			gl_Position = gl_in[0].gl_Position + offset;

			EmitVertex();

		}

		EndPrimitive();

		for (int i = 0; i <= vSides[0]; i++)
		{
			//Angle between each side in radians
			float ang = PI * 2.0 / vSides[0] * i;

			//Offset from center of point (0.3 to accomodate for aspect ratio)
			vec4 offset = vec4(cos(ang) * 0.3, -sin(ang) * 0.4, 0.0, 0.0);
			gl_Position = gl_in[0].gl_Position + 3 * offset;

			EmitVertex();

		}

		EndPrimitive();

	}
);


// Shader creation helper
GLuint createShader(GLenum type, const GLchar* src) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);
    return shader;
}


int windowInit();

GLFWwindow* window;

float zoom;
System particleSystem;

GLuint shaderProgram;

void DrawParticles (void)
{
	////This should be a vertex shader (geometry??)
	//for (int i = 1; i < particleSystem.getNumOfParticles(); i++)
	//{
	//	glPushMatrix();
	//	// set color and alpha of current particle
	//	glColor4f(particleSystem.getRGBA(i).r, particleSystem.getRGBA(i).g, particleSystem.getRGBA(i).b, particleSystem.getRGBA(i).a);
	//	//move the current particle to its new position
	//	glTranslatef(particleSystem.getPosition(i).x, particleSystem.getPosition(i).y, particleSystem.getPosition(i).z + zoom);

	//	glScalef(particleSystem.getScale(i), particleSystem.getScale(i),
	//		particleSystem.getScale(i));

	//	glDisable(GL_DEPTH_TEST);
	//	glEnable (GL_BLEND);

	//	glBlendFunc ( GL_DST_COLOR, GL_ZERO);
	//	glBindTexture(GL_TEXTURE_2D, texture[0]);

	//	glBegin (GL_QUADS);
	//	glTexCoord2d (0, 0);
	//	glVertex3f (-1, -1, 0);
	//	glTexCoord2d (1, 0);
	//	glVertex3f (1, -1, 0);
	//	glTexCoord2d (1, 1);
	//	glVertex3f (1, 1, 0);
	//	glTexCoord2d (0, 1);
	//	glVertex3f (-1, 1, 0);
	//	glEnd();

	//	glBlendFunc( GL_ONE, GL_ONE);
	//	glBindTexture( GL_TEXTURE_2D, texture[1]);

	//	glBegin (GL_QUADS);
	//	glTexCoord2d (0, 0);
	//	glVertex3f (-1, -1, 0);
	//	glTexCoord2d (1, 0);
	//	glVertex3f (1, -1, 0);
	//	glTexCoord2d (1, 1);
	//	glVertex3f (1, 1, 0);
	//	glTexCoord2d (0, 1);
	//	glVertex3f (-1, 1, 0);
	//	glEnd();

	//	glEnable(GL_DEPTH_TEST);

	//	glPopMatrix();
	//}
}

void display (void)
{
	/*glClearDepth(1);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();
	glTranslatef( 0, 0, -10);
	
	particleSystem.updateParticles();
	DrawParticles();*/

	//glutSwapBuffers();
}

void init (void)
{
	glEnable (GL_TEXTURE_2D);
	glEnable (GL_DEPTH_TEST);

	zoom = -80.0f;
	particleSystem.createParticles();

	//Soil would handle this much better
//	texture [0] = LoadTextureRAW( "particle_mask.raw", 256, 256);
//	texture [1] = LoadTextureRAW( "particle.raw", 256, 256);

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
{/*
	glViewport(0, 0, (GLsizei)w, (GLsizei)h);
	glMatrixMode (GL_PROJECTION);
	glLoadIdentity();
	gluPerspective ( 60, (GLfloat) w / (GLfloat)h, 1.0, 100.0);
	glMatrixMode(GL_MODELVIEW);*/
}

int main(int argc, char **argv)
{
	srand((unsigned int) time(0));
	/*glutInit ( &argc, argv);
	glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
	glutInitWindowSize(800, 600);
	glutInitWindowPosition( 100, 100);
	glutCreateWindow( "Particle System");
	init();
	glutDisplayFunc(display);
	glutIdleFunc(display);
	glutKeyboardFunc(handleKeypress);
	glutReshapeFunc(reshape);*/

	windowInit();

	glewExperimental = GL_TRUE;
	GLenum err = glewInit();
	if( err != GLEW_OK )
	{
		printf("GlewInit error");
		exit(1);
	}

    // Compile and activate shaders
    GLuint vertShader = createShader(GL_VERTEX_SHADER, vertexShaderSrc);
	GLuint geomShader = createShader(GL_GEOMETRY_SHADER, geomShaderSrc);
    GLuint fragShader = createShader(GL_FRAGMENT_SHADER, fragmentShaderSrc);

	GLuint shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, vertShader);
	glAttachShader(shaderProgram, geomShader);
	glAttachShader(shaderProgram, fragShader);
	glLinkProgram(shaderProgram);
	glUseProgram(shaderProgram);
	
	//Vertex buffers are neat
	GLuint vbo;
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	
	GLfloat points[] = {
	   //Coordinates    Color             Sides
		-0.45f,  0.45f, 1.0f, 0.0f, 0.0f, 4.0f,
		 0.45f,  0.45f, 0.0f, 1.0f, 0.0f, 8.0f,
		 0.45f, -0.45f, 0.0f, 0.0f, 1.0f, 16.0f,
		-0.45f, -0.45f, 1.0f, 1.0f, 0.0f, 32.0f,
	};

	glBufferData(GL_ARRAY_BUFFER, sizeof(points), points, GL_STREAM_DRAW);

	//Bind Vertex Attributes to the shaders
	GLuint vao;
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	GLint posAttrib = glGetAttribLocation(shaderProgram, "pos");
	glEnableVertexAttribArray(posAttrib);
	glVertexAttribPointer(posAttrib, 2, GL_FLOAT, GL_FALSE, 6 * sizeof(float), 0);

	GLint colAttrib = glGetAttribLocation(shaderProgram, "color");
	glEnableVertexAttribArray(colAttrib);
	glVertexAttribPointer(colAttrib, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*) (2 * sizeof(float)));

	GLint sidesAttrib = glGetAttribLocation(shaderProgram, "sides");
	glEnableVertexAttribArray(sidesAttrib);
	glVertexAttribPointer(sidesAttrib, 1, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*) (5 * sizeof(float)));

	GLuint tfBuffer; 
	glGenTransformFeedbacks(1, &tfBuffer);
	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, tfBuffer);

	do{
		
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		glDrawArrays(GL_POINTS, 0, 4);

		glfwSwapBuffers(window);
		glfwPollEvents();

	}while(glfwGetKey(window, GLFW_KEY_ENTER ) != GLFW_PRESS &&
		   glfwWindowShouldClose(window) == 0 );

	
	return 0;
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

	window = glfwCreateWindow(1440, 1280, "Particle System", NULL, NULL);

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