/*****************************************************
GL Utilities

Zach Childers

Summer 2014

/*****************************************************/

#include <ctime>
#include <iostream>
#include <fstream>
#include <string>
#include <glm/glm.hpp>
#include <GL/glew.h>
#include <GLFW\glfw3.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

using namespace std;
#define GLEW_STATIC

//Reads a shader, returns a char* for compilation
const char* loadShader(string fname)
{
	ifstream sourceFile; 
	string lineBuffer;

	string shaderSource;
	try{
		sourceFile.open(fname);
		while(getline(sourceFile, lineBuffer))
		{
			shaderSource += "\n" + lineBuffer;
		}
		sourceFile.close();
		return shaderSource.c_str();
	}
	catch(exception e)
	{
		cout << "Problem reading shader: ";
		cout << e.what() << endl;
		return nullptr;
	}
}

GLuint compileVertShader(const char* vertCode, GLFWwindow *context)
{
	GLuint vertShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertShader, 1, &vertCode, NULL);
	glCompileShader(vertShader);
	return vertShader;
};

GLuint compileFragShader(const char* fragCode, GLFWwindow *context)
{
	GLuint fragShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragShader, 1, &fragCode, NULL);
	glCompileShader(fragShader);
	return fragShader;
};

GLuint compileGeoShader(const char* geoCode, GLFWwindow *context)
{
	GLuint geoShader = glCreateShader(GL_GEOMETRY_SHADER);
	glShaderSource(geoShader, 1, &geoCode, NULL);
	glCompileShader(geoShader);
	return geoShader;
};

GLuint createShader(string vertShader, string fragShader, GLFWwindow *context)
{
	const GLchar* vertSource = loadShader(vertShader);
	const GLchar* fragSource = loadShader(fragShader);

	GLuint vertexShader = compileVertShader(vertSource, context);
	GLuint fragmentShader = compileFragShader(fragSource, context);

	GLuint shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);

	glLinkProgram(shaderProgram);
	return shaderProgram;
};

GLuint createShader(string vertShader, string geoShader, string fragShader, GLFWwindow *context)
{
	const GLchar* vertSource = loadShader(vertShader);
	const GLchar* fragSource = loadShader(fragShader);
	const GLchar* geoSource = loadShader(geoShader);

	GLuint vertexShader = compileVertShader(vertSource, context);
	GLuint fragmentShader = compileFragShader(fragSource, context);
	GLuint geometryShader = compileGeoShader(geoSource, context);

	GLuint shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	glAttachShader(shaderProgram, geometryShader);

	glLinkProgram(shaderProgram);
	return shaderProgram;
};