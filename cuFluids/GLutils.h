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

#define GLSL(src) "#version 330 core\n" #src


GLuint createShader(GLenum type, const GLchar* src)
{
	GLuint shader = glCreateShader(type);
	glShaderSource(shader, 1, &src, nullptr);
	glCompileShader(shader);
	return shader;
}
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

GLuint compileVertShader(const char* vertCode)
{
	GLuint vertShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertShader, 1, &vertCode, NULL);
	glCompileShader(vertShader);
	return vertShader;
};

GLuint compileFragShader(const char* fragCode)
{
	GLuint fragShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragShader, 1, &fragCode, NULL);
	glCompileShader(fragShader);
	return fragShader;
};

GLuint compileGeoShader(const char* geoCode)
{
	GLuint geoShader = glCreateShader(GL_GEOMETRY_SHADER);
	glShaderSource(geoShader, 1, &geoCode, NULL);
	glCompileShader(geoShader);
	return geoShader;
};

GLuint createShader(string vertShader, string fragShader)
{
	const GLchar* vertSource = loadShader(vertShader);
	const GLchar* fragSource = loadShader(fragShader);

	GLuint vertexShader = compileVertShader(vertSource);
	GLuint fragmentShader = compileFragShader(fragSource);

	GLuint shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);

	glLinkProgram(shaderProgram);
	return shaderProgram;
};

GLuint createShader(string vertShader, string geoShader, string fragShader)
{
	const GLchar* vertSource = loadShader(vertShader);
	const GLchar* fragSource = loadShader(fragShader);
	const GLchar* geoSource = loadShader(geoShader);

	GLuint vertexShader = compileVertShader(vertSource);
	GLuint fragmentShader = compileFragShader(fragSource);
	GLuint geometryShader = compileGeoShader(geoSource);

	if (vertexShader == GL_FALSE)
	{
		std::cout << "Vertex shader failed" << std::endl;
	}

	if (fragmentShader == GL_FALSE)
	{
		std::cout << "Fragment shader failed" << std::endl;
	}

	if (geometryShader == GL_FALSE)
	{
		std::cout << "Geometry shader failed" << std::endl;
	}


	GLuint shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	glAttachShader(shaderProgram, geometryShader);

	glLinkProgram(shaderProgram);
	if (shaderProgram == GL_FALSE)
	{
		std::cout << "Link failed" << std::endl;
	}
	return shaderProgram;
};