///*****************************************************
//Smooth Particle System
//
//Zach Childers
//
//Summer 2014
//
///*****************************************************/
//
//#include <ctime>
//#include <iostream>
//#include <fstream>
//#include <stdio.h>
//#include <stdlib.h>
//#include <string>
//
//#include <GL/glew.h>
//#include <GLFW/glfw3.h>
//#include <glm/gtc/matrix_transform.hpp>
//#include <glm/gtc/type_ptr.hpp>
//
//// Shader macro
//#define GLSL(src) "#version 330 core\n" #src
//
//// Vertex shader
//const GLchar* vertexShaderSrc = GLSL(
//    in vec2 pos;
//	in vec3 color;
//	in float sides;
//
//	out vec3 vColor; //output to geometry shader
//	out float vSides;
//
//	uniform mat4 trans;
//
//    void main() {
//        gl_Position = trans * vec4(pos + pos, 0.0, 1.0);
//		vColor = color;
//		vSides = sides;
//	
//    }
//);
//
//// Fragment shader
//const GLchar* fragmentShaderSrc = GLSL(
//	in vec3 fColor;
//
//    out vec4 outColor;
//    void main() {
//        outColor = vec4(fColor, 1.0);
//    }
//);
//
//const GLchar* geomShaderSrc = GLSL(
//	layout(points) in;
//	layout(triangle_strip, max_vertices = 128) out;
//
//	in vec3 vColor[]; //Output from vertex shader
//	in float vSides[];
//
//	out vec3 fColor;
//
//	const float PI = 3.1415926;
//
//	void main(){
//		fColor = vColor[0];
//		//gl_position = gl_in[0].gl_Position + 5.0;
//
//		int numShapes = 32; 
//		for (int x = 1; x <= numShapes; x++)
//		{
//
//			for (int i = 0; i <= vSides[0]; i++)
//			{
//				//Angle between each side in radians
//				float ang = PI * 2.0 / vSides[0] * i;
//
//				//Offset from center of point (0.3 to accomodate for aspect ratio)
//				vec4 offset = vec4(cos(ang) * 0.3, sin(ang) * 0.4, 0.0, 0.0);
//				gl_Position = gl_in[0].gl_Position + offset / 2.0 + x / 0.75;
//
//				EmitVertex();
//
//			}
//
//			EndPrimitive();
//		}
//	}
//);
//
//
//// Shader creation helper
//GLuint createShader(GLenum type, const GLchar* src) {
//    GLuint shader = glCreateShader(type);
//    glShaderSource(shader, 1, &src, nullptr);
//    glCompileShader(shader);
//    return shader;
//}
//
//
////int windowInit();
////GLFWwindow* window;
//
//float zoom;
//
//GLuint shaderProgram;
////
////int main(int argc, char **argv)
////{
//////
//	windowInit();
//
//	glewExperimental = GL_TRUE;
//	GLenum err = glewInit();
//	if( err != GLEW_OK )
//	{
//		printf("GlewInit error");
//		exit(1);
//	}
////
////    // Compile and activate shaders
////    GLuint vertShader = createShader(GL_VERTEX_SHADER, vertexShaderSrc);
////	GLuint geomShader = createShader(GL_GEOMETRY_SHADER, geomShaderSrc);
////    GLuint fragShader = createShader(GL_FRAGMENT_SHADER, fragmentShaderSrc);
////
////	GLuint shaderProgram = glCreateProgram();
////	glAttachShader(shaderProgram, vertShader);
////	glAttachShader(shaderProgram, geomShader);
////	glAttachShader(shaderProgram, fragShader);
////	glLinkProgram(shaderProgram);
////	glUseProgram(shaderProgram);
////	
////	
////	//Vertex buffers are neat
////	GLuint vbo;
////	glGenBuffers(1, &vbo);
////	glBindBuffer(GL_ARRAY_BUFFER, vbo);
////	
////	GLfloat points[] = {
////		//Coordinates    Color         Sides
////		-1.5f, -0.5f, 1.0f, 0.0f, 0.0f, 4.0f,
////	};
////
////	glBufferData(GL_ARRAY_BUFFER, sizeof(points), points, GL_STREAM_DRAW);
////
////	//Bind Vertex Attributes to the shaders
////	GLuint vao;
////	glGenVertexArrays(1, &vao);
////	glBindVertexArray(vao);
////
////	GLint uniTrans = glGetUniformLocation(shaderProgram, "trans");
////
////	GLint posAttrib = glGetAttribLocation(shaderProgram, "pos");
////	glEnableVertexAttribArray(posAttrib);
////	glVertexAttribPointer(posAttrib, 2, GL_FLOAT, GL_FALSE, 6 * sizeof(float), 0);
////
////	GLint colAttrib = glGetAttribLocation(shaderProgram, "color");
////	glEnableVertexAttribArray(colAttrib);
////	glVertexAttribPointer(colAttrib, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*) (2 * sizeof(float)));
////
////	GLint sidesAttrib = glGetAttribLocation(shaderProgram, "sides");
////	glEnableVertexAttribArray(sidesAttrib);
////	glVertexAttribPointer(sidesAttrib, 1, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*) (5 * sizeof(float)));
////
////
	//do{
	//	
	//	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	//	glClear(GL_COLOR_BUFFER_BIT);

	//	GLfloat s = sin((GLfloat)clock() / (GLfloat)CLOCKS_PER_SEC * 5.0f) * 0.50f + 0.75f;
	//	glm::mat4 trans;
	//	trans = glm::translate(trans, glm::vec3(1.0f, 1.0f, 0.0f) * s);
	//	glUniformMatrix4fv(uniTrans, 1, GL_FALSE, glm::value_ptr(trans));

	//	glDrawArrays(GL_POINTS, 0, 1);


	//	glfwSwapBuffers(window);
	//	glfwPollEvents();

	//}while(glfwGetKey(window, GLFW_KEY_ENTER ) != GLFW_PRESS &&
	//	   glfwWindowShouldClose(window) == 0 );
////
////	
////	return 0;
////}
//
////int windowInit()
////{
////	
////	if (!glfwInit())
////	{
////		printf("Failed to init GLFW.\n");
////		fprintf(stderr, "Failed to initialize GLFW\n");
////		return -1;
////	}
////
////	glfwWindowHint(GLFW_SAMPLES, 4);
////	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
////	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
////	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
////
////	window = glfwCreateWindow(1440, 1280, "Particle System", NULL, NULL);
////
////	if (window == NULL)
////	{
////		printf("Failed to open GLFW window.\n");
////		fprintf(stderr, "Failed to open GLFW window.");
////		glfwTerminate();
////		return -1;
////	}
////
////	glfwMakeContextCurrent(window);
////	return 0;
////}