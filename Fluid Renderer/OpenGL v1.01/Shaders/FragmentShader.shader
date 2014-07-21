#version 150 

in vec3 Color; 
in vec2 Texcoord; 
out vec4 outColor; 
uniform sampler2D texKitten; 
uniform sampler2D texPuppy;
	 
void main() 
{ 
	outColor = vec4(Color, 0.8) * mix(texture(texKitten, Texcoord), texture(texPuppy, Texcoord), 0.5); 
};