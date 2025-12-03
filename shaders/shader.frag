#version 450

layout (location = 0) in vec3 f_position;
layout (location = 1) in vec3 f_normal;
layout (location = 2) in vec2 f_uv;

layout (location = 0) out vec4 final_color;

layout (binding = 1, std140) uniform ModelUniforms {
	mat4 model;
	vec3 albedo_color;
};

layout (binding = 2) uniform sampler2D texture_sampler;

void main() {
	vec4 tex_color = texture(texture_sampler, f_uv);
	final_color = tex_color;
}
