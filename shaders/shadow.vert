#version 450

layout (location = 0) in vec3 v_position;

layout (binding = 0, std140) uniform SceneUniforms {
	mat4 view_projection;
	mat4 light_view_projection;
	vec3 light_direction;
	float _pad0;
	vec3 camera_position;
	float _pad1;
};

layout (binding = 1, std140) uniform ModelUniforms {
	mat4 model;
	vec3 albedo_color;
};

void main() {
	vec4 position = model * vec4(v_position, 1.0f);
	gl_Position = view_projection * position;
}

