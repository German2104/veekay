#version 450

layout (location = 0) in vec3 f_position;
layout (location = 1) in vec3 f_normal;
layout (location = 2) in vec2 f_uv;
layout (location = 3) in vec4 f_light_space_pos;

layout (location = 0) out vec4 final_color;

struct PointLight {
	vec4 position;
	vec4 color;
};

layout (binding = 0, std140) uniform SceneUniforms {
	mat4 view_projection;
	mat4 light_view_projection;
	vec3 light_direction;
	float _pad0;
	vec3 camera_position;
	float _pad1;
	PointLight point_lights[4];
	int point_light_count;
	vec3 _pad2;
};

layout (binding = 1, std140) uniform ModelUniforms {
	mat4 model;
	vec3 albedo_color;
	int texture_index;
};

layout (binding = 2) uniform sampler2D textures[4];
layout (binding = 3) uniform sampler2D shadow_map;

float calculateShadow(vec4 light_space_pos) {
	// Perspective divide
	vec3 proj_coords = light_space_pos.xyz / light_space_pos.w;
	
	// Transform to [0,1] range
	proj_coords = proj_coords * 0.5 + 0.5;
	
	// Check if fragment is outside light frustum
	if (proj_coords.x < 0.0 || proj_coords.x > 1.0 ||
	    proj_coords.y < 0.0 || proj_coords.y > 1.0 ||
	    proj_coords.z < 0.0 || proj_coords.z > 1.0) {
		return 0.0; // Outside shadow map, consider as shadowed
	}
	
	// Manual PCF (Percentage Closer Filtering)
	float shadow = 0.0;
	float current_depth = proj_coords.z;
	float bias = 0.005;
	
	vec2 texel_size = 1.0 / textureSize(shadow_map, 0);
	for(int x = -1; x <= 1; ++x) {
		for(int y = -1; y <= 1; ++y) {
			vec2 offset = vec2(float(x), float(y)) * texel_size;
			float closest_depth = texture(shadow_map, proj_coords.xy + offset).r;
			shadow += (current_depth - bias) > closest_depth ? 0.0 : 1.0;
		}
	}
	shadow /= 9.0;
	
	return shadow;
}

void main() {
	vec3 albedo = albedo_color;
	if (texture_index >= 0 && texture_index < 4) {
		albedo *= texture(textures[texture_index], f_uv).rgb;
	}
	
	// Normalize inputs
	vec3 N = normalize(f_normal);
	vec3 L = normalize(-light_direction);
	vec3 V = normalize(camera_position - f_position);
	vec3 H = normalize(L + V);
	
	// Ambient
	float ambient_strength = 0.2;
	vec3 ambient = ambient_strength * albedo;
	
	// Directional diffuse/specular
	float diff = max(dot(N, L), 0.0);
	vec3 diffuse = diff * albedo;
	float spec = pow(max(dot(N, H), 0.0), 32.0);
	vec3 specular = spec * vec3(1.0, 1.0, 1.0) * 0.5;
	float shadow = calculateShadow(f_light_space_pos);
	vec3 color = ambient + shadow * (diffuse + specular);

	// Add point lights (no shadows)
	for (int i = 0; i < point_light_count; ++i) {
		vec3 light_pos = point_lights[i].position.xyz;
		vec3 light_color = point_lights[i].color.rgb;
		vec3 to_light = light_pos - f_position;
		float distance = length(to_light);
		vec3 Lp = to_light / distance;

		float attenuation = 1.0 / (1.0 + 0.09 * distance + 0.032 * distance * distance);

		float diff_p = max(dot(N, Lp), 0.0);
		vec3 diffuse_p = diff_p * albedo * light_color;

		vec3 Hp = normalize(Lp + V);
		float spec_p = pow(max(dot(N, Hp), 0.0), 32.0);
		vec3 specular_p = spec_p * light_color * 0.5;

		color += attenuation * (diffuse_p + specular_p);
	}
	
	final_color = vec4(color, 1.0);
}
