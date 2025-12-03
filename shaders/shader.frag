#version 450

layout (location = 0) in vec3 f_position;
layout (location = 1) in vec3 f_normal;
layout (location = 2) in vec2 f_uv;

layout (location = 0) out vec4 final_color;

layout (binding = 0, std140) uniform SceneUniforms {
	mat4 view_projection;
	vec3 camera_position;
	float _pad0;
	vec3 ambient_color;
	float _pad1;
	vec3 directional_direction;
	float directional_intensity;
	vec3 directional_color;
	float _pad2;
};

layout (binding = 1, std140) uniform ModelUniforms {
	mat4 model;
	vec3 albedo_color;
	float shininess;
	vec3 specular_color;
	float _pad3;
};

struct PointLight {
	vec3 position;
	float intensity;
	vec3 color;
	float constant;
	float linear;
	float quadratic;
};

struct SpotLight {
	vec3 position;
	float intensity;
	vec3 direction;
	float inner_cutoff;
	vec3 color;
	float outer_cutoff;
};

layout (binding = 2, std430) readonly buffer PointLightsBuffer {
	PointLight point_lights[];
};

layout (binding = 3, std430) readonly buffer SpotLightsBuffer {
	SpotLight spot_lights[];
};

vec3 blinnPhong(vec3 light_dir, vec3 light_color, float light_intensity, vec3 view_dir, vec3 normal) {
	// Diffuse component: albedo is already applied in main()
	float diff = max(dot(normal, light_dir), 0.0);
	vec3 diffuse = diff * light_color * light_intensity * albedo_color;
	
	// Specular component (Blinn-Phong)
	vec3 half_dir = normalize(light_dir + view_dir);
	float spec = pow(max(dot(normal, half_dir), 0.0), shininess);
	vec3 specular = spec * specular_color * light_color * light_intensity;
	
	return diffuse + specular;
}

void main() {
	vec3 normal = normalize(f_normal);
	vec3 view_dir = normalize(camera_position - f_position);
	
	// Ambient component
	vec3 result = ambient_color * albedo_color;
	
	// Directional light
	vec3 dir_light_dir = normalize(-directional_direction);
	result += blinnPhong(dir_light_dir, directional_color, directional_intensity, view_dir, normal);
	
	// Point lights
	for (int i = 0; i < point_lights.length(); ++i) {
		PointLight light = point_lights[i];
		if (light.intensity <= 0.0) continue;
		
		vec3 light_dir = normalize(light.position - f_position);
		float distance = length(light.position - f_position);
		
		// Attenuation (inverse square law)
		float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * distance * distance);
		
		vec3 light_contribution = blinnPhong(light_dir, light.color, light.intensity, view_dir, normal);
		result += light_contribution * attenuation;
	}
	
	// Spot lights
	for (int i = 0; i < spot_lights.length(); ++i) {
		SpotLight light = spot_lights[i];
		if (light.intensity <= 0.0) continue;
		
		vec3 light_dir = normalize(light.position - f_position);
		float distance = length(light.position - f_position);
		
		// Spot light attenuation
		float theta = dot(light_dir, normalize(-light.direction));
		float epsilon = light.inner_cutoff - light.outer_cutoff;
		float intensity = clamp((theta - light.outer_cutoff) / epsilon, 0.0, 1.0);
		
		// Distance attenuation (simplified)
		float attenuation = 1.0 / (1.0 + 0.09 * distance + 0.032 * distance * distance);
		
		if (theta > light.outer_cutoff) {
			vec3 light_contribution = blinnPhong(light_dir, light.color, light.intensity, view_dir, normal);
			result += light_contribution * intensity * attenuation;
		}
	}
	
	final_color = vec4(result, 1.0f);
}
