#include <cstdint>
#include <climits>
#include <cstring>
#include <limits>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>

#include <veekay/veekay.hpp>

#include <vulkan/vulkan_core.h>
#include <imgui.h>
#include <lodepng.h>

namespace {

constexpr uint32_t max_models = 1024;
constexpr uint32_t max_textures = 4;

struct Vertex {
	veekay::vec3 position;
	veekay::vec3 normal;
	veekay::vec2 uv;
	// NOTE: You can add more attributes
};

struct PointLight {
	veekay::vec4 position;
	veekay::vec4 color;
};

struct SceneUniforms {
	veekay::mat4 view_projection;
	veekay::mat4 light_view_projection;
	veekay::vec3 light_direction;
	float _pad0;
	veekay::vec3 camera_position;
	float _pad1;
	PointLight point_lights[4];
	uint32_t point_light_count;
	float _pad2[3];
};

struct ModelUniforms {
	veekay::mat4 model;
	veekay::vec3 albedo_color; float _pad0;
	uint32_t texture_index;
	float _pad1[3];
};

struct Mesh {
	veekay::graphics::Buffer* vertex_buffer;
	veekay::graphics::Buffer* index_buffer;
	uint32_t indices;
};

struct Transform {
	veekay::vec3 position = {};
	veekay::vec3 scale = {1.0f, 1.0f, 1.0f};
	veekay::vec3 rotation = {};

	// NOTE: Model matrix (translation, rotation and scaling)
	veekay::mat4 matrix() const;
};

struct Model {
	Mesh mesh;
	Transform transform;
	veekay::vec3 albedo_color;
	uint32_t texture_index = 0;
};

enum class ProjectionType {
	Perspective,
	Orthographic
};

struct Camera {
	constexpr static float default_fov = 60.0f;
	constexpr static float default_near_plane = 0.01f;
	constexpr static float default_far_plane = 100.0f;

	veekay::vec3 position = {};
	veekay::vec3 rotation = {};

	float fov = default_fov;
	float near_plane = default_near_plane;
	float far_plane = default_far_plane;

	ProjectionType projection_type = ProjectionType::Perspective;

	// NOTE: View matrix of camera (inverse of a transform)
	veekay::mat4 view() const;

	// NOTE: View and projection composition
	veekay::mat4 view_projection(float aspect_ratio) const;
};

// NOTE: Scene objects
inline namespace {
	Camera camera{
		.position = {0.0f, 0.5f, -3.0f} // камера чуть выше
	};

	std::vector<Model> models;
}

// NOTE: Vulkan objects
inline namespace {
	VkShaderModule vertex_shader_module;
	VkShaderModule fragment_shader_module;

	VkDescriptorPool descriptor_pool;
	VkDescriptorSetLayout descriptor_set_layout;
	VkDescriptorSet descriptor_set;

	VkPipelineLayout pipeline_layout;
	VkPipeline pipeline;

	veekay::graphics::Buffer* scene_uniforms_buffer;
	veekay::graphics::Buffer* model_uniforms_buffer;

	Mesh plane_mesh;
	Mesh cube_mesh;

	veekay::graphics::Texture* missing_texture;
	VkSampler missing_texture_sampler;

	veekay::graphics::Texture* texture;
	VkSampler texture_sampler;
	std::vector<veekay::graphics::Texture*> color_textures;
	VkSampler color_texture_sampler;

	// Shadow mapping
	VkImage shadow_map_image;
	VkDeviceMemory shadow_map_memory;
	VkImageView shadow_map_view;
	VkSampler shadow_map_sampler;
	constexpr static uint32_t shadow_map_size = 2048;

	VkShaderModule shadow_vertex_shader_module;
	VkPipeline shadow_pipeline;
	VkPipelineLayout shadow_pipeline_layout;
}

float toRadians(float degrees) {
	return degrees * float(M_PI) / 180.0f;
}

veekay::mat4 eulerRotation(const veekay::vec3& rotation) {
	auto rx = veekay::mat4::rotation({1.0f, 0.0f, 0.0f}, rotation.x);
	auto ry = veekay::mat4::rotation({0.0f, 1.0f, 0.0f}, rotation.y);
	auto rz = veekay::mat4::rotation({0.0f, 0.0f, 1.0f}, rotation.z);

	// NOTE: Apply rotations in X -> Y -> Z order
	return rz * ry * rx;
}

veekay::mat4 Transform::matrix() const {
	auto s = veekay::mat4::scaling(scale);
	auto r = eulerRotation(rotation);
	auto t = veekay::mat4::translation(position);

	return t * r * s;
}

veekay::mat4 Camera::view() const {
	auto r = eulerRotation(rotation);
	auto t = veekay::mat4::translation(-position);

	// NOTE: Inverse of rotation is its transpose for orthonormal matrices
	return veekay::mat4::transpose(r) * t;
}

veekay::mat4 Camera::view_projection(float aspect_ratio) const {
	veekay::mat4 projection;
	if (projection_type == ProjectionType::Perspective) {
		projection = veekay::mat4::projection(fov, aspect_ratio, near_plane, far_plane);
	} else {
		float ortho_height = 2.0f; // Можно сделать настраиваемым
		float ortho_width = ortho_height * aspect_ratio;
		projection = veekay::mat4::orthographic(-ortho_width, ortho_width, -ortho_height, ortho_height, near_plane, far_plane);
	}
	return view() * projection;
}

// NOTE: Loads shader byte code from file
// NOTE: Your shaders are compiled via CMake with this code too, look it up
VkShaderModule loadShaderModule(const char* path) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file.is_open()) {
		std::cerr << "Failed to open shader file: " << path << std::endl;
		return nullptr;
	}
	
	size_t size = file.tellg();
	if (size == 0 || size == static_cast<size_t>(-1)) {
		std::cerr << "Invalid shader file size: " << path << std::endl;
		file.close();
		return nullptr;
	}
	
	if (size % sizeof(uint32_t) != 0) {
		std::cerr << "Shader file size is not a multiple of 4: " << path << std::endl;
		file.close();
		return nullptr;
	}
	
	std::vector<uint32_t> buffer(size / sizeof(uint32_t));
	file.seekg(0);
	if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
		std::cerr << "Failed to read shader file: " << path << std::endl;
		file.close();
		return nullptr;
	}
	file.close();

	VkShaderModuleCreateInfo info{
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = size,
		.pCode = buffer.data(),
	};

	VkShaderModule result;
	if (vkCreateShaderModule(veekay::app.vk_device, &
	                         info, nullptr, &result) != VK_SUCCESS) {
		std::cerr << "Failed to create shader module from: " << path << std::endl;
		return nullptr;
	}

	return result;
}

veekay::graphics::Texture* createCheckerTexture(VkCommandBuffer cmd, uint32_t size = 512, uint32_t check_size = 32) {
	std::vector<uint32_t> pixels(size * size);
	for (uint32_t y = 0; y < size; ++y) {
		for (uint32_t x = 0; x < size; ++x) {
			bool even = ((x / check_size) + (y / check_size)) % 2 == 0;
			uint8_t c = even ? 220 : 70;
			pixels[y * size + x] = 0xff000000 |
			                       (uint32_t(c) << 16) |
			                       (uint32_t(c) << 8) |
			                       uint32_t(c);
		}
	}
	return new veekay::graphics::Texture(cmd, size, size, VK_FORMAT_B8G8R8A8_UNORM, pixels.data());
}

veekay::graphics::Texture* loadTexture(VkCommandBuffer cmd, const char* path) {
	std::vector<unsigned char> image;
	unsigned width = 0, height = 0;
	unsigned error = lodepng::decode(image, width, height, path);
	if (error) {
		std::cerr << "Failed to load texture: " << path << " error: " << error << std::endl;
		return nullptr;
	}

	return new veekay::graphics::Texture(cmd, width, height, VK_FORMAT_R8G8B8A8_UNORM, image.data());
}

void initialize(VkCommandBuffer cmd) {
	VkDevice& device = veekay::app.vk_device;
	VkPhysicalDevice& physical_device = veekay::app.vk_physical_device;

	{ // NOTE: Build graphics pipeline
		vertex_shader_module = loadShaderModule("./shaders/shader.vert.spv");
		if (!vertex_shader_module) {
			std::cerr << "Failed to load Vulkan vertex shader from file\n";
			veekay::app.running = false;
			return;
		}

		fragment_shader_module = loadShaderModule("./shaders/shader.frag.spv");
		if (!fragment_shader_module) {
			std::cerr << "Failed to load Vulkan fragment shader from file\n";
			veekay::app.running = false;
			return;
		}

		VkPipelineShaderStageCreateInfo stage_infos[2];

		// NOTE: Vertex shader stage
		stage_infos[0] = VkPipelineShaderStageCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vertex_shader_module,
			.pName = "main",
		};

		// NOTE: Fragment shader stage
		stage_infos[1] = VkPipelineShaderStageCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragment_shader_module,
			.pName = "main",
		};

		// NOTE: How many bytes does a vertex take?
		VkVertexInputBindingDescription buffer_binding{
			.binding = 0,
			.stride = sizeof(Vertex),
			.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
		};

		// NOTE: Declare vertex attributes
		VkVertexInputAttributeDescription attributes[] = {
			{
				.location = 0, // NOTE: First attribute
				.binding = 0, // NOTE: First vertex buffer
				.format = VK_FORMAT_R32G32B32_SFLOAT, // NOTE: 3-component vector of floats
				.offset = offsetof(Vertex, position), // NOTE: Offset of "position" field in a Vertex struct
			},
			{
				.location = 1,
				.binding = 0,
				.format = VK_FORMAT_R32G32B32_SFLOAT,
				.offset = offsetof(Vertex, normal),
			},
			{
				.location = 2,
				.binding = 0,
				.format = VK_FORMAT_R32G32_SFLOAT,
				.offset = offsetof(Vertex, uv),
			},
		};

		// NOTE: Describe inputs
		VkPipelineVertexInputStateCreateInfo input_state_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
			.vertexBindingDescriptionCount = 1,
			.pVertexBindingDescriptions = &buffer_binding,
			.vertexAttributeDescriptionCount = sizeof(attributes) / sizeof(attributes[0]),
			.pVertexAttributeDescriptions = attributes,
		};

		// NOTE: Every three vertices make up a triangle,
		//       so our vertex buffer contains a "list of triangles"
		VkPipelineInputAssemblyStateCreateInfo assembly_state_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
			.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		};

		// NOTE: Declare clockwise triangle order as front-facing
		//       Discard triangles that are facing away
		//       Fill triangles, don't draw lines instaed
		VkPipelineRasterizationStateCreateInfo raster_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
			.polygonMode = VK_POLYGON_MODE_FILL,
			.cullMode = VK_CULL_MODE_BACK_BIT,
			.frontFace = VK_FRONT_FACE_CLOCKWISE,
			.lineWidth = 1.0f,
		};

		// NOTE: Use 1 sample per pixel
		VkPipelineMultisampleStateCreateInfo sample_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
			.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
			.sampleShadingEnable = false,
			.minSampleShading = 1.0f,
		};

		VkViewport viewport{
			.x = 0.0f,
			.y = 0.0f,
			.width = static_cast<float>(veekay::app.window_width),
			.height = static_cast<float>(veekay::app.window_height),
			.minDepth = 0.0f,
			.maxDepth = 1.0f,
		};

		VkRect2D scissor{
			.offset = {0, 0},
			.extent = {veekay::app.window_width, veekay::app.window_height},
		};

		// NOTE: Let rasterizer draw on the entire window
		VkPipelineViewportStateCreateInfo viewport_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,

			.viewportCount = 1,
			.pViewports = &viewport,

			.scissorCount = 1,
			.pScissors = &scissor,
		};

		// NOTE: Let rasterizer perform depth-testing and overwrite depth values on condition pass
		VkPipelineDepthStencilStateCreateInfo depth_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
			.depthTestEnable = true,
			.depthWriteEnable = true,
			.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
		};

		// NOTE: Let fragment shader write all the color channels
		VkPipelineColorBlendAttachmentState attachment_info{
			.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
			                  VK_COLOR_COMPONENT_G_BIT |
			                  VK_COLOR_COMPONENT_B_BIT |
			                  VK_COLOR_COMPONENT_A_BIT,
		};

		// NOTE: Let rasterizer just copy resulting pixels onto a buffer, don't blend
		VkPipelineColorBlendStateCreateInfo blend_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,

			.logicOpEnable = false,
			.logicOp = VK_LOGIC_OP_COPY,

			.attachmentCount = 1,
			.pAttachments = &attachment_info
		};

		{
		VkDescriptorPoolSize pools[] = {
			{
				.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.descriptorCount = 8,
			},
			{
					.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
					.descriptorCount = 8,
				},
			{
				.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.descriptorCount = 16,
			}
		};
			
			VkDescriptorPoolCreateInfo info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
				.maxSets = 1,
				.poolSizeCount = sizeof(pools) / sizeof(pools[0]),
				.pPoolSizes = pools,
			};

			if (vkCreateDescriptorPool(device, &info, nullptr,
			                           &descriptor_pool) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan descriptor pool\n";
				veekay::app.running = false;
				return;
			}
		}

		// NOTE: Descriptor set layout specification
		{
			VkDescriptorSetLayoutBinding bindings[] = {
				{
					.binding = 0,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
				},
				{
					.binding = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
				},
				{
					.binding = 2,
					.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.descriptorCount = max_textures,
					.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
				},
				{
					.binding = 3,
					.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
				},
			};

			VkDescriptorSetLayoutCreateInfo info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
				.bindingCount = sizeof(bindings) / sizeof(bindings[0]),
				.pBindings = bindings,
			};

			if (vkCreateDescriptorSetLayout(device, &info, nullptr,
			                                &descriptor_set_layout) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan descriptor set layout\n";
				veekay::app.running = false;
				return;
			}
		}

		{
			VkDescriptorSetAllocateInfo info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.descriptorPool = descriptor_pool,
				.descriptorSetCount = 1,
				.pSetLayouts = &descriptor_set_layout,
			};

			if (vkAllocateDescriptorSets(device, &info, &descriptor_set) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan descriptor set\n";
				veekay::app.running = false;
				return;
			}
		}

		// NOTE: Declare external data sources, only push constants this time
		VkPipelineLayoutCreateInfo layout_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = 1,
			.pSetLayouts = &descriptor_set_layout,
		};

		// NOTE: Create pipeline layout
		if (vkCreatePipelineLayout(device, &layout_info,
		                           nullptr, &pipeline_layout) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan pipeline layout\n";
			veekay::app.running = false;
			return;
		}
		
		VkGraphicsPipelineCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
			.stageCount = 2,
			.pStages = stage_infos,
			.pVertexInputState = &input_state_info,
			.pInputAssemblyState = &assembly_state_info,
			.pViewportState = &viewport_info,
			.pRasterizationState = &raster_info,
			.pMultisampleState = &sample_info,
			.pDepthStencilState = &depth_info,
			.pColorBlendState = &blend_info,
			.layout = pipeline_layout,
			.renderPass = veekay::app.vk_render_pass,
		};

		// NOTE: Create graphics pipeline
		if (vkCreateGraphicsPipelines(device, nullptr,
		                              1, &info, nullptr, &pipeline) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan pipeline\n";
			veekay::app.running = false;
			return;
		}
	}

	scene_uniforms_buffer = new veekay::graphics::Buffer(
		sizeof(SceneUniforms),
		nullptr,
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

	model_uniforms_buffer = new veekay::graphics::Buffer(
		max_models * veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms)),
		nullptr,
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

	// NOTE: This texture and sampler is used when texture could not be loaded
	{
		VkSamplerCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		};

		if (vkCreateSampler(device, &info, nullptr, &missing_texture_sampler) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan texture sampler\n";
			veekay::app.running = false;
			return;
		}

		uint32_t pixels[] = {
			0xff000000, 0xffff00ff,
			0xffff00ff, 0xff000000,
		};

		missing_texture = new veekay::graphics::Texture(cmd, 2, 2,
		                                                VK_FORMAT_B8G8R8A8_UNORM,
		                                                pixels);
	}

	// NOTE: Create shadow map
	{
		VkImageCreateInfo image_info{
			.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
			.imageType = VK_IMAGE_TYPE_2D,
			.format = VK_FORMAT_D32_SFLOAT,
			.extent = {shadow_map_size, shadow_map_size, 1},
			.mipLevels = 1,
			.arrayLayers = 1,
			.samples = VK_SAMPLE_COUNT_1_BIT,
			.tiling = VK_IMAGE_TILING_OPTIMAL,
			.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
		};

		if (vkCreateImage(device, &image_info, nullptr, &shadow_map_image) != VK_SUCCESS) {
			std::cerr << "Failed to create shadow map image\n";
			veekay::app.running = false;
			return;
		}

		VkMemoryRequirements mem_requirements;
		vkGetImageMemoryRequirements(device, shadow_map_image, &mem_requirements);

		VkPhysicalDeviceMemoryProperties mem_properties;
		vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_properties);

		uint32_t memory_type_index = std::numeric_limits<uint32_t>::max();
		for (uint32_t i = 0; i < mem_properties.memoryTypeCount; ++i) {
			if ((mem_requirements.memoryTypeBits & (1 << i)) &&
			    (mem_properties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
				memory_type_index = i;
				break;
			}
		}

		if (memory_type_index == std::numeric_limits<uint32_t>::max()) {
			std::cerr << "Failed to find memory type for shadow map\n";
			veekay::app.running = false;
			return;
		}

		VkMemoryAllocateInfo alloc_info{
			.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
			.allocationSize = mem_requirements.size,
			.memoryTypeIndex = memory_type_index,
		};

		if (vkAllocateMemory(device, &alloc_info, nullptr, &shadow_map_memory) != VK_SUCCESS) {
			std::cerr << "Failed to allocate shadow map memory\n";
			veekay::app.running = false;
			return;
		}

		if (vkBindImageMemory(device, shadow_map_image, shadow_map_memory, 0) != VK_SUCCESS) {
			std::cerr << "Failed to bind shadow map memory\n";
			veekay::app.running = false;
			return;
		}

		VkImageViewCreateInfo view_info{
			.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.image = shadow_map_image,
			.viewType = VK_IMAGE_VIEW_TYPE_2D,
			.format = VK_FORMAT_D32_SFLOAT,
			.subresourceRange = {
				.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1,
			},
		};

		if (vkCreateImageView(device, &view_info, nullptr, &shadow_map_view) != VK_SUCCESS) {
			std::cerr << "Failed to create shadow map view\n";
			veekay::app.running = false;
			return;
		}

		VkSamplerCreateInfo sampler_info{
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.magFilter = VK_FILTER_LINEAR,
			.minFilter = VK_FILTER_LINEAR,
			.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
			.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
			.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
			.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
			.mipLodBias = 0.0f,
			.anisotropyEnable = false,
			.maxAnisotropy = 1.0f,
			.compareEnable = false, // Disabled for macOS compatibility, will use manual PCF
			.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
			.minLod = 0.0f,
			.maxLod = 1.0f,
			.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
		};

		if (vkCreateSampler(device, &sampler_info, nullptr, &shadow_map_sampler) != VK_SUCCESS) {
			std::cerr << "Failed to create shadow map sampler\n";
			veekay::app.running = false;
			return;
		}

		// Transition shadow map to depth attachment layout
		VkImageMemoryBarrier barrier{
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.srcAccessMask = 0,
			.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
			.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = shadow_map_image,
			.subresourceRange = {
				.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1,
			},
		};

		vkCmdPipelineBarrier(cmd,
		                     VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
		                     VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
		                     0,
		                     0, nullptr,
		                     0, nullptr,
		                     1, &barrier);
	}

	// NOTE: Color textures and sampler
	{
		VkSamplerCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.magFilter = VK_FILTER_LINEAR,
			.minFilter = VK_FILTER_LINEAR,
			.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
			.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
			.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
			.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
			.maxAnisotropy = 1.0f,
			.anisotropyEnable = VK_FALSE,
		};

		if (vkCreateSampler(device, &info, nullptr, &color_texture_sampler) != VK_SUCCESS) {
			std::cerr << "Failed to create color texture sampler\n";
			veekay::app.running = false;
			return;
		}

		color_textures.reserve(max_textures);
		color_textures.push_back(createCheckerTexture(cmd, 512, 48)); // 0: floor
		if (auto tex = loadTexture(cmd, "./assets/lenna.png")) {       // 1: fun texture
			color_textures.push_back(tex);
		}
		// Fill remaining slots with missing texture for safe indexing
		while (color_textures.size() < max_textures) {
			color_textures.push_back(missing_texture);
		}
	}

	// NOTE: Create shadow pass pipeline
	{
		shadow_vertex_shader_module = loadShaderModule("./shaders/shadow.vert.spv");
		if (!shadow_vertex_shader_module) {
			std::cerr << "Failed to load shadow vertex shader\n";
			veekay::app.running = false;
			return;
		}

		VkPipelineShaderStageCreateInfo shadow_stage_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = shadow_vertex_shader_module,
			.pName = "main",
		};

		VkVertexInputBindingDescription buffer_binding{
			.binding = 0,
			.stride = sizeof(Vertex),
			.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
		};

		VkVertexInputAttributeDescription attributes[] = {
			{
				.location = 0,
				.binding = 0,
				.format = VK_FORMAT_R32G32B32_SFLOAT,
				.offset = offsetof(Vertex, position),
			},
		};

		VkPipelineVertexInputStateCreateInfo input_state_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
			.vertexBindingDescriptionCount = 1,
			.pVertexBindingDescriptions = &buffer_binding,
			.vertexAttributeDescriptionCount = 1,
			.pVertexAttributeDescriptions = attributes,
		};

		VkPipelineInputAssemblyStateCreateInfo assembly_state_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
			.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		};

		VkPipelineRasterizationStateCreateInfo raster_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
			.polygonMode = VK_POLYGON_MODE_FILL,
			.cullMode = VK_CULL_MODE_BACK_BIT,
			.frontFace = VK_FRONT_FACE_CLOCKWISE,
			.lineWidth = 1.0f,
			.depthBiasEnable = true,
			.depthBiasConstantFactor = 1.25f,
			.depthBiasSlopeFactor = 1.75f,
		};

		VkPipelineMultisampleStateCreateInfo sample_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
			.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
		};

		VkViewport viewport{
			.x = 0.0f,
			.y = 0.0f,
			.width = static_cast<float>(shadow_map_size),
			.height = static_cast<float>(shadow_map_size),
			.minDepth = 0.0f,
			.maxDepth = 1.0f,
		};

		VkRect2D scissor{
			.offset = {0, 0},
			.extent = {shadow_map_size, shadow_map_size},
		};

		VkPipelineViewportStateCreateInfo viewport_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
			.viewportCount = 1,
			.pViewports = &viewport,
			.scissorCount = 1,
			.pScissors = &scissor,
		};

		VkPipelineDepthStencilStateCreateInfo depth_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
			.depthTestEnable = true,
			.depthWriteEnable = true,
			.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
		};

		VkPipelineLayoutCreateInfo layout_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = 1,
			.pSetLayouts = &descriptor_set_layout,
		};

		if (vkCreatePipelineLayout(device, &layout_info, nullptr, &shadow_pipeline_layout) != VK_SUCCESS) {
			std::cerr << "Failed to create shadow pipeline layout\n";
			veekay::app.running = false;
			return;
		}

		VkPipelineRenderingCreateInfo rendering_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
			.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT,
		};

		VkGraphicsPipelineCreateInfo pipeline_info{
			.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
			.pNext = &rendering_info,
			.stageCount = 1,
			.pStages = &shadow_stage_info,
			.pVertexInputState = &input_state_info,
			.pInputAssemblyState = &assembly_state_info,
			.pViewportState = &viewport_info,
			.pRasterizationState = &raster_info,
			.pMultisampleState = &sample_info,
			.pDepthStencilState = &depth_info,
			.layout = shadow_pipeline_layout,
		};

		if (vkCreateGraphicsPipelines(device, nullptr, 1, &pipeline_info, nullptr, &shadow_pipeline) != VK_SUCCESS) {
			std::cerr << "Failed to create shadow pipeline\n";
			veekay::app.running = false;
			return;
		}
	}

	{
		VkDescriptorBufferInfo buffer_infos[] = {
			{
				.buffer = scene_uniforms_buffer->buffer,
				.offset = 0,
				.range = sizeof(SceneUniforms),
			},
			{
				.buffer = model_uniforms_buffer->buffer,
				.offset = 0,
				.range = sizeof(ModelUniforms),
			},
		};

		VkDescriptorImageInfo color_images[max_textures];
		for (size_t i = 0; i < max_textures; ++i) {
			color_images[i] = VkDescriptorImageInfo{
				.sampler = color_texture_sampler,
				.imageView = color_textures[i]->view,
				.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			};
		}

		VkDescriptorImageInfo shadow_image_info{
			.sampler = shadow_map_sampler,
			.imageView = shadow_map_view,
			.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
		};

		VkWriteDescriptorSet write_infos[] = {
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 0,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.pBufferInfo = &buffer_infos[0],
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 1,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
				.pBufferInfo = &buffer_infos[1],
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 2,
				.dstArrayElement = 0,
				.descriptorCount = max_textures,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.pImageInfo = color_images,
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 3,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.pImageInfo = &shadow_image_info,
			},
		};

		vkUpdateDescriptorSets(device, sizeof(write_infos) / sizeof(write_infos[0]),
		                       write_infos, 0, nullptr);
	}

	// NOTE: Plane mesh initialization
	{
		// (v0)------(v1)
		//  |  \       |
		//  |   `--,   |
		//  |       \  |
		// (v3)------(v2)
		std::vector<Vertex> vertices = {
			{{-5.0f, 0.0f, 5.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
			{{5.0f, 0.0f, 5.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
			{{5.0f, 0.0f, -5.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
			{{-5.0f, 0.0f, -5.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},
		};

		std::vector<uint32_t> indices = {
			0, 1, 2, 2, 3, 0
		};

		plane_mesh.vertex_buffer = new veekay::graphics::Buffer(
			vertices.size() * sizeof(Vertex), vertices.data(),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

		plane_mesh.index_buffer = new veekay::graphics::Buffer(
			indices.size() * sizeof(uint32_t), indices.data(),
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

		plane_mesh.indices = uint32_t(indices.size());
	}

	// NOTE: Cube mesh initialization
	{
		std::vector<Vertex> vertices = {
			{{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 0.0f}},
			{{+0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f}},
			{{+0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 1.0f}},
			{{-0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f}},

			{{+0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
			{{+0.5f, -0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
			{{+0.5f, +0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
			{{+0.5f, +0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},

			{{+0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}},
			{{-0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}},
			{{-0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
			{{+0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},

			{{-0.5f, -0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
			{{-0.5f, -0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
			{{-0.5f, +0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
			{{-0.5f, +0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},

			{{-0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
			{{+0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
			{{+0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
			{{-0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},

			{{-0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
			{{+0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
			{{+0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},
			{{-0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}},
		};

		std::vector<uint32_t> indices = {
			0, 1, 2, 2, 3, 0,
			4, 5, 6, 6, 7, 4,
			8, 9, 10, 10, 11, 8,
			12, 13, 14, 14, 15, 12,
			16, 17, 18, 18, 19, 16,
			20, 21, 22, 22, 23, 20,
		};

		cube_mesh.vertex_buffer = new veekay::graphics::Buffer(
			vertices.size() * sizeof(Vertex), vertices.data(),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

		cube_mesh.index_buffer = new veekay::graphics::Buffer(
			indices.size() * sizeof(uint32_t), indices.data(),
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

		cube_mesh.indices = uint32_t(indices.size());
	}

	// NOTE: Add plane to scene
	models.emplace_back(Model{
		.mesh = plane_mesh,
		.transform = Transform{
			.position = {0.0f, 0.0f, 0.0f},
		},
		.albedo_color = veekay::vec3{0.8f, 0.8f, 0.8f},
		.texture_index = 0,
	});

	// NOTE: Add cube model to scene
	models.emplace_back(Model{
		.mesh = cube_mesh,
		.transform = Transform{
			.position = {0.0f, 0.5f, 0.0f},
		},
		.albedo_color = veekay::vec3{0.0f, 0.5f, 1.0f},
		.texture_index = 1,
	});

	// NOTE: Extra cubes/pillars to show shadow falloff at different depths
	models.emplace_back(Model{
		.mesh = cube_mesh,
		.transform = Transform{
			.position = {2.0f, 0.5f, 1.5f},
		},
		.albedo_color = veekay::vec3{0.9f, 0.4f, 0.2f},
		.texture_index = 1,
	});

	models.emplace_back(Model{
		.mesh = cube_mesh,
		.transform = Transform{
			.position = {-2.5f, 0.5f, -1.0f},
		},
		.albedo_color = veekay::vec3{0.3f, 0.8f, 0.3f},
		.texture_index = 1,
	});

	// Removed tall yellow column to declutter view
}

// NOTE: Destroy resources here, do not cause leaks in your program!
void shutdown() {
	VkDevice& device = veekay::app.vk_device;

	vkDestroySampler(device, shadow_map_sampler, nullptr);
	vkDestroyImageView(device, shadow_map_view, nullptr);
	vkDestroyImage(device, shadow_map_image, nullptr);
	vkFreeMemory(device, shadow_map_memory, nullptr);

	vkDestroySampler(device, color_texture_sampler, nullptr);
	for (auto* tex : color_textures) {
		if (tex != missing_texture) {
			delete tex;
		}
	}
	color_textures.clear();

	vkDestroyPipeline(device, shadow_pipeline, nullptr);
	vkDestroyPipelineLayout(device, shadow_pipeline_layout, nullptr);
	vkDestroyShaderModule(device, shadow_vertex_shader_module, nullptr);

	vkDestroySampler(device, missing_texture_sampler, nullptr);
	delete missing_texture;

	delete cube_mesh.index_buffer;
	delete cube_mesh.vertex_buffer;

	delete plane_mesh.index_buffer;
	delete plane_mesh.vertex_buffer;

	delete model_uniforms_buffer;
	delete scene_uniforms_buffer;

	vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
	vkDestroyDescriptorPool(device, descriptor_pool, nullptr);

	vkDestroyPipeline(device, pipeline, nullptr);
	vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
	vkDestroyShaderModule(device, fragment_shader_module, nullptr);
	vkDestroyShaderModule(device, vertex_shader_module, nullptr);
}

void update(double time) {
	static float rotation_speed = 45.0f; // градусов в секунду
	static float cube_angle = 0.0f;
	static double last_time = 0.0;

	double delta_time = last_time > 0.0 ? time - last_time : 0.0;
	last_time = time;

	ImGui::Begin("Controls:");
	ImGui::SliderFloat("Rotation speed (deg/s)", &rotation_speed, 0.0f, 180.0f, "%.1f");
	ImGui::Text("Press 1: Perspective\nPress 2: Orthographic");
	ImGui::Text("RMB: toggle capture (hold RMB also rotates)\nWASD + E up / Q down + mouse: fly\nShift: speed up");
	ImGui::Text("Current projection: %s", camera.projection_type == ProjectionType::Perspective ? "Perspective" : "Orthographic");
	ImGui::End();

	cube_angle += rotation_speed * delta_time * (M_PI / 180.0f); // переводим в радианы
	
	static bool camera_captured = false;

	bool can_control = camera_captured ||
	                   veekay::input::mouse::isButtonDown(veekay::input::mouse::Button::right) ||
	                   veekay::input::mouse::isButtonDown(veekay::input::mouse::Button::left) ||
	                   !ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow);
	if (can_control) {
		using namespace veekay::input;

		if (keyboard::isKeyPressed(keyboard::Key::d1)) {
			camera.projection_type = ProjectionType::Perspective;
		}
		if (keyboard::isKeyPressed(keyboard::Key::d2)) {
			camera.projection_type = ProjectionType::Orthographic;
		}

		if (mouse::isButtonPressed(mouse::Button::right)) {
			camera_captured = !camera_captured;
			mouse::setCaptured(camera_captured);
		}

		const bool rotating = camera_captured ||
		                      mouse::isButtonDown(mouse::Button::right) ||
		                      mouse::isButtonDown(mouse::Button::left);

		if (rotating) {
			auto move_delta = mouse::cursorDelta();

			const float sensitivity = 0.01f;
			camera.rotation.y += move_delta.x * sensitivity;
			camera.rotation.x += move_delta.y * sensitivity;

			const float pitch_limit = toRadians(89.0f);
			camera.rotation.x = std::clamp(camera.rotation.x, -pitch_limit, pitch_limit);

			const float two_pi = float(M_PI) * 2.0f;
			if (camera.rotation.y > two_pi) camera.rotation.y -= two_pi;
			if (camera.rotation.y < -two_pi) camera.rotation.y += two_pi;
		}

		// NOTE: Derive camera basis from current rotation (for movement even without rotating)
		auto rotation_matrix = eulerRotation(camera.rotation);
		veekay::vec3 right = {rotation_matrix[0].x, rotation_matrix[0].y, rotation_matrix[0].z};
		veekay::vec3 up = {rotation_matrix[1].x, rotation_matrix[1].y, rotation_matrix[1].z};
		veekay::vec3 front = {rotation_matrix[2].x, rotation_matrix[2].y, rotation_matrix[2].z};

		float move_speed = 3.0f * static_cast<float>(delta_time);
		if (keyboard::isKeyDown(keyboard::Key::left_shift)) {
			move_speed *= 2.5f;
		}

		if (keyboard::isKeyDown(keyboard::Key::w))
			camera.position += front * move_speed;

		if (keyboard::isKeyDown(keyboard::Key::s))
			camera.position -= front * move_speed;

		if (keyboard::isKeyDown(keyboard::Key::d))
			camera.position += right * move_speed;

		if (keyboard::isKeyDown(keyboard::Key::a))
			camera.position -= right * move_speed;

		// Vertical: E up, Q down, also allow Z for compatibility
		if (keyboard::isKeyDown(keyboard::Key::e))
			camera.position += up * move_speed;

		if (keyboard::isKeyDown(keyboard::Key::q) || keyboard::isKeyDown(keyboard::Key::z))
			camera.position -= up * move_speed;
	}

	float aspect_ratio = float(veekay::app.window_width) / float(veekay::app.window_height);
	
	// Calculate light direction and view-projection matrix
	veekay::vec3 light_direction = veekay::vec3::normalized({0.5f, -1.0f, 0.3f});
	
	// Calculate light view matrix (looking at origin from light direction)
	veekay::vec3 light_pos = {-light_direction.x * 10.0f, -light_direction.y * 10.0f, -light_direction.z * 10.0f};
	veekay::vec3 light_target = {0.0f, 0.0f, 0.0f};
	veekay::vec3 light_up = {0.0f, 1.0f, 0.0f};
	
	veekay::vec3 light_forward = veekay::vec3::normalized(light_target - light_pos);
	veekay::vec3 light_right = veekay::vec3::normalized(veekay::vec3::cross(light_forward, light_up));
	light_up = veekay::vec3::cross(light_right, light_forward);
	
	veekay::mat4 light_view = veekay::mat4::identity();
	light_view[0][0] = light_right.x; light_view[1][0] = light_right.y; light_view[2][0] = light_right.z;
	light_view[0][1] = light_up.x; light_view[1][1] = light_up.y; light_view[2][1] = light_up.z;
	light_view[0][2] = -light_forward.x; light_view[1][2] = -light_forward.y; light_view[2][2] = -light_forward.z;
	light_view[3][0] = -veekay::vec3::dot(light_right, light_pos);
	light_view[3][1] = -veekay::vec3::dot(light_up, light_pos);
	light_view[3][2] = veekay::vec3::dot(light_forward, light_pos);
	
	// Orthographic projection for directional light
	float light_size = 15.0f;
	veekay::mat4 light_projection = veekay::mat4::orthographic(
		-light_size, light_size,
		-light_size, light_size,
		0.1f, 30.0f
	);
	
	veekay::mat4 light_view_projection = light_projection * light_view;
	
	SceneUniforms scene_uniforms{
		.view_projection = camera.view_projection(aspect_ratio),
		.light_view_projection = light_view_projection,
		.light_direction = light_direction,
		.camera_position = camera.position,
		.point_lights = {
			PointLight{
				.position = {2.5f, 2.5f, -2.0f, 1.0f},
				.color = {1.2f, 0.4f, 0.3f, 1.0f},
			},
			PointLight{
				.position = {-2.5f, 2.5f, 2.0f, 1.0f},
				.color = {0.3f, 0.5f, 1.2f, 1.0f},
			},
			PointLight{
				.position = {0.0f, 4.0f, 0.0f, 1.0f},
				.color = {0.8f, 0.8f, 0.8f, 1.0f},
			},
		},
		.point_light_count = 3,
	};

	std::vector<ModelUniforms> model_uniforms(models.size());
	for (size_t i = 0, n = models.size(); i < n; ++i) {
		const Model& model = models[i];
		ModelUniforms& uniforms = model_uniforms[i];

		// Для кубов применяем вращение вокруг Y
		if (model.mesh.indices == 36) {
			veekay::mat4 rotation = veekay::mat4::rotation({0.0f, 1.0f, 0.0f}, cube_angle);
			veekay::mat4 translation = veekay::mat4::translation(model.transform.position);
			veekay::mat4 scale = veekay::mat4::scaling(model.transform.scale);
			uniforms.model = translation * rotation * scale;
		} else {
			uniforms.model = model.transform.matrix();
		}
		uniforms.albedo_color = model.albedo_color;
		uniforms.texture_index = model.texture_index;
	}

	*(SceneUniforms*)scene_uniforms_buffer->mapped_region = scene_uniforms;

	const size_t alignment =
		veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));

	for (size_t i = 0, n = model_uniforms.size(); i < n; ++i) {
		const ModelUniforms& uniforms = model_uniforms[i];

		char* const pointer = static_cast<char*>(model_uniforms_buffer->mapped_region) + i * alignment;
		*reinterpret_cast<ModelUniforms*>(pointer) = uniforms;
	}
}

void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
	vkResetCommandBuffer(cmd, 0);

	{ // NOTE: Start recording rendering commands
		VkCommandBufferBeginInfo info{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		};

		vkBeginCommandBuffer(cmd, &info);
	}

	// NOTE: Shadow pass using Dynamic Rendering
	{
		static bool first_frame = true;
		
		// Transition shadow map to depth attachment layout
		VkImageMemoryBarrier to_depth_attachment{
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.srcAccessMask = first_frame ? static_cast<VkAccessFlags>(0) : VK_ACCESS_SHADER_READ_BIT,
			.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
			.oldLayout = first_frame ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
			.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = shadow_map_image,
			.subresourceRange = {
				.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1,
			},
		};

		vkCmdPipelineBarrier(cmd,
		                     first_frame ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
		                     VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
		                     0,
		                     0, nullptr,
		                     0, nullptr,
		                     1, &to_depth_attachment);
		
		first_frame = false;

		VkRenderingAttachmentInfo depth_attachment{
			.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
			.imageView = shadow_map_view,
			.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
			.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
			.clearValue = {.depthStencil = {1.0f, 0}},
		};

		VkRenderingInfo rendering_info{
			.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
			.renderArea = {
				.offset = {0, 0},
				.extent = {shadow_map_size, shadow_map_size},
			},
			.layerCount = 1,
			.colorAttachmentCount = 0,
			.pDepthAttachment = &depth_attachment,
		};

		vkCmdBeginRendering(cmd, &rendering_info);

		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline);
		
		VkDeviceSize zero_offset = 0;
		VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
		VkBuffer current_index_buffer = VK_NULL_HANDLE;
		const size_t model_uniorms_alignment =
			veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));

		// Update scene uniforms with light view-projection for shadow pass
		SceneUniforms* scene_uniforms = static_cast<SceneUniforms*>(scene_uniforms_buffer->mapped_region);
		veekay::mat4 light_view_proj = scene_uniforms->light_view_projection;
		
		// Temporarily set view_projection to light's view-projection
		scene_uniforms->view_projection = light_view_proj;

		for (size_t i = 0, n = models.size(); i < n; ++i) {
			const Model& model = models[i];
			const Mesh& mesh = model.mesh;

			if (current_vertex_buffer != mesh.vertex_buffer->buffer) {
				current_vertex_buffer = mesh.vertex_buffer->buffer;
				vkCmdBindVertexBuffers(cmd, 0, 1, &current_vertex_buffer, &zero_offset);
			}

			if (current_index_buffer != mesh.index_buffer->buffer) {
				current_index_buffer = mesh.index_buffer->buffer;
				vkCmdBindIndexBuffer(cmd, current_index_buffer, zero_offset, VK_INDEX_TYPE_UINT32);
			}

			uint32_t offset = i * model_uniorms_alignment;
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline_layout,
			                        0, 1, &descriptor_set, 1, &offset);

			vkCmdDrawIndexed(cmd, mesh.indices, 1, 0, 0, 0);
		}

		// Restore camera view-projection
		float aspect_ratio = float(veekay::app.window_width) / float(veekay::app.window_height);
		scene_uniforms->view_projection = camera.view_projection(aspect_ratio);

		vkCmdEndRendering(cmd);

		// Transition shadow map to shader read layout
		VkImageMemoryBarrier to_shader_read{
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
			.dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
			.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
			.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = shadow_map_image,
			.subresourceRange = {
				.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1,
			},
		};

		vkCmdPipelineBarrier(cmd,
		                     VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
		                     VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
		                     0,
		                     0, nullptr,
		                     0, nullptr,
		                     1, &to_shader_read);
	}

	{ // NOTE: Main render pass
		VkClearValue clear_color{.color = {{0.1f, 0.1f, 0.1f, 1.0f}}};
		VkClearValue clear_depth{.depthStencil = {1.0f, 0}};

		VkClearValue clear_values[] = {clear_color, clear_depth};

		VkRenderPassBeginInfo info{
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
			.renderPass = veekay::app.vk_render_pass,
			.framebuffer = framebuffer,
			.renderArea = {
				.extent = {
					veekay::app.window_width,
					veekay::app.window_height
				},
			},
			.clearValueCount = 2,
			.pClearValues = clear_values,
		};

		vkCmdBeginRenderPass(cmd, &info, VK_SUBPASS_CONTENTS_INLINE);
	}

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
	VkDeviceSize zero_offset = 0;

	VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
	VkBuffer current_index_buffer = VK_NULL_HANDLE;

	const size_t model_uniorms_alignment =
		veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));

	for (size_t i = 0, n = models.size(); i < n; ++i) {
		const Model& model = models[i];
		const Mesh& mesh = model.mesh;

		if (current_vertex_buffer != mesh.vertex_buffer->buffer) {
			current_vertex_buffer = mesh.vertex_buffer->buffer;
			vkCmdBindVertexBuffers(cmd, 0, 1, &current_vertex_buffer, &zero_offset);
		}

		if (current_index_buffer != mesh.index_buffer->buffer) {
			current_index_buffer = mesh.index_buffer->buffer;
			vkCmdBindIndexBuffer(cmd, current_index_buffer, zero_offset, VK_INDEX_TYPE_UINT32);
		}

		uint32_t offset = i * model_uniorms_alignment;
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
		                    0, 1, &descriptor_set, 1, &offset);

		vkCmdDrawIndexed(cmd, mesh.indices, 1, 0, 0, 0);
	}

	vkCmdEndRenderPass(cmd);
	vkEndCommandBuffer(cmd);
}

} // namespace

int main() {
	return veekay::run({
		.init = initialize,
		.shutdown = shutdown,
		.update = update,
		.render = render,
	});
}
