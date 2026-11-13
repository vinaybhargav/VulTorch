#pragma once

#include "vk_context.hpp"
#include "vk_memory.hpp"
#include <vulkan/vulkan.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>

namespace vultorch {

struct PipelineRecord {
    VkDevice device = VK_NULL_HANDLE;
    VkShaderModule shader_module = VK_NULL_HANDLE;
    VkDescriptorSetLayout desc_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline compute_pipeline = VK_NULL_HANDLE;
    VkDescriptorPool desc_pool = VK_NULL_HANDLE;
    uint32_t num_buffers = 0;
};

class VulkanPipelineManager {
private:
    std::string shader_dir;
    std::mutex pipeline_mutex;
    // Map: device_id -> (shader_name -> PipelineRecord)
    std::unordered_map<uint32_t, std::unordered_map<std::string, PipelineRecord>> cache;

    VulkanPipelineManager();
    ~VulkanPipelineManager();

    std::vector<uint32_t> read_spirv_file(const std::string& filepath);

public:
    void clear();
    static VulkanPipelineManager& instance();

    void set_shader_directory(const std::string& dir);
    std::string get_shader_directory() const;

    PipelineRecord& get_or_create_pipeline(uint32_t device_id, const std::string& shader_name,
                                          uint32_t num_storage_buffers, size_t push_constant_size);

    void dispatch(uint32_t device_id, const std::string& shader_name,
                  const std::vector<std::shared_ptr<VulkanBuffer>>& buffers,
                  const void* push_constants, size_t push_constants_size,
                  uint32_t group_count_x, uint32_t group_count_y = 1, uint32_t group_count_z = 1);
};

} // namespace vultorch
