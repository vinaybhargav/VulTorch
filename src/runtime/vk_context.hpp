#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <string>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <iostream>

namespace vultorch {

struct CooperativeMatrixConfig {
    uint32_t M;
    uint32_t N;
    uint32_t K;
    uint32_t a_type;
    uint32_t b_type;
    uint32_t c_type;
    uint32_t result_type;
    uint32_t scope;
};

struct DeviceInfo {
    uint32_t id;
    std::string name;
    std::string vendor;
    uint32_t api_major;
    uint32_t api_minor;
    VkDeviceSize total_memory;
    bool is_discrete;
    bool has_cooperative_matrix = false;
    std::vector<CooperativeMatrixConfig> cooperative_matrix_configs;
};

class VulkanDeviceContext {
public:
    uint32_t device_id;
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue compute_queue = VK_NULL_HANDLE;
    uint32_t compute_queue_family = 0;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties properties{};
    VkPhysicalDeviceMemoryProperties memory_properties{};
    bool has_cooperative_matrix = false;
    std::vector<CooperativeMatrixConfig> cooperative_matrix_configs;

    VulkanDeviceContext(uint32_t id, VkPhysicalDevice pdev);
    ~VulkanDeviceContext();

    uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties) const;
    VkCommandBuffer begin_single_time_commands();
    void end_single_time_commands(VkCommandBuffer command_buffer);
    void wait_idle();
};

class VulkanManager {
private:
    VkInstance instance = VK_NULL_HANDLE;
    std::vector<VkPhysicalDevice> physical_devices;
    std::vector<std::shared_ptr<VulkanDeviceContext>> active_devices;
    std::mutex manager_mutex;

    VulkanManager();
    ~VulkanManager();

public:
    static VulkanManager& instance_ref();

    VkInstance get_instance() const { return instance; }
    uint32_t get_device_count();
    DeviceInfo get_device_info(uint32_t device_id);
    std::vector<DeviceInfo> get_all_device_info();
    std::shared_ptr<VulkanDeviceContext> get_device(uint32_t device_id);
    void synchronize_device(uint32_t device_id);
    void synchronize_all();
};

} // namespace vultorch
