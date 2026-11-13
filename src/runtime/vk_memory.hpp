#pragma once

#include "vk_context.hpp"
#include <vulkan/vulkan.h>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace vultorch {

class VulkanBuffer {
public:
    std::shared_ptr<VulkanDeviceContext> device_ctx;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    void* mapped_data = nullptr;
    bool is_host_visible = false;

    VulkanBuffer(std::shared_ptr<VulkanDeviceContext> ctx, VkDeviceSize size_bytes,
                 VkBufferUsageFlags usage, VkMemoryPropertyFlags properties);
    ~VulkanBuffer();

    void* map();
    void unmap();
    void flush(VkDeviceSize offset = 0, VkDeviceSize size = VK_WHOLE_SIZE);
    void invalidate(VkDeviceSize offset = 0, VkDeviceSize size = VK_WHOLE_SIZE);
};

class VulkanMemoryManager {
private:
    std::mutex pool_mutex;
    // Free list pool binned by size (power of 2 or bucketed)
    std::unordered_map<uint32_t, std::unordered_map<VkDeviceSize, std::vector<std::shared_ptr<VulkanBuffer>>>> buffer_pool;

    VulkanMemoryManager() = default;
    ~VulkanMemoryManager() = default;

public:
    static VulkanMemoryManager& instance();

    std::shared_ptr<VulkanBuffer> allocate(uint32_t device_id, VkDeviceSize size_bytes,
                                          VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                          VkMemoryPropertyFlags properties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    std::shared_ptr<VulkanBuffer> create_staging_buffer(uint32_t device_id, VkDeviceSize size_bytes);

    void copy_host_to_device(uint32_t device_id, const void* host_ptr,
                             std::shared_ptr<VulkanBuffer> dst_buffer, VkDeviceSize size_bytes);

    void copy_device_to_host(uint32_t device_id, std::shared_ptr<VulkanBuffer> src_buffer,
                             void* host_ptr, VkDeviceSize size_bytes);

    void copy_buffer_to_buffer(uint32_t device_id, std::shared_ptr<VulkanBuffer> src,
                               std::shared_ptr<VulkanBuffer> dst, VkDeviceSize size_bytes);

    // Cross-GPU copy between vk:0 and vk:1
    void copy_cross_device(uint32_t src_device_id, std::shared_ptr<VulkanBuffer> src_buffer,
                           uint32_t dst_device_id, std::shared_ptr<VulkanBuffer> dst_buffer,
                           VkDeviceSize size_bytes);
};

} // namespace vultorch
