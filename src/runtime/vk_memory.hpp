#pragma once

#include "vk_context.hpp"
#include <vulkan/vulkan.h>
#include <atomic>
#include <map>
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
    // Recorded so the pool only hands a recycled buffer to a request with
    // identical usage and memory requirements.
    VkBufferUsageFlags usage = 0;
    VkMemoryPropertyFlags memory_properties = 0;

    VulkanBuffer(std::shared_ptr<VulkanDeviceContext> ctx, VkDeviceSize size_bytes,
                 VkBufferUsageFlags buffer_usage, VkMemoryPropertyFlags memory_flags);
    ~VulkanBuffer();

    VulkanBuffer(const VulkanBuffer&) = delete;
    VulkanBuffer& operator=(const VulkanBuffer&) = delete;

    void* map();
    void unmap();
    void flush(VkDeviceSize offset = 0, VkDeviceSize size = VK_WHOLE_SIZE);
    void invalidate(VkDeviceSize offset = 0, VkDeviceSize size = VK_WHOLE_SIZE);
};

// Caching allocator.
//
// Buffers come from per-device free lists and return to the pool when the last
// reference drops, instead of being created and destroyed per tensor. Requests
// are rounded to 4 KiB buckets, so the repeated shapes of a training loop reuse
// the same allocations. This also keeps the process well under the driver's
// maxMemoryAllocationCount (typically 4096).
//
// Reuse safety: every write into a device-local buffer is GPU-ordered (uploads
// go through staging), and every host read/write flushes and waits first, so a
// recycled buffer cannot be written by the CPU while the GPU still reads it.
class VulkanMemoryManager {
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

    // Cross-GPU copy between e.g. vk:0 and vk:1 (staged through the host).
    void copy_cross_device(uint32_t src_device_id, std::shared_ptr<VulkanBuffer> src_buffer,
                           uint32_t dst_device_id, std::shared_ptr<VulkanBuffer> dst_buffer,
                           VkDeviceSize size_bytes);

    // Diagnostics and tests.
    size_t pooled_buffer_count() const;
    size_t live_allocation_count() const { return live_allocations.load(); }

    // Destroys pooled buffers. Must run before the logical devices go away;
    // afterwards recycle() frees immediately instead of pooling.
    void shutdown();

private:
    struct PoolKey {
        VkDeviceSize bucket = 0;
        VkBufferUsageFlags usage = 0;
        VkMemoryPropertyFlags properties = 0;

        bool operator<(const PoolKey& other) const {
            if (bucket != other.bucket) return bucket < other.bucket;
            if (usage != other.usage) return usage < other.usage;
            return properties < other.properties;
        }
    };

    struct DevicePool {
        std::map<PoolKey, std::vector<std::unique_ptr<VulkanBuffer>>> free_lists;
        VkDeviceSize free_bytes = 0;
    };

    mutable std::mutex pool_mutex;
    std::unordered_map<uint32_t, DevicePool> pools;
    std::atomic<size_t> live_allocations{0};
    bool shutting_down = false;

    VulkanMemoryManager() = default;
    ~VulkanMemoryManager() = default;

    void recycle(VulkanBuffer* buffer);
    static VkDeviceSize bucket_size(VkDeviceSize size_bytes);
};

} // namespace vultorch
