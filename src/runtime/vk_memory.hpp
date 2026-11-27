#pragma once

#include "vk_context.hpp"
#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>
#include <vulkan/vulkan.h>

namespace vultorch
{

class VulkanBuffer
{
  public:
    std::shared_ptr<VulkanDeviceContext> device_ctx;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    VkDeviceSize arena_offset = 0;
    VkDeviceSize arena_block_size = 0;
    uint32_t arena_index = 0;
    bool from_arena = false;
    uint64_t allocation_id = 0;
    void *mapped_data = nullptr;
    bool is_host_visible = false;
    // Recorded so the pool only hands a recycled buffer to a request with
    // identical usage and memory requirements.
    VkBufferUsageFlags usage = 0;
    VkMemoryPropertyFlags memory_properties = 0;

    // Standalone buffer owning its VkBuffer + VkDeviceMemory.
    VulkanBuffer(std::shared_ptr<VulkanDeviceContext> ctx, VkDeviceSize size_bytes,
                 VkBufferUsageFlags buffer_usage, VkMemoryPropertyFlags memory_flags);

    // Arena sub-allocation borrowing a slice [offset_bytes, offset_bytes + block_size_bytes).
    VulkanBuffer(std::shared_ptr<VulkanDeviceContext> ctx, VkBuffer shared_buffer,
                 VkDeviceSize logical_size, VkDeviceSize offset_bytes,
                 VkDeviceSize block_size_bytes, uint32_t arena_idx, VkBufferUsageFlags buffer_usage,
                 VkMemoryPropertyFlags memory_flags, uint64_t alloc_id);

    ~VulkanBuffer();

    VulkanBuffer(const VulkanBuffer &) = delete;
    VulkanBuffer &operator=(const VulkanBuffer &) = delete;

    void *map();
    void unmap();
    void flush(VkDeviceSize byte_offset = 0, VkDeviceSize byte_size = VK_WHOLE_SIZE);
    void invalidate(VkDeviceSize byte_offset = 0, VkDeviceSize byte_size = VK_WHOLE_SIZE);
};

// Two-level Arena Sub-Allocator + Caching Pool + Persistent Mapped Staging Ring.
class VulkanMemoryManager
{
  public:
    static VulkanMemoryManager &instance();

    std::shared_ptr<VulkanBuffer>
    allocate(uint32_t device_id, VkDeviceSize size_bytes,
             VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
             VkMemoryPropertyFlags properties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    std::shared_ptr<VulkanBuffer> create_staging_buffer(uint32_t device_id,
                                                        VkDeviceSize size_bytes);

    void copy_host_to_device(uint32_t device_id, const void *host_ptr,
                             std::shared_ptr<VulkanBuffer> dst_buffer, VkDeviceSize size_bytes);

    void copy_device_to_host(uint32_t device_id, std::shared_ptr<VulkanBuffer> src_buffer,
                             void *host_ptr, VkDeviceSize size_bytes);

    void copy_buffer_to_buffer(uint32_t device_id, std::shared_ptr<VulkanBuffer> src,
                               std::shared_ptr<VulkanBuffer> dst, VkDeviceSize size_bytes,
                               VkDeviceSize src_offset = 0, VkDeviceSize dst_offset = 0);

    // Cross-GPU copy between e.g. vk:0 and vk:1 (staged through the host).
    void copy_cross_device(uint32_t src_device_id, std::shared_ptr<VulkanBuffer> src_buffer,
                           uint32_t dst_device_id, std::shared_ptr<VulkanBuffer> dst_buffer,
                           VkDeviceSize size_bytes, VkDeviceSize src_offset = 0,
                           VkDeviceSize dst_offset = 0);

    // Diagnostics and tests.
    size_t pooled_buffer_count() const;
    size_t live_allocation_count() const
    {
        return live_allocations.load();
    }
    size_t driver_allocation_count(int device_id = -1) const;
    size_t active_arena_bytes(int device_id = -1) const;
    size_t reserved_arena_bytes(int device_id = -1) const;

    // Destroys pooled buffers, arenas, and staging rings. Must run before the logical devices go
    // away; afterwards recycle() frees immediately instead of pooling.
    void shutdown();

  private:
    struct PoolKey
    {
        VkDeviceSize bucket = 0;
        VkBufferUsageFlags usage = 0;
        VkMemoryPropertyFlags properties = 0;

        bool operator<(const PoolKey &other) const
        {
            if (bucket != other.bucket)
                return bucket < other.bucket;
            if (usage != other.usage)
                return usage < other.usage;
            return properties < other.properties;
        }
    };

    struct ArenaBlock
    {
        std::shared_ptr<VulkanDeviceContext> device_ctx;
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize total_size = 0;
        VkDeviceSize used_bytes = 0;
        VkBufferUsageFlags usage = 0;
        VkMemoryPropertyFlags properties = 0;
        std::map<VkDeviceSize, VkDeviceSize> free_by_offset;
        std::set<std::pair<VkDeviceSize, VkDeviceSize>> free_by_size;

        ArenaBlock() = default;
        ~ArenaBlock();
        ArenaBlock(const ArenaBlock &) = delete;
        ArenaBlock &operator=(const ArenaBlock &) = delete;

        bool allocate_slice(VkDeviceSize aligned_size, VkDeviceSize &out_offset);
        void free_slice(VkDeviceSize slice_offset, VkDeviceSize slice_size);
    };

    struct StagingRing
    {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize capacity = 0;
        VkDeviceSize head = 0;
        void *mapped = nullptr;
        std::shared_ptr<VulkanDeviceContext> device_ctx;

        StagingRing() = default;
        ~StagingRing();
        StagingRing(const StagingRing &) = delete;
        StagingRing &operator=(const StagingRing &) = delete;
    };

    struct DevicePool
    {
        std::map<PoolKey, std::vector<std::unique_ptr<VulkanBuffer>>> free_lists;
        VkDeviceSize free_bytes = 0;
        std::vector<std::unique_ptr<ArenaBlock>> arenas;
        std::unique_ptr<StagingRing> staging_ring;
        std::mutex staging_mutex;
        size_t driver_alloc_count = 0;
    };

    mutable std::mutex pool_mutex;
    std::unordered_map<uint32_t, std::unique_ptr<DevicePool>> pools;
    std::atomic<size_t> live_allocations{0};
    std::atomic<uint64_t> next_alloc_id{1};
    bool shutting_down = false;

    VulkanMemoryManager() = default;
    ~VulkanMemoryManager() = default;

    DevicePool &get_or_create_pool_locked(uint32_t device_id);
    StagingRing &ensure_staging_ring(uint32_t device_id, DevicePool &pool);
    std::unique_ptr<ArenaBlock> create_arena_block(std::shared_ptr<VulkanDeviceContext> ctx,
                                                   VkDeviceSize requested_bytes,
                                                   VkBufferUsageFlags usage,
                                                   VkMemoryPropertyFlags properties);
    void recycle(VulkanBuffer *buffer);
    static VkDeviceSize bucket_size(VkDeviceSize size_bytes);
};

} // namespace vultorch
