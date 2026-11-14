#include "vk_memory.hpp"
#include <cstring>
#include <stdexcept>
#include <iostream>

namespace vultorch {

VulkanBuffer::VulkanBuffer(std::shared_ptr<VulkanDeviceContext> ctx, VkDeviceSize size_bytes,
                           VkBufferUsageFlags buffer_usage, VkMemoryPropertyFlags memory_flags)
    : device_ctx(ctx), size(size_bytes), usage(buffer_usage), memory_properties(memory_flags) {
    if (size == 0) size = 4; // Vulkan requires non-zero buffer sizes

    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = buffer_usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult res = vkCreateBuffer(device_ctx->device, &buffer_info, nullptr, &buffer);
    if (res != VK_SUCCESS) {
        throw std::runtime_error("vkCreateBuffer failed: code " + std::to_string(res));
    }

    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(device_ctx->device, buffer, &mem_reqs);

    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_reqs.size;
    alloc_info.memoryTypeIndex = device_ctx->find_memory_type(mem_reqs.memoryTypeBits, memory_flags);

    res = vkAllocateMemory(device_ctx->device, &alloc_info, nullptr, &memory);
    if (res != VK_SUCCESS) {
        vkDestroyBuffer(device_ctx->device, buffer, nullptr);
        buffer = VK_NULL_HANDLE;
        throw std::runtime_error("vkAllocateMemory failed: code " + std::to_string(res));
    }

    vkBindBufferMemory(device_ctx->device, buffer, memory, 0);

    is_host_visible = (memory_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
}

VulkanBuffer::~VulkanBuffer() {
    if (device_ctx && device_ctx->device != VK_NULL_HANDLE) {
        if (mapped_data != nullptr) {
            vkUnmapMemory(device_ctx->device, memory);
            mapped_data = nullptr;
        }
        if (buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device_ctx->device, buffer, nullptr);
            buffer = VK_NULL_HANDLE;
        }
        if (memory != VK_NULL_HANDLE) {
            vkFreeMemory(device_ctx->device, memory, nullptr);
            memory = VK_NULL_HANDLE;
        }
    }
}

void* VulkanBuffer::map() {
    if (!is_host_visible) {
        throw std::runtime_error("Cannot map non-host-visible Vulkan buffer!");
    }
    if (mapped_data == nullptr) {
        VkResult res = vkMapMemory(device_ctx->device, memory, 0, size, 0, &mapped_data);
        if (res != VK_SUCCESS) {
            throw std::runtime_error("vkMapMemory failed: code " + std::to_string(res));
        }
    }
    return mapped_data;
}

void VulkanBuffer::unmap() {
    if (mapped_data != nullptr) {
        vkUnmapMemory(device_ctx->device, memory);
        mapped_data = nullptr;
    }
}

void VulkanBuffer::flush(VkDeviceSize offset, VkDeviceSize flush_size) {
    if (is_host_visible) {
        VkMappedMemoryRange range{};
        range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        range.memory = memory;
        range.offset = offset;
        range.size = flush_size;
        vkFlushMappedMemoryRanges(device_ctx->device, 1, &range);
    }
}

void VulkanBuffer::invalidate(VkDeviceSize offset, VkDeviceSize inv_size) {
    if (is_host_visible) {
        VkMappedMemoryRange range{};
        range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        range.memory = memory;
        range.offset = offset;
        range.size = inv_size;
        vkInvalidateMappedMemoryRanges(device_ctx->device, 1, &range);
    }
}

VulkanMemoryManager& VulkanMemoryManager::instance() {
    // Deliberately leaked: pooled buffers hold shared_ptr<VulkanDeviceContext>,
    // and a static destructor could otherwise run after the Vulkan objects it
    // owns. shutdown() provides deterministic cleanup instead.
    static VulkanMemoryManager* mgr = new VulkanMemoryManager();
    return *mgr;
}

VkDeviceSize VulkanMemoryManager::bucket_size(VkDeviceSize size_bytes) {
    // 4 KiB granularity: repeated tensor shapes in a training loop reuse the
    // same bucket, while waste stays bounded for large allocations.
    constexpr VkDeviceSize kGranularity = 4096;
    if (size_bytes == 0) return kGranularity;
    return ((size_bytes + kGranularity - 1) / kGranularity) * kGranularity;
}

std::shared_ptr<VulkanBuffer> VulkanMemoryManager::allocate(uint32_t device_id, VkDeviceSize size_bytes,
                                                           VkBufferUsageFlags usage,
                                                           VkMemoryPropertyFlags properties) {
    const VkDeviceSize bucket = bucket_size(size_bytes);
    const PoolKey key{bucket, usage, properties};

    {
        std::lock_guard<std::mutex> lock(pool_mutex);
        auto dev_it = pools.find(device_id);
        if (dev_it != pools.end()) {
            auto list_it = dev_it->second.free_lists.find(key);
            if (list_it != dev_it->second.free_lists.end() && !list_it->second.empty()) {
                std::unique_ptr<VulkanBuffer> recycled = std::move(list_it->second.back());
                list_it->second.pop_back();
                dev_it->second.free_bytes -= recycled->size;
                live_allocations.fetch_add(1, std::memory_order_relaxed);
                return std::shared_ptr<VulkanBuffer>(recycled.release(), [](VulkanBuffer* buffer) {
                    VulkanMemoryManager::instance().recycle(buffer);
                });
            }
        }
    }

    auto dev = VulkanManager::instance_ref().get_device(device_id);
    live_allocations.fetch_add(1, std::memory_order_relaxed);
    return std::shared_ptr<VulkanBuffer>(new VulkanBuffer(dev, bucket, usage, properties),
                                         [](VulkanBuffer* buffer) {
                                             VulkanMemoryManager::instance().recycle(buffer);
                                         });
}

void VulkanMemoryManager::recycle(VulkanBuffer* raw) {
    // Adopt the buffer so every early return still destroys it.
    std::unique_ptr<VulkanBuffer> owned(raw);
    live_allocations.fetch_sub(1, std::memory_order_relaxed);

    // Cap the cache per device so a transient large allocation cannot pin
    // memory forever.
    constexpr VkDeviceSize kFreeListBudget = 1024ull * 1024ull * 1024ull;

    std::lock_guard<std::mutex> lock(pool_mutex);
    if (shutting_down) return;

    DevicePool& pool = pools[raw->device_ctx->device_id];
    if (pool.free_bytes + raw->size > kFreeListBudget) return;

    const PoolKey key{raw->size, raw->usage, raw->memory_properties};
    pool.free_bytes += raw->size;
    pool.free_lists[key].push_back(std::move(owned));
}

size_t VulkanMemoryManager::pooled_buffer_count() const {
    std::lock_guard<std::mutex> lock(pool_mutex);
    size_t count = 0;
    for (const auto& device_entry : pools) {
        for (const auto& list_entry : device_entry.second.free_lists) {
            count += list_entry.second.size();
        }
    }
    return count;
}

void VulkanMemoryManager::shutdown() {
    std::vector<std::unique_ptr<VulkanBuffer>> doomed;
    {
        std::lock_guard<std::mutex> lock(pool_mutex);
        shutting_down = true;
        for (auto& device_entry : pools) {
            for (auto& list_entry : device_entry.second.free_lists) {
                for (auto& buffer : list_entry.second) {
                    doomed.push_back(std::move(buffer));
                }
                list_entry.second.clear();
            }
            device_entry.second.free_lists.clear();
            device_entry.second.free_bytes = 0;
        }
        pools.clear();
    }
    // Destroy outside the lock: vkDestroyBuffer may block.
    doomed.clear();
}

std::shared_ptr<VulkanBuffer> VulkanMemoryManager::create_staging_buffer(uint32_t device_id, VkDeviceSize size_bytes) {
    return allocate(device_id, size_bytes,
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
}

void VulkanMemoryManager::copy_host_to_device(uint32_t device_id, const void* host_ptr,
                                              std::shared_ptr<VulkanBuffer> dst_buffer, VkDeviceSize size_bytes) {
    auto dev = VulkanManager::instance_ref().get_device(device_id);
    auto staging = create_staging_buffer(device_id, size_bytes);
    void* mapped = staging->map();
    std::memcpy(mapped, host_ptr, size_bytes);
    staging->flush();

    const VkBuffer staging_handle = staging->buffer;
    const VkBuffer dst_handle = dst_buffer->buffer;
    dev->submit_transfer([&](VkCommandBuffer cmd) {
        VkBufferCopy copy_region{};
        copy_region.srcOffset = 0;
        copy_region.dstOffset = 0;
        copy_region.size = size_bytes;
        vkCmdCopyBuffer(cmd, staging_handle, dst_handle, 1, &copy_region);
    });
}

void VulkanMemoryManager::copy_device_to_host(uint32_t device_id, std::shared_ptr<VulkanBuffer> src_buffer,
                                              void* host_ptr, VkDeviceSize size_bytes) {
    auto dev = VulkanManager::instance_ref().get_device(device_id);
    auto staging = create_staging_buffer(device_id, size_bytes);

    const VkBuffer src_handle = src_buffer->buffer;
    const VkBuffer staging_handle = staging->buffer;
    dev->submit_transfer([&](VkCommandBuffer cmd) {
        VkBufferCopy copy_region{};
        copy_region.srcOffset = 0;
        copy_region.dstOffset = 0;
        copy_region.size = size_bytes;
        vkCmdCopyBuffer(cmd, src_handle, staging_handle, 1, &copy_region);
    });

    void* mapped = staging->map();
    staging->invalidate();
    std::memcpy(host_ptr, mapped, size_bytes);
}

void VulkanMemoryManager::copy_buffer_to_buffer(uint32_t device_id, std::shared_ptr<VulkanBuffer> src,
                                                std::shared_ptr<VulkanBuffer> dst, VkDeviceSize size_bytes,
                                                VkDeviceSize src_offset, VkDeviceSize dst_offset) {
    auto dev = VulkanManager::instance_ref().get_device(device_id);
    dev->record_copy(src->buffer, dst->buffer, size_bytes, src_offset, dst_offset);
}

void VulkanMemoryManager::copy_cross_device(uint32_t src_device_id, std::shared_ptr<VulkanBuffer> src_buffer,
                                           uint32_t dst_device_id, std::shared_ptr<VulkanBuffer> dst_buffer,
                                           VkDeviceSize size_bytes,
                                           VkDeviceSize src_offset, VkDeviceSize dst_offset) {
    if (size_bytes == 0) return;
    if (src_device_id == dst_device_id) {
        copy_buffer_to_buffer(src_device_id, src_buffer, dst_buffer, size_bytes, src_offset, dst_offset);
        return;
    }

    auto src_dev = VulkanManager::instance_ref().get_device(src_device_id);
    auto dst_dev = VulkanManager::instance_ref().get_device(dst_device_id);
    auto src_staging = create_staging_buffer(src_device_id, size_bytes);
    auto dst_staging = create_staging_buffer(dst_device_id, size_bytes);

    const VkBuffer src_handle = src_buffer->buffer;
    const VkBuffer src_stage_handle = src_staging->buffer;
    src_dev->submit_transfer([&](VkCommandBuffer cmd) {
        VkBufferCopy copy_region{};
        copy_region.srcOffset = src_offset;
        copy_region.dstOffset = 0;
        copy_region.size = size_bytes;
        vkCmdCopyBuffer(cmd, src_handle, src_stage_handle, 1, &copy_region);
    });

    void* src_mapped = src_staging->map();
    src_staging->invalidate();
    void* dst_mapped = dst_staging->map();
    std::memcpy(dst_mapped, src_mapped, size_bytes);
    dst_staging->flush();

    const VkBuffer dst_stage_handle = dst_staging->buffer;
    const VkBuffer dst_handle = dst_buffer->buffer;
    dst_dev->submit_transfer([&](VkCommandBuffer cmd) {
        VkBufferCopy copy_region{};
        copy_region.srcOffset = 0;
        copy_region.dstOffset = dst_offset;
        copy_region.size = size_bytes;
        vkCmdCopyBuffer(cmd, dst_stage_handle, dst_handle, 1, &copy_region);
    });
}

} // namespace vultorch
