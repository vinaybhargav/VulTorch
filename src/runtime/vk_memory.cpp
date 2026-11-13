#include "vk_memory.hpp"
#include <cstring>
#include <stdexcept>
#include <iostream>

namespace vultorch {

VulkanBuffer::VulkanBuffer(std::shared_ptr<VulkanDeviceContext> ctx, VkDeviceSize size_bytes,
                           VkBufferUsageFlags usage, VkMemoryPropertyFlags properties)
    : device_ctx(ctx), size(size_bytes) {
    if (size == 0) size = 4; // Vulkan requires non-zero buffer sizes

    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = usage;
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
    alloc_info.memoryTypeIndex = device_ctx->find_memory_type(mem_reqs.memoryTypeBits, properties);

    res = vkAllocateMemory(device_ctx->device, &alloc_info, nullptr, &memory);
    if (res != VK_SUCCESS) {
        vkDestroyBuffer(device_ctx->device, buffer, nullptr);
        buffer = VK_NULL_HANDLE;
        throw std::runtime_error("vkAllocateMemory failed: code " + std::to_string(res));
    }

    vkBindBufferMemory(device_ctx->device, buffer, memory, 0);

    is_host_visible = (properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
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
    static VulkanMemoryManager mgr;
    return mgr;
}

std::shared_ptr<VulkanBuffer> VulkanMemoryManager::allocate(uint32_t device_id, VkDeviceSize size_bytes,
                                                           VkBufferUsageFlags usage,
                                                           VkMemoryPropertyFlags properties) {
    auto dev = VulkanManager::instance_ref().get_device(device_id);
    return std::make_shared<VulkanBuffer>(dev, size_bytes, usage, properties);
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

    VkCommandBuffer cmd = dev->begin_single_time_commands();
    VkBufferCopy copy_region{};
    copy_region.srcOffset = 0;
    copy_region.dstOffset = 0;
    copy_region.size = size_bytes;
    vkCmdCopyBuffer(cmd, staging->buffer, dst_buffer->buffer, 1, &copy_region);
    dev->end_single_time_commands(cmd);
}

void VulkanMemoryManager::copy_device_to_host(uint32_t device_id, std::shared_ptr<VulkanBuffer> src_buffer,
                                              void* host_ptr, VkDeviceSize size_bytes) {
    auto dev = VulkanManager::instance_ref().get_device(device_id);
    auto staging = create_staging_buffer(device_id, size_bytes);

    VkCommandBuffer cmd = dev->begin_single_time_commands();
    VkBufferCopy copy_region{};
    copy_region.srcOffset = 0;
    copy_region.dstOffset = 0;
    copy_region.size = size_bytes;
    vkCmdCopyBuffer(cmd, src_buffer->buffer, staging->buffer, 1, &copy_region);
    dev->end_single_time_commands(cmd);

    void* mapped = staging->map();
    staging->invalidate();
    std::memcpy(host_ptr, mapped, size_bytes);
}

void VulkanMemoryManager::copy_buffer_to_buffer(uint32_t device_id, std::shared_ptr<VulkanBuffer> src,
                                                std::shared_ptr<VulkanBuffer> dst, VkDeviceSize size_bytes) {
    auto dev = VulkanManager::instance_ref().get_device(device_id);
    VkCommandBuffer cmd = dev->begin_single_time_commands();
    VkBufferCopy copy_region{};
    copy_region.srcOffset = 0;
    copy_region.dstOffset = 0;
    copy_region.size = size_bytes;
    vkCmdCopyBuffer(cmd, src->buffer, dst->buffer, 1, &copy_region);
    dev->end_single_time_commands(cmd);
}

void VulkanMemoryManager::copy_cross_device(uint32_t src_device_id, std::shared_ptr<VulkanBuffer> src_buffer,
                                           uint32_t dst_device_id, std::shared_ptr<VulkanBuffer> dst_buffer,
                                           VkDeviceSize size_bytes) {
    // Stage from src device to host, then host to dst device
    std::vector<uint8_t> staging_host(size_bytes);
    copy_device_to_host(src_device_id, src_buffer, staging_host.data(), size_bytes);
    copy_host_to_device(dst_device_id, staging_host.data(), dst_buffer, size_bytes);
}

} // namespace vultorch
