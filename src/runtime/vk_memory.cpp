#include "vk_memory.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace vultorch
{

namespace
{

constexpr VkDeviceSize kStagingRingBytes = 16ULL * 1024ULL * 1024ULL;
constexpr VkDeviceSize kDefaultArenaSlabSmall = 64ULL * 1024ULL * 1024ULL;
constexpr VkDeviceSize kArenaMinAlign = 256ULL;

bool has_memory_type(const VulkanDeviceContext &ctx, uint32_t type_filter,
                     VkMemoryPropertyFlags required_flags, uint32_t &out_index)
{
    for (uint32_t i = 0; i < ctx.memory_properties.memoryTypeCount; ++i)
    {
        if ((type_filter & (1u << i)) != 0u &&
            (ctx.memory_properties.memoryTypes[i].propertyFlags & required_flags) == required_flags)
        {
            out_index = i;
            return true;
        }
    }
    return false;
}

VkDeviceSize align_up(VkDeviceSize value, VkDeviceSize alignment)
{
    return (value + alignment - 1ULL) & ~(alignment - 1ULL);
}

} // namespace

// ---------------------------------------------------------------------------
// VulkanBuffer
// ---------------------------------------------------------------------------

VulkanBuffer::VulkanBuffer(std::shared_ptr<VulkanDeviceContext> ctx, VkDeviceSize size_bytes,
                           VkBufferUsageFlags buffer_usage, VkMemoryPropertyFlags memory_flags)
    : device_ctx(std::move(ctx)), size(size_bytes), arena_offset(0), arena_block_size(size_bytes),
      arena_index(0), from_arena(false), usage(buffer_usage), memory_properties(memory_flags)
{
    is_host_visible = (memory_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;

    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size_bytes;
    buffer_info.usage = buffer_usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device_ctx->device, &buffer_info, nullptr, &buffer) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create Vulkan buffer of size " +
                                 std::to_string(size_bytes));
    }

    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(device_ctx->device, buffer, &mem_reqs);

    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_reqs.size;
    alloc_info.memoryTypeIndex =
        device_ctx->find_memory_type(mem_reqs.memoryTypeBits, memory_flags);

    if (vkAllocateMemory(device_ctx->device, &alloc_info, nullptr, &memory) != VK_SUCCESS)
    {
        vkDestroyBuffer(device_ctx->device, buffer, nullptr);
        buffer = VK_NULL_HANDLE;
        throw std::runtime_error("Failed to allocate Vulkan buffer memory of size " +
                                 std::to_string(mem_reqs.size));
    }

    if (vkBindBufferMemory(device_ctx->device, buffer, memory, 0) != VK_SUCCESS)
    {
        vkDestroyBuffer(device_ctx->device, buffer, nullptr);
        vkFreeMemory(device_ctx->device, memory, nullptr);
        buffer = VK_NULL_HANDLE;
        memory = VK_NULL_HANDLE;
        throw std::runtime_error("Failed to bind Vulkan buffer memory!");
    }
}

VulkanBuffer::VulkanBuffer(std::shared_ptr<VulkanDeviceContext> ctx, VkBuffer shared_buffer,
                           VkDeviceSize logical_size, VkDeviceSize offset_bytes,
                           VkDeviceSize block_size_bytes, uint32_t arena_idx,
                           VkBufferUsageFlags buffer_usage, VkMemoryPropertyFlags memory_flags,
                           uint64_t alloc_id)
    : device_ctx(std::move(ctx)), buffer(shared_buffer), memory(VK_NULL_HANDLE), size(logical_size),
      arena_offset(offset_bytes), arena_block_size(block_size_bytes), arena_index(arena_idx),
      from_arena(true), allocation_id(alloc_id), usage(buffer_usage),
      memory_properties(memory_flags)
{
    is_host_visible = (memory_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
}

VulkanBuffer::~VulkanBuffer()
{
    if (from_arena)
    {
        // The underlying VkBuffer and VkDeviceMemory belong to ArenaBlock.
        buffer = VK_NULL_HANDLE;
        memory = VK_NULL_HANDLE;
        return;
    }
    if (device_ctx && device_ctx->device != VK_NULL_HANDLE)
    {
        if (mapped_data != nullptr && memory != VK_NULL_HANDLE)
        {
            unmap();
        }
        if (buffer != VK_NULL_HANDLE)
        {
            vkDestroyBuffer(device_ctx->device, buffer, nullptr);
            buffer = VK_NULL_HANDLE;
        }
        if (memory != VK_NULL_HANDLE)
        {
            vkFreeMemory(device_ctx->device, memory, nullptr);
            memory = VK_NULL_HANDLE;
        }
    }
}

void *VulkanBuffer::map()
{
    if (from_arena)
    {
        throw std::runtime_error("Cannot map device-local arena sub-allocation directly!");
    }
    if (!is_host_visible)
    {
        throw std::runtime_error("Cannot map a non-host-visible buffer!");
    }
    if (!mapped_data)
    {
        if (vkMapMemory(device_ctx->device, memory, 0, size, 0, &mapped_data) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to map Vulkan buffer memory!");
        }
    }
    return mapped_data;
}

void VulkanBuffer::unmap()
{
    if (mapped_data)
    {
        vkUnmapMemory(device_ctx->device, memory);
        mapped_data = nullptr;
    }
}

void VulkanBuffer::flush(VkDeviceSize byte_offset, VkDeviceSize byte_size)
{
    if (from_arena || (memory_properties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0)
    {
        return;
    }
    VkMappedMemoryRange range{};
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = memory;
    range.offset = byte_offset;
    range.size = (byte_size == VK_WHOLE_SIZE) ? size : byte_size;
    vkFlushMappedMemoryRanges(device_ctx->device, 1, &range);
}

void VulkanBuffer::invalidate(VkDeviceSize byte_offset, VkDeviceSize byte_size)
{
    if (from_arena || (memory_properties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0)
    {
        return;
    }
    VkMappedMemoryRange range{};
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = memory;
    range.offset = byte_offset;
    range.size = (byte_size == VK_WHOLE_SIZE) ? size : byte_size;
    vkInvalidateMappedMemoryRanges(device_ctx->device, 1, &range);
}

// ---------------------------------------------------------------------------
// ArenaBlock: O(log N) best-fit free list with immediate neighbor coalescing
// ---------------------------------------------------------------------------

VulkanMemoryManager::ArenaBlock::~ArenaBlock()
{
    if (device_ctx && device_ctx->device != VK_NULL_HANDLE)
    {
        if (buffer != VK_NULL_HANDLE)
        {
            vkDestroyBuffer(device_ctx->device, buffer, nullptr);
            buffer = VK_NULL_HANDLE;
        }
        if (memory != VK_NULL_HANDLE)
        {
            vkFreeMemory(device_ctx->device, memory, nullptr);
            memory = VK_NULL_HANDLE;
        }
    }
}

bool VulkanMemoryManager::ArenaBlock::allocate_slice(VkDeviceSize aligned_size,
                                                     VkDeviceSize &out_offset)
{
    auto it = free_by_size.lower_bound({aligned_size, 0});
    if (it == free_by_size.end())
    {
        return false;
    }
    VkDeviceSize span_size = it->first;
    VkDeviceSize span_offset = it->second;
    free_by_size.erase(it);
    free_by_offset.erase(span_offset);

    out_offset = span_offset;
    if (span_size > aligned_size)
    {
        VkDeviceSize rem_offset = span_offset + aligned_size;
        VkDeviceSize rem_size = span_size - aligned_size;
        free_by_offset[rem_offset] = rem_size;
        free_by_size.insert({rem_size, rem_offset});
    }
    used_bytes += aligned_size;
    return true;
}

void VulkanMemoryManager::ArenaBlock::free_slice(VkDeviceSize slice_offset, VkDeviceSize slice_size)
{
    used_bytes -= slice_size;

    VkDeviceSize merged_offset = slice_offset;
    VkDeviceSize merged_size = slice_size;

    auto next_it = free_by_offset.lower_bound(merged_offset);
    if (next_it != free_by_offset.begin())
    {
        auto prev_it = std::prev(next_it);
        if (prev_it->first + prev_it->second == merged_offset)
        {
            free_by_size.erase({prev_it->second, prev_it->first});
            merged_offset = prev_it->first;
            merged_size += prev_it->second;
            free_by_offset.erase(prev_it);
        }
    }
    if (next_it != free_by_offset.end())
    {
        if (merged_offset + merged_size == next_it->first)
        {
            free_by_size.erase({next_it->second, next_it->first});
            merged_size += next_it->second;
            free_by_offset.erase(next_it);
        }
    }

    free_by_offset[merged_offset] = merged_size;
    free_by_size.insert({merged_size, merged_offset});
}

// ---------------------------------------------------------------------------
// StagingRing: Persistent host-mapped ring buffer
// ---------------------------------------------------------------------------

VulkanMemoryManager::StagingRing::~StagingRing()
{
    if (device_ctx && device_ctx->device != VK_NULL_HANDLE)
    {
        if (mapped != nullptr && memory != VK_NULL_HANDLE)
        {
            vkUnmapMemory(device_ctx->device, memory);
            mapped = nullptr;
        }
        if (buffer != VK_NULL_HANDLE)
        {
            vkDestroyBuffer(device_ctx->device, buffer, nullptr);
            buffer = VK_NULL_HANDLE;
        }
        if (memory != VK_NULL_HANDLE)
        {
            vkFreeMemory(device_ctx->device, memory, nullptr);
            memory = VK_NULL_HANDLE;
        }
    }
}

// ---------------------------------------------------------------------------
// VulkanMemoryManager
// ---------------------------------------------------------------------------

VulkanMemoryManager &VulkanMemoryManager::instance()
{
    static VulkanMemoryManager *mgr = new VulkanMemoryManager();
    return *mgr;
}

VkDeviceSize VulkanMemoryManager::bucket_size(VkDeviceSize size_bytes)
{
    if (size_bytes < 4096)
        return 4096;
    return align_up(size_bytes, 4096);
}

VulkanMemoryManager::DevicePool &VulkanMemoryManager::get_or_create_pool_locked(uint32_t device_id)
{
    auto it = pools.find(device_id);
    if (it == pools.end())
    {
        auto inserted = pools.emplace(device_id, std::make_unique<DevicePool>());
        return *inserted.first->second;
    }
    return *it->second;
}

std::unique_ptr<VulkanMemoryManager::ArenaBlock>
VulkanMemoryManager::create_arena_block(std::shared_ptr<VulkanDeviceContext> ctx,
                                        VkDeviceSize requested_bytes, VkBufferUsageFlags usage,
                                        VkMemoryPropertyFlags properties)
{
    auto block = std::make_unique<ArenaBlock>();
    block->device_ctx = std::move(ctx);
    block->total_size = requested_bytes;
    block->usage = usage;
    block->properties = properties;

    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = requested_bytes;
    buffer_info.usage = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(block->device_ctx->device, &buffer_info, nullptr, &block->buffer) !=
        VK_SUCCESS)
    {
        return nullptr;
    }

    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(block->device_ctx->device, block->buffer, &mem_reqs);

    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_reqs.size;
    alloc_info.memoryTypeIndex =
        block->device_ctx->find_memory_type(mem_reqs.memoryTypeBits, properties);

    if (vkAllocateMemory(block->device_ctx->device, &alloc_info, nullptr, &block->memory) !=
        VK_SUCCESS)
    {
        vkDestroyBuffer(block->device_ctx->device, block->buffer, nullptr);
        block->buffer = VK_NULL_HANDLE;
        return nullptr;
    }

    if (vkBindBufferMemory(block->device_ctx->device, block->buffer, block->memory, 0) !=
        VK_SUCCESS)
    {
        vkDestroyBuffer(block->device_ctx->device, block->buffer, nullptr);
        vkFreeMemory(block->device_ctx->device, block->memory, nullptr);
        block->buffer = VK_NULL_HANDLE;
        block->memory = VK_NULL_HANDLE;
        return nullptr;
    }

    block->free_by_offset[0] = requested_bytes;
    block->free_by_size.insert({requested_bytes, 0});
    return block;
}

VulkanMemoryManager::StagingRing &VulkanMemoryManager::ensure_staging_ring(uint32_t device_id,
                                                                           DevicePool &pool)
{
    if (pool.staging_ring)
    {
        return *pool.staging_ring;
    }

    auto ctx = VulkanManager::instance_ref().get_device(device_id);
    auto ring = std::make_unique<StagingRing>();
    ring->device_ctx = ctx;
    ring->capacity = kStagingRingBytes;
    ring->head = 0;

    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = ring->capacity;
    buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(ctx->device, &buffer_info, nullptr, &ring->buffer) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create persistent staging ring buffer!");
    }

    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(ctx->device, ring->buffer, &mem_reqs);

    uint32_t mem_type_idx = 0;
    VkMemoryPropertyFlags cached_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                                         VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    if (!has_memory_type(*ctx, mem_reqs.memoryTypeBits, cached_flags, mem_type_idx))
    {
        mem_type_idx = ctx->find_memory_type(mem_reqs.memoryTypeBits,
                                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }

    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_reqs.size;
    alloc_info.memoryTypeIndex = mem_type_idx;

    if (vkAllocateMemory(ctx->device, &alloc_info, nullptr, &ring->memory) != VK_SUCCESS)
    {
        vkDestroyBuffer(ctx->device, ring->buffer, nullptr);
        ring->buffer = VK_NULL_HANDLE;
        throw std::runtime_error("Failed to allocate persistent staging ring memory!");
    }

    if (vkBindBufferMemory(ctx->device, ring->buffer, ring->memory, 0) != VK_SUCCESS)
    {
        vkDestroyBuffer(ctx->device, ring->buffer, nullptr);
        vkFreeMemory(ctx->device, ring->memory, nullptr);
        ring->buffer = VK_NULL_HANDLE;
        ring->memory = VK_NULL_HANDLE;
        throw std::runtime_error("Failed to bind staging ring memory!");
    }

    if (vkMapMemory(ctx->device, ring->memory, 0, ring->capacity, 0, &ring->mapped) != VK_SUCCESS)
    {
        vkDestroyBuffer(ctx->device, ring->buffer, nullptr);
        vkFreeMemory(ctx->device, ring->memory, nullptr);
        ring->buffer = VK_NULL_HANDLE;
        ring->memory = VK_NULL_HANDLE;
        throw std::runtime_error("Failed to map persistent staging ring memory!");
    }

    pool.driver_alloc_count += 1;
    pool.staging_ring = std::move(ring);
    return *pool.staging_ring;
}

std::shared_ptr<VulkanBuffer> VulkanMemoryManager::allocate(uint32_t device_id,
                                                            VkDeviceSize size_bytes,
                                                            VkBufferUsageFlags usage,
                                                            VkMemoryPropertyFlags properties)
{
    if (size_bytes == 0)
    {
        size_bytes = 4;
    }

    auto ctx = VulkanManager::instance_ref().get_device(device_id);
    VkDeviceSize alignment = std::max<VkDeviceSize>(
        kArenaMinAlign, ctx->properties.limits.minStorageBufferOffsetAlignment);
    VkDeviceSize aligned_size = align_up(size_bytes, alignment);

    const VkDeviceSize preferred_slab = ctx->preferred_arena_slab_bytes();

    const bool arena_eligible = ((properties & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0) &&
                                ((properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0) &&
                                (aligned_size <= preferred_slab / 2);

    VulkanBuffer *raw = nullptr;
    {
        std::lock_guard<std::mutex> lock(pool_mutex);
        if (!shutting_down && arena_eligible)
        {
            DevicePool &pool = get_or_create_pool_locked(device_id);
            for (size_t idx = 0; idx < pool.arenas.size(); ++idx)
            {
                ArenaBlock &arena = *pool.arenas[idx];
                if (arena.usage == usage && arena.properties == properties)
                {
                    VkDeviceSize slice_offset = 0;
                    if (arena.allocate_slice(aligned_size, slice_offset))
                    {
                        uint64_t alloc_id = next_alloc_id.fetch_add(1, std::memory_order_relaxed);
                        raw = new VulkanBuffer(ctx, arena.buffer, size_bytes, slice_offset,
                                               aligned_size, static_cast<uint32_t>(idx), usage,
                                               properties, alloc_id);
                        break;
                    }
                }
            }

            if (!raw)
            {
                std::unique_ptr<ArenaBlock> new_arena;
                const VkDeviceSize candidates[] = {
                    preferred_slab, kDefaultArenaSlabSmall, 16ULL * 1024ULL * 1024ULL,
                    std::max(aligned_size * 2ULL, 4ULL * 1024ULL * 1024ULL)};
                for (VkDeviceSize cand : candidates)
                {
                    if (cand < aligned_size)
                        continue;
                    new_arena = create_arena_block(ctx, cand, usage, properties);
                    if (new_arena)
                        break;
                }
                if (new_arena)
                {
                    pool.driver_alloc_count += 1;
                    VkDeviceSize slice_offset = 0;
                    if (!new_arena->allocate_slice(aligned_size, slice_offset))
                    {
                        throw std::runtime_error(
                            "ArenaBlock allocation failed immediately after creation!");
                    }
                    uint32_t arena_idx = static_cast<uint32_t>(pool.arenas.size());
                    VkBuffer shared_buf = new_arena->buffer;
                    pool.arenas.push_back(std::move(new_arena));
                    uint64_t alloc_id = next_alloc_id.fetch_add(1, std::memory_order_relaxed);
                    raw = new VulkanBuffer(ctx, shared_buf, size_bytes, slice_offset, aligned_size,
                                           arena_idx, usage, properties, alloc_id);
                }
            }
        }

        if (!raw)
        {
            const VkDeviceSize bucket = bucket_size(size_bytes);
            if (!shutting_down)
            {
                auto pool_it = pools.find(device_id);
                if (pool_it != pools.end())
                {
                    DevicePool &pool = *pool_it->second;
                    PoolKey key{bucket, usage, properties};
                    auto list_it = pool.free_lists.find(key);
                    if (list_it != pool.free_lists.end() && !list_it->second.empty())
                    {
                        std::unique_ptr<VulkanBuffer> recycled = std::move(list_it->second.back());
                        list_it->second.pop_back();
                        if (list_it->second.empty())
                        {
                            pool.free_lists.erase(list_it);
                        }
                        pool.free_bytes -= bucket;
                        raw = recycled.release();
                    }
                }
            }

            if (!raw)
            {
                try
                {
                    raw = new VulkanBuffer(ctx, bucket, usage, properties);
                }
                catch (const std::runtime_error &)
                {
                    if (!shutting_down)
                    {
                        auto pool_it = pools.find(device_id);
                        if (pool_it != pools.end())
                        {
                            pool_it->second->free_lists.clear();
                            pool_it->second->free_bytes = 0;
                        }
                    }
                    raw = new VulkanBuffer(ctx, bucket, usage, properties);
                }
                raw->allocation_id = next_alloc_id.fetch_add(1, std::memory_order_relaxed);
                if (!shutting_down)
                {
                    DevicePool &pool = get_or_create_pool_locked(device_id);
                    pool.driver_alloc_count += 1;
                }
            }
        }
    }

    live_allocations.fetch_add(1, std::memory_order_relaxed);

    return std::shared_ptr<VulkanBuffer>(raw, [this](VulkanBuffer *b) { this->recycle(b); });
}

void VulkanMemoryManager::recycle(VulkanBuffer *buffer)
{
    live_allocations.fetch_sub(1, std::memory_order_relaxed);

    std::unique_ptr<VulkanBuffer> owned(buffer);
    std::lock_guard<std::mutex> lock(pool_mutex);
    if (shutting_down || !owned->device_ctx)
    {
        return;
    }

    uint32_t dev_id = owned->device_ctx->device_id;
    auto pool_it = pools.find(dev_id);
    if (pool_it == pools.end())
    {
        return;
    }
    DevicePool &pool = *pool_it->second;

    if (owned->from_arena)
    {
        if (owned->arena_index < pool.arenas.size())
        {
            pool.arenas[owned->arena_index]->free_slice(owned->arena_offset,
                                                        owned->arena_block_size);
        }
        return;
    }

    if (pool.free_bytes + owned->size > owned->device_ctx->pool_byte_budget())
    {
        return;
    }
    PoolKey key{owned->size, owned->usage, owned->memory_properties};
    pool.free_bytes += owned->size;
    pool.free_lists[key].push_back(std::move(owned));
}

size_t VulkanMemoryManager::pooled_buffer_count() const
{
    std::lock_guard<std::mutex> lock(pool_mutex);
    size_t n = 0;
    for (const auto &kv : pools)
    {
        for (const auto &entry : kv.second->free_lists)
        {
            n += entry.second.size();
        }
    }
    return n;
}

size_t VulkanMemoryManager::driver_allocation_count(int device_id) const
{
    std::lock_guard<std::mutex> lock(pool_mutex);
    size_t total = 0;
    for (const auto &kv : pools)
    {
        if (device_id < 0 || static_cast<uint32_t>(device_id) == kv.first)
        {
            total += kv.second->driver_alloc_count;
        }
    }
    return total;
}

size_t VulkanMemoryManager::active_arena_bytes(int device_id) const
{
    std::lock_guard<std::mutex> lock(pool_mutex);
    size_t total = 0;
    for (const auto &kv : pools)
    {
        if (device_id < 0 || static_cast<uint32_t>(device_id) == kv.first)
        {
            for (const auto &arena : kv.second->arenas)
            {
                total += static_cast<size_t>(arena->used_bytes);
            }
        }
    }
    return total;
}

size_t VulkanMemoryManager::reserved_arena_bytes(int device_id) const
{
    std::lock_guard<std::mutex> lock(pool_mutex);
    size_t total = 0;
    for (const auto &kv : pools)
    {
        if (device_id < 0 || static_cast<uint32_t>(device_id) == kv.first)
        {
            for (const auto &arena : kv.second->arenas)
            {
                total += static_cast<size_t>(arena->total_size);
            }
        }
    }
    return total;
}

void VulkanMemoryManager::shutdown()
{
    std::lock_guard<std::mutex> lock(pool_mutex);
    shutting_down = true;
    pools.clear();
}

std::shared_ptr<VulkanBuffer> VulkanMemoryManager::create_staging_buffer(uint32_t device_id,
                                                                         VkDeviceSize size_bytes)
{
    return allocate(device_id, size_bytes,
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
}

void VulkanMemoryManager::copy_host_to_device(uint32_t device_id, const void *host_ptr,
                                              std::shared_ptr<VulkanBuffer> dst_buffer,
                                              VkDeviceSize size_bytes)
{
    if (size_bytes == 0)
        return;

    auto ctx = VulkanManager::instance_ref().get_device(device_id);
    if (dst_buffer->is_host_visible && !dst_buffer->from_arena)
    {
        ctx->synchronize();
        void *mapped = dst_buffer->map();
        std::memcpy(mapped, host_ptr, static_cast<size_t>(size_bytes));
        dst_buffer->flush(0, size_bytes);
        return;
    }

    DevicePool *pool_ptr = nullptr;
    {
        std::lock_guard<std::mutex> lock(pool_mutex);
        if (shutting_down)
        {
            return;
        }
        pool_ptr = &get_or_create_pool_locked(device_id);
    }

    std::lock_guard<std::mutex> staging_lock(pool_ptr->staging_mutex);
    StagingRing &ring = ensure_staging_ring(device_id, *pool_ptr);

    const VkDeviceSize max_chunk = ring.capacity / 2;
    const auto *src_bytes = static_cast<const uint8_t *>(host_ptr);
    VkDeviceSize copied = 0;
    while (copied < size_bytes)
    {
        VkDeviceSize chunk = std::min(size_bytes - copied, max_chunk);
        VkDeviceSize aligned_chunk = align_up(chunk, 64ULL);

        if (ring.head + aligned_chunk > ring.capacity)
        {
            ctx->synchronize();
            ring.head = 0;
        }

        VkDeviceSize ring_offset = ring.head;
        ring.head += aligned_chunk;

        std::memcpy(static_cast<uint8_t *>(ring.mapped) + ring_offset, src_bytes + copied,
                    static_cast<size_t>(chunk));

        ctx->record_copy(ring.buffer, dst_buffer->buffer, chunk, ring_offset,
                         dst_buffer->arena_offset + copied);
        copied += chunk;
    }
}

void VulkanMemoryManager::copy_device_to_host(uint32_t device_id,
                                              std::shared_ptr<VulkanBuffer> src_buffer,
                                              void *host_ptr, VkDeviceSize size_bytes)
{
    if (size_bytes == 0)
        return;

    auto ctx = VulkanManager::instance_ref().get_device(device_id);
    if (src_buffer->is_host_visible && !src_buffer->from_arena)
    {
        ctx->synchronize();
        void *mapped = src_buffer->map();
        src_buffer->invalidate(0, size_bytes);
        std::memcpy(host_ptr, mapped, static_cast<size_t>(size_bytes));
        return;
    }

    DevicePool *pool_ptr = nullptr;
    {
        std::lock_guard<std::mutex> lock(pool_mutex);
        if (shutting_down)
        {
            return;
        }
        pool_ptr = &get_or_create_pool_locked(device_id);
    }

    std::lock_guard<std::mutex> staging_lock(pool_ptr->staging_mutex);
    StagingRing &ring = ensure_staging_ring(device_id, *pool_ptr);

    auto *dst_bytes = static_cast<uint8_t *>(host_ptr);
    VkDeviceSize copied = 0;
    while (copied < size_bytes)
    {
        VkDeviceSize chunk = std::min(size_bytes - copied, ring.capacity);
        VkDeviceSize aligned_chunk = align_up(chunk, 64ULL);

        if (ring.head + aligned_chunk > ring.capacity)
        {
            ctx->synchronize();
            ring.head = 0;
        }

        VkDeviceSize ring_offset = ring.head;
        ring.head += aligned_chunk;

        ctx->submit_transfer(
            [&](VkCommandBuffer cmd)
            {
                VkBufferCopy copy_region{};
                copy_region.srcOffset = src_buffer->arena_offset + copied;
                copy_region.dstOffset = ring_offset;
                copy_region.size = chunk;
                vkCmdCopyBuffer(cmd, src_buffer->buffer, ring.buffer, 1, &copy_region);
            });
        std::memcpy(dst_bytes + copied, static_cast<const uint8_t *>(ring.mapped) + ring_offset,
                    static_cast<size_t>(chunk));
        copied += chunk;
    }
}

void VulkanMemoryManager::copy_buffer_to_buffer(uint32_t device_id,
                                                std::shared_ptr<VulkanBuffer> src,
                                                std::shared_ptr<VulkanBuffer> dst,
                                                VkDeviceSize size_bytes, VkDeviceSize src_offset,
                                                VkDeviceSize dst_offset)
{
    if (size_bytes == 0)
        return;
    auto ctx = VulkanManager::instance_ref().get_device(device_id);
    ctx->record_copy(src->buffer, dst->buffer, size_bytes, src->arena_offset + src_offset,
                     dst->arena_offset + dst_offset, src, dst);
}

void VulkanMemoryManager::copy_cross_device(uint32_t src_device_id,
                                            std::shared_ptr<VulkanBuffer> src_buffer,
                                            uint32_t dst_device_id,
                                            std::shared_ptr<VulkanBuffer> dst_buffer,
                                            VkDeviceSize size_bytes, VkDeviceSize src_offset,
                                            VkDeviceSize dst_offset)
{
    if (size_bytes == 0)
        return;

    if (src_device_id == dst_device_id)
    {
        copy_buffer_to_buffer(src_device_id, src_buffer, dst_buffer, size_bytes, src_offset,
                              dst_offset);
        return;
    }

    std::vector<uint8_t> host_tmp(static_cast<size_t>(size_bytes));
    {
        auto src_ctx = VulkanManager::instance_ref().get_device(src_device_id);
        DevicePool *src_pool = nullptr;
        {
            std::lock_guard<std::mutex> lock(pool_mutex);
            src_pool = &get_or_create_pool_locked(src_device_id);
        }
        std::lock_guard<std::mutex> staging_lock(src_pool->staging_mutex);
        StagingRing &ring = ensure_staging_ring(src_device_id, *src_pool);
        VkDeviceSize copied = 0;
        while (copied < size_bytes)
        {
            VkDeviceSize chunk = std::min(size_bytes - copied, ring.capacity);
            VkDeviceSize aligned_chunk = align_up(chunk, 64ULL);
            if (ring.head + aligned_chunk > ring.capacity)
            {
                src_ctx->synchronize();
                ring.head = 0;
            }
            VkDeviceSize ring_off = ring.head;
            ring.head += aligned_chunk;
            src_ctx->submit_transfer(
                [&](VkCommandBuffer cmd)
                {
                    VkBufferCopy copy_region{};
                    copy_region.srcOffset = src_buffer->arena_offset + src_offset + copied;
                    copy_region.dstOffset = ring_off;
                    copy_region.size = chunk;
                    vkCmdCopyBuffer(cmd, src_buffer->buffer, ring.buffer, 1, &copy_region);
                });
            std::memcpy(host_tmp.data() + copied,
                        static_cast<const uint8_t *>(ring.mapped) + ring_off,
                        static_cast<size_t>(chunk));
            copied += chunk;
        }
    }

    {
        auto dst_ctx = VulkanManager::instance_ref().get_device(dst_device_id);
        DevicePool *dst_pool = nullptr;
        {
            std::lock_guard<std::mutex> lock(pool_mutex);
            dst_pool = &get_or_create_pool_locked(dst_device_id);
        }
        std::lock_guard<std::mutex> staging_lock(dst_pool->staging_mutex);
        StagingRing &ring = ensure_staging_ring(dst_device_id, *dst_pool);
        const VkDeviceSize max_chunk = ring.capacity / 2;
        VkDeviceSize copied = 0;
        while (copied < size_bytes)
        {
            VkDeviceSize chunk = std::min(size_bytes - copied, max_chunk);
            VkDeviceSize aligned_chunk = align_up(chunk, 64ULL);
            if (ring.head + aligned_chunk > ring.capacity)
            {
                dst_ctx->synchronize();
                ring.head = 0;
            }
            VkDeviceSize ring_off = ring.head;
            ring.head += aligned_chunk;
            std::memcpy(static_cast<uint8_t *>(ring.mapped) + ring_off, host_tmp.data() + copied,
                        static_cast<size_t>(chunk));
            dst_ctx->record_copy(ring.buffer, dst_buffer->buffer, chunk, ring_off,
                                 dst_buffer->arena_offset + dst_offset + copied);
            copied += chunk;
        }
    }
}

} // namespace vultorch
