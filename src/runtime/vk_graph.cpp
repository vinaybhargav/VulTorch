#include "vk_graph.hpp"
#include <algorithm>
#include <stdexcept>

namespace vultorch
{

namespace
{
std::mutex &live_graphs_mutex()
{
    static std::mutex m;
    return m;
}
std::vector<VulkanGraph *> &live_graphs()
{
    static std::vector<VulkanGraph *> v;
    return v;
}
} // namespace

VulkanGraph::VulkanGraph(uint32_t device_id) : dev_id(device_id)
{
    std::lock_guard<std::mutex> lk(live_graphs_mutex());
    live_graphs().push_back(this);
}

VulkanGraph::~VulkanGraph()
{
    release();
    std::lock_guard<std::mutex> lk(live_graphs_mutex());
    auto &v = live_graphs();
    v.erase(std::remove(v.begin(), v.end(), this), v.end());
}

void VulkanGraph::release_all()
{
    std::vector<VulkanGraph *> copy;
    {
        std::lock_guard<std::mutex> lk(live_graphs_mutex());
        copy = live_graphs();
    }
    for (VulkanGraph *g : copy)
    {
        if (g)
            g->release();
    }
}

void VulkanGraph::release()
{
    if (!device_ctx)
        return;
    std::lock_guard<std::mutex> lock(device_ctx->submit_mutex);
    if (device_ctx->active_graph == this)
    {
        device_ctx->active_graph = nullptr;
    }
    if (device_ctx->device != VK_NULL_HANDLE)
    {
        for (uint32_t i = 0; i < kGraphFenceRing; ++i)
        {
            if (fence_in_flight[i] && fences[i] != VK_NULL_HANDLE)
            {
                vkWaitForFences(device_ctx->device, 1, &fences[i], VK_TRUE, UINT64_MAX);
                fence_in_flight[i] = false;
            }
            if (fences[i] != VK_NULL_HANDLE)
            {
                vkDestroyFence(device_ctx->device, fences[i], nullptr);
                fences[i] = VK_NULL_HANDLE;
            }
        }
        if (cmd != VK_NULL_HANDLE)
        {
            vkFreeCommandBuffers(device_ctx->device, device_ctx->command_pool, 1, &cmd);
            cmd = VK_NULL_HANDLE;
        }
        for (const auto &d : owned_descriptors)
        {
            if (d.set != VK_NULL_HANDLE && d.pool != VK_NULL_HANDLE)
            {
                vkFreeDescriptorSets(device_ctx->device, d.pool, 1, &d.set);
            }
        }
    }
    owned_descriptors.clear();
    pinned_buffers.clear();
    capturing = false;
    captured = false;
    dispatches = 0;
    device_ctx.reset();
}

void VulkanGraph::capture_begin()
{
    if (capturing)
    {
        throw std::runtime_error("VulkanGraph: already capturing");
    }
    if (captured)
    {
        release();
    }

    device_ctx = VulkanManager::instance_ref().get_device(dev_id);
    std::lock_guard<std::mutex> lock(device_ctx->submit_mutex);
    if (device_ctx->active_graph != nullptr)
    {
        throw std::runtime_error(
            "VulkanGraph: another graph capture is already active on this device");
    }
    device_ctx->synchronize_locked();

    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = device_ctx->command_pool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(device_ctx->device, &alloc_info, &cmd) != VK_SUCCESS)
    {
        throw std::runtime_error("VulkanGraph: vkAllocateCommandBuffers failed");
    }

    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    for (uint32_t i = 0; i < kGraphFenceRing; ++i)
    {
        if (vkCreateFence(device_ctx->device, &fence_info, nullptr, &fences[i]) != VK_SUCCESS)
        {
            for (uint32_t j = 0; j < i; ++j)
            {
                vkDestroyFence(device_ctx->device, fences[j], nullptr);
                fences[j] = VK_NULL_HANDLE;
            }
            vkFreeCommandBuffers(device_ctx->device, device_ctx->command_pool, 1, &cmd);
            cmd = VK_NULL_HANDLE;
            throw std::runtime_error("VulkanGraph: vkCreateFence failed");
        }
        fence_in_flight[i] = false;
    }
    next_fence = 0;

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
    if (vkBeginCommandBuffer(cmd, &begin_info) != VK_SUCCESS)
    {
        for (uint32_t i = 0; i < kGraphFenceRing; ++i)
        {
            vkDestroyFence(device_ctx->device, fences[i], nullptr);
            fences[i] = VK_NULL_HANDLE;
        }
        vkFreeCommandBuffers(device_ctx->device, device_ctx->command_pool, 1, &cmd);
        cmd = VK_NULL_HANDLE;
        throw std::runtime_error("VulkanGraph: vkBeginCommandBuffer failed");
    }

    dispatches = 0;
    capturing = true;
    captured = false;
    device_ctx->active_graph = this;
}

void VulkanGraph::capture_end()
{
    if (!capturing || !device_ctx)
    {
        throw std::runtime_error("VulkanGraph: not currently capturing");
    }
    std::lock_guard<std::mutex> lock(device_ctx->submit_mutex);
    if (device_ctx->active_graph == this)
    {
        device_ctx->active_graph = nullptr;
    }
    if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
    {
        capturing = false;
        throw std::runtime_error("VulkanGraph: vkEndCommandBuffer failed");
    }
    capturing = false;
    captured = true;
}

void VulkanGraph::replay()
{
    if (!captured || !device_ctx || cmd == VK_NULL_HANDLE)
    {
        throw std::runtime_error("VulkanGraph: cannot replay before capture_end()");
    }
    std::lock_guard<std::mutex> lock(device_ctx->submit_mutex);
    device_ctx->flush_locked();

    const uint32_t idx = next_fence;
    next_fence = (next_fence + 1) % kGraphFenceRing;
    VkFence f = fences[idx];
    if (fence_in_flight[idx])
    {
        vkWaitForFences(device_ctx->device, 1, &f, VK_TRUE, UINT64_MAX);
        vkResetFences(device_ctx->device, 1, &f);
        fence_in_flight[idx] = false;
    }

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd;

    VkResult res = vkQueueSubmit(device_ctx->compute_queue, 1, &submit_info, f);
    if (res != VK_SUCCESS)
    {
        throw std::runtime_error("VulkanGraph: replay vkQueueSubmit failed (" +
                                 std::to_string(static_cast<int>(res)) + ")");
    }
    fence_in_flight[idx] = true;
    device_ctx->graph_submitted_since_sync = true;
    device_ctx->batches_submitted.fetch_add(1, std::memory_order_relaxed);
    device_ctx->dispatches_recorded.fetch_add(dispatches, std::memory_order_relaxed);
}

} // namespace vultorch
