#include "vk_pipeline.hpp"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace vultorch
{

namespace
{

std::string get_pipeline_cache_path(uint32_t device_id)
{
    const char *home = std::getenv("HOME");
    std::filesystem::path dir;
    if (home && *home)
    {
        dir = std::filesystem::path(home) / ".cache" / "vultorch";
    }
    else
    {
        dir = std::filesystem::temp_directory_path() / "vultorch_cache";
    }
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    return (dir / ("pipeline_cache_dev" + std::to_string(device_id) + ".bin")).string();
}

} // namespace

VulkanPipelineManager::VulkanPipelineManager()
{
    shader_dir = "shaders";
}

VkPipelineCache VulkanPipelineManager::get_or_load_pipeline_cache(uint32_t device_id,
                                                                  VkDevice device)
{
    auto it = vk_pipeline_caches.find(device_id);
    if (it != vk_pipeline_caches.end())
    {
        return it->second;
    }

    std::string cache_path = get_pipeline_cache_path(device_id);
    std::vector<char> cache_data;
    std::ifstream f(cache_path, std::ios::binary | std::ios::ate);
    if (f.is_open())
    {
        size_t sz = f.tellg();
        if (sz >= 16)
        { // Vulkan pipeline cache header is at least 16 bytes
            f.seekg(0);
            cache_data.resize(sz);
            f.read(cache_data.data(), sz);
        }
    }

    VkPipelineCacheCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    if (!cache_data.empty())
    {
        ci.initialDataSize = cache_data.size();
        ci.pInitialData = cache_data.data();
    }

    VkPipelineCache pcache = VK_NULL_HANDLE;
    VkResult res = vkCreatePipelineCache(device, &ci, nullptr, &pcache);
    if (res != VK_SUCCESS)
    {
        // Fall back to empty cache if driver rejects stale or mismatched cache data
        ci.initialDataSize = 0;
        ci.pInitialData = nullptr;
        vkCreatePipelineCache(device, &ci, nullptr, &pcache);
    }

    vk_pipeline_caches[device_id] = pcache;
    return pcache;
}

void VulkanPipelineManager::save_pipeline_cache(uint32_t device_id, VkDevice device,
                                                VkPipelineCache pcache)
{
    if (device == VK_NULL_HANDLE || pcache == VK_NULL_HANDLE)
        return;
    size_t data_size = 0;
    VkResult res = vkGetPipelineCacheData(device, pcache, &data_size, nullptr);
    if (res != VK_SUCCESS || data_size == 0)
        return;

    std::vector<char> data(data_size);
    res = vkGetPipelineCacheData(device, pcache, &data_size, data.data());
    if (res == VK_SUCCESS && data_size > 0)
    {
        std::string cache_path = get_pipeline_cache_path(device_id);
        std::ofstream out(cache_path, std::ios::binary | std::ios::trunc);
        if (out.is_open())
        {
            out.write(data.data(), data_size);
        }
    }
}

void VulkanPipelineManager::clear()
{
    std::lock_guard<std::mutex> lock(cache_mutex);
    for (auto &dev_pair : cache)
    {
        uint32_t dev_id = dev_pair.first;
        VkDevice dev = VK_NULL_HANDLE;
        for (auto &pipe_pair : dev_pair.second)
        {
            PipelineRecord &rec = pipe_pair.second;
            if (rec.device != VK_NULL_HANDLE)
            {
                dev = rec.device;
                if (rec.compute_pipeline != VK_NULL_HANDLE)
                {
                    vkDestroyPipeline(rec.device, rec.compute_pipeline, nullptr);
                    rec.compute_pipeline = VK_NULL_HANDLE;
                }
                if (rec.pipeline_layout != VK_NULL_HANDLE)
                {
                    vkDestroyPipelineLayout(rec.device, rec.pipeline_layout, nullptr);
                    rec.pipeline_layout = VK_NULL_HANDLE;
                }
                if (rec.desc_pool != VK_NULL_HANDLE)
                {
                    vkDestroyDescriptorPool(rec.device, rec.desc_pool, nullptr);
                    rec.desc_pool = VK_NULL_HANDLE;
                }
                if (rec.desc_layout != VK_NULL_HANDLE)
                {
                    vkDestroyDescriptorSetLayout(rec.device, rec.desc_layout, nullptr);
                    rec.desc_layout = VK_NULL_HANDLE;
                }
                if (rec.shader_module != VK_NULL_HANDLE)
                {
                    vkDestroyShaderModule(rec.device, rec.shader_module, nullptr);
                    rec.shader_module = VK_NULL_HANDLE;
                }
                rec.device = VK_NULL_HANDLE;
            }
        }
        auto it_c = vk_pipeline_caches.find(dev_id);
        if (it_c != vk_pipeline_caches.end() && it_c->second != VK_NULL_HANDLE &&
            dev != VK_NULL_HANDLE)
        {
            save_pipeline_cache(dev_id, dev, it_c->second);
            vkDestroyPipelineCache(dev, it_c->second, nullptr);
        }
    }
    vk_pipeline_caches.clear();
    cache.clear();
}

VulkanPipelineManager::~VulkanPipelineManager()
{
    clear();
}

VulkanPipelineManager &VulkanPipelineManager::instance()
{
    static VulkanPipelineManager *mgr = new VulkanPipelineManager();
    return *mgr;
}

void VulkanPipelineManager::set_shader_directory(const std::string &dir)
{
    std::lock_guard<std::mutex> lock(cache_mutex);
    shader_dir = dir;
}

std::string VulkanPipelineManager::get_shader_directory() const
{
    return shader_dir;
}

std::vector<uint32_t> VulkanPipelineManager::read_spirv_file(const std::string &filepath)
{
    std::ifstream file(filepath, std::ios::ate | std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open SPIR-V file: " + filepath);
    }
    size_t fileSize = (size_t)file.tellg();
    std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char *>(buffer.data()), fileSize);
    file.close();
    return buffer;
}

PipelineRecord &VulkanPipelineManager::get_or_create_pipeline(uint32_t device_id,
                                                              const std::string &shader_name,
                                                              uint32_t num_storage_buffers,
                                                              size_t push_constant_size)
{
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto &dev_cache = cache[device_id];
    auto it = dev_cache.find(shader_name);
    if (it != dev_cache.end())
    {
        return it->second;
    }

    auto dev = VulkanManager::instance_ref().get_device(device_id);
    std::string full_path = shader_dir + "/" + shader_name + ".spv";
    std::vector<uint32_t> code = read_spirv_file(full_path);

    PipelineRecord rec;
    rec.device = dev->device;
    rec.num_buffers = num_storage_buffers;

    // 1. Create Shader Module
    VkShaderModuleCreateInfo module_info{};
    module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    module_info.codeSize = code.size() * sizeof(uint32_t);
    module_info.pCode = code.data();

    VkResult res = vkCreateShaderModule(dev->device, &module_info, nullptr, &rec.shader_module);
    if (res != VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateShaderModule failed for " + full_path + ": " +
                                 std::to_string(res));
    }

    // 2. Create Descriptor Set Layout
    std::vector<VkDescriptorSetLayoutBinding> bindings(num_storage_buffers);
    for (uint32_t i = 0; i < num_storage_buffers; ++i)
    {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[i].pImmutableSamplers = nullptr;
    }

    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
    layout_info.pBindings = bindings.data();

    res = vkCreateDescriptorSetLayout(dev->device, &layout_info, nullptr, &rec.desc_layout);
    if (res != VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateDescriptorSetLayout failed for " + shader_name);
    }

    // 3. Create Pipeline Layout with Push Constants
    VkPushConstantRange pc_range{};
    pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc_range.offset = 0;
    pc_range.size = static_cast<uint32_t>(push_constant_size);

    VkPipelineLayoutCreateInfo pipeline_layout_info{};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &rec.desc_layout;
    if (push_constant_size > 0)
    {
        pipeline_layout_info.pushConstantRangeCount = 1;
        pipeline_layout_info.pPushConstantRanges = &pc_range;
    }

    res = vkCreatePipelineLayout(dev->device, &pipeline_layout_info, nullptr, &rec.pipeline_layout);
    if (res != VK_SUCCESS)
    {
        throw std::runtime_error("vkCreatePipelineLayout failed for " + shader_name);
    }

    // 4. Create Compute Pipeline (backed by persistent VkPipelineCache & Subgroup
    // Size Control)
    VkComputePipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.layout = rec.pipeline_layout;
    pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module = rec.shader_module;
    pipeline_info.stage.pName = "main";

    VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT req_subgroup_size{};
    req_subgroup_size.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT;
    if (dev->capabilities.subgroup_size_control && dev->capabilities.subgroup_size > 0)
    {
        req_subgroup_size.requiredSubgroupSize = dev->capabilities.subgroup_size;
        pipeline_info.stage.pNext = &req_subgroup_size;
        pipeline_info.stage.flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT_EXT;
    }

    VkPipelineCache pcache = get_or_load_pipeline_cache(device_id, dev->device);
    res = vkCreateComputePipelines(dev->device, pcache, 1, &pipeline_info, nullptr,
                                   &rec.compute_pipeline);
    if (res != VK_SUCCESS)
    {
        // If driver rejects subgroup size hint for this particular shader, retry
        // without it
        pipeline_info.stage.pNext = nullptr;
        pipeline_info.stage.flags = 0;
        res = vkCreateComputePipelines(dev->device, pcache, 1, &pipeline_info, nullptr,
                                       &rec.compute_pipeline);
    }
    if (res != VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateComputePipelines failed for " + shader_name);
    }
    // Periodically flush pipeline cache state
    save_pipeline_cache(device_id, dev->device, pcache);

    // 5. Create Descriptor Pool
    // One set per dispatch, and a set is only freed once its batch has retired,
    // so the pool has to cover several full batches in flight.
    constexpr uint32_t kSetsPerPool = 2048;

    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = num_storage_buffers * kSetsPerPool;

    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_info.maxSets = kSetsPerPool;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;

    res = vkCreateDescriptorPool(dev->device, &pool_info, nullptr, &rec.desc_pool);
    if (res != VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateDescriptorPool failed for " + shader_name);
    }

    dev_cache[shader_name] = rec;
    return dev_cache[shader_name];
}

VkDescriptorSet
VulkanPipelineManager::allocate_descriptor_set(const std::shared_ptr<VulkanDeviceContext> &dev,
                                               PipelineRecord &record,
                                               const std::string &shader_name)
{
    VkDescriptorSetAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = record.desc_pool;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &record.desc_layout;

    VkDescriptorSet set = VK_NULL_HANDLE;
    VkResult res = vkAllocateDescriptorSets(dev->device, &alloc_info, &set);
    if (res != VK_SUCCESS)
    {
        // Runs under the device's submission lock, so it cannot synchronise here.
        // kMaxPendingDescSets keeps the number of live sets below the pool's
        // capacity; reaching this means that accounting is broken. Never reset
        // the pool: it would invalidate sets queued in flight.
        throw std::runtime_error("vkAllocateDescriptorSets failed for " + shader_name + ": " +
                                 std::to_string(res) +
                                 " (descriptor pool exhausted; check kMaxPendingDescSets)");
    }
    return set;
}

void VulkanPipelineManager::dispatch(uint32_t device_id, const std::string &shader_name,
                                     const std::vector<std::shared_ptr<VulkanBuffer>> &buffers,
                                     const void *push_constants, size_t push_constants_size,
                                     uint32_t group_count_x, uint32_t group_count_y,
                                     uint32_t group_count_z)
{
    if (group_count_x == 0 || group_count_y == 0 || group_count_z == 0)
        return;

    auto dev = VulkanManager::instance_ref().get_device(device_id);
    PipelineRecord &rec = get_or_create_pipeline(
        device_id, shader_name, static_cast<uint32_t>(buffers.size()), push_constants_size);

    // Everything that touches the descriptor pool runs inside the device's
    // submission lock, so concurrent threads using one device cannot race on
    // vkAllocateDescriptorSets / vkUpdateDescriptorSets.
    dev->record_dispatch(
        [&](VkCommandBuffer cmd)
        {
            const VkDescriptorSet desc_set = allocate_descriptor_set(dev, rec, shader_name);

            std::vector<VkDescriptorBufferInfo> buffer_infos(buffers.size());
            std::vector<VkWriteDescriptorSet> writes(buffers.size());
            for (size_t i = 0; i < buffers.size(); ++i)
            {
                buffer_infos[i].buffer = buffers[i]->buffer;
                buffer_infos[i].offset = buffers[i]->arena_offset;
                buffer_infos[i].range =
                    buffers[i]->from_arena ? buffers[i]->arena_block_size : VK_WHOLE_SIZE;

                writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[i].dstSet = desc_set;
                writes[i].dstBinding = static_cast<uint32_t>(i);
                writes[i].dstArrayElement = 0;
                writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writes[i].descriptorCount = 1;
                writes[i].pBufferInfo = &buffer_infos[i];
            }
            vkUpdateDescriptorSets(dev->device, static_cast<uint32_t>(writes.size()), writes.data(),
                                   0, nullptr);

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, rec.compute_pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, rec.pipeline_layout, 0, 1,
                                    &desc_set, 0, nullptr);

            if (push_constants != nullptr && push_constants_size > 0)
            {
                vkCmdPushConstants(cmd, rec.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                   static_cast<uint32_t>(push_constants_size), push_constants);
            }

            vkCmdDispatch(cmd, group_count_x, group_count_y, group_count_z);
            return DescriptorAllocation{rec.desc_pool, desc_set};
        },
        buffers);
}

} // namespace vultorch
