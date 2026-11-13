#include "vk_pipeline.hpp"
#include <fstream>
#include <stdexcept>
#include <iostream>

namespace vultorch {

VulkanPipelineManager::VulkanPipelineManager() {
    shader_dir = "shaders";
}

void VulkanPipelineManager::clear() {
    std::lock_guard<std::mutex> lock(pipeline_mutex);
    for (auto& dev_pair : cache) {
        for (auto& pipe_pair : dev_pair.second) {
            PipelineRecord& rec = pipe_pair.second;
            if (rec.device != VK_NULL_HANDLE) {
                if (rec.compute_pipeline != VK_NULL_HANDLE) {
                    vkDestroyPipeline(rec.device, rec.compute_pipeline, nullptr);
                    rec.compute_pipeline = VK_NULL_HANDLE;
                }
                if (rec.pipeline_layout != VK_NULL_HANDLE) {
                    vkDestroyPipelineLayout(rec.device, rec.pipeline_layout, nullptr);
                    rec.pipeline_layout = VK_NULL_HANDLE;
                }
                if (rec.desc_pool != VK_NULL_HANDLE) {
                    vkDestroyDescriptorPool(rec.device, rec.desc_pool, nullptr);
                    rec.desc_pool = VK_NULL_HANDLE;
                }
                if (rec.desc_layout != VK_NULL_HANDLE) {
                    vkDestroyDescriptorSetLayout(rec.device, rec.desc_layout, nullptr);
                    rec.desc_layout = VK_NULL_HANDLE;
                }
                if (rec.shader_module != VK_NULL_HANDLE) {
                    vkDestroyShaderModule(rec.device, rec.shader_module, nullptr);
                    rec.shader_module = VK_NULL_HANDLE;
                }
                rec.device = VK_NULL_HANDLE;
            }
        }
    }
    cache.clear();
}

VulkanPipelineManager::~VulkanPipelineManager() {
    clear();
}

VulkanPipelineManager& VulkanPipelineManager::instance() {
    static VulkanPipelineManager mgr;
    return mgr;
}

void VulkanPipelineManager::set_shader_directory(const std::string& dir) {
    std::lock_guard<std::mutex> lock(pipeline_mutex);
    shader_dir = dir;
}

std::string VulkanPipelineManager::get_shader_directory() const {
    return shader_dir;
}

std::vector<uint32_t> VulkanPipelineManager::read_spirv_file(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open SPIR-V file: " + filepath);
    }
    size_t fileSize = (size_t)file.tellg();
    std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
    file.close();
    return buffer;
}

PipelineRecord& VulkanPipelineManager::get_or_create_pipeline(uint32_t device_id, const std::string& shader_name,
                                                             uint32_t num_storage_buffers, size_t push_constant_size) {
    auto& dev_cache = cache[device_id];
    auto it = dev_cache.find(shader_name);
    if (it != dev_cache.end()) {
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
    if (res != VK_SUCCESS) {
        throw std::runtime_error("vkCreateShaderModule failed for " + full_path + ": " + std::to_string(res));
    }

    // 2. Create Descriptor Set Layout
    std::vector<VkDescriptorSetLayoutBinding> bindings(num_storage_buffers);
    for (uint32_t i = 0; i < num_storage_buffers; ++i) {
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
    if (res != VK_SUCCESS) {
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
    if (push_constant_size > 0) {
        pipeline_layout_info.pushConstantRangeCount = 1;
        pipeline_layout_info.pPushConstantRanges = &pc_range;
    }

    res = vkCreatePipelineLayout(dev->device, &pipeline_layout_info, nullptr, &rec.pipeline_layout);
    if (res != VK_SUCCESS) {
        throw std::runtime_error("vkCreatePipelineLayout failed for " + shader_name);
    }

    // 4. Create Compute Pipeline
    VkComputePipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.layout = rec.pipeline_layout;
    pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module = rec.shader_module;
    pipeline_info.stage.pName = "main";

    res = vkCreateComputePipelines(dev->device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &rec.compute_pipeline);
    if (res != VK_SUCCESS) {
        throw std::runtime_error("vkCreateComputePipelines failed for " + shader_name);
    }

    // 5. Create Descriptor Pool
    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = num_storage_buffers * 64;

    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_info.maxSets = 64;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;

    res = vkCreateDescriptorPool(dev->device, &pool_info, nullptr, &rec.desc_pool);
    if (res != VK_SUCCESS) {
        throw std::runtime_error("vkCreateDescriptorPool failed for " + shader_name);
    }

    dev_cache[shader_name] = rec;
    return dev_cache[shader_name];
}

void VulkanPipelineManager::dispatch(uint32_t device_id, const std::string& shader_name,
                                     const std::vector<std::shared_ptr<VulkanBuffer>>& buffers,
                                     const void* push_constants, size_t push_constants_size,
                                     uint32_t group_count_x, uint32_t group_count_y, uint32_t group_count_z) {
    if (group_count_x == 0 || group_count_y == 0 || group_count_z == 0) return;

    std::lock_guard<std::mutex> lock(pipeline_mutex);
    auto dev = VulkanManager::instance_ref().get_device(device_id);
    PipelineRecord& rec = get_or_create_pipeline(device_id, shader_name,
                                                 static_cast<uint32_t>(buffers.size()),
                                                 push_constants_size);

    // Allocate Descriptor Set
    VkDescriptorSetAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = rec.desc_pool;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &rec.desc_layout;

    VkDescriptorSet desc_set;
    VkResult res = vkAllocateDescriptorSets(dev->device, &alloc_info, &desc_set);
    if (res != VK_SUCCESS) {
        // Reset pool if full
        vkResetDescriptorPool(dev->device, rec.desc_pool, 0);
        res = vkAllocateDescriptorSets(dev->device, &alloc_info, &desc_set);
        if (res != VK_SUCCESS) {
            throw std::runtime_error("vkAllocateDescriptorSets failed: " + std::to_string(res));
        }
    }

    // Update Descriptor Set with Buffers
    std::vector<VkDescriptorBufferInfo> buffer_infos(buffers.size());
    std::vector<VkWriteDescriptorSet> writes(buffers.size());
    for (size_t i = 0; i < buffers.size(); ++i) {
        buffer_infos[i].buffer = buffers[i]->buffer;
        buffer_infos[i].offset = 0;
        buffer_infos[i].range = VK_WHOLE_SIZE;

        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = desc_set;
        writes[i].dstBinding = static_cast<uint32_t>(i);
        writes[i].dstArrayElement = 0;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].descriptorCount = 1;
        writes[i].pBufferInfo = &buffer_infos[i];
    }
    vkUpdateDescriptorSets(dev->device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    // Record & Submit Command Buffer
    VkCommandBuffer cmd = dev->begin_single_time_commands();
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, rec.compute_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, rec.pipeline_layout, 0, 1, &desc_set, 0, nullptr);

    if (push_constants != nullptr && push_constants_size > 0) {
        vkCmdPushConstants(cmd, rec.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           static_cast<uint32_t>(push_constants_size), push_constants);
    }

    vkCmdDispatch(cmd, group_count_x, group_count_y, group_count_z);

    // Memory barrier to make writes visible to host / subsequent dispatches
    VkMemoryBarrier mem_barrier{};
    mem_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mem_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mem_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_HOST_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;

    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 1, &mem_barrier, 0, nullptr, 0, nullptr);

    dev->end_single_time_commands(cmd);

    // Free descriptor set
    vkFreeDescriptorSets(dev->device, rec.desc_pool, 1, &desc_set);
}

} // namespace vultorch
