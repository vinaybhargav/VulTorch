#include "vk_context.hpp"
#include "vk_pipeline.hpp"
#include <cstring>
#include <sstream>

namespace vultorch {

VulkanDeviceContext::VulkanDeviceContext(uint32_t id, VkPhysicalDevice pdev)
    : device_id(id), physical_device(pdev) {
    vkGetPhysicalDeviceProperties(physical_device, &properties);
    vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);

    // Capabilities consulted by kernel selection: a variant is only considered
    // when the device reports it can run it.
    //
    // Subgroup data is a *property* struct and must be chained to
    // vkGetPhysicalDeviceProperties2; chaining it to the features query leaves
    // it silently zero-initialised.
    bool supported_shader_int64 = false;
    bool supported_shader_int16 = false;
    bool supported_uniform_storage_16bit = false;
    bool supported_shader_int8 = false;
    {
        VkPhysicalDeviceSubgroupProperties subgroup{};
        subgroup.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;

        VkPhysicalDeviceProperties2 properties2{};
        properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        properties2.pNext = &subgroup;

        vkGetPhysicalDeviceProperties2(physical_device, &properties2);

        VkPhysicalDeviceShaderFloat16Int8Features float16{};
        float16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;

        VkPhysicalDevice16BitStorageFeatures storage_16bit{};
        storage_16bit.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
        storage_16bit.pNext = &float16;

        VkPhysicalDeviceFeatures2 features2{};
        features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features2.pNext = &storage_16bit;

        vkGetPhysicalDeviceFeatures2(physical_device, &features2);

        capabilities.subgroup_size = subgroup.subgroupSize;
        capabilities.max_workgroup_invocations = properties.limits.maxComputeWorkGroupInvocations;
        capabilities.max_shared_memory_bytes = properties.limits.maxComputeSharedMemorySize;
        capabilities.float16_arithmetic = float16.shaderFloat16 == VK_TRUE;
        capabilities.storage_16bit = storage_16bit.storageBuffer16BitAccess == VK_TRUE;
        capabilities.subgroup_basic =
            (subgroup.supportedOperations & VK_SUBGROUP_FEATURE_BASIC_BIT) != 0;
        capabilities.subgroup_arithmetic =
            (subgroup.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0;

        supported_shader_int64 = features2.features.shaderInt64 == VK_TRUE;
        supported_shader_int16 = features2.features.shaderInt16 == VK_TRUE;
        supported_uniform_storage_16bit = storage_16bit.uniformAndStorageBuffer16BitAccess == VK_TRUE;
        supported_shader_int8 = float16.shaderInt8 == VK_TRUE;
    }

    // Find compute queue family
    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, nullptr);
    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, queue_families.data());

    int selected_family = -1;
    // Prefer a dedicated compute queue if available, otherwise any queue with compute bit
    for (uint32_t i = 0; i < queue_family_count; ++i) {
        if ((queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
            !(queue_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
            selected_family = i;
            break;
        }
    }
    if (selected_family == -1) {
        for (uint32_t i = 0; i < queue_family_count; ++i) {
            if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                selected_family = i;
                break;
            }
        }
    }

    if (selected_family == -1) {
        throw std::runtime_error("No queue family with VK_QUEUE_COMPUTE_BIT found on device " + std::to_string(id));
    }
    compute_queue_family = static_cast<uint32_t>(selected_family);

    // Check for extensions (e.g. VK_KHR_cooperative_matrix)
    uint32_t ext_count = 0;
    vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &ext_count, nullptr);
    std::vector<VkExtensionProperties> available_exts(ext_count);
    vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &ext_count, available_exts.data());

    std::vector<const char*> enabled_extensions;
    for (const auto& e : available_exts) {
        if (std::strcmp(e.extensionName, "VK_KHR_cooperative_matrix") == 0) {
            has_cooperative_matrix = true;
            enabled_extensions.push_back("VK_KHR_cooperative_matrix");
            break;
        }
    }

    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coop_features{};
    coop_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    coop_features.cooperativeMatrix = VK_TRUE;

    VkPhysicalDeviceShaderFloat16Int8Features enabled_float16{};
    enabled_float16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
    enabled_float16.shaderFloat16 = capabilities.float16_arithmetic ? VK_TRUE : VK_FALSE;
    enabled_float16.shaderInt8 = supported_shader_int8 ? VK_TRUE : VK_FALSE;
    if (has_cooperative_matrix) {
        enabled_float16.pNext = &coop_features;
    }

    VkPhysicalDevice16BitStorageFeatures enabled_16bit{};
    enabled_16bit.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
    enabled_16bit.storageBuffer16BitAccess = capabilities.storage_16bit ? VK_TRUE : VK_FALSE;
    enabled_16bit.uniformAndStorageBuffer16BitAccess = supported_uniform_storage_16bit ? VK_TRUE : VK_FALSE;
    enabled_16bit.pNext = &enabled_float16;

    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_create_info{};
    queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_create_info.queueFamilyIndex = compute_queue_family;
    queue_create_info.queueCount = 1;
    queue_create_info.pQueuePriorities = &queue_priority;

    VkPhysicalDeviceFeatures device_features{};
    device_features.shaderFloat64 = VK_FALSE;
    device_features.shaderInt64 = supported_shader_int64 ? VK_TRUE : VK_FALSE;
    device_features.shaderInt16 = supported_shader_int16 ? VK_TRUE : VK_FALSE;

    VkDeviceCreateInfo device_create_info{};
    device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_create_info.pNext = &enabled_16bit;
    device_create_info.queueCreateInfoCount = 1;
    device_create_info.pQueueCreateInfos = &queue_create_info;
    device_create_info.pEnabledFeatures = &device_features;
    device_create_info.enabledExtensionCount = static_cast<uint32_t>(enabled_extensions.size());
    device_create_info.ppEnabledExtensionNames = enabled_extensions.data();

    VkResult res = vkCreateDevice(physical_device, &device_create_info, nullptr, &device);
    if (res != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Vulkan logical device " + std::to_string(id));
    }

    // Query cooperative matrix properties
    if (has_cooperative_matrix) {
        auto pfnGetProps = (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)
            vkGetInstanceProcAddr(VulkanManager::instance_ref().get_instance(), "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
        if (pfnGetProps) {
            uint32_t num_props = 0;
            pfnGetProps(physical_device, &num_props, nullptr);
            if (num_props > 0) {
                std::vector<VkCooperativeMatrixPropertiesKHR> props(num_props);
                for (auto& pr : props) pr.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
                pfnGetProps(physical_device, &num_props, props.data());
                for (const auto& pr : props) {
                    cooperative_matrix_configs.push_back({
                        pr.MSize, pr.NSize, pr.KSize,
                        static_cast<uint32_t>(pr.AType),
                        static_cast<uint32_t>(pr.BType),
                        static_cast<uint32_t>(pr.CType),
                        static_cast<uint32_t>(pr.ResultType),
                        static_cast<uint32_t>(pr.scope)
                    });
                }
            }
        }

        // Does this device expose the shape the coopmat kernel is written for?
        for (const auto& config : cooperative_matrix_configs) {
            if (config.M == 16 && config.N == 16 && config.K == 16 &&
                config.a_type == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                config.b_type == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                config.c_type == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                config.scope == VK_SCOPE_SUBGROUP_KHR) {
                capabilities.fp16_coopmat_16x16 = true;
                break;
            }
        }
    }

    vkGetDeviceQueue(device, compute_queue_family, 0, &compute_queue);

    // Create command pool
    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = compute_queue_family;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    res = vkCreateCommandPool(device, &pool_info, nullptr, &command_pool);
    if (res != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Vulkan command pool on device " + std::to_string(id));
    }

    // Command buffers and fences are allocated once and recycled in a ring;
    // creating them per dispatch was a large part of the per-op overhead.
    slots.resize(kSlotCount);
    std::vector<VkCommandBuffer> ring_cmds(kSlotCount);
    VkCommandBufferAllocateInfo cmd_alloc{};
    cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_alloc.commandPool = command_pool;
    cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_alloc.commandBufferCount = kSlotCount;
    res = vkAllocateCommandBuffers(device, &cmd_alloc, ring_cmds.data());
    if (res != VK_SUCCESS) {
        throw std::runtime_error("vkAllocateCommandBuffers failed on device " + std::to_string(id));
    }

    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    for (uint32_t i = 0; i < kSlotCount; ++i) {
        slots[i].cmd = ring_cmds[i];
        res = vkCreateFence(device, &fence_info, nullptr, &slots[i].fence);
        if (res != VK_SUCCESS) {
            throw std::runtime_error("vkCreateFence failed on device " + std::to_string(id));
        }
    }
    res = vkCreateFence(device, &fence_info, nullptr, &transfer_fence);
    if (res != VK_SUCCESS) {
        throw std::runtime_error("vkCreateFence (transfer) failed on device " + std::to_string(id));
    }
}

VulkanDeviceContext::~VulkanDeviceContext() {
    if (device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device);
        for (auto& slot : slots) {
            if (slot.fence != VK_NULL_HANDLE) {
                vkDestroyFence(device, slot.fence, nullptr);
                slot.fence = VK_NULL_HANDLE;
            }
        }
        if (transfer_fence != VK_NULL_HANDLE) {
            vkDestroyFence(device, transfer_fence, nullptr);
            transfer_fence = VK_NULL_HANDLE;
        }
        if (command_pool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device, command_pool, nullptr);
            command_pool = VK_NULL_HANDLE;
        }
        vkDestroyDevice(device, nullptr);
        device = VK_NULL_HANDLE;
    }
}

uint32_t VulkanDeviceContext::find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags required_flags) const {
    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) &&
            (memory_properties.memoryTypes[i].propertyFlags & required_flags) == required_flags) {
            return i;
        }
    }
    throw std::runtime_error("Failed to find suitable Vulkan memory type!");
}

// ---------------------------------------------------------------------------
// Batched execution
//
// Every dispatch is recorded into an open command buffer instead of being
// submitted and waited on immediately. Submission happens when a batch fills,
// when something needs results, or on an explicit synchronize(). Because the
// device has a single compute queue, submissions complete in FIFO order, so
// waiting on the newest fence also proves that all older work has finished.
// ---------------------------------------------------------------------------

namespace {
// Kept as a named constant so the header and implementation cannot drift.
constexpr size_t kMaxPendingDescSets = VulkanDeviceContext::kMaxPendingDescSets;
} // namespace

void VulkanDeviceContext::add_compute_barrier(VkCommandBuffer cmd) {
    VkMemoryBarrier mem_barrier{};
    mem_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mem_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    mem_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                                VK_ACCESS_HOST_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT |
                                VK_ACCESS_TRANSFER_WRITE_BIT;

    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_HOST_BIT |
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 1, &mem_barrier, 0, nullptr, 0, nullptr);
}

void VulkanDeviceContext::record_copy(VkBuffer src, VkBuffer dst, VkDeviceSize size_bytes,
                                      VkDeviceSize src_offset, VkDeviceSize dst_offset) {
    std::lock_guard<std::mutex> lock(submit_mutex);

    open_batch_locked();
    VkCommandBuffer cmd = slots[static_cast<size_t>(open_slot)].cmd;

    VkBufferCopy copy_region{};
    copy_region.srcOffset = src_offset;
    copy_region.dstOffset = dst_offset;
    copy_region.size = size_bytes;
    vkCmdCopyBuffer(cmd, src, dst, 1, &copy_region);

    add_compute_barrier(cmd);

    ++ops_in_batch;
    dispatches_recorded.fetch_add(1, std::memory_order_relaxed);

    if (ops_in_batch >= kMaxOpsPerBatch) {
        flush_locked();
    }
}

void VulkanDeviceContext::open_batch_locked() {
    if (open_slot >= 0) return;

    Slot& slot = slots[next_slot];
    if (slot.in_use) {
        // Reusing a slot: wait for its previous submission to drain.
        vkWaitForFences(device, 1, &slot.fence, VK_TRUE, UINT64_MAX);
        slot.in_use = false;
    }

    // The fence is unsignalled at this point, so resetting is safe.
    vkResetFences(device, 1, &slot.fence);

    VkResult res = vkResetCommandBuffer(slot.cmd, 0);
    if (res != VK_SUCCESS) {
        throw std::runtime_error("vkResetCommandBuffer failed on device " + std::to_string(device_id));
    }

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    res = vkBeginCommandBuffer(slot.cmd, &begin_info);
    if (res != VK_SUCCESS) {
        throw std::runtime_error("vkBeginCommandBuffer failed on device " + std::to_string(device_id));
    }

    open_slot = static_cast<int32_t>(next_slot);
    next_slot = (next_slot + 1) % kSlotCount;
    ops_in_batch = 0;
}

void VulkanDeviceContext::flush_locked() {
    if (open_slot < 0) return;

    Slot& slot = slots[static_cast<size_t>(open_slot)];

    VkResult res = vkEndCommandBuffer(slot.cmd);
    if (res != VK_SUCCESS) {
        throw std::runtime_error("vkEndCommandBuffer failed on device " + std::to_string(device_id));
    }

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &slot.cmd;

    res = vkQueueSubmit(compute_queue, 1, &submit_info, slot.fence);
    if (res != VK_SUCCESS) {
        throw std::runtime_error("vkQueueSubmit failed on device " + std::to_string(device_id));
    }

    slot.in_use = true;
    last_submitted_slot = open_slot;
    open_slot = -1;
    ops_in_batch = 0;
    batches_submitted.fetch_add(1, std::memory_order_relaxed);
}

void VulkanDeviceContext::synchronize_locked() {
    flush_locked();

    if (last_submitted_slot >= 0) {
        Slot& slot = slots[static_cast<size_t>(last_submitted_slot)];
        vkWaitForFences(device, 1, &slot.fence, VK_TRUE, UINT64_MAX);
        // FIFO on a single queue: every earlier submission has completed too.
        for (auto& s : slots) {
            s.in_use = false;
        }
        last_submitted_slot = -1;
    }

    for (const auto& descriptor : pending_descriptors) {
        if (descriptor.set != VK_NULL_HANDLE) {
            vkFreeDescriptorSets(device, descriptor.pool, 1, &descriptor.set);
        }
    }
    pending_descriptors.clear();
}

void VulkanDeviceContext::flush() {
    std::lock_guard<std::mutex> lock(submit_mutex);
    flush_locked();
}

void VulkanDeviceContext::synchronize() {
    std::lock_guard<std::mutex> lock(submit_mutex);
    synchronize_locked();
}

void VulkanDeviceContext::submit_transfer(const std::function<void(VkCommandBuffer)>& record) {
    std::lock_guard<std::mutex> lock(submit_mutex);

    // Pending compute writes must be ordered ahead of the copy.
    flush_locked();

    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandPool = command_pool;
    alloc_info.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkResult res = vkAllocateCommandBuffers(device, &alloc_info, &cmd);
    if (res != VK_SUCCESS) {
        throw std::runtime_error("vkAllocateCommandBuffers (transfer) failed on device " +
                                 std::to_string(device_id));
    }

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin_info);

    record(cmd);

    VkMemoryBarrier mem_barrier{};
    mem_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mem_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mem_barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mem_barrier, 0, nullptr, 0, nullptr);

    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
        vkFreeCommandBuffers(device, command_pool, 1, &cmd);
        throw std::runtime_error("vkEndCommandBuffer (transfer) failed on device " +
                                 std::to_string(device_id));
    }

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd;

    vkResetFences(device, 1, &transfer_fence);
    if (vkQueueSubmit(compute_queue, 1, &submit_info, transfer_fence) != VK_SUCCESS) {
        vkFreeCommandBuffers(device, command_pool, 1, &cmd);
        throw std::runtime_error("vkQueueSubmit (transfer) failed on device " +
                                 std::to_string(device_id));
    }
    vkWaitForFences(device, 1, &transfer_fence, VK_TRUE, UINT64_MAX);

    // This wait also covers the batch flushed above, so reclamation is safe.
    for (auto& s : slots) {
        s.in_use = false;
    }
    last_submitted_slot = -1;

    for (const auto& descriptor : pending_descriptors) {
        if (descriptor.set != VK_NULL_HANDLE) {
            vkFreeDescriptorSets(device, descriptor.pool, 1, &descriptor.set);
        }
    }
    pending_descriptors.clear();

    vkFreeCommandBuffers(device, command_pool, 1, &cmd);
}

VulkanManager::VulkanManager() {
    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "VulTorch Engine";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "VulTorch";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;

    VkResult res = vkCreateInstance(&create_info, nullptr, &instance);
    if (res != VK_SUCCESS) {
        throw std::runtime_error("Failed to initialize Vulkan instance: code " + std::to_string(res));
    }

    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
    if (device_count == 0) {
        throw std::runtime_error("No Vulkan physical devices found on system!");
    }
    physical_devices.resize(device_count);
    vkEnumeratePhysicalDevices(instance, &device_count, physical_devices.data());
    active_devices.resize(device_count, nullptr);
}

VulkanManager::~VulkanManager() {
    // Order matters: drain the queues, release pooled buffers, then destroy the
    // resources that depend on the logical devices.
    for (auto& dev : active_devices) {
        if (dev) {
            dev->synchronize();
        }
    }
    VulkanMemoryManager::instance().shutdown();
    VulkanPipelineManager::instance().clear();
    active_devices.clear();
    if (instance != VK_NULL_HANDLE) {
        vkDestroyInstance(instance, nullptr);
        instance = VK_NULL_HANDLE;
    }
}

VulkanManager& VulkanManager::instance_ref() {
    // Deliberately heap-allocated so C++ static exit order cannot race with
    // NVIDIA/Intel driver internal static destructors at process exit.
    // Cleanup is driven deterministically via Python's atexit hook.
    static VulkanManager* manager = new VulkanManager();
    return *manager;
}

uint32_t VulkanManager::get_device_count() {
    return static_cast<uint32_t>(physical_devices.size());
}

DeviceInfo VulkanManager::get_device_info(uint32_t device_id) {
    if (device_id >= physical_devices.size()) {
        throw std::out_of_range("Device ID out of range: " + std::to_string(device_id));
    }
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physical_devices[device_id], &props);

    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(physical_devices[device_id], &mem_props);

    VkDeviceSize total_vram = 0;
    for (uint32_t i = 0; i < mem_props.memoryHeapCount; ++i) {
        if (mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            total_vram += mem_props.memoryHeaps[i].size;
        }
    }

    std::string vendor = "Unknown";
    if (props.vendorID == 0x1002) vendor = "AMD";
    else if (props.vendorID == 0x10DE) vendor = "NVIDIA";
    else if (props.vendorID == 0x8086) vendor = "Intel";
    else if (props.vendorID == 0x13B5) vendor = "ARM";
    else if (props.vendorID == 0x5143) vendor = "Qualcomm";

    DeviceInfo info;
    info.id = device_id;
    info.name = props.deviceName;
    info.vendor = vendor;
    info.vendor_id = props.vendorID;
    info.device_id = props.deviceID;
    info.api_major = VK_API_VERSION_MAJOR(props.apiVersion);
    info.api_minor = VK_API_VERSION_MINOR(props.apiVersion);
    info.total_memory = total_vram;
    info.is_discrete = (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU);

    // Populate cooperative matrix capabilities
    auto dev = get_device(device_id);
    info.has_cooperative_matrix = dev->has_cooperative_matrix;
    info.cooperative_matrix_configs = dev->cooperative_matrix_configs;
    info.capabilities = dev->capabilities;

    return info;
}

std::vector<DeviceInfo> VulkanManager::get_all_device_info() {
    std::vector<DeviceInfo> list;
    for (uint32_t i = 0; i < physical_devices.size(); ++i) {
        list.push_back(get_device_info(i));
    }
    return list;
}

std::shared_ptr<VulkanDeviceContext> VulkanManager::get_device(uint32_t device_id) {
    std::lock_guard<std::mutex> lock(manager_mutex);
    if (device_id >= physical_devices.size()) {
        throw std::out_of_range("Device ID " + std::to_string(device_id) + " does not exist!");
    }
    if (!active_devices[device_id]) {
        active_devices[device_id] = std::make_shared<VulkanDeviceContext>(device_id, physical_devices[device_id]);
    }
    return active_devices[device_id];
}

void VulkanManager::synchronize_device(uint32_t device_id) {
    auto dev = get_device(device_id);
    dev->wait_idle();
}

void VulkanManager::synchronize_all() {
    // Snapshot under the manager lock, then synchronize without holding it:
    // a dispatch takes manager_mutex (to fetch the device) before submit_mutex,
    // so holding manager_mutex while waiting on submit_mutex would invert the
    // order and risk a deadlock.
    std::vector<std::shared_ptr<VulkanDeviceContext>> devices;
    {
        std::lock_guard<std::mutex> lock(manager_mutex);
        devices = active_devices;
    }

    // Submit everywhere first, then wait everywhere. Waiting device-by-device
    // would submit device N only after device N-1 drained, serialising work
    // that could have run concurrently.
    for (auto& dev : devices) {
        if (dev) {
            dev->flush();
        }
    }
    for (auto& dev : devices) {
        if (dev) {
            dev->synchronize();
        }
    }
}

} // namespace vultorch
