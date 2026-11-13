#include "vk_context.hpp"
#include "vk_pipeline.hpp"
#include <cstring>
#include <sstream>

namespace vultorch {

VulkanDeviceContext::VulkanDeviceContext(uint32_t id, VkPhysicalDevice pdev)
    : device_id(id), physical_device(pdev) {
    vkGetPhysicalDeviceProperties(physical_device, &properties);
    vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);

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

    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_create_info{};
    queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_create_info.queueFamilyIndex = compute_queue_family;
    queue_create_info.queueCount = 1;
    queue_create_info.pQueuePriorities = &queue_priority;

    VkPhysicalDeviceFeatures device_features{};
    device_features.shaderFloat64 = VK_FALSE;

    VkDeviceCreateInfo device_create_info{};
    device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    if (has_cooperative_matrix) {
        device_create_info.pNext = &coop_features;
    }
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
}

VulkanDeviceContext::~VulkanDeviceContext() {
    if (device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device);
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

VkCommandBuffer VulkanDeviceContext::begin_single_time_commands() {
    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandPool = command_pool;
    alloc_info.commandBufferCount = 1;

    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device, &alloc_info, &cmd);

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(cmd, &begin_info);
    return cmd;
}

void VulkanDeviceContext::end_single_time_commands(VkCommandBuffer cmd) {
    vkEndCommandBuffer(cmd);

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd;

    VkFence fence;
    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    vkCreateFence(device, &fence_info, nullptr, &fence);

    vkQueueSubmit(compute_queue, 1, &submit_info, fence);
    vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);

    vkDestroyFence(device, fence, nullptr);
    vkFreeCommandBuffers(device, command_pool, 1, &cmd);
}

void VulkanDeviceContext::wait_idle() {
    if (device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device);
    }
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
    // Destroy pipelines before destroying logical devices
    VulkanPipelineManager::instance().clear();
    active_devices.clear();
    if (instance != VK_NULL_HANDLE) {
        vkDestroyInstance(instance, nullptr);
        instance = VK_NULL_HANDLE;
    }
}

VulkanManager& VulkanManager::instance_ref() {
    static VulkanManager manager;
    return manager;
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
    info.api_major = VK_API_VERSION_MAJOR(props.apiVersion);
    info.api_minor = VK_API_VERSION_MINOR(props.apiVersion);
    info.total_memory = total_vram;
    info.is_discrete = (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU);

    // Populate cooperative matrix capabilities
    auto dev = get_device(device_id);
    info.has_cooperative_matrix = dev->has_cooperative_matrix;
    info.cooperative_matrix_configs = dev->cooperative_matrix_configs;

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
    std::lock_guard<std::mutex> lock(manager_mutex);
    for (size_t i = 0; i < active_devices.size(); ++i) {
        if (active_devices[i]) {
            active_devices[i]->wait_idle();
        }
    }
}

} // namespace vultorch
