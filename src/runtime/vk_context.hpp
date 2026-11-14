#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <string>
#include <memory>
#include <mutex>
#include <atomic>
#include <functional>
#include <stdexcept>
#include <iostream>

namespace vultorch {

struct CooperativeMatrixConfig {
    uint32_t M;
    uint32_t N;
    uint32_t K;
    uint32_t a_type;
    uint32_t b_type;
    uint32_t c_type;
    uint32_t result_type;
    uint32_t scope;
};

// What a device can actually do. Kernel selection filters on this: a variant
// is only ever considered if the device reports it can run it.
struct DeviceCapabilities {
    uint32_t subgroup_size = 0;
    uint32_t max_workgroup_invocations = 0;
    uint32_t max_shared_memory_bytes = 0;
    bool float16_arithmetic = false;
    bool storage_16bit = false;
    bool subgroup_basic = false;
    bool subgroup_arithmetic = false;
    // A 16x16x16 f16 x f16 -> f32 cooperative-matrix configuration at subgroup
    // scope. Required by shaders/matmul_coopmat.comp.
    bool fp16_coopmat_16x16 = false;
};

struct DeviceInfo {
    uint32_t id;
    std::string name;
    std::string vendor;
    uint32_t vendor_id = 0;
    uint32_t device_id = 0;
    uint32_t api_major;
    uint32_t api_minor;
    VkDeviceSize total_memory;
    bool is_discrete;
    bool has_cooperative_matrix = false;
    std::vector<CooperativeMatrixConfig> cooperative_matrix_configs;
    DeviceCapabilities capabilities;
};

// A descriptor set cannot be freed until the batch that references it has
// completed, so ownership travels with the dispatch until synchronize().
struct DescriptorAllocation {
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
};

class VulkanDeviceContext {
public:
    // Command buffers and fences are recycled in a ring. A slot may only be
    // reused once its fence has signalled.
    static constexpr uint32_t kSlotCount = 4;
    // Dispatches per batch. Together with kSlotCount this bounds how many
    // descriptor sets can be outstanding.
    static constexpr uint32_t kMaxOpsPerBatch = 256;
    // When this many sets await reclamation, the device synchronises so they
    // return to their pools. Must stay well below the per-pool set limit.
    static constexpr size_t kMaxPendingDescSets = 1024;

    uint32_t device_id;
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue compute_queue = VK_NULL_HANDLE;
    uint32_t compute_queue_family = 0;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties properties{};
    VkPhysicalDeviceMemoryProperties memory_properties{};
    DeviceCapabilities capabilities;
    bool has_cooperative_matrix = false;
    std::vector<CooperativeMatrixConfig> cooperative_matrix_configs;

    VulkanDeviceContext(uint32_t id, VkPhysicalDevice pdev);
    ~VulkanDeviceContext();

    uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags required_flags) const;

    // Records one dispatch into the open batch, opening or flushing as needed.
    //
    // The callback runs while the device's submission lock is held, so all
    // descriptor-pool access (vkAllocateDescriptorSets/vkUpdateDescriptorSets)
    // is serialised per device, as Vulkan requires. It returns the descriptor
    // set it allocated; the device keeps it alive until the batch completes.
    //
    // Templated rather than std::function so the hot path does not allocate.
    template <typename Recorder>
    void record_dispatch(Recorder&& recorder) {
        std::lock_guard<std::mutex> lock(submit_mutex);

        if (pending_descriptors.size() >= kMaxPendingDescSets) {
            // Pool pressure: complete outstanding work so sets can be recycled.
            synchronize_locked();
        }

        open_batch_locked();
        VkCommandBuffer cmd = slots[static_cast<size_t>(open_slot)].cmd;

        const DescriptorAllocation descriptor = recorder(cmd);
        add_compute_barrier(cmd);

        if (descriptor.set != VK_NULL_HANDLE) {
            pending_descriptors.push_back(descriptor);
        }

        ++ops_in_batch;
        dispatches_recorded.fetch_add(1, std::memory_order_relaxed);

        if (ops_in_batch >= kMaxOpsPerBatch) {
            flush_locked();
        }
    }

    // Records a device-to-device buffer copy into the open batch without
    // flushing or waiting on the host.
    void record_copy(VkBuffer src, VkBuffer dst, VkDeviceSize size_bytes);

    // Submits the open batch without waiting. Called automatically when a batch
    // fills or when anything needs to read results back.
    void flush();

    // Submits any open batch, waits for all previously submitted work on this
    // queue, and reclaims descriptor sets / recycled buffers.
    void synchronize();

    // Ordering point for host transfers: flushes pending compute, runs the
    // recorded copy, and waits for it to complete before returning.
    void submit_transfer(const std::function<void(VkCommandBuffer)>& record);

    void wait_idle() { synchronize(); }

    uint64_t submitted_batches() const { return batches_submitted.load(); }
    uint64_t recorded_dispatches() const { return dispatches_recorded.load(); }
    uint32_t open_batch_ops() const { return ops_in_batch; }

private:
    struct Slot {
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        bool in_use = false;
    };

    mutable std::mutex submit_mutex;
    std::vector<Slot> slots;
    uint32_t next_slot = 0;
    int32_t open_slot = -1;
    uint32_t ops_in_batch = 0;
    int32_t last_submitted_slot = -1;
    VkFence transfer_fence = VK_NULL_HANDLE;
    std::vector<DescriptorAllocation> pending_descriptors;

    std::atomic<uint64_t> batches_submitted{0};
    std::atomic<uint64_t> dispatches_recorded{0};

    void open_batch_locked();
    void flush_locked();
    void synchronize_locked();
    static void add_compute_barrier(VkCommandBuffer cmd);
};

class VulkanManager {
private:
    VkInstance instance = VK_NULL_HANDLE;
    std::vector<VkPhysicalDevice> physical_devices;
    std::vector<std::shared_ptr<VulkanDeviceContext>> active_devices;
    std::mutex manager_mutex;

    VulkanManager();
    ~VulkanManager();

public:
    static VulkanManager& instance_ref();

    VkInstance get_instance() const { return instance; }
    uint32_t get_device_count();
    DeviceInfo get_device_info(uint32_t device_id);
    std::vector<DeviceInfo> get_all_device_info();
    std::shared_ptr<VulkanDeviceContext> get_device(uint32_t device_id);
    void synchronize_device(uint32_t device_id);
    void synchronize_all();
};

} // namespace vultorch
