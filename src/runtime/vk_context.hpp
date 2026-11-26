#pragma once

#include <atomic>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

namespace vultorch {

class VulkanBuffer;
class VulkanGraph;

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
  bool subgroup_size_control = false;
  bool shader_atomic_float = false;
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
  // Dispatches per batch ceiling. Together with kSlotCount this bounds how many
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
  VkDeviceSize total_vram_bytes = 0;
  DeviceCapabilities capabilities;
  bool has_cooperative_matrix = false;
  std::vector<CooperativeMatrixConfig> cooperative_matrix_configs;

  VulkanDeviceContext(uint32_t id, VkPhysicalDevice pdev);
  ~VulkanDeviceContext();

  // VRAM-proportional memory and command-batching budgets so 2 GB laptop GPUs
  // (MX450) and 10-24 GB desktop GPUs (RTX 3080, RX 7900 XTX) each scale
  // automatically to their hardware capacity.
  VkDeviceSize preferred_arena_slab_bytes() const {
    constexpr VkDeviceSize kGiB = 1024ULL * 1024ULL * 1024ULL;
    constexpr VkDeviceSize kMiB = 1024ULL * 1024ULL;
    if (total_vram_bytes >= 16ULL * kGiB) return 512ULL * kMiB;
    if (total_vram_bytes > 4ULL * kGiB) return 256ULL * kMiB;
    return 64ULL * kMiB;
  }

  VkDeviceSize pool_byte_budget() const {
    constexpr VkDeviceSize kMiB = 1024ULL * 1024ULL;
    VkDeviceSize raw = total_vram_bytes / 16ULL;
    if (raw < 128ULL * kMiB) return 128ULL * kMiB;
    if (raw > 2048ULL * kMiB) return 2048ULL * kMiB;
    return raw;
  }

  VkDeviceSize max_in_flight_temp_bytes() const {
    constexpr VkDeviceSize kMiB = 1024ULL * 1024ULL;
    VkDeviceSize raw = total_vram_bytes / 32ULL;
    if (raw < 48ULL * kMiB) return 48ULL * kMiB;
    if (raw > 1024ULL * kMiB) return 1024ULL * kMiB;
    return raw;
  }

  VkDeviceSize max_workspace_bytes() const {
    return preferred_arena_slab_bytes() / 4ULL;
  }

  uint32_t effective_max_ops_per_batch() const {
    constexpr VkDeviceSize kGiB = 1024ULL * 1024ULL * 1024ULL;
    return (total_vram_bytes > 0 && total_vram_bytes <= 4ULL * kGiB) ? 32u : kMaxOpsPerBatch;
  }

  uint32_t find_memory_type(uint32_t type_filter,
                            VkMemoryPropertyFlags required_flags) const;

  // Records one dispatch into the open batch, opening or flushing as needed.
  //
  // The callback runs while the device's submission lock is held, so all
  // descriptor-pool access (vkAllocateDescriptorSets/vkUpdateDescriptorSets)
  // is serialised per device, as Vulkan requires. It returns the descriptor
  // set it allocated; the device keeps it alive until the batch completes.
  //
  // Templated rather than std::function so the hot path does not allocate.
  template <typename Recorder>
  void record_dispatch(
      Recorder &&recorder,
      const std::vector<std::shared_ptr<VulkanBuffer>> &buffers = {}) {
    std::lock_guard<std::mutex> lock(submit_mutex);

    if (active_graph != nullptr) {
      VkCommandBuffer gcmd = active_graph_cmd();
      const DescriptorAllocation descriptor = recorder(gcmd);
      add_compute_barrier(gcmd);
      finish_graph_dispatch(descriptor, buffers);
      return;
    }

    if (pending_descriptors.size() >= kMaxPendingDescSets) {
      // Pool pressure: complete outstanding work so sets can be recycled.
      synchronize_locked();
    }

    open_batch_locked();
    Slot &slot = slots[static_cast<size_t>(open_slot)];
    VkCommandBuffer cmd = slot.cmd;

    const DescriptorAllocation descriptor = recorder(cmd);
    add_compute_barrier(cmd);

    if (descriptor.set != VK_NULL_HANDLE) {
      pending_descriptors.push_back(descriptor);
    }
    if (!buffers.empty()) {
      slot.pending_buffers.insert(slot.pending_buffers.end(), buffers.begin(),
                                  buffers.end());
    }

    ++ops_in_batch;
    dispatches_recorded.fetch_add(1, std::memory_order_relaxed);

    if (ops_in_batch >= effective_max_ops_per_batch()) {
      flush_locked();
    }
  }

  // Records a device-to-device buffer copy into the open batch without
  // flushing or waiting on the host.
  void record_copy(VkBuffer src, VkBuffer dst, VkDeviceSize size_bytes,
                   VkDeviceSize src_offset = 0, VkDeviceSize dst_offset = 0,
                   std::shared_ptr<VulkanBuffer> src_keep = nullptr,
                   std::shared_ptr<VulkanBuffer> dst_keep = nullptr);

  // Submits the open batch without waiting. Called automatically when a batch
  // fills or when anything needs to read results back.
  void flush();

  // Submits any open batch, waits for all previously submitted work on this
  // queue, and reclaims descriptor sets / recycled buffers.
  void synchronize();

  // Ordering point for host transfers: flushes pending compute, runs the
  // recorded copy, and waits for it to complete before returning.
  void submit_transfer(const std::function<void(VkCommandBuffer)> &record);

  void wait_idle() { synchronize(); }

  uint64_t submitted_batches() const { return batches_submitted.load(); }
  uint64_t recorded_dispatches() const { return dispatches_recorded.load(); }
  uint32_t open_batch_ops() const { return ops_in_batch; }

private:
  friend class VulkanGraph;

  struct Slot {
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    bool in_use = false;
    std::vector<std::shared_ptr<VulkanBuffer>> pending_buffers;
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
  VulkanGraph *active_graph = nullptr;
  bool graph_submitted_since_sync = false;

  void open_batch_locked();
  void flush_locked();
  void synchronize_locked();
  static void add_compute_barrier(VkCommandBuffer cmd);

  VkCommandBuffer active_graph_cmd() const;
  void finish_graph_dispatch(
      const DescriptorAllocation &descriptor,
      const std::vector<std::shared_ptr<VulkanBuffer>> &buffers);
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
  static VulkanManager &instance_ref();

  VkInstance get_instance() const { return instance; }
  uint32_t get_device_count();
  DeviceInfo get_device_info(uint32_t device_id);
  std::vector<DeviceInfo> get_all_device_info();
  std::shared_ptr<VulkanDeviceContext> get_device(uint32_t device_id);
  void synchronize_device(uint32_t device_id);
  void synchronize_all();
};

} // namespace vultorch
