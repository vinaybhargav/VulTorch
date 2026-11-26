#pragma once

#include "vk_context.hpp"
#include "vk_memory.hpp"
#include <array>
#include <memory>
#include <mutex>
#include <vector>
#include <vulkan/vulkan.h>

namespace vultorch {

class VulkanGraph {
public:
  explicit VulkanGraph(uint32_t device_id = 0);
  ~VulkanGraph();

  VulkanGraph(const VulkanGraph &) = delete;
  VulkanGraph &operator=(const VulkanGraph &) = delete;

  void capture_begin();
  void capture_end();
  void replay();
  void release();

  uint32_t device_id() const { return dev_id; }
  uint32_t num_dispatches() const { return dispatches; }
  bool is_captured() const { return captured; }
  bool is_capturing() const { return capturing; }

  static void release_all();

private:
  friend class VulkanDeviceContext;

  static constexpr uint32_t kGraphFenceRing = 4;

  uint32_t dev_id = 0;
  std::shared_ptr<VulkanDeviceContext> device_ctx;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  std::array<VkFence, kGraphFenceRing> fences{};
  std::array<bool, kGraphFenceRing> fence_in_flight{};
  uint32_t next_fence = 0;
  bool capturing = false;
  bool captured = false;
  uint32_t dispatches = 0;

  std::vector<DescriptorAllocation> owned_descriptors;
  std::vector<std::shared_ptr<VulkanBuffer>> pinned_buffers;
};

} // namespace vultorch
