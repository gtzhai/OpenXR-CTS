// Copyright (c) 2019-2025 The Khronos Group Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "utilities/array_size.h"
#ifdef XR_USE_GRAPHICS_API_VULKAN

#include "RGBAImage.h"
#include "conformance_utils.h"
#include "graphics_plugin.h"
#include "graphics_plugin_impl_helpers.h"
#include "graphics_plugin_vulkan_gltf.h"
#include "report.h"
#include "swapchain_image_data.h"

#include "common/hex_and_handles.h"
#include "common/vulkan_debug_object_namer.hpp"
#include "common/xr_linear.h"

#include "pbr/GltfLoader.h"
#include "pbr/PbrCommon.h"
#include "pbr/PbrModel.h"
#include "pbr/Vulkan/VkCommon.h"
#include "pbr/Vulkan/VkModel.h"
#include "pbr/Vulkan/VkResources.h"
#include "pbr/Vulkan/VkTexture.h"
#include "utilities/Geometry.h"
#include "utilities/swapchain_format_data.h"
#include "utilities/swapchain_parameters.h"
#include "utilities/throw_helpers.h"
#include "utilities/utils.h"
#include "utilities/vulkan_utils.h"

#include <catch2/catch_test_macros.hpp>
#include <nonstd/span.hpp>
#include <nonstd/type.hpp>
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>
#include <vulkan/vk_platform.h>
#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <array>
#include <assert.h>
#include <cstdint>
#include <inttypes.h>
#include <initializer_list>
#include <iterator>
#include <map>
#include <memory>
#include <stddef.h>
#include <stdexcept>
#include <string.h>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "common/android_logging.h"

namespace tinygltf
{
    class Model;
}  // namespace tinygltf

#ifdef USE_CHECKPOINTS
#include <unordered_set>
#endif

namespace Conformance
{
    struct IPlatformPlugin;
    PFN_vkCreateRenderPass2KHR vkCreateRenderPass2KHR{nullptr};

#ifdef USE_ONLINE_VULKAN_SHADERC
    constexpr char VertexShaderGlsl[] =
        R"_(
    #version 430
    #extension GL_ARB_separate_shader_objects : enable

    layout (std140, push_constant) uniform buf
    {
        mat4 mvp;
        mat4 vp;
        mat4 prevVp;
        mat4 model;
        mat4 prevModel;
        vec4 tintColor;
        vec4 alpha;
    } ubuf;

    layout (location = 0) in vec3 Position;
    layout (location = 1) in vec3 Color;

    layout (location = 0) out vec4 oColor;
    out gl_PerVertex
    {
        vec4 gl_Position;
    };

    void main()
    {
        oColor.rgb = mix(Color.rgb, ubuf.tintColor.rgb, ubuf.tintColor.a);
        oColor.a  = ubuf.alpha.x;
        gl_Position = ubuf.mvp * vec4(Position, 1);
    }
)_";

    constexpr char FragmentShaderGlsl[] =
        R"_(
    #version 430
    #extension GL_ARB_separate_shader_objects : enable

    layout (location = 0) in vec4 oColor;

    layout (location = 0) out vec4 FragColor;

    void main()
    {
        FragColor = oColor;
    }
)_";
    constexpr char VertexShaderMotionVectorGlsl[] =
        R"_(
    #version 430
    #extension GL_ARB_separate_shader_objects : enable

    layout (std140, push_constant) uniform buf
    {
        mat4 mvp;
        mat4 vp;
        mat4 prevVp;
        mat4 model;
        mat4 prevModel;
        vec4 tintColor;
        float alpha;
    } ubuf;

    layout (location = 0) in vec3 Position;
    layout (location = 1) in vec3 Color;

    layout (location = 0) out vec4 clipPos;
    layout (location = 1) out vec4 prevClipPos;
    out gl_PerVertex
    {
        vec4 gl_Position;
    };

    void main()
    {
        clipPos = ubuf.vp * ( ubuf.model * vec4( Position, 1.0 ) );
        prevClipPos = ubuf.prevVp * ( ubuf.prevModel* vec4( Position, 1.0 ) );
        gl_Position = clipPos;
    }
)_";

    constexpr char FragmentShaderMotionVectorGlsl[] =
        R"_(
    #version 430
    #extension GL_ARB_separate_shader_objects : enable

    layout (location = 0) in vec4 clipPos;
    layout (location = 1) in vec4 prevClipPos;

    layout (location = 0) out vec4 FragColor;

    void main()
    {
        highp vec4 motionVector = ( clipPos / clipPos.w - prevClipPos / prevClipPos.w );
        outColor = motionVector;
    }
)_";
#endif  // USE_ONLINE_VULKAN_SHADERC

    struct VulkanArraySliceState
    {
        VulkanArraySliceState() = default;
        VulkanArraySliceState(const VulkanArraySliceState&) = delete;
        std::vector<RenderTarget> m_renderTarget;  // per swapchain index
        RenderPass m_rp{};
        Pipeline m_pipe{};
        Pipeline m_pipeCompute{};

        void init(const VulkanDebugObjectNamer& namer, VkDevice device, uint32_t capacity, const VkExtent2D size, VkFormat colorFormat,
                  VkFormat depthFormat, VkSampleCountFlagBits sampleCount, const PipelineLayout& layout,
                  const PipelineLayout& computeLayout, const ShaderProgram& sp, const ShaderProgram& spCompute,
                  const VkVertexInputBindingDescription& bindDesc, span<const VkVertexInputAttributeDescription> attrDesc)
        {
            bool msaa_enable = GetGlobalData().IsUsingMSAA();
            bool multiview_enable =  Conformance::GetGlobalData().IsUsingMultiview();

            m_renderTarget.resize(capacity);
            m_rp.Create(namer, device, colorFormat, depthFormat, sampleCount, msaa_enable, multiview_enable);
            VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_VIEWPORT};
            m_pipe.Create(device, size, layout, m_rp, sp, bindDesc, attrDesc, dynamicStates);

            m_pipeCompute.Create(device, size, computeLayout, m_rp, spCompute, bindDesc, attrDesc, dynamicStates);
        }

        void Reset()
        {
            m_pipe.Reset();
            m_pipeCompute.Reset();
            m_rp.Reset();
            m_renderTarget.clear();
        }
    };

    /// Vulkan data used per swapchain. One per XrSwapchain handle.
    class VulkanSwapchainImageData : public SwapchainImageDataBase<XrSwapchainImageVulkanKHR>
    {
        void init(uint32_t capacity, VkFormat colorFormat, const PipelineLayout& layout, const PipelineLayout& computeLayout,
                  const ShaderProgram& sp, const ShaderProgram& spCompute, const VkVertexInputBindingDescription& bindDesc,
                  span<const VkVertexInputAttributeDescription> attrDesc)
        {
            m_depthBuffer.resize(capacity);
            #ifdef FEATURE_ADD_MSAA
            m_depthBufferMSAA.resize(capacity);
            m_colorBufferMSAA.resize(capacity);
            #endif

            for (auto& slice : m_slices) {
                slice.init(m_namer, m_vkDevice, capacity, m_size, colorFormat, m_depthFormat, m_sampleCount, layout, computeLayout, sp,
                           spCompute, bindDesc, attrDesc);
            }

            #ifdef FEATURE_ADD_MSAA
            bool msaa_enable = GetGlobalData().IsUsingMSAA();
            if(msaa_enable){
                prepareMSAAImages(colorFormat, m_depthFormat);
            }
            #endif
        }

    public:
        VulkanSwapchainImageData(const VulkanDebugObjectNamer& namer, uint32_t capacity, const XrSwapchainCreateInfo& swapchainCreateInfo,
                                 VkDevice device, MemoryAllocator* memAllocator, const PipelineLayout& layout,
                                 const PipelineLayout& computeLayout, const ShaderProgram& sp, const ShaderProgram& spCompute,
                                 const VkVertexInputBindingDescription& bindDesc, span<const VkVertexInputAttributeDescription> attrDesc)
            : SwapchainImageDataBase(XR_TYPE_SWAPCHAIN_IMAGE_VULKAN_KHR, capacity, swapchainCreateInfo)
            , m_namer(namer)
            , m_vkDevice(device)
            , m_memAllocator(memAllocator)
            , m_size{swapchainCreateInfo.width, swapchainCreateInfo.height}
            , m_sampleCount{(VkSampleCountFlagBits)swapchainCreateInfo.sampleCount}
            , m_slices(swapchainCreateInfo.arraySize)
        {
            init(capacity, (VkFormat)swapchainCreateInfo.format, layout, computeLayout, sp, spCompute, bindDesc, attrDesc);
        }

        VulkanSwapchainImageData(const VulkanDebugObjectNamer& namer, uint32_t capacity, const XrSwapchainCreateInfo& swapchainCreateInfo,
                                 XrSwapchain depthSwapchain, const XrSwapchainCreateInfo& depthSwapchainCreateInfo, VkDevice device,
                                 MemoryAllocator* memAllocator, const PipelineLayout& layout, const PipelineLayout& computeLayout,
                                 const ShaderProgram& sp, const ShaderProgram& spCompute, const VkVertexInputBindingDescription& bindDesc,
                                 span<const VkVertexInputAttributeDescription> attrDesc)
            : SwapchainImageDataBase(XR_TYPE_SWAPCHAIN_IMAGE_VULKAN_KHR, capacity, swapchainCreateInfo, depthSwapchain,
                                     depthSwapchainCreateInfo)
            , m_namer(namer)
            , m_vkDevice(device)
            , m_memAllocator(memAllocator)
            , m_size{swapchainCreateInfo.width, swapchainCreateInfo.height}
            , m_sampleCount{(VkSampleCountFlagBits)swapchainCreateInfo.sampleCount}
            , m_depthFormat((VkFormat)depthSwapchainCreateInfo.format)
            , m_slices(swapchainCreateInfo.arraySize)
        {
            init(capacity, (VkFormat)swapchainCreateInfo.format, layout, computeLayout, sp, spCompute, bindDesc, attrDesc);
        }

        ~VulkanSwapchainImageData() override
        {
            // Calling a virtual function from a destructor doesn't work the way you'd expect, so we do this here.
            VulkanSwapchainImageData::Reset();
        }

        void BindRenderTarget(uint32_t index, uint32_t arraySlice, const VkRect2D& renderArea, VkImageAspectFlags secondAttachmentAspect,
                              VkRenderPassBeginInfo* renderPassBeginInfo)
        {
            bool msaa_enable = GetGlobalData().IsUsingMSAA();
            RenderTarget& rt = m_slices[arraySlice].m_renderTarget[index];
            RenderPass& rp = m_slices[arraySlice].m_rp;
            if (rt.fb == VK_NULL_HANDLE) {
                bool multiview_enable = GetGlobalData().IsUsingMultiview();
            #ifdef FEATURE_ADD_MSAA
                rt.Create(m_namer, m_vkDevice, GetTypedImage(index).image, GetDepthImageForColorIndex(index).image, secondAttachmentAspect,
                          arraySlice, m_size, rp, m_colorBufferMSAA[index].GetTexture().image, m_depthBufferMSAA[index].GetTexture().image, 
                          msaa_enable, multiview_enable);
            #else
                rt.Create(m_namer, m_vkDevice, GetTypedImage(index).image, GetDepthImageForColorIndex(index).image, secondAttachmentAspect,
                          arraySlice, m_size, rp, VK_NULL_HANDLE, VK_NULL_HANDLE, 
                          false, multiview_enable);
            #endif

            }
            renderPassBeginInfo->renderPass = rp.pass;
            renderPassBeginInfo->framebuffer = rt.fb;
            renderPassBeginInfo->renderArea = renderArea;
        }

        void BindPipeline(VkCommandBuffer buf, uint32_t arraySlice, enum ShaderProgramType programType = SHADER_PROGRAM_TYPE_GRAPHICS)
        {
            switch (programType) {
            case SHADER_PROGRAM_TYPE_GRAPHICS:
                vkCmdBindPipeline(buf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_slices[arraySlice].m_pipe.pipe);
                break;
            case SHADER_PROGRAM_TYPE_COMPUTE:
                vkCmdBindPipeline(buf, VK_PIPELINE_BIND_POINT_COMPUTE, m_slices[arraySlice].m_pipeCompute.pipe);
                break;
            default:
                XRC_THROW("unknown programType");
            }
        }

        void TransitionLayout(uint32_t imageIndex, CmdBuffer* cmdBuffer, VkImageLayout newLayout)
        {
            m_depthBuffer[imageIndex].TransitionLayout(cmdBuffer, newLayout);
            #ifdef FEATURE_ADD_MSAA
            bool msaa_enable = GetGlobalData().IsUsingMSAA();
            if(msaa_enable){
                m_depthBufferMSAA[imageIndex].TransitionLayout(cmdBuffer, newLayout);
            }
            #endif
        }

        void Reset() override
        {
            for (auto& slice : m_slices) {
                slice.Reset();
            }
            m_depthBuffer.clear();
            #ifdef FEATURE_ADD_MSAA
            m_depthBufferMSAA.clear();
            m_colorBufferMSAA.clear();
            #endif

            SwapchainImageDataBase::Reset();
        }

        int64_t GetDepthFormat() const
        {
            return m_depthFormat;
        }

        const std::vector<VulkanArraySliceState>& GetSlices() const
        {
            return m_slices;
        }

    protected:
        const XrSwapchainImageVulkanKHR& GetFallbackDepthSwapchainImage(uint32_t i) override
        {
            if (!m_depthBuffer[i].Allocated()) {
                bool isProtectedMemory = GetGlobalData().IsProtectedMemory();
                m_depthBuffer[i].Allocate(m_namer, m_vkDevice, m_memAllocator, m_depthFormat, this->Width(), this->Height(),
                                          this->ArraySize(), this->SampleCount(), isProtectedMemory);
            }

            return m_depthBuffer[i].GetTexture();
        }

    private:
        #ifdef FEATURE_ADD_MSAA
        void prepareMSAAImages( VkFormat colorFormat, VkFormat depthFormat){
            bool isProtectedMemory = GetGlobalData().IsProtectedMemory();
            for (auto& depthBuffer : m_depthBufferMSAA) {
                depthBuffer.Allocate(m_namer, m_vkDevice, m_memAllocator, depthFormat, this->Width(), this->Height(),
                                     this->ArraySize(), 4, isProtectedMemory);
            }
            for (auto& colorBuffer : m_colorBufferMSAA) {
                colorBuffer.Allocate(m_namer, m_vkDevice, m_memAllocator, colorFormat, this->Width(), this->Height(),
                                     this->ArraySize(), 4, isProtectedMemory);
            }
        }
        #endif

        VulkanDebugObjectNamer m_namer;
        VkDevice m_vkDevice{VK_NULL_HANDLE};
        MemoryAllocator* m_memAllocator{nullptr};
        VkExtent2D m_size{};
        VkSampleCountFlagBits m_sampleCount;
        std::vector<DepthBuffer> m_depthBuffer;  // per swapchain index
        #ifdef FEATURE_ADD_MSAA
        std::vector<DepthBuffer> m_depthBufferMSAA;
        std::vector<ColorBuffer> m_colorBufferMSAA;
        #endif
        VkFormat m_depthFormat{VK_FORMAT_D32_SFLOAT};

        std::vector<VulkanArraySliceState> m_slices;
    };

#if defined(USE_MIRROR_WINDOW)
    // Swapchain
    struct Swapchain
    {
        VkFormat format{VK_FORMAT_B8G8R8A8_SRGB};
        VkSurfaceKHR surface{VK_NULL_HANDLE};
        VkSwapchainKHR swapchain{VK_NULL_HANDLE};
        VkFence readyFence{VK_NULL_HANDLE};
        VkFence presentFence{VK_NULL_HANDLE};
        static const uint32_t maxImages = 4;
        uint32_t swapchainCount = 0;
        uint32_t renderImageIdx = 0;
        VkImage image[maxImages]{VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};

        Swapchain()
        {
        }
        ~Swapchain()
        {
            Reset();
        }

        void Create(VkInstance instance, VkPhysicalDevice physDevice, VkDevice device, uint32_t queueFamilyIndex);
        void Prepare(VkCommandBuffer buf);
        void Wait();
        void Acquire(VkSemaphore readySemaphore = VK_NULL_HANDLE);
        void Present(VkQueue queue, VkSemaphore drawComplete = VK_NULL_HANDLE);
        void Reset()
        {
            if (m_vkDevice) {
                // Flush any pending Present() calls which are using the fence
                Wait();
                if (swapchain)
                    vkDestroySwapchainKHR(m_vkDevice, swapchain, nullptr);
                if (readyFence)
                    vkDestroyFence(m_vkDevice, readyFence, nullptr);
            }

            if (m_vkInstance && surface)
                vkDestroySurfaceKHR(m_vkInstance, surface, nullptr);

            readyFence = VK_NULL_HANDLE;
            presentFence = VK_NULL_HANDLE;
            swapchain = VK_NULL_HANDLE;
            surface = VK_NULL_HANDLE;
            for (uint32_t i = 0; i < swapchainCount; ++i) {
                image[i] = VK_NULL_HANDLE;
            }
            swapchainCount = 0;

#if defined(VK_USE_PLATFORM_WIN32_KHR)
            if (hWnd) {
                DestroyWindow(hWnd);
                hWnd = nullptr;
                UnregisterClassW(L"conformance_test", hInst);
            }
            if (hUser32Dll) {
                ::FreeLibrary(hUser32Dll);
                hUser32Dll = nullptr;
            }
#endif

            m_vkDevice = nullptr;
        }
        void Recreate()
        {
            Reset();
            Create(m_vkInstance, m_vkPhysicalDevice, m_vkDevice, m_queueFamilyIndex);
        }

    private:
#if defined(VK_USE_PLATFORM_WIN32_KHR)
        HINSTANCE hInst{NULL};
        HWND hWnd{NULL};
        HINSTANCE hUser32Dll{NULL};
#endif
        const VkExtent2D size{640, 480};
        VkInstance m_vkInstance{VK_NULL_HANDLE};
        VkPhysicalDevice m_vkPhysicalDevice{VK_NULL_HANDLE};
        VkDevice m_vkDevice{VK_NULL_HANDLE};
        uint32_t m_queueFamilyIndex = 0;
    };

    void Swapchain::Create(VkInstance instance, VkPhysicalDevice physDevice, VkDevice device, uint32_t queueFamilyIndex)
    {
        m_vkInstance = instance;
        m_vkPhysicalDevice = physDevice;
        m_vkDevice = device;
        m_queueFamilyIndex = queueFamilyIndex;

// Create a WSI surface for the window:
#if defined(VK_USE_PLATFORM_WIN32_KHR)
        hInst = GetModuleHandle(NULL);

        WNDCLASSW wc{};
        wc.style = CS_CLASSDC;
        wc.lpfnWndProc = DefWindowProcW;
        wc.cbWndExtra = sizeof(this);
        wc.hInstance = hInst;
        wc.lpszClassName = L"conformance_test";
        RegisterClassW(&wc);

// adjust the window size and show at InitDevice time
#if defined(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)
        typedef DPI_AWARENESS_CONTEXT(WINAPI * PFN_SetThreadDpiAwarenessContext)(DPI_AWARENESS_CONTEXT);
        hUser32Dll = ::LoadLibraryA("user32.dll");
        if (PFN_SetThreadDpiAwarenessContext SetThreadDpiAwarenessContextFn =
                reinterpret_cast<PFN_SetThreadDpiAwarenessContext>(::GetProcAddress(hUser32Dll, "SetThreadDpiAwarenessContext"))) {
            // Make sure we're 1:1 for HMD pixels
            SetThreadDpiAwarenessContextFn(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
        }
#endif
        RECT rect{0, 0, (LONG)size.width, (LONG)size.height};
        AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, false);
        hWnd = CreateWindowW(wc.lpszClassName, L"conformance_test (Vulkan)", WS_OVERLAPPEDWINDOW | WS_VISIBLE, CW_USEDEFAULT, CW_USEDEFAULT,
                             rect.right - rect.left, rect.bottom - rect.top, 0, 0, hInst, 0);
        assert(hWnd != NULL);

        SetWindowLongPtr(hWnd, 0, LONG_PTR(this));

        VkWin32SurfaceCreateInfoKHR surfCreateInfo{VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR};
        surfCreateInfo.flags = 0;
        surfCreateInfo.hinstance = hInst;
        surfCreateInfo.hwnd = hWnd;
        XRC_CHECK_THROW_VKCMD(vkCreateWin32SurfaceKHR(m_vkInstance, &surfCreateInfo, nullptr, &surface));
#else
#error CreateSurface not supported on this OS
#endif  // defined(VK_USE_PLATFORM_WIN32_KHR)

        VkSurfaceCapabilitiesKHR surfCaps;
        XRC_CHECK_THROW_VKCMD(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_vkPhysicalDevice, surface, &surfCaps));
        XRC_CHECK_THROW(surfCaps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT);

        uint32_t surfFmtCount = 0;
        XRC_CHECK_THROW_VKCMD(vkGetPhysicalDeviceSurfaceFormatsKHR(m_vkPhysicalDevice, surface, &surfFmtCount, nullptr));
        std::vector<VkSurfaceFormatKHR> surfFmts(surfFmtCount);
        XRC_CHECK_THROW_VKCMD(vkGetPhysicalDeviceSurfaceFormatsKHR(m_vkPhysicalDevice, surface, &surfFmtCount, &surfFmts[0]));
        uint32_t foundFmt;
        for (foundFmt = 0; foundFmt < surfFmtCount; ++foundFmt) {
            if (surfFmts[foundFmt].format == format)
                break;
        }

        XRC_CHECK_THROW(foundFmt < surfFmtCount);

        uint32_t presentModeCount = 0;
        XRC_CHECK_THROW_VKCMD(vkGetPhysicalDeviceSurfacePresentModesKHR(m_vkPhysicalDevice, surface, &presentModeCount, nullptr));
        std::vector<VkPresentModeKHR> presentModes(presentModeCount);
        XRC_CHECK_THROW_VKCMD(vkGetPhysicalDeviceSurfacePresentModesKHR(m_vkPhysicalDevice, surface, &presentModeCount, &presentModes[0]));

        // Do not use VSYNC for the mirror window, but Nvidia doesn't support IMMEDIATE so fall back to MAILBOX
        VkPresentModeKHR presentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
        for (uint32_t i = 0; i < presentModeCount; ++i) {
            if ((presentModes[i] == VK_PRESENT_MODE_IMMEDIATE_KHR) || (presentModes[i] == VK_PRESENT_MODE_MAILBOX_KHR)) {
                presentMode = presentModes[i];
                break;
            }
        }

        VkBool32 presentable = false;
        XRC_CHECK_THROW_VKCMD(vkGetPhysicalDeviceSurfaceSupportKHR(m_vkPhysicalDevice, m_queueFamilyIndex, surface, &presentable));
        XRC_CHECK_THROW(presentable);

        VkSwapchainCreateInfoKHR swapchainInfo{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
        swapchainInfo.flags = 0;
        swapchainInfo.surface = surface;
        swapchainInfo.minImageCount = surfCaps.minImageCount;
        swapchainInfo.imageFormat = format;
        swapchainInfo.imageColorSpace = surfFmts[foundFmt].colorSpace;
        swapchainInfo.imageExtent = size;
        swapchainInfo.imageArrayLayers = 1;
        swapchainInfo.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        swapchainInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        swapchainInfo.queueFamilyIndexCount = 0;
        swapchainInfo.pQueueFamilyIndices = nullptr;
        swapchainInfo.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
        swapchainInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        swapchainInfo.presentMode = presentMode;
        swapchainInfo.clipped = true;
        swapchainInfo.oldSwapchain = VK_NULL_HANDLE;
        XRC_CHECK_THROW_VKCMD(vkCreateSwapchainKHR(m_vkDevice, &swapchainInfo, nullptr, &swapchain));

        // Fence to throttle host on Acquire
        VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        XRC_CHECK_THROW_VKCMD(vkCreateFence(m_vkDevice, &fenceInfo, nullptr, &readyFence));

        swapchainCount = 0;
        XRC_CHECK_THROW_VKCMD(vkGetSwapchainImagesKHR(m_vkDevice, swapchain, &swapchainCount, nullptr));
        assert(swapchainCount < maxImages);
        XRC_CHECK_THROW_VKCMD(vkGetSwapchainImagesKHR(m_vkDevice, swapchain, &swapchainCount, image));
        if (swapchainCount > maxImages) {
            //Log::Write(Log::Level::Info, "Reducing swapchain length from " + std::to_string(swapchainCount) + " to " + std::to_string(maxImages));
            swapchainCount = maxImages;
        }

        //Log::Write(Log::Level::Info, "Swapchain length " + std::to_string(swapchainCount));
    }

    void Swapchain::Prepare(VkCommandBuffer buf)
    {
        // Convert swapchain images to VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
        for (uint32_t i = 0; i < swapchainCount; ++i) {
            VkImageMemoryBarrier imgBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            imgBarrier.srcAccessMask = 0;  // XXX was VK_ACCESS_TRANSFER_READ_BIT wrong?
            imgBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            imgBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            imgBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
            imgBarrier.image = image[i];
            imgBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdPipelineBarrier(buf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                                 &imgBarrier);
        }
    }

    void Swapchain::Wait()
    {
        if (presentFence) {
            // Wait for the fence...
            XRC_CHECK_THROW_VKCMD(vkWaitForFences(m_vkDevice, 1, &presentFence, VK_TRUE, UINT64_MAX));
            // ...then reset the fence for future Acquire calls
            XRC_CHECK_THROW_VKCMD(vkResetFences(m_vkDevice, 1, &presentFence));
            presentFence = VK_NULL_HANDLE;
        }
    }

    void Swapchain::Acquire(VkSemaphore readySemaphore)
    {
        // If we're not using a semaphore to rate-limit the GPU, rate limit the host with a fence instead
        if (readySemaphore == VK_NULL_HANDLE) {
            Wait();
            presentFence = readyFence;
        }

        XRC_CHECK_THROW_VKCMD(vkAcquireNextImageKHR(m_vkDevice, swapchain, UINT64_MAX, readySemaphore, presentFence, &renderImageIdx));
    }

    void Swapchain::Present(VkQueue queue, VkSemaphore drawComplete)
    {
        VkPresentInfoKHR presentInfo{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        if (drawComplete) {
            presentInfo.waitSemaphoreCount = 1;
            presentInfo.pWaitSemaphores = &drawComplete;
        }
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &swapchain;
        presentInfo.pImageIndices = &renderImageIdx;
        auto res = vkQueuePresentKHR(queue, &presentInfo);
        if (res == VK_ERROR_OUT_OF_DATE_KHR) {
            Recreate();
            return;
        }
        XRC_CHECK_THROW_VKRESULT(res, "vkQueuePresentKHR");
    }
#endif  // defined(USE_MIRROR_WINDOW)

    struct VulkanMesh
    {
        static constexpr VkVertexInputAttributeDescription c_attrDesc[2] = {
            {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Geometry::Vertex, Position)},
            {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Geometry::Vertex, Color)}};
        static constexpr VkVertexInputBindingDescription c_bindingDesc = VertexBuffer<Geometry::Vertex>::c_bindingDesc;

        VertexBuffer<Geometry::Vertex> m_DrawBuffer;

        VulkanMesh(VkDevice device, const VulkanDebugObjectNamer& namer,  //
                   const MemoryAllocator* memAllocator,                   //
                   const uint16_t* idx_data, uint32_t idx_count,          //
                   const Geometry::Vertex* vtx_data, uint32_t vtx_count)
        {
            std::vector<VkVertexInputAttributeDescription> attrDesc(std::begin(c_attrDesc), std::end(c_attrDesc));
            m_DrawBuffer.Init(device, memAllocator, attrDesc);
            m_DrawBuffer.Create(idx_count, vtx_count);
            XRC_CHECK_THROW_VKCMD(namer.SetName(VK_OBJECT_TYPE_BUFFER, (uint64_t)m_DrawBuffer.idx.buf, "CTS mesh draw index buffer"));
            XRC_CHECK_THROW_VKCMD(namer.SetName(VK_OBJECT_TYPE_BUFFER, (uint64_t)m_DrawBuffer.vtx.buf, "CTS mesh draw vertex buffer"));

            m_DrawBuffer.UpdateIndices(span<const uint16_t>(idx_data, idx_count), 0);
            m_DrawBuffer.UpdateVertices(span<const Geometry::Vertex>(vtx_data, vtx_count), 0);
        }

        VulkanMesh(VulkanMesh&& other) noexcept
        {
            using std::swap;
            swap(m_DrawBuffer, other.m_DrawBuffer);
        }

        VulkanMesh(const VulkanMesh&) = delete;
    };
    constexpr VkVertexInputAttributeDescription VulkanMesh::c_attrDesc[];
    constexpr VkVertexInputBindingDescription VulkanMesh::c_bindingDesc;

    struct VulkanGraphicsPlugin : public IGraphicsPlugin
    {
        VulkanGraphicsPlugin(const std::shared_ptr<IPlatformPlugin>& /*unused*/);

        ~VulkanGraphicsPlugin() override;

        void Flush() override
        {
            if (m_vkDevice != VK_NULL_HANDLE) {
                vkDeviceWaitIdle(m_vkDevice);
            }
        }

        std::vector<std::string> GetInstanceExtensions() const override
        {
            return {XR_KHR_VULKAN_ENABLE2_EXTENSION_NAME};
        }
        virtual XrStructureType GetGraphicsBindingType() const
        {
            return XR_TYPE_GRAPHICS_BINDING_VULKAN2_KHR;
        }
        virtual XrStructureType GetSwapchainImageType() const
        {
            return XR_TYPE_SWAPCHAIN_IMAGE_VULKAN2_KHR;
        }
        virtual XrResult GetVulkanGraphicsRequirements2KHR(XrInstance instance, XrSystemId systemId,
                                                           XrGraphicsRequirementsVulkan2KHR* graphicsRequirements);
        virtual XrResult CreateVulkanInstanceKHR(XrInstance instance, const XrVulkanInstanceCreateInfoKHR* createInfo,
                                                 VkInstance* vulkanInstance, VkResult* vulkanResult);
        virtual XrResult GetVulkanGraphicsDevice2KHR(XrInstance instance, const XrVulkanGraphicsDeviceGetInfoKHR* getInfo,
                                                     VkPhysicalDevice* vulkanPhysicalDevice);
        virtual XrResult CreateVulkanDeviceKHR(XrInstance instance, const XrVulkanDeviceCreateInfoKHR* createInfo, VkDevice* vulkanDevice,
                                               VkResult* vulkanResult);

        // Note: The output must not outlive the input - this modifies the input and returns a collection of views into that modified
        // input!
        std::vector<const char*> ParseExtensionString(char* names)
        {
            std::vector<const char*> list;
            while (*names != 0) {
                list.push_back(names);
                while (*(++names) != 0) {
                    if (*names == ' ') {
                        *names++ = '\0';
                        break;
                    }
                }
            }
            return list;
        }

        bool Initialize() override;

        bool IsInitialized() const override;

        void Shutdown() override;
        void enum_vulkan_instance_exts();
        void enum_vulkan_device_exts();

        std::string DescribeGraphics() const override;

        bool InitializeDevice(XrInstance instance, XrSystemId systemId, bool checkGraphicsRequirements,
                              uint32_t deviceCreationFlags) override;

#ifdef USE_ONLINE_VULKAN_SHADERC
        std::vector<uint32_t> CompileGlslShader(const std::string& name, shaderc_shader_kind kind, const std::string& source);
#endif

        void InitializeResources();

        void ClearSwapchainCache() override;

        void ShutdownDevice() override;

        const XrBaseInStructure* GetGraphicsBinding() const override;

        std::string GetImageFormatName(int64_t imageFormat) const override;

        bool IsImageFormatKnown(int64_t imageFormat) const override;

        bool GetSwapchainCreateTestParameters(int64_t imageFormat, SwapchainCreateTestParameters* swapchainTestParameters) override;

        bool ValidateSwapchainImages(int64_t imageFormat, const SwapchainCreateTestParameters* tp, XrSwapchain swapchain,
                                     uint32_t* imageCount) const override;
        bool ValidateSwapchainImageState(XrSwapchain swapchain, uint32_t index, int64_t imageFormat) const override;

        int64_t SelectColorSwapchainFormat(bool throwIfNotFound, span<const int64_t> imageFormatArray) const override;

        int64_t SelectDepthSwapchainFormat(bool throwIfNotFound, span<const int64_t> imageFormatArray) const override;

        int64_t SelectMotionVectorSwapchainFormat(bool throwIfNotFound, span<const int64_t> imageFormatArray) const override;

        // Format required by RGBAImage type.
        int64_t GetSRGBA8Format() const override;

        ISwapchainImageData* AllocateSwapchainImageData(size_t size, const XrSwapchainCreateInfo& colorSwapchainCreateInfo) override;

        ISwapchainImageData* AllocateSwapchainImageDataWithDepthSwapchain(size_t size,
                                                                          const XrSwapchainCreateInfo& colorSwapchainCreateInfo,
                                                                          XrSwapchain depthSwapchain,
                                                                          const XrSwapchainCreateInfo& depthSwapchainCreateInfo, bool isMotionVector = false) override;

        // ISwapchainImageData * EnumerateSwapchainImageData(XrSwapchain colorSwapchain,
        //                                                                  const XrSwapchainCreateInfo& swapchainCreateInfo) override;

        void CopyRGBAImage(const XrSwapchainImageBaseHeader* swapchainImageBase, uint32_t arraySlice, const RGBAImage& image, int faceId = -1) override;

        void SetViewportAndScissor(const VkRect2D& rect);

        void ClearImageSlice(const XrSwapchainImageBaseHeader* colorSwapchainImage, uint32_t imageArrayIndex, XrColor4f color) override;

        MeshHandle MakeSimpleMesh(span<const uint16_t> idx, span<const Geometry::Vertex> vtx) override;

        GLTFModelHandle LoadGLTF(Gltf::ModelBuilder&& modelBuilder) override;
        std::shared_ptr<Pbr::Model> GetPbrModel(GLTFModelHandle handle) const override;
        GLTFModelInstanceHandle CreateGLTFModelInstance(GLTFModelHandle handle) override;
        Pbr::ModelInstance& GetModelInstance(GLTFModelInstanceHandle handle) override;

        void RenderView(const XrCompositionLayerProjectionView& layerView, const XrSwapchainImageBaseHeader* colorSwapchainImage,
                        const RenderParams& params, bool isMotionVectorPass = false, const XrCompositionLayerProjectionView* prevLayerView = nullptr,
                        const XrCompositionLayerProjectionView* nextLayerView = nullptr, const XrCompositionLayerProjectionView* nextPrevLayerView = nullptr,
                     const EnvDepthOcclusionParams* edoParams = nullptr) override;

        void RenderClearImageSliceCompute(const XrCompositionLayerProjectionView& layerView,
                                          const XrSwapchainImageBaseHeader* colorSwapchainImage, XrColor4f color) override;

        /// Get data on a known swapchain format
        const SwapchainFormatData& FindFormatData(int64_t format) const;

#if defined(USE_CHECKPOINTS)
        void Checkpoint(std::string msg)
        {
            auto check = checkpoints.emplace(std::move(msg));
            bool isProtectedMemory = GetGlobalData().IsProtectedMemory();
            VkCommandBuffer buf = isProtectedMemory?m_cmdBuffer.bufP:m_cmdBuffer.buf;
            vkCmdSetCheckpointNV(buf, check.first->c_str());
        }

        void ShowCheckpoints()
        {
            bool isProtectedMemory = GetGlobalData().IsProtectedMemory();
            VkQueue queue = isProtectedMemory?m_vkQueueProtected:m_vkQueue;
            if (queue != VK_NULL_HANDLE) {
                uint32_t count = 0;
                vkGetQueueCheckpointDataNV(queue, &count, nullptr);
                ReportF("ShowCheckpoints found %u checkpoints", count);
                if (count > 0) {
                    std::vector<VkCheckpointDataNV> data(count);
                    vkGetQueueCheckpointDataNV(queue, &count, &data[0]);
                    for (uint32_t i = 0; i < count; ++i) {
                        auto& c = data[i];
                        std::string stages = GetPipelineStages(c.stage);
                        ReportF("%3d: %s -%s", i, (const char*)c.pCheckpointMarker, stages.c_str());
                    }
                }
            }
        }
#endif  // defined(USE_CHECKPOINTS)

    protected:
        bool initialized{false};
        VkInstance m_vkInstance{VK_NULL_HANDLE};
        VkPhysicalDevice m_vkPhysicalDevice{VK_NULL_HANDLE};

    private:
        XrGraphicsBindingVulkan2KHR m_graphicsBinding{XR_TYPE_GRAPHICS_BINDING_VULKAN2_KHR};
        SwapchainImageDataMap<VulkanSwapchainImageData> m_swapchainImageDataMap;

        VkDevice m_vkDevice{VK_NULL_HANDLE};
        VulkanDebugObjectNamer m_namer{};
        uint32_t m_queueFamilyIndex = 0;
        VkQueue m_vkQueue{VK_NULL_HANDLE};
        VkQueue m_vkQueueProtected{VK_NULL_HANDLE};
        VkSemaphore m_vkDrawDone{VK_NULL_HANDLE};

        MemoryAllocator m_memAllocator{};

        ShaderProgram m_shaderProgram{SHADER_PROGRAM_TYPE_GRAPHICS};
        ShaderProgram m_computeShaderProgram{SHADER_PROGRAM_TYPE_COMPUTE};
        ShaderProgram m_mvShaderProgram{SHADER_PROGRAM_TYPE_GRAPHICS};
        ShaderProgram m_multiviewShaderProgram{SHADER_PROGRAM_TYPE_GRAPHICS};
        ShaderProgram m_envDepthShaderProgram{SHADER_PROGRAM_TYPE_GRAPHICS};
        CmdBuffer m_cmdBuffer{};
        PipelineLayout m_pipelineLayout{};

        PipelineLayout m_computePipelineLayout{};
        Conformance::ScopedVkDescriptorPool m_computeDescriptorPool;
        VkDescriptorSet m_ComputeDescriptorSet;

        Conformance::ScopedVkDescriptorPool m_gfxDescriptorPool;
        VkDescriptorSet m_gfxDescriptorSet;

        MeshHandle m_cubeMesh{};
        VectorWithGenerationCountedHandles<VulkanMesh, MeshHandle> m_meshes;
        // This is fine to be a shared_ptr because Model doesn't directly hold any graphics state.
        VectorWithGenerationCountedHandles<std::shared_ptr<Pbr::Model>, GLTFModelHandle> m_gltfModels;
        VectorWithGenerationCountedHandles<VulkanGLTF, GLTFModelInstanceHandle> m_gltfInstances;
        std::unique_ptr<Pbr::VulkanResources> m_pbrResources;

#if defined(USE_MIRROR_WINDOW)
        Swapchain m_swapchain{};
#endif

#if defined(USE_CHECKPOINTS)
        PFN_vkCmdSetCheckpointNV vkCmdSetCheckpointNV{};
        PFN_vkGetQueueCheckpointDataNV vkGetQueueCheckpointDataNV{};
        std::unordered_set<std::string> checkpoints{};
#endif

        PFN_vkCreateDebugUtilsMessengerEXT vkCreateDebugUtilsMessengerEXT{nullptr};
        PFN_vkDestroyDebugUtilsMessengerEXT vkDestroyDebugUtilsMessengerEXT{nullptr};
        VkDebugUtilsMessengerEXT m_vkDebugUtilsMessenger{VK_NULL_HANDLE};

        static std::string vkObjectTypeToString(VkObjectType objectType)
        {
            std::string objName;

#define LIST_OBJECT_TYPES(_)          \
    _(UNKNOWN)                        \
    _(INSTANCE)                       \
    _(PHYSICAL_DEVICE)                \
    _(DEVICE)                         \
    _(QUEUE)                          \
    _(SEMAPHORE)                      \
    _(COMMAND_BUFFER)                 \
    _(FENCE)                          \
    _(DEVICE_MEMORY)                  \
    _(BUFFER)                         \
    _(IMAGE)                          \
    _(EVENT)                          \
    _(QUERY_POOL)                     \
    _(BUFFER_VIEW)                    \
    _(IMAGE_VIEW)                     \
    _(SHADER_MODULE)                  \
    _(PIPELINE_CACHE)                 \
    _(PIPELINE_LAYOUT)                \
    _(RENDER_PASS)                    \
    _(PIPELINE)                       \
    _(DESCRIPTOR_SET_LAYOUT)          \
    _(SAMPLER)                        \
    _(DESCRIPTOR_POOL)                \
    _(DESCRIPTOR_SET)                 \
    _(FRAMEBUFFER)                    \
    _(COMMAND_POOL)                   \
    _(SURFACE_KHR)                    \
    _(SWAPCHAIN_KHR)                  \
    _(DISPLAY_KHR)                    \
    _(DISPLAY_MODE_KHR)               \
    _(DESCRIPTOR_UPDATE_TEMPLATE_KHR) \
    _(DEBUG_UTILS_MESSENGER_EXT)

            switch (objectType) {
            default:
#define MK_OBJECT_TYPE_CASE(name) \
    case VK_OBJECT_TYPE_##name:   \
        objName = #name;          \
        break;
                LIST_OBJECT_TYPES(MK_OBJECT_TYPE_CASE)
            }

            return objName;
        }
        VkBool32 debugMessage(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageTypes,
                              const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData)
        {

            std::string flagNames;
            std::string objName;

            if ((messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT) != 0u) {
                flagNames += "DEBUG:";
            }
            if ((messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) != 0u) {
                flagNames += "INFO:";
            }
            if ((messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) != 0u) {
                flagNames += "WARN:";
            }
            if ((messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0u) {
                flagNames += "ERROR:";
            }
            if ((messageTypes & VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT) != 0u) {
                flagNames += "PERF:";
            }

            uint64_t object = 0;
            // skip loader messages about device extensions
            if (pCallbackData->objectCount > 0) {
                auto objectType = pCallbackData->pObjects[0].objectType;
                if ((objectType == VK_OBJECT_TYPE_INSTANCE) && (strncmp(pCallbackData->pMessage, "Device Extension:", 17) == 0)) {
                    return VK_FALSE;
                }
                objName = vkObjectTypeToString(objectType);
                object = pCallbackData->pObjects[0].objectHandle;
                if (pCallbackData->pObjects[0].pObjectName != nullptr) {
                    objName += " " + std::string(pCallbackData->pObjects[0].pObjectName);
                }
            }

            ALOGE("%s (%s 0x%" PRIu64 ") %s", flagNames.c_str(), objName.c_str(), object, pCallbackData->pMessage);

            return VK_FALSE;
        }

        static VKAPI_ATTR VkBool32 VKAPI_CALL debugMessageThunk(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                                                VkDebugUtilsMessageTypeFlagsEXT messageTypes,
                                                                const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData)
        {
            return static_cast<VulkanGraphicsPlugin*>(pUserData)->debugMessage(messageSeverity, messageTypes, pCallbackData);
        }
    };

    struct VulkanGraphicsPluginLegacy : public VulkanGraphicsPlugin
    {
        VulkanGraphicsPluginLegacy(const std::shared_ptr<IPlatformPlugin>& platformPlugin);

        std::vector<std::string> GetInstanceExtensions() const override
        {
            return {XR_KHR_VULKAN_ENABLE_EXTENSION_NAME};
        }
        XrStructureType GetGraphicsBindingType() const override
        {
            return XR_TYPE_GRAPHICS_BINDING_VULKAN_KHR;
        }
        XrStructureType GetSwapchainImageType() const override
        {
            return XR_TYPE_SWAPCHAIN_IMAGE_VULKAN_KHR;
        }
        XrResult GetVulkanGraphicsRequirements2KHR(XrInstance instance, XrSystemId systemId,
                                                   XrGraphicsRequirementsVulkan2KHR* graphicsRequirements) override;
        XrResult CreateVulkanInstanceKHR(XrInstance instance, const XrVulkanInstanceCreateInfoKHR* createInfo, VkInstance* vulkanInstance,
                                         VkResult* vulkanResult) override;
        XrResult GetVulkanGraphicsDevice2KHR(XrInstance instance, const XrVulkanGraphicsDeviceGetInfoKHR* getInfo,
                                             VkPhysicalDevice* vulkanPhysicalDevice) override;
        XrResult CreateVulkanDeviceKHR(XrInstance instance, const XrVulkanDeviceCreateInfoKHR* createInfo, VkDevice* vulkanDevice,
                                       VkResult* vulkanResult) override;

        // do not override ShutdownDevice or Shutdown here!
    };

    VulkanGraphicsPlugin::VulkanGraphicsPlugin(const std::shared_ptr<IPlatformPlugin>& /*unused*/)
    {
    }

    VulkanGraphicsPlugin::~VulkanGraphicsPlugin()
    {
        ShutdownDevice();
        Shutdown();
    }

    VulkanGraphicsPluginLegacy::VulkanGraphicsPluginLegacy(const std::shared_ptr<IPlatformPlugin>& platformPlugin)
        : VulkanGraphicsPlugin(platformPlugin)
    {
    }

    XrResult VulkanGraphicsPlugin::GetVulkanGraphicsRequirements2KHR(XrInstance instance, XrSystemId systemId,
                                                                     XrGraphicsRequirementsVulkan2KHR* graphicsRequirements)
    {
        PFN_xrGetVulkanGraphicsRequirements2KHR pfnGetVulkanGraphicsRequirements2KHR = nullptr;
        XRC_CHECK_THROW_XRCMD(xrGetInstanceProcAddr(instance, "xrGetVulkanGraphicsRequirements2KHR",
                                                    reinterpret_cast<PFN_xrVoidFunction*>(&pfnGetVulkanGraphicsRequirements2KHR)));

        return pfnGetVulkanGraphicsRequirements2KHR(instance, systemId, graphicsRequirements);
    }

    XrResult VulkanGraphicsPluginLegacy::GetVulkanGraphicsRequirements2KHR(XrInstance instance, XrSystemId systemId,
                                                                           XrGraphicsRequirementsVulkan2KHR* graphicsRequirements)
    {
        PFN_xrGetVulkanGraphicsRequirementsKHR pfnGetVulkanGraphicsRequirementsKHR = nullptr;
        XRC_CHECK_THROW_XRCMD(xrGetInstanceProcAddr(instance, "xrGetVulkanGraphicsRequirementsKHR",
                                                    reinterpret_cast<PFN_xrVoidFunction*>(&pfnGetVulkanGraphicsRequirementsKHR)));

        XrGraphicsRequirementsVulkanKHR legacyRequirements{XR_TYPE_GRAPHICS_REQUIREMENTS_VULKAN_KHR};
        XRC_CHECK_THROW_XRCMD(pfnGetVulkanGraphicsRequirementsKHR(instance, systemId, &legacyRequirements));

        graphicsRequirements->maxApiVersionSupported = legacyRequirements.maxApiVersionSupported;
        graphicsRequirements->minApiVersionSupported = legacyRequirements.minApiVersionSupported;

        return XR_SUCCESS;
    }

    XrResult VulkanGraphicsPlugin::CreateVulkanInstanceKHR(XrInstance instance, const XrVulkanInstanceCreateInfoKHR* createInfo,
                                                           VkInstance* vulkanInstance, VkResult* vulkanResult)
    {
        PFN_xrCreateVulkanInstanceKHR pfnCreateVulkanInstanceKHR = nullptr;
        XRC_CHECK_THROW_XRCMD(xrGetInstanceProcAddr(instance, "xrCreateVulkanInstanceKHR",
                                                    reinterpret_cast<PFN_xrVoidFunction*>(&pfnCreateVulkanInstanceKHR)));

        return pfnCreateVulkanInstanceKHR(instance, createInfo, vulkanInstance, vulkanResult);
    }

    XrResult VulkanGraphicsPluginLegacy::CreateVulkanInstanceKHR(XrInstance instance, const XrVulkanInstanceCreateInfoKHR* createInfo,
                                                                 VkInstance* vulkanInstance, VkResult* vulkanResult)
    {
        PFN_xrGetVulkanInstanceExtensionsKHR pfnGetVulkanInstanceExtensionsKHR = nullptr;
        XRC_CHECK_THROW_XRCMD(xrGetInstanceProcAddr(instance, "xrGetVulkanInstanceExtensionsKHR",
                                                    reinterpret_cast<PFN_xrVoidFunction*>(&pfnGetVulkanInstanceExtensionsKHR)));

        uint32_t extensionNamesSize = 0;
        XRC_CHECK_THROW_XRCMD(pfnGetVulkanInstanceExtensionsKHR(instance, createInfo->systemId, 0, &extensionNamesSize, nullptr));

        std::vector<char> extensionNames(extensionNamesSize);
        XRC_CHECK_THROW_XRCMD(
            pfnGetVulkanInstanceExtensionsKHR(instance, createInfo->systemId, extensionNamesSize, &extensionNamesSize, &extensionNames[0]));
        {
            // Note: This cannot outlive the extensionNames above, since it's just a collection of views into that string!
            std::vector<const char*> extensions = ParseExtensionString(&extensionNames[0]);

            // Merge the runtime's request with the applications requests
            for (uint32_t i = 0; i < createInfo->vulkanCreateInfo->enabledExtensionCount; ++i) {
                extensions.push_back(createInfo->vulkanCreateInfo->ppEnabledExtensionNames[i]);
            }

            VkInstanceCreateInfo instInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
            memcpy(&instInfo, createInfo->vulkanCreateInfo, sizeof(instInfo));
            instInfo.enabledExtensionCount = (uint32_t)extensions.size();
            instInfo.ppEnabledExtensionNames = extensions.empty() ? nullptr : extensions.data();

            bool multiview_enable = GetGlobalData().IsUsingMultiview();
            if(multiview_enable){
                extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
            }

            auto pfnCreateInstance = (PFN_vkCreateInstance)createInfo->pfnGetInstanceProcAddr(nullptr, "vkCreateInstance");
            *vulkanResult = pfnCreateInstance(&instInfo, createInfo->vulkanAllocator, vulkanInstance);
        }

        return XR_SUCCESS;
    }

    XrResult VulkanGraphicsPlugin::GetVulkanGraphicsDevice2KHR(XrInstance instance, const XrVulkanGraphicsDeviceGetInfoKHR* getInfo,
                                                               VkPhysicalDevice* vulkanPhysicalDevice)
    {
        PFN_xrGetVulkanGraphicsDevice2KHR pfnGetVulkanGraphicsDevice2KHR = nullptr;
        XRC_CHECK_THROW_XRCMD(xrGetInstanceProcAddr(instance, "xrGetVulkanGraphicsDevice2KHR",
                                                    reinterpret_cast<PFN_xrVoidFunction*>(&pfnGetVulkanGraphicsDevice2KHR)));

        return pfnGetVulkanGraphicsDevice2KHR(instance, getInfo, vulkanPhysicalDevice);
    }

    XrResult VulkanGraphicsPluginLegacy::GetVulkanGraphicsDevice2KHR(XrInstance instance, const XrVulkanGraphicsDeviceGetInfoKHR* getInfo,
                                                                     VkPhysicalDevice* vulkanPhysicalDevice)
    {
        PFN_xrGetVulkanGraphicsDeviceKHR pfnGetVulkanGraphicsDeviceKHR = nullptr;
        XRC_CHECK_THROW_XRCMD(xrGetInstanceProcAddr(instance, "xrGetVulkanGraphicsDeviceKHR",
                                                    reinterpret_cast<PFN_xrVoidFunction*>(&pfnGetVulkanGraphicsDeviceKHR)));

        if (getInfo->next != nullptr) {
            return XR_ERROR_FEATURE_UNSUPPORTED;
        }

        XRC_CHECK_THROW_XRCMD(pfnGetVulkanGraphicsDeviceKHR(instance, getInfo->systemId, getInfo->vulkanInstance, vulkanPhysicalDevice));

        return XR_SUCCESS;
    }

    XrResult VulkanGraphicsPlugin::CreateVulkanDeviceKHR(XrInstance instance, const XrVulkanDeviceCreateInfoKHR* createInfo,
                                                         VkDevice* vulkanDevice, VkResult* vulkanResult)
    {
        PFN_xrCreateVulkanDeviceKHR pfnCreateVulkanDeviceKHR = nullptr;
        XRC_CHECK_THROW_XRCMD(
            xrGetInstanceProcAddr(instance, "xrCreateVulkanDeviceKHR", reinterpret_cast<PFN_xrVoidFunction*>(&pfnCreateVulkanDeviceKHR)));

        return pfnCreateVulkanDeviceKHR(instance, createInfo, vulkanDevice, vulkanResult);
    }

    static VkDescriptorPool CreateDescriptorPool(VkDevice device, uint32_t maxSets, uint32_t descriptorCount, bool isGfx=false)
    {
        VkDescriptorPoolSize poolSizes[2]{};
        if(isGfx){
            poolSizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            poolSizes[0].descriptorCount = descriptorCount;

            poolSizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            poolSizes[1].descriptorCount = descriptorCount;
        } else {
            poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            poolSizes[0].descriptorCount = descriptorCount;

            poolSizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            poolSizes[1].descriptorCount = descriptorCount;
        }

        VkDescriptorPoolCreateInfo descriptorPoolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        descriptorPoolInfo.maxSets = maxSets;
        descriptorPoolInfo.poolSizeCount = (uint32_t)ArraySize(poolSizes);
        descriptorPoolInfo.pPoolSizes = poolSizes;

        VkDescriptorPool descriptorPool;
        XRC_CHECK_THROW_VKCMD(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool));
        return descriptorPool;
    };

    XrResult VulkanGraphicsPluginLegacy::CreateVulkanDeviceKHR(XrInstance instance, const XrVulkanDeviceCreateInfoKHR* createInfo,
                                                               VkDevice* vulkanDevice, VkResult* vulkanResult)
    {
        PFN_xrGetVulkanDeviceExtensionsKHR pfnGetVulkanDeviceExtensionsKHR = nullptr;
        XRC_CHECK_THROW_XRCMD(xrGetInstanceProcAddr(instance, "xrGetVulkanDeviceExtensionsKHR",
                                                    reinterpret_cast<PFN_xrVoidFunction*>(&pfnGetVulkanDeviceExtensionsKHR)));

        uint32_t deviceExtensionNamesSize = 0;
        XRC_CHECK_THROW_XRCMD(pfnGetVulkanDeviceExtensionsKHR(instance, createInfo->systemId, 0, &deviceExtensionNamesSize, nullptr));
        std::vector<char> deviceExtensionNames(deviceExtensionNamesSize);
        XRC_CHECK_THROW_XRCMD(pfnGetVulkanDeviceExtensionsKHR(instance, createInfo->systemId, deviceExtensionNamesSize,
                                                              &deviceExtensionNamesSize, &deviceExtensionNames[0]));
        {
            // Note: This cannot outlive the extensionNames above, since it's just a collection of views into that string!
            std::vector<const char*> extensions = ParseExtensionString(&deviceExtensionNames[0]);

            // Merge the runtime's request with the applications requests
            for (uint32_t i = 0; i < createInfo->vulkanCreateInfo->enabledExtensionCount; ++i) {
                extensions.push_back(createInfo->vulkanCreateInfo->ppEnabledExtensionNames[i]);
            }

            #ifdef FEATURE_ADD_MSAA
            bool msaa_enable = GetGlobalData().IsUsingMSAA();
            if(msaa_enable){
                extensions.push_back("VK_KHR_multiview");
                extensions.push_back("VK_KHR_maintenance2");
                extensions.push_back("VK_KHR_create_renderpass2");
                extensions.push_back("VK_KHR_depth_stencil_resolve");
            }
            #endif

            bool multiview_enable = GetGlobalData().IsUsingMultiview();
            if(multiview_enable){
                extensions.push_back("VK_KHR_multiview");
            }

            VkPhysicalDeviceFeatures features{};
            memcpy(&features, createInfo->vulkanCreateInfo->pEnabledFeatures, sizeof(features));

            VkPhysicalDeviceMultiviewFeaturesKHR physicalDeviceMultiviewFeatures = {};
            if(multiview_enable)
            {
                VkPhysicalDeviceFeatures2KHR deviceFeatures2{};
                VkPhysicalDeviceMultiviewFeaturesKHR extFeatures{};
                extFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES_KHR;
                deviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2_KHR;
                deviceFeatures2.pNext = &extFeatures;
                PFN_vkGetPhysicalDeviceFeatures2KHR vkGetPhysicalDeviceFeatures2KHR = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2KHR>(createInfo->pfnGetInstanceProcAddr(m_vkInstance, "vkGetPhysicalDeviceFeatures2KHR"));
                vkGetPhysicalDeviceFeatures2KHR(m_vkPhysicalDevice, &deviceFeatures2);

                if (!extFeatures.multiview) {
                    ALOGE("multiview not supported");
                    return XR_ERROR_RUNTIME_FAILURE;
                }

                VkPhysicalDeviceProperties2KHR deviceProps2{};
                VkPhysicalDeviceMultiviewPropertiesKHR extProps{};
                extProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_PROPERTIES_KHR;
                deviceProps2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2_KHR;
                deviceProps2.pNext = &extProps;
                PFN_vkGetPhysicalDeviceProperties2KHR vkGetPhysicalDeviceProperties2KHR = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2KHR>(createInfo->pfnGetInstanceProcAddr(m_vkInstance, "vkGetPhysicalDeviceProperties2KHR"));
                vkGetPhysicalDeviceProperties2KHR(m_vkPhysicalDevice, &deviceProps2);
                if (extProps.maxMultiviewViewCount < 2) {
                   ALOGE("multiview not supported 2");
                    return XR_ERROR_RUNTIME_FAILURE;
                }

                // Enable extension required for multiview.
                extensions.push_back(VK_KHR_MULTIVIEW_EXTENSION_NAME);
                physicalDeviceMultiviewFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES_KHR;
                physicalDeviceMultiviewFeatures.multiview = VK_TRUE;
                physicalDeviceMultiviewFeatures.pNext = NULL;
            }

#if !defined(XR_USE_PLATFORM_ANDROID)
            VkPhysicalDeviceFeatures availableFeatures{};
            vkGetPhysicalDeviceFeatures(m_vkPhysicalDevice, &availableFeatures);
            if (availableFeatures.shaderStorageImageMultisample == VK_TRUE) {
                // Setting this quiets down a validation error triggered by the Oculus runtime
                features.shaderStorageImageMultisample = VK_TRUE;
            }
#endif

            VkDeviceCreateInfo deviceInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
            memcpy(&deviceInfo, createInfo->vulkanCreateInfo, sizeof(deviceInfo));
            deviceInfo.pEnabledFeatures = &features;
            deviceInfo.enabledExtensionCount = (uint32_t)extensions.size();
            deviceInfo.ppEnabledExtensionNames = extensions.empty() ? nullptr : extensions.data();

            VkPhysicalDeviceVulkan11Features physicalDeviceFeatures{
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
                .pNext = NULL,
                .multiview = VK_TRUE,
                .protectedMemory = VK_TRUE,
            };
            
            if(multiview_enable){
                physicalDeviceFeatures.pNext = &physicalDeviceMultiviewFeatures;
            }

            deviceInfo.pNext = &physicalDeviceFeatures;
            deviceInfo.queueCreateInfoCount = 2;

            auto pfnCreateDevice = (PFN_vkCreateDevice)createInfo->pfnGetInstanceProcAddr(m_vkInstance, "vkCreateDevice");
            *vulkanResult = pfnCreateDevice(m_vkPhysicalDevice, &deviceInfo, createInfo->vulkanAllocator, vulkanDevice);
            ALOGE("%s:xxxxxx:vulkanDevice:%p", __func__, *vulkanDevice);
        }

        return XR_SUCCESS;
    }

    bool VulkanGraphicsPlugin::Initialize()
    {
        if (initialized) {
            return false;
        }
        m_graphicsBinding.type = GetGraphicsBindingType();

        // To do.
        initialized = true;
        return initialized;
    }

    bool VulkanGraphicsPlugin::IsInitialized() const
    {
        return initialized;
    }

    void VulkanGraphicsPlugin::Shutdown()
    {
        if (initialized) {
            // To do.
            initialized = false;
        }
    }

    void VulkanGraphicsPlugin::enum_vulkan_instance_exts() {
        uint32_t extensionCount = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> extensions(extensionCount);

        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());

        ALOGE("Available Vulkan instance extensions:\n");
        for (const auto& ext : extensions) {
            ALOGE("\t%s", ext.extensionName);
        }
    }

    void VulkanGraphicsPlugin::enum_vulkan_device_exts() {
        uint32_t extensionCount = 0;

       /* vkEnumerateDeviceExtensionProperties = (PFN_vkEnumerateDeviceExtensionProperties)vkGetInstanceProcAddr(m_vkInstance, "vkEnumerateDeviceExtensionProperties");
        if(vkEnumerateDeviceExtensionProperties == NULL){
            Log::Write(Log::Level::Error,"vkEnumerateDeviceExtensionProperties is null");
            return;
        }*/

        vkEnumerateDeviceExtensionProperties(m_vkPhysicalDevice, nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> extensions(extensionCount);

        vkEnumerateDeviceExtensionProperties(m_vkPhysicalDevice, nullptr, &extensionCount, extensions.data());

        ALOGE("Available Vulkan device extensions:\n");
        for (const auto& ext : extensions) {
            ALOGE("\t%s", ext.extensionName);
        }
    }

    std::string VulkanGraphicsPlugin::DescribeGraphics() const
    {
        std::string gpu = "";
        if (m_vkPhysicalDevice != VK_NULL_HANDLE) {
            PFN_vkGetPhysicalDeviceProperties2KHR pfnvkGetPhysicalDeviceProperties2KHR =
                (PFN_vkGetPhysicalDeviceProperties2KHR)vkGetInstanceProcAddr(m_vkInstance, "vkGetPhysicalDeviceProperties2KHR");
            if (pfnvkGetPhysicalDeviceProperties2KHR) {
                VkPhysicalDeviceIDPropertiesKHR gpuDevID{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES_KHR};
                VkPhysicalDeviceProperties2KHR gpuProps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2_KHR, &gpuDevID};
                pfnvkGetPhysicalDeviceProperties2KHR(m_vkPhysicalDevice, &gpuProps);
                std::string deviceType = "unknown";
                switch (gpuProps.properties.deviceType) {
                case VK_PHYSICAL_DEVICE_TYPE_OTHER:
                    deviceType = "<other>";
                    break;
                case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
                    deviceType = "<integrated>";
                    break;
                case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
                    deviceType = "<discrete>";
                    break;
                case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
                    deviceType = "<virtual>";
                    break;
                case VK_PHYSICAL_DEVICE_TYPE_CPU:
                    deviceType = "<cpu>";
                    break;
                // VK_PHYSICAL_DEVICE_TYPE_RANGE_SIZE was removed from vulkan headers
                // case VK_PHYSICAL_DEVICE_TYPE_RANGE_SIZE:
                case VK_PHYSICAL_DEVICE_TYPE_MAX_ENUM:
                default:
                    break;
                }

                gpu += "\nGPU: " + std::string(gpuProps.properties.deviceName) + " " + deviceType;
#if !defined(NDEBUG)  // CMAKE defines this
                gpu += "\nLUID: " + (gpuDevID.deviceLUIDValid ? to_hex(gpuDevID.deviceLUID) : std::string("<invalid>"));
#endif
            }
        }

        return std::string("Vulkan" + gpu);
    }

    bool VulkanGraphicsPlugin::InitializeDevice(XrInstance instance, XrSystemId systemId, bool checkGraphicsRequirements,
                                                uint32_t deviceCreationFlags)
    {
        // Create the Vulkan device for the adapter associated with the system.
        // Extension function must be loaded by name

        if (checkGraphicsRequirements) {
            XrGraphicsRequirementsVulkanKHR graphicsRequirements{XR_TYPE_GRAPHICS_REQUIREMENTS_VULKAN_KHR};
            XRC_CHECK_THROW_XRCMD(GetVulkanGraphicsRequirements2KHR(instance, systemId, &graphicsRequirements));
            const XrVersion vulkanVersion = XR_MAKE_VERSION(VK_VERSION_MAJOR(VK_API_VERSION_1_0), VK_VERSION_MINOR(VK_API_VERSION_1_0), 0);
            if ((vulkanVersion < graphicsRequirements.minApiVersionSupported) ||
                (vulkanVersion > graphicsRequirements.maxApiVersionSupported)) {
                // Log?
                return false;
            }
        }

        VkDebugUtilsMessengerCreateInfoEXT debugInfo{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
        debugInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT;
        debugInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
//#if !defined(NDEBUG)
        debugInfo.messageSeverity |= VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT;
//#endif
        debugInfo.pfnUserCallback = debugMessageThunk;
        debugInfo.pUserData = this;

        VkResult err;
        {
            // Note: This cannot outlive the extensionNames above, since it's just a collection of views into that string!
            std::vector<const char*> extensions;
            {
                uint32_t extensionCount = 0;
                XRC_CHECK_THROW_VKCMD(vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr));

                std::vector<VkExtensionProperties> availableExtensions(extensionCount);
                XRC_CHECK_THROW_VKCMD(vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, availableExtensions.data()));
                const auto b = availableExtensions.begin();
                const auto e = availableExtensions.end();

                auto isExtSupported = [&](const char* extName) -> bool {
                    auto it = std::find_if(
                        b, e, [&](const VkExtensionProperties& properties) { return (0 == strcmp(extName, properties.extensionName)); });
                    return (it != e);
                };

                // Debug utils is optional and not always available
                if (isExtSupported(VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
                    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
                }
                // TODO add back VK_EXT_debug_report code for compatibility with older systems? (Android)
            }

            bool multiview_enable = GetGlobalData().IsUsingMultiview();
            if(multiview_enable){
                extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
            }

            std::vector<const char*> layers;
//#if !defined(NDEBUG)
            auto GetValidationLayerName = []() -> const char* {
                uint32_t layerCount;
                vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
                std::vector<VkLayerProperties> availableLayers(layerCount);
                vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

                // Enable only one validation layer, prefer KHRONOS.
                for (auto validationLayerName : {"VK_LAYER_KHRONOS_validation", "VK_LAYER_LUNARG_standard_validation"}) {
                    for (const auto& layerProperties : availableLayers) {
                        if (0 == strcmp(validationLayerName, layerProperties.layerName)) {
                            return validationLayerName;
                        }
                    }
                }

                return nullptr;
            };
            const char* validationLayerName = GetValidationLayerName();
            if (validationLayerName)
            {
                ALOGE("%s:xxxxxx:validationLayerName:%s", __func__, validationLayerName);
                layers.push_back(validationLayerName);
            }
            else
                ReportF("No Vulkan validation layers found, running without them");
//#endif
#if defined(USE_CHECKPOINTS)
            layers.push_back("VK_NV_device_diagnostic_checkpoints");
#endif

            VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
            appInfo.pApplicationName = "conformance_test";
            appInfo.applicationVersion = 1;
            appInfo.pEngineName = "conformance_test";
            appInfo.engineVersion = 1;
            appInfo.apiVersion = VK_API_VERSION_1_2;

            VkInstanceCreateInfo instInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
            instInfo.pNext = &debugInfo;
            instInfo.pApplicationInfo = &appInfo;
            instInfo.enabledLayerCount = (uint32_t)layers.size();
            instInfo.ppEnabledLayerNames = layers.empty() ? nullptr : layers.data();
            instInfo.enabledExtensionCount = (uint32_t)extensions.size();
            instInfo.ppEnabledExtensionNames = extensions.empty() ? nullptr : extensions.data();

            XrVulkanInstanceCreateInfoKHR createInfo{XR_TYPE_VULKAN_INSTANCE_CREATE_INFO_KHR};
            createInfo.systemId = systemId;
            createInfo.pfnGetInstanceProcAddr = &vkGetInstanceProcAddr;
            createInfo.vulkanCreateInfo = &instInfo;
            createInfo.vulkanAllocator = nullptr;
            XRC_CHECK_THROW_XRCMD(CreateVulkanInstanceKHR(instance, &createInfo, &m_vkInstance, &err));
            XRC_CHECK_THROW_VKCMD(err);
        }

#if defined(USE_CHECKPOINTS)
        vkCmdSetCheckpointNV = (PFN_vkCmdSetCheckpointNV)vkGetInstanceProcAddr(m_vkInstance, "vkCmdSetCheckpointNV");
        vkGetQueueCheckpointDataNV = (PFN_vkGetQueueCheckpointDataNV)vkGetInstanceProcAddr(m_vkInstance, "vkGetQueueCheckpointDataNV");
#endif

        vkCreateDebugUtilsMessengerEXT =
            (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(m_vkInstance, "vkCreateDebugUtilsMessengerEXT");
        vkDestroyDebugUtilsMessengerEXT =
            (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(m_vkInstance, "vkDestroyDebugUtilsMessengerEXT");

        if (vkCreateDebugUtilsMessengerEXT != nullptr && vkDestroyDebugUtilsMessengerEXT != nullptr) {
            XRC_CHECK_THROW_VKCMD(vkCreateDebugUtilsMessengerEXT(m_vkInstance, &debugInfo, nullptr, &m_vkDebugUtilsMessenger));
        }

        vkCreateRenderPass2KHR =
            (PFN_vkCreateRenderPass2KHR)vkGetInstanceProcAddr(m_vkInstance, "vkCreateRenderPass2KHR");
        ALOGE("%s:xxxxxx:vkCreateRenderPass2KHR:%p", __func__, vkCreateRenderPass2KHR);


        XrVulkanGraphicsDeviceGetInfoKHR deviceGetInfo{XR_TYPE_VULKAN_GRAPHICS_DEVICE_GET_INFO_KHR};
        deviceGetInfo.systemId = systemId;
        deviceGetInfo.vulkanInstance = m_vkInstance;
        XRC_CHECK_THROW_XRCMD(GetVulkanGraphicsDevice2KHR(instance, &deviceGetInfo, &m_vkPhysicalDevice));

        VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        float queuePriorities = 0;
        queueInfo.queueCount = 1;
        queueInfo.pQueuePriorities = &queuePriorities;

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(m_vkPhysicalDevice, &queueFamilyCount, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilyProps(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(m_vkPhysicalDevice, &queueFamilyCount, &queueFamilyProps[0]);

        for (uint32_t i = 0; i < queueFamilyCount; ++i) {
            // Only need graphics (not presentation) for draw queue
            if ((queueFamilyProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0u) {
                m_queueFamilyIndex = queueInfo.queueFamilyIndex = i;
                break;
            }
        }

        std::vector<const char*> deviceExtensions;

        VkPhysicalDeviceFeatures features{};
        // features.samplerAnisotropy = VK_TRUE;
        // Setting this quiets down a validation error triggered by the Oculus runtime
        // features.shaderStorageImageMultisample = VK_TRUE;

        VkPhysicalDeviceMultiviewFeaturesKHR physicalDeviceMultiviewFeatures = {};
        bool multiview_enable = GetGlobalData().IsUsingMultiview();
        if(multiview_enable)
        {
            VkPhysicalDeviceFeatures2KHR deviceFeatures2{};
            VkPhysicalDeviceMultiviewFeaturesKHR extFeatures{};
            extFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES_KHR;
            deviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2_KHR;
            deviceFeatures2.pNext = &extFeatures;
            PFN_vkGetPhysicalDeviceFeatures2KHR vkGetPhysicalDeviceFeatures2KHR = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2KHR>(vkGetInstanceProcAddr(m_vkInstance, "vkGetPhysicalDeviceFeatures2KHR"));
            vkGetPhysicalDeviceFeatures2KHR(m_vkPhysicalDevice, &deviceFeatures2);

            if (!extFeatures.multiview) {
                ALOGE("multiview not supported");
                return XR_ERROR_RUNTIME_FAILURE;
            }

            VkPhysicalDeviceProperties2KHR deviceProps2{};
            VkPhysicalDeviceMultiviewPropertiesKHR extProps{};
            extProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_PROPERTIES_KHR;
            deviceProps2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2_KHR;
            deviceProps2.pNext = &extProps;
            PFN_vkGetPhysicalDeviceProperties2KHR vkGetPhysicalDeviceProperties2KHR = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2KHR>(vkGetInstanceProcAddr(m_vkInstance, "vkGetPhysicalDeviceProperties2KHR"));
            vkGetPhysicalDeviceProperties2KHR(m_vkPhysicalDevice, &deviceProps2);
            if (extProps.maxMultiviewViewCount < 2) {
               ALOGE("multiview not supported 2");
                return XR_ERROR_RUNTIME_FAILURE;
            }

            // Enable extension required for multiview.
            deviceExtensions.push_back(VK_KHR_MULTIVIEW_EXTENSION_NAME);
            physicalDeviceMultiviewFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES_KHR;
            physicalDeviceMultiviewFeatures.multiview = VK_TRUE;
            physicalDeviceMultiviewFeatures.pNext = NULL;
        }

        VkDeviceQueueCreateInfo queueCreateInfo[2] = {
            {
                .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .queueCount = 1,
                .queueFamilyIndex = m_queueFamilyIndex,
                .pQueuePriorities = &queuePriorities,  //for swapchain create & ATW, promote the queue priority
            },
            {
                .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                .pNext = nullptr,
                .flags = VK_DEVICE_QUEUE_CREATE_PROTECTED_BIT,
                .queueCount = 1,
                .queueFamilyIndex = m_queueFamilyIndex,
                .pQueuePriorities = &queuePriorities,  //for swapchain create & ATW, promote the queue priority
            }
        };

        VkDeviceCreateInfo deviceInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        deviceInfo.flags = VkDeviceCreateFlags(deviceCreationFlags);
        deviceInfo.queueCreateInfoCount = 1;
        deviceInfo.pQueueCreateInfos =queueCreateInfo;
        deviceInfo.enabledLayerCount = 0;
        deviceInfo.ppEnabledLayerNames = nullptr;
        deviceInfo.enabledExtensionCount = (uint32_t)deviceExtensions.size();
        deviceInfo.ppEnabledExtensionNames = deviceExtensions.empty() ? nullptr : deviceExtensions.data();
        deviceInfo.pEnabledFeatures = &features;


        VkPhysicalDeviceVulkan11Features physicalDeviceFeatures{
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
                .pNext = NULL,
                .multiview = VK_TRUE,
                .protectedMemory = VK_TRUE,
        };
        
        if(multiview_enable){
            physicalDeviceFeatures.pNext = &physicalDeviceMultiviewFeatures;
        }

        bool isProtectedMemory = true;
        if(isProtectedMemory){
            deviceInfo.pNext = &physicalDeviceFeatures;
            deviceInfo.queueCreateInfoCount = 2;
        }

        XrVulkanDeviceCreateInfoKHR deviceCreateInfo{XR_TYPE_VULKAN_DEVICE_CREATE_INFO_KHR};
        deviceCreateInfo.systemId = systemId;
        deviceCreateInfo.pfnGetInstanceProcAddr = &vkGetInstanceProcAddr;
        deviceCreateInfo.vulkanCreateInfo = &deviceInfo;
        deviceCreateInfo.vulkanPhysicalDevice = m_vkPhysicalDevice;
        deviceCreateInfo.vulkanAllocator = nullptr;
        XRC_CHECK_THROW_XRCMD(CreateVulkanDeviceKHR(instance, &deviceCreateInfo, &m_vkDevice, &err));
        XRC_CHECK_THROW_VKCMD(err);

        ALOGE("%s:xxxxxx:m_vkDevice:%p", __func__, m_vkDevice);

        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(m_vkPhysicalDevice, &props);
        ALOGE("xxxxxx:Max push constants size: %u", props.limits.maxPushConstantsSize);
        m_namer.Init(m_vkInstance, m_vkDevice);

        vkGetDeviceQueue(m_vkDevice, queueInfo.queueFamilyIndex, 0, &m_vkQueue);
        if(isProtectedMemory){

            PFN_vkGetDeviceQueue2 vkGetDeviceQueue2 = NULL;
            PFN_vkGetDeviceProcAddr vkGetDeviceProcAddr = (PFN_vkGetDeviceProcAddr )vkGetInstanceProcAddr(m_vkInstance, "vkGetDeviceProcAddr");;
            ALOGE("%s:vkGetDeviceProcAddr:%p", __func__, vkGetDeviceProcAddr);
            if(vkGetDeviceProcAddr != NULL){
                vkGetDeviceQueue2 = (PFN_vkGetDeviceQueue2)vkGetDeviceProcAddr(m_vkDevice, "vkGetDeviceQueue2");
            }

            ALOGE("%s:vkGetDeviceQueue2:%p", __func__, vkGetDeviceQueue2);

            if(vkGetDeviceQueue2 != NULL){
                VkDeviceQueueInfo2 info = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_INFO_2};
                info.queueFamilyIndex = m_queueFamilyIndex;
                info.queueIndex       = 0;
                info.flags            = VK_DEVICE_QUEUE_CREATE_PROTECTED_BIT;
                vkGetDeviceQueue2(m_vkDevice, &info, &m_vkQueueProtected);
            }
        }

        m_memAllocator.Init(m_vkPhysicalDevice, m_vkDevice);

        enum_vulkan_instance_exts();
        enum_vulkan_device_exts();

        InitializeResources();

        m_graphicsBinding.instance = m_vkInstance;
        m_graphicsBinding.physicalDevice = m_vkPhysicalDevice;
        m_graphicsBinding.device = m_vkDevice;
        m_graphicsBinding.queueFamilyIndex = queueInfo.queueFamilyIndex;
        m_graphicsBinding.queueIndex = 0;

        return true;
    }

#ifdef USE_ONLINE_VULKAN_SHADERC
    // Compile a shader to a SPIR-V binary.
    std::vector<uint32_t> VulkanGraphicsPlugin::CompileGlslShader(const std::string& name, shaderc_shader_kind kind,
                                                                  const std::string& source)
    {
        shaderc::Compiler compiler;
        shaderc::CompileOptions options;

        options.SetOptimizationLevel(shaderc_optimization_level_size);

        shaderc::SpvCompilationResult module = compiler.CompileGlslToSpv(source, kind, name.c_str(), options);

        if (module.GetCompilationStatus() != shaderc_compilation_status_success) {
            XRC_THROW("Shader " + name + " compilation failed: " + module.GetErrorMessage());
        }

        return {module.cbegin(), module.cend()};
    }
#endif

    void VulkanGraphicsPlugin::InitializeResources()
    {
#ifdef USE_ONLINE_VULKAN_SHADERC
        auto vertexSPIRV = CompileGlslShader("vertex", shaderc_glsl_default_vertex_shader, VertexShaderGlsl);
        auto fragmentSPIRV = CompileGlslShader("fragment", shaderc_glsl_default_fragment_shader, FragmentShaderGlsl);
        #ifdef FEATURE_ADD_MOTION_VECTOR
        auto mvVertexSPIRV = CompileGlslShader("vertex", shaderc_glsl_default_vertex_shader, VertexShaderMotionVectorGlsl);
        auto mvFragmentSPIRV = CompileGlslShader("fragment", shaderc_glsl_default_fragment_shader, FragmentShaderMotionVectorGlsl);
        #endif
#else
        std::vector<uint32_t> vertexSPIRV = SPV_PREFIX
#include "vert.spv"  // IWYU pragma: keep

            SPV_SUFFIX;
        std::vector<uint32_t> fragmentSPIRV = SPV_PREFIX
#include "frag.spv"  // IWYU pragma: keep
            SPV_SUFFIX;
        std::vector<uint32_t> computeSPIRV = SPV_PREFIX
#include "comp.spv"  // IWYU pragma: keep
            SPV_SUFFIX;
#endif
        #ifdef FEATURE_ADD_MOTION_VECTOR
        std::vector<uint32_t> mvVertexSPIRV = SPV_PREFIX
#include "mvvert.spv"  // IWYU pragma: keep
            SPV_SUFFIX;
        std::vector<uint32_t> mvFragmentSPIRV = SPV_PREFIX
#include "mvfrag.spv"  // IWYU pragma: keep
            SPV_SUFFIX;
        #endif

        std::vector<uint32_t> multiviewVertexSPIRV = SPV_PREFIX
#include "multiviewvert.spv"  // IWYU pragma: keep
            SPV_SUFFIX;
        std::vector<uint32_t> multiviewFragmentSPIRV = SPV_PREFIX
#include "multiviewfrag.spv"  // IWYU pragma: keep
            SPV_SUFFIX;

        std::vector<uint32_t> envDepthVertexSPIRV = SPV_PREFIX
#include "envdepthvert.spv"  // IWYU pragma: keep
            SPV_SUFFIX;
        std::vector<uint32_t> envDepthFragmentSPIRV = SPV_PREFIX
#include "envdepthfrag.spv"  // IWYU pragma: keep
            SPV_SUFFIX;

        if (vertexSPIRV.empty())
            XRC_THROW("Failed to compile vertex shader");
        if (fragmentSPIRV.empty())
            XRC_THROW("Failed to compile fragment shader");
        if (computeSPIRV.empty())
            XRC_THROW("Failed to compile compute shader");

        m_shaderProgram.Init(m_vkDevice);
        m_shaderProgram.LoadVertexShader(vertexSPIRV);
        m_shaderProgram.LoadFragmentShader(fragmentSPIRV);

        #ifdef FEATURE_ADD_MOTION_VECTOR
        ALOGE("%s:xxxxxx:1:%d,2:%d", __func__, vertexSPIRV.size(), mvVertexSPIRV.size());
        m_mvShaderProgram.Init(m_vkDevice);
        m_mvShaderProgram.LoadVertexShader(mvVertexSPIRV);
        m_mvShaderProgram.LoadFragmentShader(mvFragmentSPIRV);
        #endif

        ALOGE("%s:xxxxxx:3:%d,2:%d", __func__, multiviewVertexSPIRV.size(), multiviewFragmentSPIRV.size());
        m_multiviewShaderProgram.Init(m_vkDevice);
        m_multiviewShaderProgram.LoadVertexShader(multiviewVertexSPIRV);
        m_multiviewShaderProgram.LoadFragmentShader(multiviewFragmentSPIRV);

        ALOGE("%s:xxxxxx:4:%d,2:%d", __func__, multiviewVertexSPIRV.size(), multiviewFragmentSPIRV.size());
        m_envDepthShaderProgram.Init(m_vkDevice);
        m_envDepthShaderProgram.LoadVertexShader(envDepthVertexSPIRV);
        m_envDepthShaderProgram.LoadFragmentShader(envDepthFragmentSPIRV);

        m_computeShaderProgram.Init(m_vkDevice);
        m_computeShaderProgram.LoadComputeShader(computeSPIRV);

        // Semaphore to block on draw complete
        VkSemaphoreCreateInfo semInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        XRC_CHECK_THROW_VKCMD(vkCreateSemaphore(m_vkDevice, &semInfo, nullptr, &m_vkDrawDone));
        XRC_CHECK_THROW_VKCMD(m_namer.SetName(VK_OBJECT_TYPE_SEMAPHORE, (uint64_t)m_vkDrawDone, "CTS draw done semaphore"));

        if (!m_cmdBuffer.Init(m_namer, m_vkDevice, m_queueFamilyIndex))
            XRC_THROW("Failed to create command buffer");

        bool edo_enalble      = GetGlobalData().IsUsingEnvDepthOcclusion();

        m_pipelineLayout.Create(m_vkDevice, SHADER_PROGRAM_TYPE_GRAPHICS, edo_enalble);
        XRC_CHECK_THROW_VKCMD(
            m_namer.SetName(VK_OBJECT_TYPE_PIPELINE_LAYOUT, (uint64_t)m_pipelineLayout.layout, "CTS graphics pipeline layout"));
        if(edo_enalble){
            XRC_CHECK_THROW_VKCMD(m_namer.SetName(VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, (uint64_t)m_pipelineLayout.descriptorSetLayout,
                                              "CTS graphics env depth descriptor set layout"));
        }

        m_computePipelineLayout.Create(m_vkDevice, SHADER_PROGRAM_TYPE_COMPUTE);
        XRC_CHECK_THROW_VKCMD(
            m_namer.SetName(VK_OBJECT_TYPE_PIPELINE_LAYOUT, (uint64_t)m_computePipelineLayout.layout, "CTS compute pipeline layout"));
        XRC_CHECK_THROW_VKCMD(m_namer.SetName(VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, (uint64_t)m_computePipelineLayout.descriptorSetLayout,
                                              "CTS compute descriptor set layout"));

        // resources for compute pipeline
        m_computeDescriptorPool.adopt(CreateDescriptorPool(m_vkDevice, 1, 1), m_vkDevice);

        VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocInfo.descriptorPool = m_computeDescriptorPool.get();
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &m_computePipelineLayout.descriptorSetLayout;
        XRC_CHECK_THROW_VKCMD(vkAllocateDescriptorSets(m_vkDevice, &allocInfo, &m_ComputeDescriptorSet));

        if(edo_enalble){
            m_gfxDescriptorPool.adopt(CreateDescriptorPool(m_vkDevice, 1, 1, true), m_vkDevice);

            VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            allocInfo.descriptorPool = m_gfxDescriptorPool.get();
            allocInfo.descriptorSetCount = 1;
            allocInfo.pSetLayouts = &m_pipelineLayout.descriptorSetLayout;
            XRC_CHECK_THROW_VKCMD(vkAllocateDescriptorSets(m_vkDevice, &allocInfo, &m_gfxDescriptorSet));
        }

        static_assert(sizeof(Geometry::Vertex) == 24, "Unexpected Vertex size");

        m_cubeMesh = MakeCubeMesh();

        m_pbrResources = std::make_unique<Pbr::VulkanResources>(m_namer, m_vkPhysicalDevice, m_vkDevice, m_queueFamilyIndex);
        m_pbrResources->SetLight({0.0f, 0.7071067811865475f, 0.7071067811865475f}, Pbr::RGB::White);

        auto blackCubeMap =
            std::make_shared<Pbr::VulkanTextureBundle>(Pbr::VulkanTexture::CreateFlatCubeTexture(*m_pbrResources, Pbr::RGBA::Black, false));
        m_pbrResources->SetEnvironmentMap(blackCubeMap, blackCubeMap);

        // Read the BRDF Lookup Table used by the PBR system into a Vulkan texture.
        std::vector<unsigned char> brdfLutFileData = ReadFileBytes("brdf_lut.png");
        auto brdLutResourceView = std::make_shared<Pbr::VulkanTextureBundle>(
            Pbr::VulkanTexture::LoadTextureImage(*m_pbrResources, false, brdfLutFileData.data(), (uint32_t)brdfLutFileData.size()));
        m_pbrResources->SetBrdfLut(brdLutResourceView);

#if defined(USE_MIRROR_WINDOW)
        m_swapchain.Create(m_vkInstance, m_vkPhysicalDevice, m_vkDevice, m_graphicsBinding.queueFamilyIndex);

        bool isProtectedMemory = GetGlobalData().IsProtectedMemory();
        VkQueue queue = isProtectedMemory?m_vkQueueProtected:m_vkQueue;
        VkCommandBuffer buf = isProtectedMemory?m_cmdBuffer.bufP:m_cmdBuffer.buf;
        m_cmdBuffer.Clear();
        m_cmdBuffer.Begin(isProtectedMemory);
        m_swapchain.Prepare(buf);
        m_cmdBuffer.End(isProtectedMemory);
        m_cmdBuffer.Exec(queue, isProtectedMemory);
        m_cmdBuffer.Wait();
#endif
    }

    void VulkanGraphicsPlugin::ClearSwapchainCache()
    {
        m_swapchainImageDataMap.Reset();
    }

    void VulkanGraphicsPlugin::ShutdownDevice()
    {
        if (m_vkDevice != VK_NULL_HANDLE) {
            // Make sure we're idle.
            vkDeviceWaitIdle(m_vkDevice);

            // Reset the swapchains to avoid calling Vulkan functions in the dtors after
            // we've shut down the device.

            m_pbrResources.reset();

            m_swapchainImageDataMap.Reset();
            m_cubeMesh = {};
            m_meshes.clear();
            m_gltfInstances.clear();
            m_gltfModels.clear();
            m_pbrResources.reset();

            m_queueFamilyIndex = 0;
            m_vkQueue = VK_NULL_HANDLE;
            m_vkQueueProtected = VK_NULL_HANDLE;
            if (m_vkDrawDone) {
                vkDestroySemaphore(m_vkDevice, m_vkDrawDone, nullptr);
                m_vkDrawDone = VK_NULL_HANDLE;
            }

            m_cmdBuffer.Reset();
            m_pipelineLayout.Reset();
            m_computePipelineLayout.Reset();
            m_shaderProgram.Reset();
        #ifdef FEATURE_ADD_MOTION_VECTOR
            m_mvShaderProgram.Reset();
        #endif
            m_multiviewShaderProgram.Reset();
            m_envDepthShaderProgram.Reset();


            m_computeShaderProgram.Reset();
            m_memAllocator.Reset();

            m_computeDescriptorPool.reset();
            m_gfxDescriptorPool.reset();

#if defined(USE_MIRROR_WINDOW)
            m_swapchain.Reset();
            m_swapchainImageData.clear();
#endif
            vkDestroyDevice(m_vkDevice, nullptr);
            m_vkDevice = VK_NULL_HANDLE;
        }
        if (m_vkDebugUtilsMessenger != VK_NULL_HANDLE && vkDestroyDebugUtilsMessengerEXT != nullptr) {
            vkDestroyDebugUtilsMessengerEXT(m_vkInstance, m_vkDebugUtilsMessenger, nullptr);
            m_vkDebugUtilsMessenger = VK_NULL_HANDLE;
        }
        if (m_vkInstance != VK_NULL_HANDLE) {
            vkDestroyInstance(m_vkInstance, nullptr);
            m_vkInstance = VK_NULL_HANDLE;
        }
    }

    const XrBaseInStructure* VulkanGraphicsPlugin::GetGraphicsBinding() const
    {
        if (m_graphicsBinding.device) {
            return reinterpret_cast<const XrBaseInStructure*>(&m_graphicsBinding);
        }
        return nullptr;
    }

    static const SwapchainFormatDataMap& GetSwapchainFormatData()
    {
        using namespace SwapchainFormat;

        // Add SwapchainCreateTestParameters for other Vulkan formats if they are supported by a runtime
        static SwapchainFormatDataMap map{{

            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_R8G8B8A8_UNORM).rgba().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_R8G8B8A8_SRGB).rgba().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_B8G8R8A8_UNORM).rgba().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_B8G8R8A8_SRGB).rgba().ToPair(),

            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_R8G8B8_UNORM).rgb().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_R8G8B8_SRGB).rgb().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_B8G8R8_UNORM).rgb().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_B8G8R8_SRGB).rgb().ToPair(),

            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_R8G8_UNORM).rg().ToPair(),

            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_R8_UNORM).r().ToPair(),

            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_R8_SNORM).r().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_R8G8_SNORM).rg().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_R8G8B8_SNORM).rgb().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_R8G8B8A8_SNORM).rgba().ToPair(),

            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_R8_UINT).r().Int(u8).ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_R8G8_UINT).rg().Int(u8).ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_R8G8B8_UINT).rgb().Int(u8).ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_R8G8B8A8_UINT).rgba().Int(u8).ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_R8_SINT).r().Int(s8).ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_R8G8_SINT).rg().Int(s8).ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_R8G8B8_SINT).rgb().Int(s8).ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_R8_UNORM).r().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_R8G8B8A8_SINT).rgba().Int(s8).ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_R8_SRGB).r().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_R8G8_SRGB).rg().ToPair(),

            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_R16_UNORM).r().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_R16G16_UNORM).rg().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_R16G16B16_UNORM).rgb().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_R16G16B16A16_UNORM).rgba().ToPair(),

            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_R16_SNORM).r().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_R16G16_SNORM).rg().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_R16G16B16_SNORM).rgb().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_R16G16B16A16_SNORM).rgba().ToPair(),

            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_R16_UINT).r().Int(u16).ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_R16G16_UINT).rg().Int(u16).ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_R16G16B16_UINT).rgb().Int(u16).ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_R16G16B16A16_UINT).rgba().Int(u16).ToPair(),

            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_R16_SINT).r().Int(s16).ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_R16G16_SINT).rg().Int(s16).ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_R16G16B16_SINT).rgb().Int(s16).ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_R16G16B16A16_SINT).rgba().Int(s16).ToPair(),

            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_R16_SFLOAT).r().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_R16G16_SFLOAT).rg().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_R16G16B16_SFLOAT).rgb().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_R16G16B16A16_SFLOAT).rgba().ToPair(),

            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_R32_SINT).r().Int(s32).ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_R32G32_SINT).rg().Int(s32).ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_R32G32B32_SINT).rgb().Int(s32).ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_R32G32B32A32_SINT).rgba().Int(s32).ToPair(),

            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_R32_UINT).r().Int(u32).ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_R32G32_UINT).rg().Int(u32).ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_R32G32B32_UINT).rgb().Int(u32).ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_R32G32B32A32_UINT).rgba().Int(u32).ToPair(),

            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_R32_SFLOAT).r().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_R32G32_SFLOAT).rg().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_R32G32B32_SFLOAT).rgb().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_R32G32B32A32_SFLOAT).rgba().ToPair(),

            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_R5G5B5A1_UNORM_PACK16).rgba().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_R5G6B5_UNORM_PACK16).rgb().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_A2B10G10R10_UNORM_PACK32).rgba().ToPair(),

            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_R4G4B4A4_UNORM_PACK16).rgba().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_A1R5G5B5_UNORM_PACK16).rgba().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_A2R10G10B10_UNORM_PACK32).rgba().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_A2R10G10B10_UINT_PACK32).rgba().Int(uRGB10A2).ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_A2B10G10R10_UNORM_PACK32).rgba().ToPair(),

            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_A2B10G10R10_UINT_PACK32).rgba().Int(uRGB10A2).ToPair(),
            // Runtimes with D3D11 back-ends map VK_FORMAT_B10G11R11_UFLOAT_PACK32 to DXGI_FORMAT_R11G11B10_FLOAT and that format doesn't have a TYPELESS equivalent.
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_B10G11R11_UFLOAT_PACK32).rgb().NotMutable().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_E5B9G9R9_UFLOAT_PACK32).rgb().ToPair(),

            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_R16G16B16A16_SFLOAT).rgba().ToPair(),

            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_D16_UNORM).Depth().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_D24_UNORM_S8_UINT).DepthStencil().ToPair(),

            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_X8_D24_UNORM_PACK32).Depth().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_S8_UINT).Stencil().ToPair(),

            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_D32_SFLOAT).Depth().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_D32_SFLOAT_S8_UINT).DepthStencil().ToPair(),

            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK).rgb().Compressed().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK).rgba().Compressed().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK).rgba().Compressed().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK).rgb().Compressed().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK).rgba().Compressed().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK).rgba().Compressed().ToPair(),

            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_EAC_R11_UNORM_BLOCK).r().Compressed().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_EAC_R11G11_UNORM_BLOCK).rg().Compressed().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_EAC_R11_SNORM_BLOCK).r().Compressed().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_EAC_R11G11_SNORM_BLOCK).rg().Compressed().ToPair(),

            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_ASTC_4x4_UNORM_BLOCK).rgba().Compressed().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_ASTC_5x4_UNORM_BLOCK).rgba().Compressed().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_ASTC_5x5_UNORM_BLOCK).rgba().Compressed().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_ASTC_6x5_UNORM_BLOCK).rgba().Compressed().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_ASTC_6x6_UNORM_BLOCK).rgba().Compressed().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_ASTC_8x5_UNORM_BLOCK).rgba().Compressed().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_ASTC_8x6_UNORM_BLOCK).rgba().Compressed().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_ASTC_8x8_UNORM_BLOCK).rgba().Compressed().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_ASTC_10x5_UNORM_BLOCK).rgba().Compressed().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_ASTC_10x6_UNORM_BLOCK).rgba().Compressed().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_ASTC_10x8_UNORM_BLOCK).rgba().Compressed().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_ASTC_10x10_UNORM_BLOCK).rgba().Compressed().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_ASTC_12x10_UNORM_BLOCK).rgba().Compressed().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_ASTC_12x12_UNORM_BLOCK).rgba().Compressed().ToPair(),

            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_ASTC_4x4_SRGB_BLOCK).rgba().Compressed().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_ASTC_5x4_SRGB_BLOCK).rgba().Compressed().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_ASTC_5x5_SRGB_BLOCK).rgba().Compressed().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_ASTC_6x5_SRGB_BLOCK).rgba().Compressed().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_ASTC_6x6_SRGB_BLOCK).rgba().Compressed().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_ASTC_8x5_SRGB_BLOCK).rgba().Compressed().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_ASTC_8x6_SRGB_BLOCK).rgba().Compressed().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_ASTC_8x8_SRGB_BLOCK).rgba().Compressed().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_ASTC_10x5_SRGB_BLOCK).rgba().Compressed().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_ASTC_10x6_SRGB_BLOCK).rgba().Compressed().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_ASTC_10x8_SRGB_BLOCK).rgba().Compressed().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_ASTC_10x10_SRGB_BLOCK).rgba().Compressed().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_ASTC_12x10_SRGB_BLOCK).rgba().Compressed().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_ASTC_12x12_SRGB_BLOCK).rgba().Compressed().ToPair(),

            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_BC1_RGBA_UNORM_BLOCK).rgba().Compressed().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_BC1_RGBA_SRGB_BLOCK).rgba().Compressed().ToPair(),

            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_BC2_UNORM_BLOCK).rgba().Compressed().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_BC2_SRGB_BLOCK).rgba().Compressed().ToPair(),

            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_BC3_UNORM_BLOCK).rgba().Compressed().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_BC3_SRGB_BLOCK).rgba().Compressed().ToPair(),

            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_BC6H_UFLOAT_BLOCK).rgb().Compressed().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_BC6H_SFLOAT_BLOCK).rgb().Compressed().ToPair(),

            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_BC7_UNORM_BLOCK).rgba().Compressed().ToPair(),
            XRC_SWAPCHAIN_FORMAT(VK_FORMAT_BC7_SRGB_BLOCK).rgba().Compressed().ToPair(),
        }};
        return map;
    }

    std::string VulkanGraphicsPlugin::GetImageFormatName(int64_t imageFormat) const
    {
        return ::Conformance::GetImageFormatName(GetSwapchainFormatData(), imageFormat);
    }

    bool VulkanGraphicsPlugin::IsImageFormatKnown(int64_t imageFormat) const
    {
        return ::Conformance::IsImageFormatKnown(GetSwapchainFormatData(), imageFormat);
    }

    bool VulkanGraphicsPlugin::GetSwapchainCreateTestParameters(int64_t imageFormat, SwapchainCreateTestParameters* swapchainTestParameters)
    {
        *swapchainTestParameters = ::Conformance::GetSwapchainCreateTestParameters(GetSwapchainFormatData(), imageFormat);

        // Conformance::GetSwapchainCreateTestParameters has no access to our Vulkan objects, therefore the queries have to be made here.

        VkFormatProperties formatProperties{};
        vkGetPhysicalDeviceFormatProperties(m_vkPhysicalDevice, (VkFormat)imageFormat, &formatProperties);

        // we only care about optimal tiling
        bool supportsStorageImage = (formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) != 0;

        swapchainTestParameters->usageFlagsVector.erase(
            std::remove_if(swapchainTestParameters->usageFlagsVector.begin(), swapchainTestParameters->usageFlagsVector.end(),
                           [&](auto uf) {
                               if ((uf & XR_SWAPCHAIN_USAGE_UNORDERED_ACCESS_BIT) != 0) {
                                   if (!supportsStorageImage) {
                                       return true;
                                   }
                               }
                               return false;
                           }),
            swapchainTestParameters->usageFlagsVector.end());

        return true;
    }

    bool VulkanGraphicsPlugin::ValidateSwapchainImages(int64_t /*imageFormat*/, const SwapchainCreateTestParameters* /*tp*/,
                                                       XrSwapchain swapchain, uint32_t* imageCount) const noexcept(false)
    {
        // OK to use CHECK and REQUIRE in here because this is always called from within a test.
        *imageCount = 0;  // Zero until set below upon success.

        std::vector<XrSwapchainImageVulkanKHR> swapchainImageVector;
        uint32_t countOutput;

        XrResult result = xrEnumerateSwapchainImages(swapchain, 0, &countOutput, nullptr);
        CHECK(ValidateResultAllowed("xrEnumerateSwapchainImages", result));
        REQUIRE(result == XR_SUCCESS);
        REQUIRE(countOutput > 0);

        swapchainImageVector.resize(countOutput, XrSwapchainImageVulkanKHR{XR_TYPE_SWAPCHAIN_IMAGE_VULKAN_KHR});

        // Exercise XR_ERROR_SIZE_INSUFFICIENT
        if (countOutput >= 2) {  // Need at least two in order to exercise XR_ERROR_SIZE_INSUFFICIENT
            result = xrEnumerateSwapchainImages(swapchain, 1, &countOutput,
                                                reinterpret_cast<XrSwapchainImageBaseHeader*>(swapchainImageVector.data()));
            CHECK(ValidateResultAllowed("xrEnumerateSwapchainImages", result));
            CHECK(result == XR_ERROR_SIZE_INSUFFICIENT);
            CHECK(countOutput == swapchainImageVector.size());
            // Contents of swapchainImageVector is undefined, so nothing to validate about the output.
        }

        countOutput = (uint32_t)swapchainImageVector.size();  // Restore countOutput if it was (mistakenly) modified.
        swapchainImageVector.clear();                         // Who knows what the runtime may have mistakely written into our vector.
        swapchainImageVector.resize(countOutput, XrSwapchainImageVulkanKHR{XR_TYPE_SWAPCHAIN_IMAGE_VULKAN_KHR});
        result = xrEnumerateSwapchainImages(swapchain, countOutput, &countOutput,
                                            reinterpret_cast<XrSwapchainImageBaseHeader*>(swapchainImageVector.data()));
        CHECK(ValidateResultAllowed("xrEnumerateSwapchainImages", result));
        REQUIRE(result == XR_SUCCESS);
        REQUIRE(countOutput == swapchainImageVector.size());
        REQUIRE(ValidateStructVectorType(swapchainImageVector, XR_TYPE_SWAPCHAIN_IMAGE_VULKAN_KHR));

        for (const XrSwapchainImageVulkanKHR& image : swapchainImageVector) {
            // Verify that the image is a valid handle
            CHECK(image.image != VK_NULL_HANDLE);
        }

        *imageCount = countOutput;
        return true;
    }

    bool VulkanGraphicsPlugin::ValidateSwapchainImageState(XrSwapchain /*swapchain*/, uint32_t /*index*/, int64_t /*imageFormat*/) const
    {
        // OK to use CHECK and REQUIRE in here because this is always called from within a test.
        // TODO: check VK_ACCESS_*? Mandate this in the spec?
        return true;
    }

    // Select the preferred swapchain format from the list of available formats.
    int64_t VulkanGraphicsPlugin::SelectColorSwapchainFormat(bool throwIfNotFound, span<const int64_t> imageFormatArray) const
    {
        // List of supported color swapchain formats.
        return SelectSwapchainFormat(  //
            throwIfNotFound, imageFormatArray,
            {
                VK_FORMAT_R8G8B8A8_SRGB,
                VK_FORMAT_B8G8R8A8_SRGB,
                VK_FORMAT_R8G8B8A8_UNORM,
                VK_FORMAT_B8G8R8A8_UNORM,
            });
    }

    // Select the preferred swapchain format from the list of available formats.
    int64_t VulkanGraphicsPlugin::SelectDepthSwapchainFormat(bool throwIfNotFound, span<const int64_t> imageFormatArray) const
    {
        // List of supported depth swapchain formats.
        return SelectSwapchainFormat(  //
            throwIfNotFound, imageFormatArray,
            {
                VK_FORMAT_D32_SFLOAT,
                VK_FORMAT_D24_UNORM_S8_UINT,
                VK_FORMAT_D16_UNORM,
                VK_FORMAT_D32_SFLOAT_S8_UINT,
            });
    }

    // Select the preferred swapchain format from the list of available formats.
    int64_t VulkanGraphicsPlugin::SelectMotionVectorSwapchainFormat(bool throwIfNotFound, span<const int64_t> imageFormatArray) const
    {
        // Implementation must select a signed format with four components unless there are none with alpha.
        int64_t alphaFormat = SelectSwapchainFormat(  //
            false, imageFormatArray, {VK_FORMAT_R16G16B16A16_SFLOAT});
        if (alphaFormat != -1) {
            return alphaFormat;
        }
        return SelectSwapchainFormat(  //
            throwIfNotFound, imageFormatArray, {VK_FORMAT_R16G16B16_SFLOAT});
    }

    int64_t VulkanGraphicsPlugin::GetSRGBA8Format() const
    {
        return VK_FORMAT_R8G8B8A8_SRGB;
    }

    ISwapchainImageData* VulkanGraphicsPlugin::AllocateSwapchainImageData(size_t size, const XrSwapchainCreateInfo& swapchainCreateInfo)
    {
        auto typedResult = std::make_unique<VulkanSwapchainImageData>(
            m_namer, uint32_t(size), swapchainCreateInfo, m_vkDevice, &m_memAllocator, m_pipelineLayout, m_computePipelineLayout,
            m_shaderProgram, m_computeShaderProgram, VulkanMesh::c_bindingDesc, VulkanMesh::c_attrDesc);

        // Cast our derived type to the caller-expected type.
        auto ret = static_cast<ISwapchainImageData*>(typedResult.get());

        m_swapchainImageDataMap.Adopt(std::move(typedResult));

        return ret;
    }

    inline ISwapchainImageData* VulkanGraphicsPlugin::AllocateSwapchainImageDataWithDepthSwapchain(
        size_t size, const XrSwapchainCreateInfo& colorSwapchainCreateInfo, XrSwapchain depthSwapchain,
        const XrSwapchainCreateInfo& depthSwapchainCreateInfo, bool isMotionVector)
    {
        bool multiview_enable = GetGlobalData().IsUsingMultiview();
        ALOGE("%s:xxxxxx:%d,%d", __func__, isMotionVector, multiview_enable);
        #ifndef FEATURE_ADD_MOTION_VECTOR
        isMotionVector = 0;
        #endif

        if(!isMotionVector){
            if(!multiview_enable){
                auto typedResult = std::make_unique<VulkanSwapchainImageData>(
                    m_namer, uint32_t(size), colorSwapchainCreateInfo, depthSwapchain, depthSwapchainCreateInfo, m_vkDevice, &m_memAllocator,
                    m_pipelineLayout, m_computePipelineLayout, m_shaderProgram, m_computeShaderProgram, VulkanMesh::c_bindingDesc,
                    VulkanMesh::c_attrDesc);

                // Cast our derived type to the caller-expected type.
                auto ret = static_cast<ISwapchainImageData*>(typedResult.get());

                m_swapchainImageDataMap.Adopt(std::move(typedResult));
                return ret;
            } else {
                bool edo_enalble      = GetGlobalData().IsUsingEnvDepthOcclusion();
                auto typedResult = std::make_unique<VulkanSwapchainImageData>(
                    m_namer, uint32_t(size), colorSwapchainCreateInfo, depthSwapchain, depthSwapchainCreateInfo, m_vkDevice, &m_memAllocator,
                    m_pipelineLayout, m_computePipelineLayout, edo_enalble?m_envDepthShaderProgram:m_multiviewShaderProgram, m_computeShaderProgram, VulkanMesh::c_bindingDesc,
                    VulkanMesh::c_attrDesc);

                // Cast our derived type to the caller-expected type.
                auto ret = static_cast<ISwapchainImageData*>(typedResult.get());

                m_swapchainImageDataMap.Adopt(std::move(typedResult));
                return ret;
            }
        } else {
            auto typedResult = std::make_unique<VulkanSwapchainImageData>(
                m_namer, uint32_t(size), colorSwapchainCreateInfo, depthSwapchain, depthSwapchainCreateInfo, m_vkDevice, &m_memAllocator,
                m_pipelineLayout, m_computePipelineLayout, m_mvShaderProgram, m_computeShaderProgram, VulkanMesh::c_bindingDesc,
                VulkanMesh::c_attrDesc);

            // Cast our derived type to the caller-expected type.
            auto ret = static_cast<ISwapchainImageData*>(typedResult.get());

            m_swapchainImageDataMap.Adopt(std::move(typedResult));

            return ret;
        }
    }

    void VulkanGraphicsPlugin::CopyRGBAImage(const XrSwapchainImageBaseHeader* swapchainImageBase, uint32_t arraySlice,
                                             const RGBAImage& image, int faceId)
    {
        const XrSwapchainImageVulkanKHR* swapchainImageVk = reinterpret_cast<const XrSwapchainImageVulkanKHR*>(swapchainImageBase);

        VulkanSwapchainImageData* swapchainData;
        uint32_t imageIndex;

        std::tie(swapchainData, imageIndex) = m_swapchainImageDataMap.GetDataAndIndexFromBasePointer(swapchainImageBase);

        if(swapchainData->FaceCount() == 6){
            if(faceId < 0 || faceId >= 6) {
                throw std::runtime_error("Invalid face id"); 
            }
            arraySlice = arraySlice * 6 + faceId;
        }

        uint32_t w = image.width;
        uint32_t h = image.height;

        // Create a linear staging buffer
        VkImageCreateInfo imgInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        imgInfo.imageType = VK_IMAGE_TYPE_2D;

        int64_t imageFormat = swapchainData->GetCreateInfo().format;
        XRC_CHECK_THROW(imageFormat == GetSRGBA8Format());

        imgInfo.format = static_cast<VkFormat>(imageFormat);
        imgInfo.extent = {w, h, 1};
        imgInfo.mipLevels = 1;
        imgInfo.arrayLayers = 1;
        imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imgInfo.tiling = VK_IMAGE_TILING_LINEAR;
        imgInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        imgInfo.initialLayout = VK_IMAGE_LAYOUT_PREINITIALIZED;
        VkImage stagingImage{VK_NULL_HANDLE};
        XRC_CHECK_THROW_VKCMD(vkCreateImage(m_vkDevice, &imgInfo, nullptr, &stagingImage));

        VkMemoryRequirements memReq{};
        vkGetImageMemoryRequirements(m_vkDevice, stagingImage, &memReq);
        VkDeviceMemory stagingMemory{VK_NULL_HANDLE};
        m_memAllocator.Allocate(memReq, &stagingMemory, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        XRC_CHECK_THROW_VKCMD(vkBindImageMemory(m_vkDevice, stagingImage, stagingMemory, 0));

        VkImageSubresource imgSubRes{};
        imgSubRes.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imgSubRes.mipLevel = 0;
        imgSubRes.arrayLayer = 0;
        VkSubresourceLayout layout{};
        vkGetImageSubresourceLayout(m_vkDevice, stagingImage, &imgSubRes, &layout);

        uint8_t* data{nullptr};
        XRC_CHECK_THROW_VKCMD(vkMapMemory(m_vkDevice, stagingMemory, layout.offset, layout.size, 0, (void**)&data));
        image.CopyWithStride(data, static_cast<uint32_t>(layout.rowPitch), static_cast<uint32_t>(layout.offset));
        vkUnmapMemory(m_vkDevice, stagingMemory);

        bool isProtectedMemory = GetGlobalData().IsProtectedMemory();
        VkCommandBuffer buf = isProtectedMemory?m_cmdBuffer.bufP:m_cmdBuffer.buf;

        m_cmdBuffer.Clear();
        m_cmdBuffer.Begin(isProtectedMemory);

        // Switch the staging buffer from PREINITIALIZED -> TRANSFER_SRC_OPTIMAL
        VkImageMemoryBarrier imgBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        imgBarrier.srcAccessMask = 0;
        imgBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        imgBarrier.oldLayout = VK_IMAGE_LAYOUT_PREINITIALIZED;
        imgBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        imgBarrier.srcQueueFamilyIndex = m_queueFamilyIndex;
        imgBarrier.dstQueueFamilyIndex = m_queueFamilyIndex;
        imgBarrier.image = stagingImage;
        imgBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(buf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                             &imgBarrier);

        // Switch the destination image from COLOR_ATTACHMENT_OPTIMAL -> TRANSFER_DST_OPTIMAL
        //
        // XR_KHR_vulkan_enable / XR_KHR_vulkan_enable2
        // When an application acquires a swapchain image by calling xrAcquireSwapchainImage
        // in a session created using XrGraphicsBindingVulkanKHR, the OpenXR runtime must
        // guarantee that:
        // - The image has a memory layout compatible with VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
        //   for color images, or VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL for depth images.
        // - The VkQueue specified in XrGraphicsBindingVulkanKHR has ownership of the image.
        imgBarrier.srcAccessMask = 0;
        imgBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        imgBarrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        imgBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        imgBarrier.srcQueueFamilyIndex = m_queueFamilyIndex;
        imgBarrier.dstQueueFamilyIndex = m_queueFamilyIndex;
        imgBarrier.image = swapchainImageVk->image;
        imgBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, arraySlice, 1};
        vkCmdPipelineBarrier(buf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                             &imgBarrier);

        // Blit staging -> swapchain
        VkImageBlit blit = {{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                            {{0, 0, 0}, {(int32_t)w, (int32_t)h, 1}},
                            {VK_IMAGE_ASPECT_COLOR_BIT, 0, arraySlice, 1},
                            {{0, 0, 0}, {(int32_t)w, (int32_t)h, 1}}};
        vkCmdBlitImage(buf, stagingImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, swapchainImageVk->image,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_NEAREST);

        // Switch the destination image from TRANSFER_DST_OPTIMAL -> COLOR_ATTACHMENT_OPTIMAL
        //
        // XR_KHR_vulkan_enable / XR_KHR_vulkan_enable2:
        // When an application releases a swapchain image by calling xrReleaseSwapchainImage,
        // in a session created using XrGraphicsBindingVulkanKHR, the OpenXR runtime must
        // interpret the image as:
        //  - Having a memory layout compatible with VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
        //    for color images, or VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL for depth
        //    images.
        //  - Being owned by the VkQueue specified in XrGraphicsBindingVulkanKHR.
        imgBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        imgBarrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
        imgBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        imgBarrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        imgBarrier.srcQueueFamilyIndex = m_queueFamilyIndex;
        imgBarrier.dstQueueFamilyIndex = m_queueFamilyIndex;
        imgBarrier.image = swapchainImageVk->image;
        imgBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, arraySlice, 1};
        vkCmdPipelineBarrier(buf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, nullptr,
                             0, nullptr, 1, &imgBarrier);

        m_cmdBuffer.End(isProtectedMemory);
        m_cmdBuffer.Exec(isProtectedMemory?m_vkQueueProtected:m_vkQueue, isProtectedMemory);
        m_cmdBuffer.Wait();

        vkDestroyImage(m_vkDevice, stagingImage, nullptr);
        vkFreeMemory(m_vkDevice, stagingMemory, nullptr);
    }

    void VulkanGraphicsPlugin::SetViewportAndScissor(const VkRect2D& rect)
    {
        bool isProtectedMemory = GetGlobalData().IsProtectedMemory();
        VkCommandBuffer buf = isProtectedMemory?m_cmdBuffer.bufP:m_cmdBuffer.buf;

        VkViewport viewport{float(rect.offset.x), float(rect.offset.y), float(rect.extent.width), float(rect.extent.height), 0.0f, 1.0f};
        vkCmdSetViewport(buf, 0, 1, &viewport);
        vkCmdSetScissor(buf, 0, 1, &rect);
    }

    /// Compute image layout for the "second image" format (depth and/or stencil)
    static inline VkImageLayout ComputeLayout(const SwapchainFormatData& secondFormatData)
    {
        // can't use VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL /
        //           VK_IMAGE_LAYOUT_STENCIL_ATTACHMENT_OPTIMAL
        // because they're Vulkan 1.2
        if (secondFormatData.GetDepthFormat() || secondFormatData.GetStencilFormat()) {
            return VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        }
        if (secondFormatData.GetColorFormat()) {
            return VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        }
        throw std::runtime_error("No idea what layout to use for depth/stencil image");
    }

    /// Compute image aspect flags for the "second image" format (depth and/or stencil)
    static inline VkImageAspectFlags ComputeAspectFlags(const SwapchainFormatData& secondFormatData)
    {
        VkImageAspectFlags secondAttachmentAspect = 0;
        if (secondFormatData.GetDepthFormat()) {
            secondAttachmentAspect |= VK_IMAGE_ASPECT_DEPTH_BIT;
        }
        if (secondFormatData.GetStencilFormat()) {
            secondAttachmentAspect |= VK_IMAGE_ASPECT_STENCIL_BIT;
        }
        if (secondFormatData.GetColorFormat()) {
            throw std::logic_error("this is just for the second image");
        }
        return secondAttachmentAspect;
    }

    const SwapchainFormatData& VulkanGraphicsPlugin::FindFormatData(int64_t format) const
    {
        auto& vkFormatMap = GetSwapchainFormatData();
        auto it = vkFormatMap.find(format);
        assert(it != vkFormatMap.end());
        return it->second;
    }

    void VulkanGraphicsPlugin::ClearImageSlice(const XrSwapchainImageBaseHeader* colorSwapchainImage, uint32_t imageArrayIndex,
                                               XrColor4f color)
    {
        VulkanSwapchainImageData* swapchainData;
        uint32_t imageIndex;

        std::tie(swapchainData, imageIndex) = m_swapchainImageDataMap.GetDataAndIndexFromBasePointer(colorSwapchainImage);

        bool isProtectedMemory = GetGlobalData().IsProtectedMemory();
        VkCommandBuffer buf = isProtectedMemory?m_cmdBuffer.bufP:m_cmdBuffer.buf;

        m_cmdBuffer.Clear();
        m_cmdBuffer.Begin(isProtectedMemory);

        VkRect2D renderArea = {{0, 0}, {swapchainData->Width(), swapchainData->Height()}};
        SetViewportAndScissor(renderArea);

        // may be depth, stencil, or both
        int64_t secondImageFormat = swapchainData->GetDepthFormat();
        const SwapchainFormatData& secondFormatData = FindFormatData(secondImageFormat);

        VkImageAspectFlags secondAttachmentAspect = ComputeAspectFlags(secondFormatData);

        // Bind eye render target
        VkRenderPassBeginInfo renderPassBeginInfo{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        swapchainData->BindRenderTarget(imageIndex, imageArrayIndex, renderArea, secondAttachmentAspect, &renderPassBeginInfo);

        if (!swapchainData->DepthSwapchainEnabled()) {
            // Ensure self-made fallback depth is in the right layout
            VkImageLayout layout = ComputeLayout(secondFormatData);
            swapchainData->TransitionLayout(imageIndex, &m_cmdBuffer, layout);
        }

        vkCmdBeginRenderPass(buf, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

        swapchainData->BindPipeline(buf, imageArrayIndex);

        // Clear the buffers
        static std::array<VkClearValue, 4> clearValues;
        clearValues[0].color.float32[0] = color.r;
        clearValues[0].color.float32[1] = color.g;
        clearValues[0].color.float32[2] = color.b;
        clearValues[0].color.float32[3] = color.a;
        clearValues[1].depthStencil.depth = 1.0f;
        clearValues[1].depthStencil.stencil = 0;
        std::array<VkClearAttachment, 4> clearAttachments{{
            {VK_IMAGE_ASPECT_COLOR_BIT, 0, clearValues[0]},
            {secondAttachmentAspect, 0, clearValues[1]},
        }};

        // imageArrayIndex already included in the VkImageView
        VkClearRect clearRect{renderArea, 0, 1};
        #ifdef FEATURE_ADD_MSAA
        bool msaa_enable = GetGlobalData().IsUsingMSAA();
        if(msaa_enable){
            clearValues[2].color.float32[0] = color.r;
            clearValues[2].color.float32[1] = color.g;
            clearValues[2].color.float32[2] = color.b;
            clearValues[2].color.float32[3] = color.a;
            clearValues[3].depthStencil.depth = 1.0f;
            clearValues[3].depthStencil.stencil = 0;

            clearAttachments[2] = {VK_IMAGE_ASPECT_COLOR_BIT, 0, clearValues[2]};
            clearAttachments[3] = {secondAttachmentAspect, 0, clearValues[3]};
            vkCmdClearAttachments(buf, 4, &clearAttachments[0], 1, &clearRect);
        } else {
            vkCmdClearAttachments(buf, 2, &clearAttachments[0], 1, &clearRect);
        }
        #else
        vkCmdClearAttachments(buf, 2, &clearAttachments[0], 1, &clearRect);
        #endif

        vkCmdEndRenderPass(buf);

        m_cmdBuffer.End(isProtectedMemory);
        m_cmdBuffer.Exec(isProtectedMemory?m_vkQueueProtected:m_vkQueue, isProtectedMemory);
        // XXX Should double-buffer the command buffers, for now just flush
        m_cmdBuffer.Wait();
    }

    MeshHandle VulkanGraphicsPlugin::MakeSimpleMesh(span<const uint16_t> idx, span<const Geometry::Vertex> vtx)
    {
        auto handle =
            m_meshes.emplace_back(m_vkDevice, m_namer, &m_memAllocator, idx.data(), (uint32_t)idx.size(), vtx.data(), (uint32_t)vtx.size());

        return handle;
    }

    GLTFModelHandle VulkanGraphicsPlugin::LoadGLTF(Gltf::ModelBuilder&& modelBuilder)
    {
        auto handle = m_gltfModels.emplace_back(modelBuilder.Build(*m_pbrResources));
        return handle;
    }

    std::shared_ptr<Pbr::Model> VulkanGraphicsPlugin::GetPbrModel(GLTFModelHandle handle) const
    {
        return m_gltfModels[handle];
    }

    GLTFModelInstanceHandle VulkanGraphicsPlugin::CreateGLTFModelInstance(GLTFModelHandle handle)
    {
        auto pbrModelInstance = Pbr::VulkanModelInstance(*m_pbrResources, GetPbrModel(handle));
        auto instanceHandle = m_gltfInstances.emplace_back(std::move(pbrModelInstance));
        return instanceHandle;
    }

    Pbr::ModelInstance& VulkanGraphicsPlugin::GetModelInstance(GLTFModelInstanceHandle handle)
    {
        return m_gltfInstances[handle].GetModelInstance();
    }

    void VulkanGraphicsPlugin::RenderView(const XrCompositionLayerProjectionView& layerView,
                                          const XrSwapchainImageBaseHeader* colorSwapchainImage, const RenderParams& params, bool isMotionVectorPass, const XrCompositionLayerProjectionView* prevLayerView, 
                                          const XrCompositionLayerProjectionView* nextLayerView, const XrCompositionLayerProjectionView* nextPrevLayerView,
                                         const EnvDepthOcclusionParams* edoParams)

    {
        VulkanSwapchainImageData* swapchainData;
        uint32_t imageIndex;
        bool multiview_enable = GetGlobalData().IsUsingMultiview();
        bool isProtectedMemory = GetGlobalData().IsProtectedMemory();
        VkCommandBuffer buf = isProtectedMemory?m_cmdBuffer.bufP:m_cmdBuffer.buf;
        (nextPrevLayerView);

        ALOGE("%s:xxxxxx:%d", __func__, isMotionVectorPass);
        #ifndef FEATURE_ADD_MOTION_VECTOR
        isMotionVectorPass = 0;
        #endif

        std::tie(swapchainData, imageIndex) = m_swapchainImageDataMap.GetDataAndIndexFromBasePointer(colorSwapchainImage);

        bool edo_enalble      = GetGlobalData().IsUsingEnvDepthOcclusion();
        struct
        {
            XrMatrix4x4f vp[2];
        } ubo;

        uint8_t* uboData = (uint8_t*)&ubo;

        VkBufferCreateInfo bufferCreateInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bufferCreateInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        bufferCreateInfo.size = static_cast<VkDeviceSize>(sizeof(ubo));

        Conformance::BufferAndMemory uboBuffer;
        uboBuffer.Create(m_vkDevice, m_memAllocator, bufferCreateInfo);
        XRC_CHECK_THROW_VKCMD(m_namer.SetName(VK_OBJECT_TYPE_BUFFER, (uint64_t)uboBuffer.buf, "CTS ENV depth UBO buffer"));
        ALOGE("%s:xxxxxx::2", __func__);

        VkImageView depthView{VK_NULL_HANDLE};
        auto defaultSampler = std::make_shared<Conformance::ScopedVkSampler>(Pbr::VulkanTexture::CreateSampler(m_vkDevice), m_vkDevice);
        ALOGE("%s:xxxxxx::2.1", __func__);
        if(edo_enalble){
            VkImageViewCreateInfo depthViewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};

            ALOGE("%s:xxxxxx::2.2:%p", __func__, edoParams->depthTex);
            depthViewInfo.image = (VkImage)edoParams->depthTex;
            depthViewInfo.viewType = multiview_enable?VK_IMAGE_VIEW_TYPE_2D_ARRAY:VK_IMAGE_VIEW_TYPE_2D;
            depthViewInfo.format = VK_FORMAT_D16_UNORM;
            depthViewInfo.components.r = VK_COMPONENT_SWIZZLE_R;
            depthViewInfo.components.g = VK_COMPONENT_SWIZZLE_G;
            depthViewInfo.components.b = VK_COMPONENT_SWIZZLE_B;
            depthViewInfo.components.a = VK_COMPONENT_SWIZZLE_A;
            depthViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            depthViewInfo.subresourceRange.baseMipLevel = 0;
            depthViewInfo.subresourceRange.levelCount = 1;
            depthViewInfo.subresourceRange.baseArrayLayer = 0;
            depthViewInfo.subresourceRange.layerCount = multiview_enable?2:1;
            XRC_CHECK_THROW_VKCMD(vkCreateImageView(m_vkDevice, &depthViewInfo, nullptr, &depthView));
            ALOGE("%s:xxxxxx::2.3:%p", __func__, depthView);
            XRC_CHECK_THROW_VKCMD(m_namer.SetName(VK_OBJECT_TYPE_IMAGE_VIEW, (uint64_t)depthView, "CTS env depth image view"));

            ubo.vp[0] = edoParams->depthViewProj[0];
            ubo.vp[1] = edoParams->depthViewProj[1];
            ubo.vp[0].m[0] = 1.0f;
            ubo.vp[1].m[0] = 1.0f;
            uboBuffer.Update<uint8_t>(m_vkDevice, {uboData, static_cast<size_t>(bufferCreateInfo.size)}, 0);

            VkImageView depthImageView = depthView;
            VkImageLayout depthImageLayout = VK_IMAGE_LAYOUT_GENERAL;
            uint32_t depthImageBinding = 0;
            uint32_t uboBinding = 1;

            VkDescriptorImageInfo depthImageInfo;
            depthImageInfo.imageLayout = depthImageLayout;
            depthImageInfo.imageView = depthImageView;
            depthImageInfo.sampler = defaultSampler->get();

            VkDescriptorBufferInfo uboInfo;
            uboInfo.buffer = uboBuffer.buf;
            uboInfo.offset = 0;
            uboInfo.range = VK_WHOLE_SIZE;

            VkWriteDescriptorSet writeDescriptorSets[2]{};
            writeDescriptorSets[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeDescriptorSets[0].dstSet = m_gfxDescriptorSet;
            writeDescriptorSets[0].dstBinding = depthImageBinding;
            writeDescriptorSets[0].descriptorCount = 1;
            writeDescriptorSets[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;//VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            writeDescriptorSets[0].pImageInfo = &depthImageInfo;

            writeDescriptorSets[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeDescriptorSets[1].dstSet = m_gfxDescriptorSet;
            writeDescriptorSets[1].dstBinding = uboBinding;
            writeDescriptorSets[1].descriptorCount = 1;
            writeDescriptorSets[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writeDescriptorSets[1].pBufferInfo = &uboInfo;

            vkUpdateDescriptorSets(m_vkDevice, (uint32_t)ArraySize(writeDescriptorSets), writeDescriptorSets, 0, NULL);
        }

        m_cmdBuffer.Clear();
        m_cmdBuffer.Begin(isProtectedMemory);

        CHECKPOINT();

        const XrRect2Di& r = layerView.subImage.imageRect;
        VkRect2D renderArea = {{r.offset.x, r.offset.y}, {uint32_t(r.extent.width), uint32_t(r.extent.height)}};
        SetViewportAndScissor(renderArea);

        // may be depth, stencil, or both
        int64_t secondImageFormat = swapchainData->GetDepthFormat();
        const SwapchainFormatData& secondFormatData = FindFormatData(secondImageFormat);
        VkImageAspectFlags secondAttachmentAspect = ComputeAspectFlags(secondFormatData);

        // Just bind the eye render target, ClearImageSlice will have cleared it.
        VkRenderPassBeginInfo renderPassBeginInfo{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};

        // aka slice
        auto imageArrayIndex = layerView.subImage.imageArrayIndex;

        swapchainData->BindRenderTarget(imageIndex, imageArrayIndex, renderArea, secondAttachmentAspect, &renderPassBeginInfo);

        vkCmdBeginRenderPass(buf, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

        CHECKPOINT();
        ALOGE("%s:xxxxxx::1", __func__);

        swapchainData->BindPipeline(buf, imageArrayIndex);

        CHECKPOINT();


        ALOGE("%s:xxxxxx::3", __func__);
        CHECKPOINT();

        if(!isMotionVectorPass){
            if(!multiview_enable){
                // Compute the view-projection transform.
                // Note all matrixes (including OpenXR's) are column-major, right-handed.
                const auto& pose = layerView.pose;
                XrMatrix4x4f proj;
                XrMatrix4x4f_CreateProjectionFov(&proj, GRAPHICS_VULKAN, layerView.fov, 0.05f, 100.0f);
                XrMatrix4x4f toView = Matrix::FromPose(pose);
                XrMatrix4x4f view = Matrix::InvertRigidBody(toView);
                XrMatrix4x4f vp = proj * view;
                MeshHandle lastMeshHandle;

                const auto drawMesh = [this, &vp, &lastMeshHandle, &buf](const MeshDrawable mesh) {
                    VulkanMesh& vkMesh = m_meshes[mesh.handle];
                    if (mesh.handle != lastMeshHandle) {
                        // We are now rendering a new mesh

                        // Bind index and vertex buffers
                        vkCmdBindIndexBuffer(buf, vkMesh.m_DrawBuffer.idx.buf, 0, VK_INDEX_TYPE_UINT16);

                        CHECKPOINT();

                        VkDeviceSize offset = 0;
                        vkCmdBindVertexBuffers(buf, 0, 1, &vkMesh.m_DrawBuffer.vtx.buf, &offset);

                        CHECKPOINT();
                        lastMeshHandle = mesh.handle;
                    }

                    // Compute the model-view-projection transform and push it.
                    XrMatrix4x4f model =
                        Matrix::FromTranslationRotationScale(mesh.params.pose.position, mesh.params.pose.orientation, mesh.params.scale);
                    VulkanUniformBuffer ubuf;
                    ALOGE("%s:xxxxxx:%d", __func__, sizeof(VulkanUniformBuffer));
                    ubuf.tintColor = mesh.tintColor;
                    ubuf.mvp = vp * model;
                    ubuf.alpha.x = mesh.alpha;
                    vkCmdPushConstants(buf, m_pipelineLayout.layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(VulkanUniformBuffer), &ubuf);

                    CHECKPOINT();

                    // Draw the mesh.
                    vkCmdDrawIndexed(buf, vkMesh.m_DrawBuffer.count.idx, 1, 0, 0, 0);

                    CHECKPOINT();
                };

                // Render each cube
                for (const Cube& cube : params.cubes) {
                    drawMesh(MeshDrawable{m_cubeMesh, cube.params.pose, cube.params.scale, cube.tintColor, cube.alpha});
                }

                // Render each mesh
                for (const auto& mesh : params.meshes) {
                    drawMesh(mesh);
                }

                // Render each gltf
                for (const auto& gltfDrawable : params.glTFs) {
                    VulkanGLTF& gltf = m_gltfInstances[gltfDrawable.handle];
                    // Compute and update the model transform.

                    XrMatrix4x4f modelToWorld = Matrix::FromTranslationRotationScale(
                        gltfDrawable.params.pose.position, gltfDrawable.params.pose.orientation, gltfDrawable.params.scale);
                    // XrMatrix4x4f viewMatrix = Matrix::FromPose(layerView.pose);
                    // XrMatrix4x4f viewMatrixInverse = Matrix::InvertRigidBody(viewMatrix);
                    m_pbrResources->SetViewProjection(view, proj);

                    gltf.Render(m_cmdBuffer, *m_pbrResources, modelToWorld, renderPassBeginInfo.renderPass,
                                (VkSampleCountFlagBits)swapchainData->GetCreateInfo().sampleCount);
                }

            } else {
                // Compute the view-projection transform.
                // Note all matrixes (including OpenXR's) are column-major, right-handed.
                int size = 2;

                XrVector3f scale{1.f, 1.f, 1.f};
                XrMatrix4x4f proj[size];
                XrMatrix4x4f toView[size];
                XrMatrix4x4f view[size];
                XrMatrix4x4f vp[size];

                XrMatrix4x4f_CreateProjectionFov(&proj[0], GRAPHICS_VULKAN, layerView.fov, 0.05f, 100.0f);
                XrMatrix4x4f_CreateTranslationRotationScale(&toView[0], &layerView.pose.position, &layerView.pose.orientation, &scale);
                XrMatrix4x4f_InvertRigidBody(&view[0], &toView[0]);
                XrMatrix4x4f_Multiply(&vp[0], &proj[0], &view[0]);

                XrMatrix4x4f_CreateProjectionFov(&proj[1], GRAPHICS_VULKAN, nextLayerView->fov, 0.05f, 100.0f);
                XrMatrix4x4f_CreateTranslationRotationScale(&toView[1], &nextLayerView->pose.position, &nextLayerView->pose.orientation, &scale);
                XrMatrix4x4f_InvertRigidBody(&view[1], &toView[1]);
                XrMatrix4x4f_Multiply(&vp[1], &proj[1], &view[1]);
                MeshHandle lastMeshHandle;

                const auto drawMesh = [this, &vp, &lastMeshHandle, &edo_enalble, &edoParams, &ubo, &uboBuffer, &uboData, &depthView, &defaultSampler, &size, &bufferCreateInfo, &buf](const MeshDrawable mesh) {
                    VulkanMesh& vkMesh = m_meshes[mesh.handle];
                    if (mesh.handle != lastMeshHandle) {
                        // We are now rendering a new mesh

                        // Bind index and vertex buffers
                        vkCmdBindIndexBuffer(buf, vkMesh.m_DrawBuffer.idx.buf, 0, VK_INDEX_TYPE_UINT16);

                        CHECKPOINT();

                        VkDeviceSize offset = 0;
                        vkCmdBindVertexBuffers(buf, 0, 1, &vkMesh.m_DrawBuffer.vtx.buf, &offset);

                        CHECKPOINT();
                        lastMeshHandle = mesh.handle;
                    }

                    ALOGE("%s:xxxxxx::4", __func__);
                    if(edo_enalble)
                    {
                        vkCmdBindDescriptorSets(
                            buf,
                            VK_PIPELINE_BIND_POINT_GRAPHICS,
                            m_pipelineLayout.layout,     
                            0,                  
                            1,               
                            &m_gfxDescriptorSet,        
                            0,
                            nullptr);
                    }


                    // Compute the model-view-projection transform and push it.
                    XrMatrix4x4f model =
                    Matrix::FromTranslationRotationScale(mesh.params.pose.position, mesh.params.pose.orientation, mesh.params.scale);
                    XrMatrix4x4f mvp[size];
                    mvp[0] = vp[0] * model;
                    mvp[1] = vp[1] * model;

                    VulkanUniformBuffer ubuf;
                    ALOGE("%s:xxxxxx:%d", __func__, sizeof(VulkanUniformBuffer));
                    ubuf.tintColor = mesh.tintColor;
                    ubuf.mvp = mvp[0];
                    ubuf.p0  = mvp[1];
                    ubuf.p1  = model;
                    ubuf.alpha.x = mesh.alpha;
                    vkCmdPushConstants(buf, m_pipelineLayout.layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(VulkanUniformBuffer), &ubuf);

                    CHECKPOINT();

                    // Draw the mesh.
                    vkCmdDrawIndexed(buf, vkMesh.m_DrawBuffer.count.idx, 1, 0, 0, 0);

                    CHECKPOINT();
                };

                // Render each cube
                for (const Cube& cube : params.cubes) {
                    drawMesh(MeshDrawable{m_cubeMesh, cube.params.pose, cube.params.scale, cube.tintColor, cube.alpha});
                }

                // Render each mesh
                for (const auto& mesh : params.meshes) {
                    drawMesh(mesh);
                }
            }
        } else {
            const auto& pose = layerView.pose;
            XrMatrix4x4f proj;
            XrMatrix4x4f_CreateProjectionFov(&proj, GRAPHICS_VULKAN, layerView.fov, 0.05f, 100.0f);
            XrMatrix4x4f toView = Matrix::FromPose(pose);
            XrMatrix4x4f view = Matrix::InvertRigidBody(toView);
            XrMatrix4x4f vp = proj * view;
            MeshHandle lastMeshHandle;

            const auto& posePrev = prevLayerView->pose;
            XrMatrix4x4f projPrev;
            XrMatrix4x4f_CreateProjectionFov(&projPrev, GRAPHICS_VULKAN, prevLayerView->fov, 0.05f, 100.0f);
            XrMatrix4x4f toViewPrev = Matrix::FromPose(posePrev);
            XrMatrix4x4f viewPrev = Matrix::InvertRigidBody(toViewPrev);
            XrMatrix4x4f vpPrev = projPrev * viewPrev;

            const auto drawMesh = [this, &vp, &vpPrev, &lastMeshHandle, &buf](const MeshDrawable mesh) {
                VulkanMesh& vkMesh = m_meshes[mesh.handle];
                if (mesh.handle != lastMeshHandle) {
                    // We are now rendering a new mesh

                    // Bind index and vertex buffers
                    vkCmdBindIndexBuffer(buf, vkMesh.m_DrawBuffer.idx.buf, 0, VK_INDEX_TYPE_UINT16);

                    CHECKPOINT();

                    VkDeviceSize offset = 0;
                    vkCmdBindVertexBuffers(buf, 0, 1, &vkMesh.m_DrawBuffer.vtx.buf, &offset);


                    CHECKPOINT();
                    lastMeshHandle = mesh.handle;
                }

                // Compute the model-view-projection transform and push it.
                XrMatrix4x4f model =
                    Matrix::FromTranslationRotationScale(mesh.params.pose.position, mesh.params.pose.orientation, mesh.params.scale);
                XrMatrix4x4f modelPrev =
                    Matrix::FromTranslationRotationScale(mesh.params.posePrev.position, mesh.params.posePrev.orientation, mesh.params.scalePrev);

                VulkanUniformBufferMV ubuf;
                //#ifdef FEATURE_ADD_MOTION_VECTOR
                ubuf.vp = vp;
                ubuf.prevVp = vpPrev;
                ubuf.model = model;
                ubuf.prevModel = modelPrev;
                //#endif

                vkCmdPushConstants(buf, m_pipelineLayout.layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(VulkanUniformBufferMV), &ubuf);

                CHECKPOINT();

                // Draw the mesh.
                vkCmdDrawIndexed(buf, vkMesh.m_DrawBuffer.count.idx, 1, 0, 0, 0);

                CHECKPOINT();
            };

            // Render each cube
            for (const Cube& cube : params.cubes) {
                drawMesh(MeshDrawable{m_cubeMesh, cube.params.pose, cube.params.scale, cube.tintColor, cube.alpha});
            }

            // Render each mesh
            for (const auto& mesh : params.meshes) {
                drawMesh(mesh);
            }

        }

        vkCmdEndRenderPass(buf);

        CHECKPOINT();

        VkQueue queue = isProtectedMemory?m_vkQueueProtected:m_vkQueue;
        m_pbrResources->SubmitFrameResources(queue);

        m_cmdBuffer.End(isProtectedMemory);
        m_cmdBuffer.Exec(queue, isProtectedMemory);
        // XXX Should double-buffer the command buffers, for now just flush
        m_cmdBuffer.Wait();

        m_pbrResources->Wait();

        uboBuffer.Reset(m_vkDevice);
        if(depthView != VK_NULL_HANDLE){
            vkDestroyImageView(m_vkDevice, depthView, nullptr);
        }

#if defined(USE_MIRROR_WINDOW)
        // Cycle the window's swapchain on the last view rendered
        if (swapchainData == &m_swapchainImageData.back()) {
            m_swapchain.Acquire();
            m_swapchain.Present(queue);
        }
#endif
    }

    void VulkanGraphicsPlugin::RenderClearImageSliceCompute(const XrCompositionLayerProjectionView& layerView,
                                                            const XrSwapchainImageBaseHeader* colorSwapchainImage, XrColor4f color)
    {
        (void)colorSwapchainImage;
        (void)color;
        bool isProtectedMemory = GetGlobalData().IsProtectedMemory();
        VkCommandBuffer buf = isProtectedMemory?m_cmdBuffer.bufP:m_cmdBuffer.buf;

        VulkanSwapchainImageData* swapchainData;
        uint32_t imageIndex;

        std::tie(swapchainData, imageIndex) = m_swapchainImageDataMap.GetDataAndIndexFromBasePointer(colorSwapchainImage);

        m_cmdBuffer.Clear();
        m_cmdBuffer.Begin(isProtectedMemory);

        CHECKPOINT();

        const XrRect2Di& r = layerView.subImage.imageRect;
        VkRect2D renderArea = {{r.offset.x, r.offset.y}, {uint32_t(r.extent.width), uint32_t(r.extent.height)}};
        SetViewportAndScissor(renderArea);

        // may be depth, stencil, or both
        int64_t secondImageFormat = swapchainData->GetDepthFormat();
        const SwapchainFormatData& secondFormatData = FindFormatData(secondImageFormat);
        VkImageAspectFlags secondAttachmentAspect = ComputeAspectFlags(secondFormatData);

        // Bind eye render target
        VkRenderPassBeginInfo renderPassBeginInfo{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};

        // aka slice
        auto imageArrayIndex = layerView.subImage.imageArrayIndex;

        swapchainData->BindRenderTarget(imageIndex, imageArrayIndex, renderArea, secondAttachmentAspect, &renderPassBeginInfo);

        if (!swapchainData->DepthSwapchainEnabled()) {
            // Ensure self-made fallback depth is in the right layout
            VkImageLayout layout = ComputeLayout(secondFormatData);
            swapchainData->TransitionLayout(imageIndex, &m_cmdBuffer, layout);
        }

        CHECKPOINT();

        auto defaultSampler = std::make_shared<Conformance::ScopedVkSampler>(Pbr::VulkanTexture::CreateSampler(m_vkDevice), m_vkDevice);

        struct
        {
            float color[4];
        } ubo = {color.r, color.r, color.g, color.a};
        uint8_t* uboData = (uint8_t*)&ubo;

        VkBufferCreateInfo bufferCreateInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bufferCreateInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        bufferCreateInfo.size = static_cast<VkDeviceSize>(sizeof(ubo));

        Conformance::BufferAndMemory uboBuffer;
        uboBuffer.Create(m_vkDevice, m_memAllocator, bufferCreateInfo);
        XRC_CHECK_THROW_VKCMD(m_namer.SetName(VK_OBJECT_TYPE_BUFFER, (uint64_t)uboBuffer.buf, "CTS UBO buffer"));
        uboBuffer.Update<uint8_t>(m_vkDevice, {uboData, static_cast<size_t>(bufferCreateInfo.size)}, 0);

        {
            VkImageMemoryBarrier imgBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            // Switch the destination image from COLOR_ATTACHMENT_OPTIMAL -> GENERAL
            //
            // XR_KHR_vulkan_enable / XR_KHR_vulkan_enable2
            // When an application acquires a swapchain image by calling xrAcquireSwapchainImage
            // in a session created using XrGraphicsBindingVulkanKHR, the OpenXR runtime must
            // guarantee that:
            // - The image has a memory layout compatible with VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
            //   for color images, or VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL for depth images.
            // - The VkQueue specified in XrGraphicsBindingVulkanKHR has ownership of the image.
            imgBarrier.srcAccessMask = 0;
            imgBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            imgBarrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            imgBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            imgBarrier.srcQueueFamilyIndex = m_queueFamilyIndex;
            imgBarrier.dstQueueFamilyIndex = m_queueFamilyIndex;
            imgBarrier.image = swapchainData->GetTypedImage(imageIndex).image;
            imgBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdPipelineBarrier(buf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr,
                                 1, &imgBarrier);
        }

        {
            // TODO texture array with multiple slices
            assert(swapchainData->GetSlices().size() == 1);
            VkImageView targetImageView = swapchainData->GetSlices()[0].m_renderTarget[imageIndex].colorView;
            VkImageLayout targetImageLayout = VK_IMAGE_LAYOUT_GENERAL;
            uint32_t targetImageBinding = 0;
            uint32_t uboBinding = 1;

            VkDescriptorImageInfo targetImageInfo;
            targetImageInfo.imageLayout = targetImageLayout;
            targetImageInfo.imageView = targetImageView;
            targetImageInfo.sampler = defaultSampler->get();

            VkDescriptorBufferInfo uboInfo;
            uboInfo.buffer = uboBuffer.buf;
            uboInfo.offset = 0;
            uboInfo.range = VK_WHOLE_SIZE;

            VkWriteDescriptorSet writeDescriptorSets[2]{};
            writeDescriptorSets[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeDescriptorSets[0].dstSet = m_ComputeDescriptorSet;
            writeDescriptorSets[0].dstBinding = targetImageBinding;
            writeDescriptorSets[0].descriptorCount = 1;
            writeDescriptorSets[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            writeDescriptorSets[0].pImageInfo = &targetImageInfo;

            writeDescriptorSets[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeDescriptorSets[1].dstSet = m_ComputeDescriptorSet;
            writeDescriptorSets[1].dstBinding = uboBinding;
            writeDescriptorSets[1].descriptorCount = 1;
            writeDescriptorSets[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writeDescriptorSets[1].pBufferInfo = &uboInfo;

            vkUpdateDescriptorSets(m_vkDevice, (uint32_t)ArraySize(writeDescriptorSets), writeDescriptorSets, 0, NULL);
        }

        swapchainData->BindPipeline(buf, imageArrayIndex, SHADER_PROGRAM_TYPE_COMPUTE);

        vkCmdBindDescriptorSets(buf, VK_PIPELINE_BIND_POINT_COMPUTE, m_computePipelineLayout.layout, 0, 1,
                                &m_ComputeDescriptorSet, 0, NULL);

        CHECKPOINT();

        uint32_t widthDiv = (r.extent.width + 15) / 16;
        uint32_t heightDiv = (r.extent.height + 15) / 16;

        vkCmdDispatch(buf, widthDiv, heightDiv, 1);

        CHECKPOINT();

        {
            VkImageMemoryBarrier imgBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            // Switch the destination image from GENERAL -> COLOR_ATTACHMENT_OPTIMAL
            //
            // XR_KHR_vulkan_enable / XR_KHR_vulkan_enable2
            // When an application releases a swapchain image by calling xrReleaseSwapchainImage, in a
            // session created using XrGraphicsBindingVulkanKHR, the OpenXR runtime must interpret the
            // image as:
            //
            // - Having a memory layout compatible with VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL for
            //   color images, or VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL for depth images.
            // - Being owned by the VkQueue specified in XrGraphicsBindingVulkanKHR.
            imgBarrier.srcAccessMask = 0;
            imgBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            imgBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            imgBarrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            imgBarrier.srcQueueFamilyIndex = m_queueFamilyIndex;
            imgBarrier.dstQueueFamilyIndex = m_queueFamilyIndex;
            imgBarrier.image = swapchainData->GetTypedImage(imageIndex).image;
            imgBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdPipelineBarrier(buf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr,
                                 1, &imgBarrier);
        }

        CHECKPOINT();

        VkQueue queue = isProtectedMemory?m_vkQueueProtected:m_vkQueue;
        m_cmdBuffer.End(isProtectedMemory);
        m_cmdBuffer.Exec(queue, isProtectedMemory);
        // XXX Should double-buffer the command buffers, for now just flush
        m_cmdBuffer.Wait();

        uboBuffer.Reset(m_vkDevice);
    }

#if defined(USE_CHECKPOINTS)
    static void ShowCheckpoints()
    {
        VulkanGraphicsPlugin* graphics = static_cast<VulkanGraphicsPlugin*>(GetGlobalData().graphicsPlugin.get());
        if (graphics)
            graphics->ShowCheckpoints();
    }
#endif

    std::shared_ptr<IGraphicsPlugin> CreateGraphicsPlugin_Vulkan2(std::shared_ptr<IPlatformPlugin> platformPlugin)
    {
        return std::make_shared<VulkanGraphicsPlugin>(std::move(platformPlugin));
    }

    std::shared_ptr<IGraphicsPlugin> CreateGraphicsPlugin_Vulkan(std::shared_ptr<IPlatformPlugin> platformPlugin)
    {
        return std::make_shared<VulkanGraphicsPluginLegacy>(std::move(platformPlugin));
    }

}  // namespace Conformance

#endif  // XR_USE_GRAPHICS_API_VULKAN
