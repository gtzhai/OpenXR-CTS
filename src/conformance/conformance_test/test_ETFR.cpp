// Copyright (c) 2019-2024, The Khronos Group Inc.
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

#ifdef XR_USE_PLATFORM_ANDROID
#include <android/log.h>
#include <jni.h>
#define ALOGE(...) __android_log_print(ANDROID_LOG_ERROR, "ETFR", __VA_ARGS__)
#define ALOGV(...) __android_log_print(ANDROID_LOG_VERBOSE, "ETFR", __VA_ARGS__)
#endif

#include "common/xr_linear.h"
#include "composition_utils.h"
#include "conformance_framework.h"
#include "utilities/throw_helpers.h"
#include "utilities/types_and_constants.h"
#include "utilities/xrduration_literals.h"

#include <catch2/catch_test_macros.hpp>
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

#include <algorithm>
#include <array>
#include <numeric>


using namespace Conformance;

//adb shell am start-activity -S -n org.khronos.openxr.cts/android.app.NativeActivity --esa args "[FFR]" -e graphicsPlugin opengles -e xmlFilename test.xml
//adb shell am start-activity -S -n org.khronos.openxr.cts/android.app.NativeActivity --esa args "[FFR]" -e graphicsPlugin vulkan   -e xmlFilename test.xml

namespace Conformance
{
    using namespace openxr::math_operators;


    class FFRHelper{

    public:
        enum class FFRLevel
        {
	        UNDEFINED,
	        LOW,
	        MEDIUM,
	        HIGH
        };

        void setupFFRProfile(XrInstance xrInstance, XrSession xrSession, FFRLevel level, bool et_enable)
        {
            if(m_ffrLevel != level){

                if(m_ffrProfile != XR_NULL_HANDLE){
	                PFN_xrDestroyFoveationProfileFB xrDestroyFoveationProfileFB = nullptr;
	                XRC_CHECK_THROW_XRCMD(xrGetInstanceProcAddr(xrInstance, "xrDestroyFoveationProfileFB",
	                                                (PFN_xrVoidFunction *) &xrDestroyFoveationProfileFB));
	                XRC_CHECK_THROW_XRCMD(xrDestroyFoveationProfileFB(m_ffrProfile));
                    m_ffrProfile = XR_NULL_HANDLE;
                }

	            PFN_xrCreateFoveationProfileFB xrCreateFoveationProfileFB = nullptr;
	            XRC_CHECK_THROW_XRCMD(xrGetInstanceProcAddr(xrInstance, "xrCreateFoveationProfileFB",
	                                            (PFN_xrVoidFunction *) &xrCreateFoveationProfileFB));

	            XrFoveationProfileCreateInfoFB info{XR_TYPE_FOVEATION_PROFILE_CREATE_INFO_FB};
	            XrFoveationLevelProfileCreateInfoFB levelInfo{XR_TYPE_FOVEATION_LEVEL_PROFILE_CREATE_INFO_FB};
	            switch (level)
	            {
		            case FFRLevel::HIGH:
			            levelInfo.level = XR_FOVEATION_LEVEL_HIGH_FB;
			            break;
		            case FFRLevel::MEDIUM:
			            levelInfo.level = XR_FOVEATION_LEVEL_MEDIUM_FB;
			            break;
		            case FFRLevel::LOW:
			            levelInfo.level = XR_FOVEATION_LEVEL_LOW_FB;
			            break;
		            default:
			            levelInfo.level = XR_FOVEATION_LEVEL_LOW_FB;
			            break;
	            }

	            levelInfo.dynamic = XR_FOVEATION_DYNAMIC_DISABLED_FB;
	            levelInfo.verticalOffset = 0.0f;
	            info.next = &levelInfo;

                XrFoveationEyeTrackedProfileCreateInfoMETA fepci{XR_TYPE_FOVEATION_EYE_TRACKED_PROFILE_CREATE_INFO_META};
                if(et_enable){//both for gles and vulkan 
                    fepci.flags = 0;
                    levelInfo.next = &fepci; 
                }

	            XRC_CHECK_THROW_XRCMD(xrCreateFoveationProfileFB(xrSession, &info, &m_ffrProfile));
                m_ffrLevel = level;
            }
        }

        void applyFFRProfile(XrInstance xrInstance, XrSwapchain sc){
            PFN_xrUpdateSwapchainFB xrUpdateSwapchainFB = nullptr;
	        XRC_CHECK_THROW_XRCMD(xrGetInstanceProcAddr(xrInstance, "xrUpdateSwapchainFB", (PFN_xrVoidFunction *) &xrUpdateSwapchainFB));
	        XrSwapchainStateFoveationFB ffrState = {
			    .type = XR_TYPE_SWAPCHAIN_STATE_FOVEATION_FB,
			    .profile = m_ffrProfile
	        };
	        XRC_CHECK_THROW_XRCMD(xrUpdateSwapchainFB(sc, reinterpret_cast<XrSwapchainStateBaseHeaderFB *>(&ffrState)));
        }

        void getEyeTrackingCenter(XrInstance xrInstance, XrSession xrSession, XrVector2f& left, XrVector2f& right){
            PFN_xrGetFoveationEyeTrackedStateMETA xrGetFoveationEyeTrackedStateMETA = nullptr;
	        XRC_CHECK_THROW_XRCMD(xrGetInstanceProcAddr(xrInstance, "xrGetFoveationEyeTrackedStateMETA", (PFN_xrVoidFunction *) &xrGetFoveationEyeTrackedStateMETA));

            XrFoveationEyeTrackedStateMETA ets = {XR_TYPE_FOVEATION_EYE_TRACKED_STATE_META};
	        XRC_CHECK_THROW_XRCMD(xrGetFoveationEyeTrackedStateMETA(xrSession, &ets));
            if(XR_FOVEATION_EYE_TRACKED_STATE_VALID_BIT_META & ets.flags){
                left  = ets.foveationCenter[0];
                right = ets.foveationCenter[1];

                GlobalData& globalData = GetGlobalData();

                //maybe pico only
                if (globalData.IsGraphicsPluginGlES()) {
                    left.y  *= -1.0f;
                    right.y *= -1.0f;
                }
            } else {
                left.x = left.y = right.x = right.y = 0.0f;
            }
        }

        void *GetSwapchainCreateInfoNext(bool et_enable=false, bool subsample=false){
            sci_foveation.next  = nullptr;
	        sci_foveation.flags = XR_SWAPCHAIN_CREATE_FOVEATION_FRAGMENT_DENSITY_MAP_BIT_FB;

            if(et_enable){
                GlobalData& globalData = GetGlobalData();
                if (globalData.IsGraphicsPluginVulkan()) {
	                sci_vulkan.additionalUsageFlags  = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
	                sci_vulkan.additionalCreateFlags = 0x00008000;//VK_IMAGE_CREATE_FRAGMENT_DENSITY_MAP_OFFSET_BIT_QCOM;;
	                sci_foveation.next = &sci_vulkan;
                } else {//gles
                    //qcom did offset and fdm
                }
            }

	        if (subsample)
	        {
                //gles it's weired
		        sci_vulkan.additionalCreateFlags |= VK_IMAGE_CREATE_SUBSAMPLED_BIT_EXT;
	            sci_foveation.next = &sci_vulkan;
	        }

            return &sci_foveation;
        }

        static std::shared_ptr<FFRHelper> m_instance;

        static std::shared_ptr<FFRHelper> getInstance(){
            if(m_instance == nullptr){
                m_instance = std::shared_ptr<FFRHelper>(new FFRHelper());
            }
            return m_instance;
        }


    private:
	    XrSwapchainCreateInfoFoveationFB sci_foveation{XR_TYPE_SWAPCHAIN_CREATE_INFO_FOVEATION_FB};
        XrFoveationProfileFB m_ffrProfile{XR_NULL_HANDLE};
        FFRLevel  m_ffrLevel = FFRLevel::UNDEFINED;


        //vulkan 
    	XrVulkanSwapchainCreateInfoMETA sci_vulkan{XR_TYPE_VULKAN_SWAPCHAIN_CREATE_INFO_META};
    };

    std::shared_ptr<FFRHelper> FFRHelper::m_instance = nullptr;


    TEST_CASE("ProjectionVulkanFFR", "[composition][interactive][FFR]")
    {
        GlobalData& globalData = GetGlobalData();

        globalData.SetUsingFFR(true);
        globalData.SetUsingAppFDM(false);
        globalData.SetUsingFFRSoftFDMOffset(false);
#ifdef XR_USE_PLATFORM_ANDROID
        ALOGE("test ffr validation");
#endif

        if (!globalData.IsUsingGraphicsPlugin()) {
            SKIP("Test run not using graphics plugin or not vulkan");
        }

        if (!globalData.IsInstanceExtensionSupported(XR_FB_FOVEATION_EXTENSION_NAME)) {
            SKIP(XR_FB_FOVEATION_EXTENSION_NAME " not supported");
        }

        if (!globalData.IsInstanceExtensionSupported(XR_FB_FOVEATION_CONFIGURATION_EXTENSION_NAME)) {
            SKIP(XR_FB_FOVEATION_CONFIGURATION_EXTENSION_NAME " not supported");
        }

        if (!globalData.IsInstanceExtensionSupported(XR_FB_SWAPCHAIN_UPDATE_STATE_EXTENSION_NAME)) {
            SKIP(XR_FB_SWAPCHAIN_UPDATE_STATE_EXTENSION_NAME " not supported");
        }
        

        if (!globalData.IsInstanceExtensionSupported(XR_FB_FOVEATION_VULKAN_EXTENSION_NAME)) {
            SKIP(XR_FB_FOVEATION_VULKAN_EXTENSION_NAME " not supported");
        }

#ifdef XR_USE_PLATFORM_ANDROID
        ALOGE("test ffr in");
#endif

        CompositionHelper compositionHelper("Projection Vulkan FFR", {XR_FB_FOVEATION_EXTENSION_NAME, XR_FB_FOVEATION_CONFIGURATION_EXTENSION_NAME, XR_FB_SWAPCHAIN_UPDATE_STATE_EXTENSION_NAME, XR_FB_FOVEATION_VULKAN_EXTENSION_NAME});
        InteractiveLayerManager interactiveLayerManager(
            compositionHelper, "projection_array.png",
            "test swapchain created with fixed foveation, graphic using Vulkan");
        XrSession  session  = compositionHelper.GetSession();
        XrInstance instance = compositionHelper.GetInstance();
        InteractionManager& interactionManager = compositionHelper.GetInteractionManager();
        interactionManager.AttachActionSets();
        compositionHelper.BeginSession();

        const XrSpace localSpace = compositionHelper.CreateReferenceSpace(XR_REFERENCE_SPACE_TYPE_LOCAL, Pose::Identity);

        const std::vector<XrViewConfigurationView> viewProperties = compositionHelper.EnumerateConfigurationViews();

        // Because a single swapchain is being used for all views (each view is a slice of the texture array), the maximum dimensions must be used
        // since the dimensions of all slices are the same.
        const auto maxWidth = std::max_element(viewProperties.begin(), viewProperties.end(),
                                               [](const XrViewConfigurationView& l, const XrViewConfigurationView& r) {
                                                   return l.recommendedImageRectWidth < r.recommendedImageRectWidth;
                                               })
                                  ->recommendedImageRectWidth;
        const auto maxHeight = std::max_element(viewProperties.begin(), viewProperties.end(),
                                                [](const XrViewConfigurationView& l, const XrViewConfigurationView& r) {
                                                    return l.recommendedImageRectHeight < r.recommendedImageRectHeight;
                                                })
                                   ->recommendedImageRectHeight;

        auto fh = FFRHelper::getInstance();
        // Create swapchain with array type.
        auto swapchainCreateInfo = compositionHelper.DefaultColorSwapchainCreateInfo(maxWidth, maxHeight);
        swapchainCreateInfo.next      = fh->GetSwapchainCreateInfoNext(false, true);
        fh->setupFFRProfile(instance, session, FFRHelper::FFRLevel::MEDIUM, false);

        std::vector<XrSwapchain> swapchains;
        // Set up the projection layer
        XrCompositionLayerProjection* const projLayer = compositionHelper.CreateProjectionLayer(localSpace);

        for (uint32_t j = 0; j < projLayer->viewCount; j++) {
            const XrSwapchain swapchain = compositionHelper.CreateSwapchain(swapchainCreateInfo);
            const_cast<XrSwapchainSubImage&>(projLayer->views[j].subImage) = compositionHelper.MakeDefaultSubImage(swapchain, 0);
            swapchains.push_back(swapchain);
            fh->applyFFRProfile(instance, swapchain);
        }

        const std::vector<Cube> cubes = {Cube::Make({-1, 0, -2}), Cube::Make({1, 0, -2}), Cube::Make({0, -1, -2}), Cube::Make({0, 1, -2})};

        auto updateLayers = [&](const XrFrameState& frameState) {
            auto viewData = compositionHelper.LocateViews(localSpace, frameState.predictedDisplayTime);
            const auto& viewState = std::get<XrViewState>(viewData);

            std::vector<XrCompositionLayerBaseHeader*> layers;
            if (viewState.viewStateFlags & XR_VIEW_STATE_POSITION_VALID_BIT &&
                viewState.viewStateFlags & XR_VIEW_STATE_ORIENTATION_VALID_BIT) {
                const auto& views = std::get<std::vector<XrView>>(viewData);

                for (uint32_t j = 0; j < projLayer->viewCount; j++) {
                    compositionHelper.AcquireWaitReleaseImage(swapchains[j], [&](const XrSwapchainImageBaseHeader* swapchainImage) {
                        const_cast<XrFovf&>(projLayer->views[j].fov) = views[j].fov;
                        const_cast<XrPosef&>(projLayer->views[j].pose) = views[j].pose;
                        GetGlobalData().graphicsPlugin->ClearImageSlice(swapchainImage);
                        GetGlobalData().graphicsPlugin->RenderView(projLayer->views[j], swapchainImage, RenderParams().Draw(cubes));
                    });
                }

                layers.push_back(reinterpret_cast<XrCompositionLayerBaseHeader*>(projLayer));
            }

            return interactiveLayerManager.EndFrame(frameState, layers);
        };

        RenderLoop(session, updateLayers).Loop();
#ifdef XR_USE_PLATFORM_ANDROID
        ALOGE("test ffr out");
#endif
    }

    TEST_CASE("ProjectionVulkanETFR", "[composition][interactive][ETFR]")
    {
        GlobalData& globalData = GetGlobalData();
#ifdef XR_USE_PLATFORM_ANDROID
        ALOGE("test etfr validation");
#endif
        globalData.SetUsingFFR(true);
        globalData.SetUsingAppFDM(true);
        globalData.SetUsingFFRSoftFDMOffset(true);

        if (!globalData.IsUsingGraphicsPlugin()) {
            SKIP("Test run not using graphics plugin or not vulkan");
        }

        if (!globalData.IsInstanceExtensionSupported(XR_FB_FOVEATION_EXTENSION_NAME)) {
            SKIP(XR_FB_FOVEATION_EXTENSION_NAME " not supported");
        }

        if (!globalData.IsInstanceExtensionSupported(XR_FB_FOVEATION_CONFIGURATION_EXTENSION_NAME)) {
            SKIP(XR_FB_FOVEATION_CONFIGURATION_EXTENSION_NAME " not supported");
        }

        if (!globalData.IsInstanceExtensionSupported(XR_FB_SWAPCHAIN_UPDATE_STATE_EXTENSION_NAME)) {
            SKIP(XR_FB_SWAPCHAIN_UPDATE_STATE_EXTENSION_NAME " not supported");
        }
        

        if (!globalData.IsInstanceExtensionSupported(XR_FB_FOVEATION_VULKAN_EXTENSION_NAME)) {
            SKIP(XR_FB_FOVEATION_VULKAN_EXTENSION_NAME " not supported");
        }

        if (!globalData.IsInstanceExtensionSupported(XR_META_FOVEATION_EYE_TRACKED_EXTENSION_NAME)) {
            SKIP(XR_META_FOVEATION_EYE_TRACKED_EXTENSION_NAME " not supported");
        }

        if (!globalData.IsInstanceExtensionSupported(XR_META_VULKAN_SWAPCHAIN_CREATE_INFO_EXTENSION_NAME)) {
            SKIP(XR_META_VULKAN_SWAPCHAIN_CREATE_INFO_EXTENSION_NAME" not supported");
        }

#ifdef XR_USE_PLATFORM_ANDROID
        ALOGE("test etfr in");
#endif

        CompositionHelper compositionHelper("Projection Vulkan ETFR", 
        {XR_FB_FOVEATION_EXTENSION_NAME, XR_FB_FOVEATION_CONFIGURATION_EXTENSION_NAME, XR_FB_SWAPCHAIN_UPDATE_STATE_EXTENSION_NAME, XR_FB_FOVEATION_VULKAN_EXTENSION_NAME, 
        XR_META_FOVEATION_EYE_TRACKED_EXTENSION_NAME, XR_META_VULKAN_SWAPCHAIN_CREATE_INFO_EXTENSION_NAME});

        InteractiveLayerManager interactiveLayerManager(
            compositionHelper, "projection_array.png",
            "test swapchain created with foveation and eye tracked enable, graphic using Vulkan");
        XrSession  session  = compositionHelper.GetSession();
        XrInstance instance = compositionHelper.GetInstance();
        XrSystemFoveationEyeTrackedPropertiesMETA  etProperties = {XR_TYPE_SYSTEM_FOVEATION_EYE_TRACKED_PROPERTIES_META};

        XrSystemProperties systemProperties = {XR_TYPE_SYSTEM_PROPERTIES};
        systemProperties.next = &etProperties;
        REQUIRE(xrGetSystemProperties(instance, compositionHelper.GetSystemId(), &systemProperties) == XR_SUCCESS);

        if(etProperties.supportsFoveationEyeTracked == XR_FALSE){
            SKIP("eye tracking not supported");
        }

        InteractionManager& interactionManager = compositionHelper.GetInteractionManager();
        interactionManager.AttachActionSets();
        compositionHelper.BeginSession();

        const XrSpace localSpace = compositionHelper.CreateReferenceSpace(XR_REFERENCE_SPACE_TYPE_LOCAL, Pose::Identity);

        const std::vector<XrViewConfigurationView> viewProperties = compositionHelper.EnumerateConfigurationViews();

        // Because a single swapchain is being used for all views (each view is a slice of the texture array), the maximum dimensions must be used
        // since the dimensions of all slices are the same.
        const auto maxWidth = std::max_element(viewProperties.begin(), viewProperties.end(),
                                               [](const XrViewConfigurationView& l, const XrViewConfigurationView& r) {
                                                   return l.recommendedImageRectWidth < r.recommendedImageRectWidth;
                                               })
                                  ->recommendedImageRectWidth;
        const auto maxHeight = std::max_element(viewProperties.begin(), viewProperties.end(),
                                                [](const XrViewConfigurationView& l, const XrViewConfigurationView& r) {
                                                    return l.recommendedImageRectHeight < r.recommendedImageRectHeight;
                                                })
                                   ->recommendedImageRectHeight;

        auto fh = FFRHelper::getInstance();
        // Create swapchain with array type.
        auto swapchainCreateInfo = compositionHelper.DefaultColorSwapchainCreateInfo(maxWidth, maxHeight);
        swapchainCreateInfo.next      = fh->GetSwapchainCreateInfoNext(true, true);
        fh->setupFFRProfile(instance, session, FFRHelper::FFRLevel::MEDIUM, true);

        std::vector<XrSwapchain> swapchains;
        // Set up the projection layer
        XrCompositionLayerProjection* const projLayer = compositionHelper.CreateProjectionLayer(localSpace);

        for (uint32_t j = 0; j < projLayer->viewCount; j++) {
            const XrSwapchain swapchain = compositionHelper.CreateSwapchain(swapchainCreateInfo);
            const_cast<XrSwapchainSubImage&>(projLayer->views[j].subImage) = compositionHelper.MakeDefaultSubImage(swapchain, 0);
            swapchains.push_back(swapchain);
            fh->applyFFRProfile(instance, swapchain);
        }

        XrVector2f left  = {0.0f, 0.0f};
        XrVector2f right = {0.0f, 0.0f};
        fh->getEyeTrackingCenter(instance, session, left, right);
        GetGlobalData().graphicsPlugin->SetEyeTrackedCenter(left, right);

        const std::vector<Cube> cubes = {Cube::Make({-1, 0, -2}), Cube::Make({1, 0, -2}), Cube::Make({0, -1, -2}), Cube::Make({0, 1, -2})};

        auto updateLayers = [&](const XrFrameState& frameState) {
            auto viewData = compositionHelper.LocateViews(localSpace, frameState.predictedDisplayTime);
            const auto& viewState = std::get<XrViewState>(viewData);

            std::vector<XrCompositionLayerBaseHeader*> layers;
            if (viewState.viewStateFlags & XR_VIEW_STATE_POSITION_VALID_BIT &&
                viewState.viewStateFlags & XR_VIEW_STATE_ORIENTATION_VALID_BIT) {
                const auto& views = std::get<std::vector<XrView>>(viewData);

                for (uint32_t j = 0; j < projLayer->viewCount; j++) {
                    compositionHelper.AcquireWaitReleaseImage(swapchains[j], [&](const XrSwapchainImageBaseHeader* swapchainImage) {
                        const_cast<XrFovf&>(projLayer->views[j].fov) = views[j].fov;
                        const_cast<XrPosef&>(projLayer->views[j].pose) = views[j].pose;
                        GetGlobalData().graphicsPlugin->ClearImageSlice(swapchainImage);
                        GetGlobalData().graphicsPlugin->RenderView(projLayer->views[j], swapchainImage, RenderParams().Draw(cubes));
                    });
                }

                layers.push_back(reinterpret_cast<XrCompositionLayerBaseHeader*>(projLayer));
            }

            return interactiveLayerManager.EndFrame(frameState, layers);
        };

        RenderLoop(session, updateLayers).Loop();
#ifdef XR_USE_PLATFORM_ANDROID
        ALOGE("test etfr out");
#endif
    }
}  // namespace Conformance
