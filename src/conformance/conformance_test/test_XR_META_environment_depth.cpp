// Copyright (c) 2019-2025 The Khronos Group Inc.
// Copyright (c) Meta Platforms, LLC and its affiliates. All rights reserved.
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

#include "common/xr_linear.h"
#include "composition_utils.h"
#include "conformance_framework.h"
#include "conformance_options.h"
#include "utilities/colors.h"
#include "utilities/throw_helpers.h"
#include "utilities/types_and_constants.h"
#include "utilities/xr_math_operators.h"
#include "utilities/xrduration_literals.h"

#include <catch2/catch_test_macros.hpp>
#include <openxr/openxr.h>
#include <jni.h>
#include <openxr/openxr_platform.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <sstream>
#include "common/android_logging.h"

namespace Conformance
{

    class EnvironmentDepthMeta {
    public:
        EnvironmentDepthMeta(CompositionHelper& compositionHelper): compositionHelper_(compositionHelper){
            XrInstance instance = compositionHelper.GetInstance();

            xrCreateEnvironmentDepthProviderMETA = GetInstanceExtensionFunction<PFN_xrCreateEnvironmentDepthProviderMETA>(instance, "xrCreateEnvironmentDepthProviderMETA");
            xrDestroyEnvironmentDepthProviderMETA = GetInstanceExtensionFunction<PFN_xrDestroyEnvironmentDepthProviderMETA>(instance, "xrDestroyEnvironmentDepthProviderMETA");
            xrStartEnvironmentDepthProviderMETA = GetInstanceExtensionFunction<PFN_xrStartEnvironmentDepthProviderMETA>(instance, "xrStartEnvironmentDepthProviderMETA");
            xrStopEnvironmentDepthProviderMETA = GetInstanceExtensionFunction<PFN_xrStopEnvironmentDepthProviderMETA>(instance, "xrStopEnvironmentDepthProviderMETA");    
            xrCreateEnvironmentDepthSwapchainMETA = GetInstanceExtensionFunction<PFN_xrCreateEnvironmentDepthSwapchainMETA>(instance, "xrCreateEnvironmentDepthSwapchainMETA");
            xrDestroyEnvironmentDepthSwapchainMETA = GetInstanceExtensionFunction<PFN_xrDestroyEnvironmentDepthSwapchainMETA>(instance, "xrDestroyEnvironmentDepthSwapchainMETA");
            xrGetEnvironmentDepthSwapchainStateMETA = GetInstanceExtensionFunction<PFN_xrGetEnvironmentDepthSwapchainStateMETA>(instance, "xrGetEnvironmentDepthSwapchainStateMETA");
            xrAcquireEnvironmentDepthImageMETA = GetInstanceExtensionFunction<PFN_xrAcquireEnvironmentDepthImageMETA>(instance, "xrAcquireEnvironmentDepthImageMETA");
            xrSetEnvironmentDepthHandRemovalMETA = GetInstanceExtensionFunction<PFN_xrSetEnvironmentDepthHandRemovalMETA>(instance, "xrSetEnvironmentDepthHandRemovalMETA");
            xrEnumerateEnvironmentDepthSwapchainImagesMETA = GetInstanceExtensionFunction<PFN_xrEnumerateEnvironmentDepthSwapchainImagesMETA>(instance, "xrEnumerateEnvironmentDepthSwapchainImagesMETA");
        }

        virtual ~EnvironmentDepthMeta() = default;

        int CreateEnvironmentDepthProvider(const XrEnvironmentDepthProviderCreateInfoMETA* createInfo = nullptr);
        int DestroyEnvironmentDepthProvider();
        int StartEnvironmentDepthProvider();
        int StopEnvironmentDepthProvider();
        int CreateEnvironmentDepthSwapchain(const XrEnvironmentDepthSwapchainCreateInfoMETA* createInfo = nullptr);
        int DestroyEnvironmentDepthSwapchain();
        int EnumerateEnvironmentDepthSwapchainImages(uint32_t imageCapacityInput, uint32_t* imageCountOutput, XrSwapchainImageBaseHeader* images);
        int GetEnvironmentDepthSwapchainStateMETA(XrEnvironmentDepthSwapchainStateMETA* state);
        int AcquireEnvironmentDepthImageMETA(const XrEnvironmentDepthImageAcquireInfoMETA* acquireInfo, XrEnvironmentDepthImageMETA* environmentDepthImage);
        int SetEnvironmentDepthHandRemovalMETA(const XrEnvironmentDepthHandRemovalSetInfoMETA* setInfo);

    private:
        PFN_xrCreateEnvironmentDepthProviderMETA xrCreateEnvironmentDepthProviderMETA = nullptr;
        PFN_xrDestroyEnvironmentDepthProviderMETA xrDestroyEnvironmentDepthProviderMETA = nullptr;
        PFN_xrStartEnvironmentDepthProviderMETA xrStartEnvironmentDepthProviderMETA = nullptr;
        PFN_xrStopEnvironmentDepthProviderMETA xrStopEnvironmentDepthProviderMETA = nullptr;    
        PFN_xrCreateEnvironmentDepthSwapchainMETA xrCreateEnvironmentDepthSwapchainMETA = nullptr;
        PFN_xrDestroyEnvironmentDepthSwapchainMETA xrDestroyEnvironmentDepthSwapchainMETA = nullptr;
        PFN_xrGetEnvironmentDepthSwapchainStateMETA xrGetEnvironmentDepthSwapchainStateMETA = nullptr;
        PFN_xrAcquireEnvironmentDepthImageMETA xrAcquireEnvironmentDepthImageMETA = nullptr;
        PFN_xrSetEnvironmentDepthHandRemovalMETA xrSetEnvironmentDepthHandRemovalMETA = nullptr;
        PFN_xrEnumerateEnvironmentDepthSwapchainImagesMETA xrEnumerateEnvironmentDepthSwapchainImagesMETA = nullptr;

        XrEnvironmentDepthProviderMETA environment_depth_provider_ = XR_NULL_HANDLE;
        XrEnvironmentDepthSwapchainMETA swapchain_ = XR_NULL_HANDLE;

        CompositionHelper& compositionHelper_;
        const char *module = "EnvironmentDepthMeta";
    };


    int EnvironmentDepthMeta::CreateEnvironmentDepthProvider(const XrEnvironmentDepthProviderCreateInfoMETA* createInfo) {
        auto xrSession = compositionHelper_.GetSession();
        if (xrSession == XR_NULL_HANDLE) {
            ALOGE("%s::%s failed for xrSession is null", module, __FUNCTION__);
            return -1;
        }

        if (nullptr != createInfo) {
            XRC_CHECK_THROW_XRCMD(xrCreateEnvironmentDepthProviderMETA(xrSession, createInfo, &environment_depth_provider_));
        } else {
            XrEnvironmentDepthProviderCreateInfoMETA createInfoMETA = {XR_TYPE_ENVIRONMENT_DEPTH_PROVIDER_CREATE_INFO_META};
            createInfoMETA.next = nullptr;
            createInfoMETA.createFlags = 0;
            XRC_CHECK_THROW_XRCMD(xrCreateEnvironmentDepthProviderMETA(xrSession, &createInfoMETA, &environment_depth_provider_)); 
        }

        return 0; 
    }

    int EnvironmentDepthMeta::DestroyEnvironmentDepthProvider(){
        if (environment_depth_provider_ == XR_NULL_HANDLE) {
            ALOGE("%s::%s failed for environment_depth_provider_ is null", module, __FUNCTION__);
            return -1;
        }
        XRC_CHECK_THROW_XRCMD(xrDestroyEnvironmentDepthProviderMETA(environment_depth_provider_));
        return 0;
    }

    int EnvironmentDepthMeta::StartEnvironmentDepthProvider(){
        if (environment_depth_provider_ == XR_NULL_HANDLE) {
            ALOGE("%s::%s failed for environment_depth_provider_ is null", module, __FUNCTION__);
            return -1;
        }

        XRC_CHECK_THROW_XRCMD(xrStartEnvironmentDepthProviderMETA(environment_depth_provider_));
        return 0;
    }

    int EnvironmentDepthMeta::StopEnvironmentDepthProvider(){
        if (environment_depth_provider_ == XR_NULL_HANDLE) {
            ALOGE("%s::%s failed for environment_depth_provider_ is null", module, __FUNCTION__);
            return -1;
        }
        XRC_CHECK_THROW_XRCMD(xrStopEnvironmentDepthProviderMETA(environment_depth_provider_));
        return 0;
    }

    int EnvironmentDepthMeta::CreateEnvironmentDepthSwapchain(const XrEnvironmentDepthSwapchainCreateInfoMETA* createInfo){
        if (environment_depth_provider_ == XR_NULL_HANDLE) {
            ALOGE("%s::%s failed for environment_depth_provider_ is null", module, __FUNCTION__);
            return -1;
        }
        
        if (nullptr != createInfo) {
            XRC_CHECK_THROW_XRCMD(xrCreateEnvironmentDepthSwapchainMETA(environment_depth_provider_, createInfo, &swapchain_));
        } else {
            XrEnvironmentDepthSwapchainCreateInfoMETA createInfoMETA = {XR_TYPE_ENVIRONMENT_DEPTH_SWAPCHAIN_CREATE_INFO_META};
            createInfoMETA.next = nullptr;
            createInfoMETA.createFlags = 0;
            XRC_CHECK_THROW_XRCMD(xrCreateEnvironmentDepthSwapchainMETA(environment_depth_provider_, &createInfoMETA, &swapchain_)); 
        }

        return 0; 
    }

    int EnvironmentDepthMeta::DestroyEnvironmentDepthSwapchain(){
        if (environment_depth_provider_ == XR_NULL_HANDLE) {
            ALOGE("%s::%s failed for environment_depth_provider_ is null", module, __FUNCTION__);
            return -1;
        }

        if (swapchain_ == XR_NULL_HANDLE) {
            ALOGE("%s::%s failed for swapchain_ is null", module, __FUNCTION__);
            return -1;
        }
        return 0;
    }

    int EnvironmentDepthMeta::EnumerateEnvironmentDepthSwapchainImages(uint32_t imageCapacityInput, uint32_t* imageCountOutput, XrSwapchainImageBaseHeader* images){
        if (environment_depth_provider_ == XR_NULL_HANDLE) {
            ALOGE("%s::%s failed for environment_depth_provider_ is null", module, __FUNCTION__);
            return -1;
        }

        if (swapchain_ == XR_NULL_HANDLE) {
            ALOGE("%s::%s failed for swapchain_ is null", module, __FUNCTION__);
            return -1;
        }

        XRC_CHECK_THROW_XRCMD(xrEnumerateEnvironmentDepthSwapchainImagesMETA(swapchain_, imageCapacityInput, imageCountOutput, images));
        
        return 0; 
    }

    int EnvironmentDepthMeta::GetEnvironmentDepthSwapchainStateMETA(XrEnvironmentDepthSwapchainStateMETA* state){

        if (environment_depth_provider_ == XR_NULL_HANDLE) {
            ALOGE("%s::%s failed for environment_depth_provider_ is null", module, __FUNCTION__);
            return -1;
        }

        if (swapchain_ == XR_NULL_HANDLE) {
            ALOGE("%s::%s failed for swapchain_ is null", module, __FUNCTION__);
            return -1;
        }

        if (state == XR_NULL_HANDLE) {
            ALOGE("%s::%s failed for state is null", module, __FUNCTION__);
            return -1;
        }

        state->type = XR_TYPE_ENVIRONMENT_DEPTH_SWAPCHAIN_STATE_META;
        XRC_CHECK_THROW_XRCMD(xrGetEnvironmentDepthSwapchainStateMETA(swapchain_, state));
        return 0;
    }

    int EnvironmentDepthMeta::AcquireEnvironmentDepthImageMETA(const XrEnvironmentDepthImageAcquireInfoMETA* acquireInfo, XrEnvironmentDepthImageMETA* environmentDepthImage){

        if (environment_depth_provider_ == XR_NULL_HANDLE) {
            ALOGE("%s::%s failed for environment_depth_provider_ is null", module, __FUNCTION__);
            return -1;
        }

        if (acquireInfo == nullptr) {
            ALOGE("%s::%s param in acquireInfo is null", module, __FUNCTION__);
            return -1;
        }

        if (environmentDepthImage == nullptr) {
            ALOGE("%s::%s param in environmentDepthImage is null", module, __FUNCTION__);
            return -1;
        }


        XRC_CHECK_THROW_XRCMD(xrAcquireEnvironmentDepthImageMETA(environment_depth_provider_, acquireInfo, environmentDepthImage));
        return 0;
    }

    int EnvironmentDepthMeta::SetEnvironmentDepthHandRemovalMETA(const XrEnvironmentDepthHandRemovalSetInfoMETA* setInfo){

        if (environment_depth_provider_ == XR_NULL_HANDLE) {
            ALOGE("%s::%s failed for environment_depth_provider_ is null", module, __FUNCTION__);
            return -1;
        }

        if (setInfo == nullptr) {
            ALOGE("%s::%s param in setInfo is null", module, __FUNCTION__);
            return -1;
        }

        XRC_CHECK_THROW_XRCMD(xrSetEnvironmentDepthHandRemovalMETA(environment_depth_provider_, setInfo));

        return 0;
    }

    TEST_CASE("EnvironmentDepth", "[xxxxxx][envdepth]")
    {
        ALOGE("Test Case:EnvironmentDepth");
        GlobalData& globalData = GetGlobalData();
        if (!globalData.IsUsingGraphicsPlugin()) {
            SKIP("Cannot test without a graphics plugin");
        }

        if (!globalData.IsInstanceExtensionSupported(XR_META_ENVIRONMENT_DEPTH_EXTENSION_NAME)) {
            SKIP(XR_META_ENVIRONMENT_DEPTH_EXTENSION_NAME " not supported");
        }

        if (!globalData.IsInstanceExtensionSupported(XR_FB_PASSTHROUGH_EXTENSION_NAME)) {
            SKIP(XR_FB_PASSTHROUGH_EXTENSION_NAME " not supported");
        }

        bool isVulkan = globalData.IsGraphicsPluginVulkan();

        CompositionHelper compositionHelper(
            "Environment Depth", {XR_META_ENVIRONMENT_DEPTH_EXTENSION_NAME, XR_FB_PASSTHROUGH_EXTENSION_NAME});

        InteractiveLayerManager interactiveLayerManager(compositionHelper, "projection_depth.png",
                                                        "Four cubes each are drawn on two different layers, with the front face"
                                                        " appearing darker on the second layer. All eight cubes should be visible,"
                                                        " with the darker blue front face appearing closer on the left and bottom,"
                                                        " and further away on the right and top.");
        XrSession session = compositionHelper.GetSession();
        InteractionManager& interactionManager = compositionHelper.GetInteractionManager();
        interactionManager.AttachActionSets();
        compositionHelper.BeginSession();

        const XrSpace localSpace = compositionHelper.CreateReferenceSpace(XR_REFERENCE_SPACE_TYPE_LOCAL);

        const std::vector<XrViewConfigurationView> viewProperties = compositionHelper.EnumerateConfigurationViews();

        std::vector<XrSwapchainCreateInfo> colorSwapchainCreateInfo;
        std::vector<XrSwapchainCreateInfo> depthSwapchainCreateInfo;
        bool isUsingMultiView = globalData.IsUsingMultiview();

        if(!isUsingMultiView){
            for (auto& view : viewProperties) {
                colorSwapchainCreateInfo.push_back(
                    compositionHelper.DefaultColorSwapchainCreateInfo(view.recommendedImageRectWidth, view.recommendedImageRectHeight));
                depthSwapchainCreateInfo.push_back(
                    compositionHelper.DefaultDepthSwapchainCreateInfo(view.recommendedImageRectWidth, view.recommendedImageRectHeight));
            }
        } else {
            colorSwapchainCreateInfo.push_back(
                    compositionHelper.DefaultColorSwapchainCreateInfo(viewProperties[0].recommendedImageRectWidth, viewProperties[0].recommendedImageRectHeight, 0, -1, 2));
            depthSwapchainCreateInfo.push_back(
                    compositionHelper.DefaultDepthSwapchainCreateInfo(viewProperties[0].recommendedImageRectWidth, viewProperties[0].recommendedImageRectHeight, 0, -1, 2));
        }

        XrInstance instance = compositionHelper.GetInstance();

        auto xrCreatePassthroughFB = GetInstanceExtensionFunction<PFN_xrCreatePassthroughFB>(instance, "xrCreatePassthroughFB");
        auto xrDestroyPassthroughFB = GetInstanceExtensionFunction<PFN_xrDestroyPassthroughFB>(instance, "xrDestroyPassthroughFB");
        auto xrPassthroughStartFB = GetInstanceExtensionFunction<PFN_xrPassthroughStartFB>(instance, "xrPassthroughStartFB");
        auto xrPassthroughPauseFB = GetInstanceExtensionFunction<PFN_xrPassthroughPauseFB>(instance, "xrPassthroughPauseFB");
        auto xrCreatePassthroughLayerFB = GetInstanceExtensionFunction<PFN_xrCreatePassthroughLayerFB>(instance, "xrCreatePassthroughLayerFB");
        auto xrDestroyPassthroughLayerFB = GetInstanceExtensionFunction<PFN_xrDestroyPassthroughLayerFB>(instance, "xrDestroyPassthroughLayerFB");
        auto xrPassthroughLayerSetStyleFB = GetInstanceExtensionFunction<PFN_xrPassthroughLayerSetStyleFB>(instance, "xrPassthroughLayerSetStyleFB");
        auto xrPassthroughLayerPauseFB = GetInstanceExtensionFunction<PFN_xrPassthroughLayerPauseFB>(instance, "xrPassthroughLayerPauseFB");
        auto xrPassthroughLayerResumeFB = GetInstanceExtensionFunction<PFN_xrPassthroughLayerResumeFB>(instance, "xrPassthroughLayerResumeFB");

        XrPassthroughFB passthrough = XR_NULL_HANDLE;
        XrPassthroughLayerFB passthroughLayer = XR_NULL_HANDLE;
        {
            XrPassthroughCreateInfoFB ptci = {XR_TYPE_PASSTHROUGH_CREATE_INFO_FB};
            XrResult result;
            XRC_CHECK_THROW_XRCMD(result = xrCreatePassthroughFB(session, &ptci, &passthrough));

            if (XR_SUCCEEDED(result)) {
                XrPassthroughLayerCreateInfoFB plci = {XR_TYPE_PASSTHROUGH_LAYER_CREATE_INFO_FB};
                plci.passthrough = passthrough;
                plci.purpose = XR_PASSTHROUGH_LAYER_PURPOSE_RECONSTRUCTION_FB;
                XRC_CHECK_THROW_XRCMD(xrCreatePassthroughLayerFB(session, &plci, &passthroughLayer));
            }
        }

        XRC_CHECK_THROW_XRCMD(xrPassthroughStartFB(passthrough));
        XRC_CHECK_THROW_XRCMD(xrPassthroughLayerResumeFB(passthroughLayer));

        struct EnvDepthAcqResult {
            bool valid = false;

            uint64_t texture = 0;
            float nearZ = 0.0f;
            float farZ = 0.0f;

            XrFovf fov[2]{};
            XrPosef pose[2]{};

            XrTime predictedDisplayTime;
        };
        struct EnvDepthAcqResult env_depth_acq_result_{};

        XrSystemEnvironmentDepthPropertiesMETA env_depth_properties_{};

        std::shared_ptr<EnvironmentDepthMeta> env_depth_context_ = std::make_shared<EnvironmentDepthMeta>(compositionHelper);
        std::vector<uint64_t> env_depth_gl_textures_;
        std::vector<uint64_t> env_depth_vulkan_textures_;

        env_depth_context_->CreateEnvironmentDepthProvider();
        env_depth_context_->CreateEnvironmentDepthSwapchain();

        //enum images
        uint32_t imageCapacity = 0;
        env_depth_context_->EnumerateEnvironmentDepthSwapchainImages(imageCapacity, &imageCapacity, nullptr);

        ALOGE("%s:xxxxxx:imageCapacity :%d", __func__, imageCapacity );

        if(!isVulkan){
            std::vector<XrSwapchainImageOpenGLESKHR> images_gles(imageCapacity);
            for (uint32_t i = 0; i < imageCapacity; ++i) {
                images_gles[i] = {XR_TYPE_SWAPCHAIN_IMAGE_OPENGL_ES_KHR};
            }
            env_depth_context_->EnumerateEnvironmentDepthSwapchainImages(imageCapacity, &imageCapacity,
                                                                            (XrSwapchainImageBaseHeader *)images_gles.data());

            env_depth_gl_textures_.resize(imageCapacity);
            for (uint32_t i = 0; i < imageCapacity; ++i) {
                env_depth_gl_textures_[i] = uint64_t(images_gles[i].image);
            }
        } else {
            std::vector<XrSwapchainImageVulkanKHR> images_gles(imageCapacity);
            for (uint32_t i = 0; i < imageCapacity; ++i) {
                images_gles[i] = {XR_TYPE_SWAPCHAIN_IMAGE_VULKAN_KHR};
            }
            env_depth_context_->EnumerateEnvironmentDepthSwapchainImages(imageCapacity, &imageCapacity,
                                                                            (XrSwapchainImageBaseHeader *)images_gles.data());

            env_depth_vulkan_textures_.resize(imageCapacity);
            for (uint32_t i = 0; i < imageCapacity; ++i) {
                env_depth_vulkan_textures_[i] = uint64_t(images_gles[i].image);
                ALOGE("%s:xxxxxx:vulkan_textures:%p", __func__, env_depth_vulkan_textures_[i] );
            }
        }

        //start provider
        env_depth_context_->StartEnvironmentDepthProvider();

        const int LayerCount = 1;
        XrCompositionLayerProjection* projLayers[LayerCount];
        std::vector<std::pair<XrSwapchain, XrSwapchain>> swapchain[LayerCount];

        // Set up the projection layers
        for (int layer = 0; layer < LayerCount; layer++) {
            projLayers[layer] = compositionHelper.CreateProjectionLayer(localSpace);
            projLayers[layer]->layerFlags = XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT | XR_COMPOSITION_LAYER_UNPREMULTIPLIED_ALPHA_BIT;

            if(!isUsingMultiView){
                for (uint32_t j = 0; j < projLayers[layer]->viewCount; j++) {
                    // create color and depth swapchains
                    swapchain[layer].push_back(
                        compositionHelper.CreateSwapchainWithDepth(colorSwapchainCreateInfo[j], depthSwapchainCreateInfo[j]));
                    const_cast<XrSwapchainSubImage&>(projLayers[layer]->views[j].subImage) =
                        compositionHelper.MakeDefaultSubImage(swapchain[layer][j].first);
                }
            } else {
                swapchain[layer].push_back(
                        compositionHelper.CreateSwapchainWithDepth(colorSwapchainCreateInfo[0], depthSwapchainCreateInfo[0]));
                const_cast<XrSwapchainSubImage&>(projLayers[layer]->views[0].subImage) =
                        compositionHelper.MakeDefaultSubImage(swapchain[layer][0].first);
                const_cast<XrSwapchainSubImage&>(projLayers[layer]->views[1].subImage) =
                        compositionHelper.MakeDefaultSubImage(swapchain[layer][0].first);
            }
        }

        ALOGE("xxxxxx:Env Depth Cases:");

        XrCompositionLayerPassthroughFB passthrough_layer = {
                XR_TYPE_COMPOSITION_LAYER_PASSTHROUGH_FB};
        if (passthroughLayer != XR_NULL_HANDLE) {
            passthrough_layer.layerHandle = passthroughLayer;
            passthrough_layer.flags = XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT;
            passthrough_layer.space = XR_NULL_HANDLE;
        }

        // Alternate which cube should be in front. Rotate every cube in the second layer to tell them apart
        const std::vector<Cube> cubes[LayerCount] = {
            {Cube::Make({-1, 0, -2.5}, 2.0f), Cube::Make({1, 0, -2}, 2.0f), Cube::Make({0, -1, -2.5}, 2.0f), Cube::Make({0, 1, -2}, 2.0f)}};

        auto updateLayers = [&](const XrFrameState& frameState) {
            auto viewData = compositionHelper.LocateViews(localSpace, frameState.predictedDisplayTime);
            const auto& viewState = std::get<XrViewState>(viewData);


            XrEnvironmentDepthImageAcquireInfoMETA envDepthAcquireInfo{
                    XR_TYPE_ENVIRONMENT_DEPTH_IMAGE_ACQUIRE_INFO_META};

            env_depth_acq_result_.predictedDisplayTime = frameState.predictedDisplayTime;
            envDepthAcquireInfo.space = localSpace;
            envDepthAcquireInfo.displayTime = env_depth_acq_result_.predictedDisplayTime;
            XrEnvironmentDepthImageMETA envDepthImage{XR_TYPE_ENVIRONMENT_DEPTH_IMAGE_META};
            envDepthImage.views[0].type = XR_TYPE_ENVIRONMENT_DEPTH_IMAGE_VIEW_META;
            envDepthImage.views[1].type = XR_TYPE_ENVIRONMENT_DEPTH_IMAGE_VIEW_META;

            int ret = env_depth_context_->AcquireEnvironmentDepthImageMETA(&envDepthAcquireInfo, &envDepthImage);

            env_depth_acq_result_.valid = false;
            if (ret == 0) {
                env_depth_acq_result_.valid = true;
                env_depth_acq_result_.nearZ = envDepthImage.nearZ;
                env_depth_acq_result_.farZ = envDepthImage.farZ;
                env_depth_acq_result_.texture = isVulkan?env_depth_vulkan_textures_[envDepthImage.swapchainIndex]:env_depth_gl_textures_[envDepthImage.swapchainIndex];
                env_depth_acq_result_.pose[0] = envDepthImage.views[0].pose;
                env_depth_acq_result_.pose[1] = envDepthImage.views[1].pose;
                env_depth_acq_result_.fov[0] = envDepthImage.views[0].fov;
                env_depth_acq_result_.fov[1] = envDepthImage.views[1].fov;
                ALOGE("%s:index:%d, texture:%p, isVulkan:%d", __FUNCTION__, envDepthImage.swapchainIndex, env_depth_acq_result_.texture, isVulkan);
            }

            const int size = 2;
            std::vector<XrMatrix4x4f> depthProj(size);
            std::vector<XrMatrix4x4f> depthView(size);
            XrMatrix4x4f depthToView[size];
            XrMatrix4x4f depthViewProj[size];
            XrMatrix4x4f depthViewProjInv[size];
            uint64_t depthTex = 0;
            XrVector3f scale{1.f, 1.f, 1.f};
            std::vector<XrMatrix4x4f> depthUV2World(size);

            const XrMatrix4x4f UVD2NDCMAT = {{2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, -1, -1, -1, 1}};
            const XrMatrix4x4f NDC2UVDMAT = {{0.5, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0.5, 0, 0.5, 0.5, 0.5, 1}};

            if (env_depth_acq_result_.valid) {
                for (int i = 0; i < size; i++) {
                    XrMatrix4x4f_CreateProjectionFov(&depthProj[i], GRAPHICS_OPENGL_ES, env_depth_acq_result_.fov[i],
                                                     env_depth_acq_result_.nearZ, env_depth_acq_result_.farZ);
                    XrMatrix4x4f_CreateTranslationRotationScale(&depthToView[i],
                                                                &env_depth_acq_result_.pose[i].position,
                                                                &env_depth_acq_result_.pose[i].orientation, &scale);
                    XrMatrix4x4f_InvertRigidBody(&depthView[i], &depthToView[i]);
                    XrMatrix4x4f_Multiply(&depthViewProj[i], &depthProj[i], &depthView[i]);
                    XrMatrix4x4f_InvertRigidBody(&depthViewProjInv[i], &depthViewProj[i]);

                    XrMatrix4x4f_Multiply(&depthUV2World[i], &depthViewProjInv[i], &UVD2NDCMAT);
                }
                depthTex = env_depth_acq_result_.texture;
            } else {
                ALOGE("depth buffer not valid");
            }

            std::vector<XrCompositionLayerBaseHeader*> layers;
            layers.push_back(reinterpret_cast<XrCompositionLayerBaseHeader*>(&passthrough_layer));
            if (viewState.viewStateFlags & XR_VIEW_STATE_POSITION_VALID_BIT &&
                viewState.viewStateFlags & XR_VIEW_STATE_ORIENTATION_VALID_BIT) {
                const auto& views = std::get<std::vector<XrView>>(viewData);

                for (int layer = 0; layer < LayerCount; layer++) {
                    if(!isUsingMultiView){
                        for (size_t j = 0; j < views.size(); j++) {
                            // Render into each view's swapchain using the projection layer view fov and pose.
                            compositionHelper.AcquireWaitReleaseImage(
                                swapchain[layer][j].first, [&](const XrSwapchainImageBaseHeader* swapchainImage) {
                                    GetGlobalData().graphicsPlugin->ClearImageSlice(swapchainImage, 0, {0.0f, 0.0f, 0.0f, 0.0f});

                                    const_cast<XrFovf&>(projLayers[layer]->views[j].fov) = views[j].fov;
                                    const_cast<XrPosef&>(projLayers[layer]->views[j].pose) = views[j].pose;
                                    GetGlobalData().graphicsPlugin->RenderView(projLayers[layer]->views[j], swapchainImage,
                                                                            RenderParams().Draw(cubes[layer]));
                                });
                        }
                    } else {
                        EnvDepthOcclusionParams edoParams(depthViewProj[0], depthViewProj[1], depthTex);
                        compositionHelper.AcquireWaitReleaseImage(
                            swapchain[layer][0].first, [&](const XrSwapchainImageBaseHeader* swapchainImage) {
                                GetGlobalData().graphicsPlugin->ClearImageSlice(swapchainImage, 0, {0.0f, 0.0f, 0.0f, 0.0f});

                                for (size_t j = 0; j < views.size(); j++) {
                                    const_cast<XrFovf&>(projLayers[layer]->views[j].fov) = views[j].fov;
                                    const_cast<XrPosef&>(projLayers[layer]->views[j].pose) = views[j].pose;
                                    const_cast<uint32_t&>(projLayers[layer]->views[j].subImage.imageArrayIndex) = j;
                                }
                                GetGlobalData().graphicsPlugin->RenderView(projLayers[layer]->views[0], swapchainImage,
                                                                        RenderParams().Draw(cubes[layer]), false, nullptr, 
                                                                        &projLayers[layer]->views[1], nullptr,
                                                                        &edoParams);
                            });
                    }
                    layers.push_back(reinterpret_cast<XrCompositionLayerBaseHeader*>(projLayers[layer]));
                }
            }
            return interactiveLayerManager.EndFrame(frameState, layers);
        };

        RenderLoop(session, updateLayers).Loop();

        env_depth_context_->StopEnvironmentDepthProvider();
        env_depth_context_->DestroyEnvironmentDepthProvider();

        XRC_CHECK_THROW_XRCMD(xrPassthroughPauseFB(passthrough));
        XRC_CHECK_THROW_XRCMD(xrDestroyPassthroughFB(passthrough));
    }

}  // namespace Conformance
