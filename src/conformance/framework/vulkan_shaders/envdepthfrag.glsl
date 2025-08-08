// Copyright (c) 2017-2025 The Khronos Group Inc.
//
// SPDX-License-Identifier: Apache-2.0

#version 400
#define NUM_VIEWS 2
#define VIEW_ID gl_ViewID_OVR
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable
#extension GL_OVR_multiview : enable
#extension GL_OVR_multiview2 : enable
#extension GL_EXT_multiview : enable

layout(num_views=2) in;

#pragma fragment

layout (location = 0) in vec4 oColor;
layout (location = 1) in vec4 oCubeWorldPosition;

layout (location = 0) out vec4 FragColor;

layout (set = 0, binding = 1, std140) uniform depthbuf
{
    mat4 vp[2];
} dbuf;

layout (set = 0, binding = 0) uniform highp sampler2DArray EnvironmentDepthTexture;

void main()
{
    // Transform from world space to depth camera space using 6-DOF matrix
    highp vec4 cubeDepthCameraPosition = dbuf.vp[VIEW_ID] * oCubeWorldPosition;
  
    // 3D point --> Homogeneous Coordinates --> Normalized Coordinates in [0,1]
    highp vec2 cubeDepthCameraPositionHC = cubeDepthCameraPosition.xy / cubeDepthCameraPosition.w;
    cubeDepthCameraPositionHC = cubeDepthCameraPositionHC * 0.5f + 0.5f;
  
    // Sample from Environment Depth API texture
    highp vec3 depthViewCoord = vec3(cubeDepthCameraPositionHC, VIEW_ID);
    highp float depthViewEyeZ = texture(EnvironmentDepthTexture, depthViewCoord).r;
    //depthViewEyeZ = texture(EnvironmentDepthTexture, vec3(0.5f, 0.5f, VIEW_ID)).r;
  
    // Get virtual object depth
    highp float cubeDepth = cubeDepthCameraPosition.z / cubeDepthCameraPosition.w;
    cubeDepth = cubeDepth * 0.5f + 0.5f;
  
    // Test virtual object depth with environment depth.
    // If the virtual object is further away (occluded) output a transparent color so real scene content from PT layer is displayed.
    FragColor = oColor;
    if (cubeDepth < depthViewEyeZ) {
    FragColor.a = 1.0f; // fully opaque
    }
    else {
    FragColor = vec4(0.0f, 0.0f, 0.0f, 0.0f); // invisible
    }
    //FragColor = vec4(depthViewEyeZ, depthViewEyeZ, depthViewEyeZ, 1.0f); // invisible
    //FragColor = vec4(cubeDepthCameraPositionHC.x, cubeDepthCameraPositionHC.y, 0.0f, 1.0f); // invisible
    //mat4 tmp = dbuf.vp[VIEW_ID];
    //FragColor = vec4(tmp[0][0], tmp[0][0], VIEW_ID, 1.0f); // invisible
  
    gl_FragDepth = cubeDepth;
}
