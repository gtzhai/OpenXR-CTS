// Copyright (c) 2017-2025 The Khronos Group Inc.
//
// SPDX-License-Identifier: Apache-2.0

#version 400
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable
#extension GL_OVR_multiview : enable
#extension GL_OVR_multiview2 : enable
#extension GL_EXT_multiview : enable

layout(num_views=2) in;

#pragma vertex

layout (std140, push_constant) uniform buf
{
    mat4 vp[2];
    mat4 model;
    vec4 tintColor;
    vec4 alpha;
    vec4 p2;
    vec4 p3;
} ubuf;

layout (location = 0) in vec3 Position;
layout (location = 1) in vec3 Color;

layout (location = 0) out vec4 oColor;
layout (location = 1) out vec4 oCubeWorldPosition;

out gl_PerVertex
{
    vec4 gl_Position;
};

void main()
{
    oCubeWorldPosition = ubuf.model * vec4(Position, 1.0f);
    oColor.rgb = mix(Color.rgb, ubuf.tintColor.rgb, ubuf.tintColor.a);
    oColor.a  = ubuf.alpha.x;
    gl_Position = ubuf.vp[gl_ViewIndex] * oCubeWorldPosition;
    //gl_Position = vec4(Position, 1.0f);
}
