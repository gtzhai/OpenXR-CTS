// Copyright (c) 2017-2025 The Khronos Group Inc.
//
// SPDX-License-Identifier: Apache-2.0

#version 400
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

#pragma vertex

layout (std140, push_constant) uniform buf
{
    mat4 vp;
    mat4 prevVp;
    mat4 model;
    mat4 prevModel;
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
