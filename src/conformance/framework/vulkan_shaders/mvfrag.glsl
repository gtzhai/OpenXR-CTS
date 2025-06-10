// Copyright (c) 2017-2025 The Khronos Group Inc.
//
// SPDX-License-Identifier: Apache-2.0

#version 400
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

#pragma fragment

layout (location = 0) in vec4 clipPos;
layout (location = 1) in vec4 prevClipPos;

layout (location = 0) out vec4 FragColor;

void main()
{
    vec4 motionVector = ( clipPos / clipPos.w - prevClipPos / prevClipPos.w );
    FragColor = motionVector;
}
