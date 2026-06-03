/**
  ******************************************************************************
  * @file    network.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-06-01T19:12:36+0200
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */

#include "ai_lite_inspect.h"
#include "ai_platform_interface.h"
#include "layers.h"
#include "core_convert.h"
#include "network.h"
#include "network_details.h"
#include "network_data.h"
#include "stai_events.h"

#include "lite_operators.h"

#include "ai_lite_inspect.h"
/*****************************************************************************/
#define STAI_INTERNAL_API_MAJOR               (1)
#define STAI_INTERNAL_API_MINOR               (0)
#define STAI_INTERNAL_API_MICRO               (0)

#define STAI_MAGIC                            (0xB1C00100)

/*****************************************************************************/
#define _STAI_CONCAT_ARG(a, b)     a ## b
#define STAI_CONCAT(a, b)         _STAI_CONCAT_ARG(a, b)

/*!  STAI_CAST SECTION                       *********************************/
#define STAI_CAST(type, expr) \
  ((type)(expr))


/*****************************************************************************/
#define STAI_SIZE(_size) \
  ((stai_size)(_size))

/*****************************************************************************/
#define STAI_INIT_BUFFER(_flags, _size, _address) \
  { \
    .size = (_size), \
    .address = (uintptr_t)(_address), \
    .flags = (_flags), \
  }

#define STAI_INIT_TENSOR(_name, _flags, _fmt, _size_bytes, _shape, _scale, _zeropoint) \
  { \
    .size_bytes = (_size_bytes), \
    .flags = (_flags), \
    .format = (stai_format)(_fmt), \
    .shape = STAI_PACK(_shape), \
    .scale = STAI_PACK(_scale), \
    .zeropoint = STAI_PACK(_zeropoint), \
    .name = (_name) \
  }

#define STAI_INIT_ARRAY(_size, _ptr) \
  { .size = STAI_SIZE(_size), .data = STAI_PACK(_ptr) }


#define STAI_CAST_ARRAY(_type, _size, _ptr) \
  { .size = STAI_SIZE(_size), .data = (_type)STAI_PACK(_ptr) }


#define STAI_DECLARE_ARRAY(_type, _size, ...) \
  { .size = STAI_SIZE(_size), .data = (_type[_size]) { STAI_PACK(__VA_ARGS__) } }


#define STAI_EMPTY_ARRAY() \
  { .size = 0, .data = NULL }


#define STAI_INIT_VERSION(_major, _minor, _micro) \
  { .major = (_major), .minor = (_minor), .micro = (_micro), .reserved = 0x0 }

/*****************************************************************************/
/**  Getters and setters  **/

#define STAI_GET_ARRAY_SIZE(nd_array) \
  (nd_array.size)


#define STAI_GET_ARRAY_ELEM(nd_array, pos) \
  (nd_array.data[(pos)])

#define _STAI_SET_ERROR(net_ctx, cond, value, exit) { \
  if (!(net_ctx)) { return STAI_ERROR_NETWORK_INVALID_CONTEXT_HANDLE; } \
  if (((uintptr_t)net_ctx) & (_STAI_CONTEXT_ALIGNMENT-1)) { return STAI_ERROR_NETWORK_INVALID_CONTEXT_ALIGNMENT; } \
  if (((value) >= STAI_ERROR_GENERIC) && (cond)) { \
    if ((net_ctx)->_return_code == STAI_SUCCESS) { \
      (net_ctx)->_return_code = (value); \
    } \
    return (exit); \
  } \
}

/*****************************************************************************/
/* TODO REMOVE THESE TWO MACROS */
#define STAI_EVENT_NODE_START_CB
#define STAI_EVENT_NODE_STOP_CB

#ifdef STAI_EVENT_NODE_START_CB
#ifndef _STAI_NETWORK_EVENT_NODE_START_CB
  #define _STAI_NETWORK_EVENT_NODE_START_CB(_node_id, _buffers_size, ...) \
  if (net_ctx->_callback) { \
    const stai_event_node_start_stop _start_event = { \
      .node_id=(_node_id), \
      .buffers={ \
        .size=(_buffers_size), \
        .data=(stai_ptr const*)(const stai_ptr[_buffers_size])STAI_PACK(__VA_ARGS__) \
      } \
    }; \
    net_ctx->_callback(net_ctx->_callback_cookie, STAI_EVENT_NODE_START, (const void*)&_start_event); \
  }
#endif
#else
  #define _STAI_NETWORK_EVENT_NODE_START_CB(_node_id, _buffers_size, ...) \
    do { /* _STAI_NETWORK_EVENT_NODE_START_CB() */ } while(0);
#endif      /* STAI_EVENT_NODE_START_CB */

#ifdef STAI_EVENT_NODE_STOP_CB
#ifndef _STAI_NETWORK_EVENT_NODE_STOP_CB
  #define _STAI_NETWORK_EVENT_NODE_STOP_CB(_node_id, _buffers_size, ...) \
  if (net_ctx->_callback) { \
    const stai_event_node_start_stop _stop_event = { \
      .node_id=(_node_id), \
      .buffers={ \
        .size=(_buffers_size), \
        .data=(stai_ptr const*)(stai_ptr[_buffers_size])STAI_PACK(__VA_ARGS__) \
      } \
    }; \
    net_ctx->_callback(net_ctx->_callback_cookie, STAI_EVENT_NODE_STOP, (const void*)&_stop_event); \
  }
#endif
#else
  #define _STAI_NETWORK_EVENT_NODE_STOP_CB(_node_id, _buffers_size, ...) \
    do { /* _STAI_NETWORK_EVENT_NODE_STOP_CB() */ } while(0);
#endif      /* STAI_EVENT_NODE_STOP_CB */


/*****************************************************************************/
#define _STAI_NETWORK_MODEL_SIGNATURE     "0x9bab4f2918ec644103bee84b1c013277"
#define _STAI_NETWORK_DATETIME            "2026-06-01T19:12:36+0200"
#define _STAI_NETWORK_COMPILE_DATETIME    __DATE__ " " __TIME__

#define _STAI_CONTEXT_ALIGNMENT        STAI_NETWORK_CONTEXT_ALIGNMENT

/*****************************************************************************/
/* Declare and allocate activation buffer #1 */
STAI_ALIGNED(STAI_NETWORK_ACTIVATION_1_ALIGNMENT)
static uint8_t g_network_activations_1[STAI_NETWORK_ACTIVATION_1_SIZE];
/* Declare and allocate activation buffer #2 */
STAI_ALIGNED(STAI_NETWORK_ACTIVATION_2_ALIGNMENT)
static uint8_t g_network_activations_2[STAI_NETWORK_ACTIVATION_2_SIZE];
/* Declare and allocate activation buffer #3 */
STAI_ALIGNED(STAI_NETWORK_ACTIVATION_3_ALIGNMENT)
static uint8_t g_network_activations_3[STAI_NETWORK_ACTIVATION_3_SIZE];


#if defined(HAVE_NETWORK_INFO)
/*****************************************************************************/
static const stai_network_info g_network_info = {
  .model_signature = _STAI_NETWORK_MODEL_SIGNATURE,
  .c_compile_datetime = _STAI_NETWORK_COMPILE_DATETIME,
  .c_model_name = STAI_NETWORK_MODEL_NAME,
  .c_model_datetime = _STAI_NETWORK_DATETIME,
  .c_model_signature = 0x0,
  .runtime_version = STAI_INIT_VERSION(12, 0, 0),
  .tool_version = STAI_INIT_VERSION(4, 0, 0),
  .api_version = STAI_INIT_VERSION(1, 0, 0),
  .n_macc = STAI_NETWORK_MACC_NUM,
  .n_nodes = STAI_NETWORK_NODES_NUM,
  .flags = STAI_NETWORK_FLAGS,
  .n_inputs = STAI_NETWORK_IN_NUM,
  .n_outputs = STAI_NETWORK_OUT_NUM,
  .n_activations = STAI_NETWORK_ACTIVATIONS_NUM,
  .n_weights = STAI_NETWORK_WEIGHTS_NUM,
  .n_states = STAI_NETWORK_STATES_NUM,
  .inputs = (stai_tensor[STAI_NETWORK_IN_NUM]) {
    STAI_INIT_TENSOR(
      STAI_NETWORK_IN_1_NAME,
      STAI_NETWORK_IN_1_FLAGS,
      STAI_NETWORK_IN_1_FORMAT,
      STAI_NETWORK_IN_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 4, 1, 128, 128, 1),
      STAI_DECLARE_ARRAY(float, 1, 0.9999597668647766f),
      STAI_DECLARE_ARRAY(int16_t, 1, 0)),
    },
    .outputs = (stai_tensor[STAI_NETWORK_OUT_NUM]) {
    STAI_INIT_TENSOR(
      STAI_NETWORK_OUT_1_NAME,
      STAI_NETWORK_OUT_1_FLAGS,
      STAI_NETWORK_OUT_1_FORMAT,
      STAI_NETWORK_OUT_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 2, 1, 3),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    },
  .activations = (stai_tensor[STAI_NETWORK_ACTIVATIONS_NUM]) {
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_ACTIVATION_1_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_ACTIVATION_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 71316),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_ACTIVATION_2_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_ACTIVATION_2_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 65536),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_ACTIVATION_3_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_ACTIVATION_3_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 209088),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    },
  .weights = (stai_tensor[STAI_NETWORK_WEIGHTS_NUM]) {
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_1_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 414076),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    },

  .states = NULL
};
#endif

#define _STAI_CONTEXT_ACQUIRE(_net_ctx, _net_handle) \
  _stai_network_context* _net_ctx = (_stai_network_context*)(_net_handle); \
  STAI_ASSERT(_net_ctx != NULL) \
  _STAI_SET_ERROR(_net_ctx, _net_ctx->_magic != STAI_MAGIC, \
                  STAI_ERROR_NETWORK_INVALID_CONTEXT_HANDLE, _net_ctx->_return_code)


/*****************************************************************************/
static
void _stai_network_check(_stai_network_context* net_ctx)
{
  stai_size idx;

// Check activations status
  for (idx=0; idx<STAI_NETWORK_ACTIVATIONS_NUM; idx++) {
    if (net_ctx->_activations[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_NETWORK_ACTIVATIONS_NUM) ? STAI_FLAG_ACTIVATIONS : STAI_FLAG_NONE;
// Check inputs status
  for (idx=0; idx<STAI_NETWORK_IN_NUM; idx++) {
    if (net_ctx->_inputs[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_NETWORK_IN_NUM) ? STAI_FLAG_INPUTS : STAI_FLAG_NONE;

  // Check outputs status
  for (idx=0; idx<STAI_NETWORK_OUT_NUM; idx++) {
    if (net_ctx->_outputs[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_NETWORK_OUT_NUM) ? STAI_FLAG_OUTPUTS : STAI_FLAG_NONE;

// Check weights status
  for (idx=0; idx<STAI_NETWORK_WEIGHTS_NUM; idx++) {
    if (net_ctx->_weights[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_NETWORK_WEIGHTS_NUM) ? STAI_FLAG_WEIGHTS : STAI_FLAG_NONE;
STAI_PRINT("  [_stai_network_check] flags: 0x%08x\n", net_ctx->_flags)
}


/*****************************************************************************/
STAI_API_ENTRY
stai_return_code stai_network_init(
  stai_network* network)
{
  /* Memory where to store internal context is provided by applications as a raw byte buffer */
  _stai_network_context* net_ctx = (_stai_network_context*)(network);
  net_ctx->_return_code = STAI_SUCCESS;
  STAI_PRINT("[Entering Network Init] network(%p) context_size(%d)\n", net_ctx, (int32_t)sizeof(_stai_network_context))

  _STAI_SET_ERROR(net_ctx, STAI_NETWORK_CONTEXT_SIZE != sizeof(_stai_network_context),
                 STAI_ERROR_NETWORK_INVALID_CONTEXT_SIZE, net_ctx->_return_code)

  {
    const _stai_network_context _network_context = {
      ._magic = STAI_MAGIC,
      ._signature = STAI_NETWORK_MODEL_SIGNATURE,
      ._flags = STAI_NETWORK_FLAGS,
      ._return_code = STAI_SUCCESS,
      ._callback = NULL,
      ._callback_cookie = NULL,
      ._activations = {
      (stai_ptr)g_network_activations_1,(stai_ptr)g_network_activations_2,(stai_ptr)g_network_activations_3
      },
      ._weights = {
      (stai_ptr)g_network_weights_array
      },
      ._inputs = {
    (stai_ptr)g_network_activations_2 + 24896},
      ._outputs = {
    (stai_ptr)g_network_activations_2 + 4},
    };

    // Deep copy of internal context to opaque buffer provided by app
    *net_ctx = _network_context;

    _stai_network_check(net_ctx);
  }

  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_deinit(
  stai_network* network)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  /*  Reset flags to initial state  */
  net_ctx->_flags = STAI_NETWORK_FLAGS;
  return net_ctx->_return_code;
}

/*****************************************************************************/



/* Int quant #0 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_6_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.3467442989349365f),
    AI_PACK_INTQ_ZP(6)))

/* Int quant #1 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_9_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.6685433387756348f),
    AI_PACK_INTQ_ZP(-14)))

/* Int quant #2 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_10_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.6306323409080505f),
    AI_PACK_INTQ_ZP(-7)))

/* Int quant #3 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_13_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.29803141951560974f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #4 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_16_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.3349054753780365f),
    AI_PACK_INTQ_ZP(5)))

/* Int quant #5 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_17_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.3349054753780365f),
    AI_PACK_INTQ_ZP(5)))

/* Int quant #6 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_20_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.38780033588409424f),
    AI_PACK_INTQ_ZP(-13)))

/* Int quant #7 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_21_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.38780033588409424f),
    AI_PACK_INTQ_ZP(-13)))

/* Int quant #8 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_24_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.2426532357931137f),
    AI_PACK_INTQ_ZP(12)))

/* Int quant #9 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_27_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.24610859155654907f),
    AI_PACK_INTQ_ZP(6)))

/* Int quant #10 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_28_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.24610859155654907f),
    AI_PACK_INTQ_ZP(6)))

/* Int quant #11 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_31_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.24313980340957642f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #12 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_32_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.24313980340957642f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #13 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_35_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.25250938534736633f),
    AI_PACK_INTQ_ZP(13)))

/* Int quant #14 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_36_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.25250938534736633f),
    AI_PACK_INTQ_ZP(13)))

/* Int quant #15 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_39_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.18943023681640625f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #16 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_42_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.2056785374879837f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #17 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_43_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.2056785374879837f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #18 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_46_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.30521687865257263f),
    AI_PACK_INTQ_ZP(-20)))

/* Int quant #19 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_47_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.30521687865257263f),
    AI_PACK_INTQ_ZP(-20)))

/* Int quant #20 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_50_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.16475240886211395f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #21 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_53_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.17767465114593506f),
    AI_PACK_INTQ_ZP(6)))

/* Int quant #22 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_54_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.17767465114593506f),
    AI_PACK_INTQ_ZP(6)))

/* Int quant #23 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_57_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.18914462625980377f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #24 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_58_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.18914462625980377f),
    AI_PACK_INTQ_ZP(3)))

/* Int quant #25 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_62_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0235294122248888f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #26 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(pool_63_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.019416170194745064f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #27 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_64_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.17528976500034332f),
    AI_PACK_INTQ_ZP(2)))

/* Int quant #28 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_64_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 3,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003487570444121957f, 0.0030479049310088158f, 0.0028619312215596437f),
    AI_PACK_INTQ_ZP(0, 0, 0)))



/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_6_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_9_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_10_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_13_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_16_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_17_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_20_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_21_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_24_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1536, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_27_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1536, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_28_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1536, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_31_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1536, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_32_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1536, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_35_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1536, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_36_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1536, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_39_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2048, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_42_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2048, AI_STATIC)

/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_43_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2048, AI_STATIC)

/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_46_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2048, AI_STATIC)

/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_47_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2048, AI_STATIC)

/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_50_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 896, AI_STATIC)

/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_53_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 896, AI_STATIC)

/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_54_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 896, AI_STATIC)

/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_57_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 896, AI_STATIC)

/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_58_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 896, AI_STATIC)

/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_62_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 20480, AI_STATIC)

/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  pool_63_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1280, AI_STATIC)

/* Array#27 */
AI_ARRAY_OBJ_DECLARE(
  gemm_64_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3, AI_STATIC)

/* Array#28 */
AI_ARRAY_OBJ_DECLARE(
  gemm_64_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3840, AI_STATIC)

/* Array#29 */
AI_ARRAY_OBJ_DECLARE(
  gemm_64_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 3, AI_STATIC)

/* Array#30 */
AI_ARRAY_OBJ_DECLARE(
  gemm_64_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 1295, AI_STATIC)



/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_6_output, AI_STATIC,
  213, 0x1,
  AI_SHAPE_INIT(4, 1, 8, 32, 32), AI_STRIDE_INIT(4, 1, 1, 8, 256),
  1, &conv2d_6_output_array, &conv2d_6_output_array_intq)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_9_output, AI_STATIC,
  226, 0x1,
  AI_SHAPE_INIT(4, 1, 8, 32, 32), AI_STRIDE_INIT(4, 1, 1, 8, 256),
  1, &conv2d_9_output_array, &conv2d_9_output_array_intq)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_10_output, AI_STATIC,
  230, 0x1,
  AI_SHAPE_INIT(4, 1, 8, 32, 32), AI_STRIDE_INIT(4, 1, 1, 8, 256),
  1, &eltwise_10_output_array, &eltwise_10_output_array_intq)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_13_output, AI_STATIC,
  14, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 16, 16), AI_STRIDE_INIT(4, 1, 1, 16, 256),
  1, &conv2d_13_output_array, &conv2d_13_output_array_intq)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_16_output, AI_STATIC,
  27, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 16, 16), AI_STRIDE_INIT(4, 1, 1, 16, 256),
  1, &conv2d_16_output_array, &conv2d_16_output_array_intq)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_17_output, AI_STATIC,
  231, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 16, 16), AI_STRIDE_INIT(4, 1, 1, 16, 256),
  1, &eltwise_17_output_array, &eltwise_17_output_array_intq)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_20_output, AI_STATIC,
  44, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 16, 16), AI_STRIDE_INIT(4, 1, 1, 16, 256),
  1, &conv2d_20_output_array, &conv2d_20_output_array_intq)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_21_output, AI_STATIC,
  232, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 16, 16), AI_STRIDE_INIT(4, 1, 1, 16, 256),
  1, &eltwise_21_output_array, &eltwise_21_output_array_intq)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_24_output, AI_STATIC,
  57, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 8, 8), AI_STRIDE_INIT(4, 1, 1, 24, 192),
  1, &conv2d_24_output_array, &conv2d_24_output_array_intq)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_27_output, AI_STATIC,
  70, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 8, 8), AI_STRIDE_INIT(4, 1, 1, 24, 192),
  1, &conv2d_27_output_array, &conv2d_27_output_array_intq)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_28_output, AI_STATIC,
  233, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 8, 8), AI_STRIDE_INIT(4, 1, 1, 24, 192),
  1, &eltwise_28_output_array, &eltwise_28_output_array_intq)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_31_output, AI_STATIC,
  88, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 8, 8), AI_STRIDE_INIT(4, 1, 1, 24, 192),
  1, &conv2d_31_output_array, &conv2d_31_output_array_intq)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_32_output, AI_STATIC,
  234, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 8, 8), AI_STRIDE_INIT(4, 1, 1, 24, 192),
  1, &eltwise_32_output_array, &eltwise_32_output_array_intq)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_35_output, AI_STATIC,
  101, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 8, 8), AI_STRIDE_INIT(4, 1, 1, 24, 192),
  1, &conv2d_35_output_array, &conv2d_35_output_array_intq)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_36_output, AI_STATIC,
  235, 0x1,
  AI_SHAPE_INIT(4, 1, 24, 8, 8), AI_STRIDE_INIT(4, 1, 1, 24, 192),
  1, &eltwise_36_output_array, &eltwise_36_output_array_intq)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_39_output, AI_STATIC,
  114, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 8, 8), AI_STRIDE_INIT(4, 1, 1, 32, 256),
  1, &conv2d_39_output_array, &conv2d_39_output_array_intq)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_42_output, AI_STATIC,
  131, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 8, 8), AI_STRIDE_INIT(4, 1, 1, 32, 256),
  1, &conv2d_42_output_array, &conv2d_42_output_array_intq)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_43_output, AI_STATIC,
  236, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 8, 8), AI_STRIDE_INIT(4, 1, 1, 32, 256),
  1, &eltwise_43_output_array, &eltwise_43_output_array_intq)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_46_output, AI_STATIC,
  144, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 8, 8), AI_STRIDE_INIT(4, 1, 1, 32, 256),
  1, &conv2d_46_output_array, &conv2d_46_output_array_intq)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_47_output, AI_STATIC,
  237, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 8, 8), AI_STRIDE_INIT(4, 1, 1, 32, 256),
  1, &eltwise_47_output_array, &eltwise_47_output_array_intq)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_50_output, AI_STATIC,
  161, 0x1,
  AI_SHAPE_INIT(4, 1, 56, 4, 4), AI_STRIDE_INIT(4, 1, 1, 56, 224),
  1, &conv2d_50_output_array, &conv2d_50_output_array_intq)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_53_output, AI_STATIC,
  174, 0x1,
  AI_SHAPE_INIT(4, 1, 56, 4, 4), AI_STRIDE_INIT(4, 1, 1, 56, 224),
  1, &conv2d_53_output_array, &conv2d_53_output_array_intq)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_54_output, AI_STATIC,
  238, 0x1,
  AI_SHAPE_INIT(4, 1, 56, 4, 4), AI_STRIDE_INIT(4, 1, 1, 56, 224),
  1, &eltwise_54_output_array, &eltwise_54_output_array_intq)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_57_output, AI_STATIC,
  187, 0x1,
  AI_SHAPE_INIT(4, 1, 56, 4, 4), AI_STRIDE_INIT(4, 1, 1, 56, 224),
  1, &conv2d_57_output_array, &conv2d_57_output_array_intq)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_58_output, AI_STATIC,
  239, 0x1,
  AI_SHAPE_INIT(4, 1, 56, 4, 4), AI_STRIDE_INIT(4, 1, 1, 56, 224),
  1, &eltwise_58_output_array, &eltwise_58_output_array_intq)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_62_output, AI_STATIC,
  209, 0x1,
  AI_SHAPE_INIT(4, 1, 1280, 4, 4), AI_STRIDE_INIT(4, 1, 1, 1280, 5120),
  1, &conv2d_62_output_array, &conv2d_62_output_array_intq)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  pool_63_output, AI_STATIC,
  246, 0x1,
  AI_SHAPE_INIT(4, 1, 1280, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1280, 1280),
  1, &pool_63_output_array, &pool_63_output_array_intq)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  gemm_64_bias, AI_STATIC,
  240, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 1, 1), AI_STRIDE_INIT(4, 4, 4, 12, 12),
  1, &gemm_64_bias_array, NULL)

/* Tensor #28 */
AI_TENSOR_OBJ_DECLARE(
  gemm_64_output, AI_STATIC,
  241, 0x1,
  AI_SHAPE_INIT(4, 1, 3, 1, 1), AI_STRIDE_INIT(4, 1, 1, 3, 3),
  1, &gemm_64_output_array, &gemm_64_output_array_intq)

/* Tensor #29 */
AI_TENSOR_OBJ_DECLARE(
  gemm_64_scratch0, AI_STATIC,
  242, 0x0,
  AI_SHAPE_INIT(4, 1, 1295, 1, 1), AI_STRIDE_INIT(4, 2, 2, 2590, 2590),
  1, &gemm_64_scratch0_array, NULL)

/* Tensor #30 */
AI_TENSOR_OBJ_DECLARE(
  gemm_64_weights, AI_STATIC,
  243, 0x1,
  AI_SHAPE_INIT(4, 1280, 3, 1, 1), AI_STRIDE_INIT(4, 1, 1280, 3840, 3840),
  1, &gemm_64_weights_array, &gemm_64_weights_array_intq)


AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_10_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_6_output, &conv2d_9_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_10_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_10_layer, 10,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_10_chain,
  NULL, &eltwise_10_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_17_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_13_output, &conv2d_16_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_17_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_17_layer, 17,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_17_chain,
  NULL, &eltwise_17_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_21_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_17_output, &conv2d_20_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_21_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_21_layer, 21,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_21_chain,
  NULL, &eltwise_21_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_28_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_24_output, &conv2d_27_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_28_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_28_layer, 28,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_28_chain,
  NULL, &eltwise_28_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_32_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_28_output, &conv2d_31_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_32_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_32_layer, 32,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_32_chain,
  NULL, &eltwise_32_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_36_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_32_output, &conv2d_35_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_36_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_36_layer, 36,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_36_chain,
  NULL, &eltwise_36_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_43_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_39_output, &conv2d_42_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_43_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_43_layer, 43,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_43_chain,
  NULL, &eltwise_43_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_47_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_43_output, &conv2d_46_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_47_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_47_layer, 47,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_47_chain,
  NULL, &eltwise_47_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_54_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_50_output, &conv2d_53_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_54_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_54_layer, 54,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_54_chain,
  NULL, &eltwise_54_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_58_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &eltwise_54_output, &conv2d_57_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_58_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_58_layer, 58,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_58_chain,
  NULL, &eltwise_58_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_63_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_62_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_63_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_63_layer, 63,
  POOL_TYPE, 0x0, NULL,
  pool, forward_ap_integer_INT8,
  &pool_63_chain,
  NULL, &pool_63_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(4, 4), 
  .pool_stride = AI_SHAPE_2D_INIT(4, 4), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_64_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_63_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_64_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_64_weights, &gemm_64_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_64_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_64_layer, 64,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_64_chain,
  NULL, &gemm_64_layer, AI_STATIC, 
)
/**  Hybrid layers declarations section  *************************************/
void forward_lite_eltwise_integer_INT8_eltwise_10(_stai_network_context* net_ctx)
{
  conv2d_6_output_array.data = AI_PTR(net_ctx->_activations[1] + 57280);
  conv2d_6_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 57280);
  conv2d_9_output_array.data = AI_PTR(net_ctx->_activations[1] + 0);
  conv2d_9_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 0);
  eltwise_10_output_array.data = AI_PTR(net_ctx->_activations[1] + 8192);
  eltwise_10_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 8192);
  _STAI_NETWORK_EVENT_NODE_START_CB(10, 2, { conv2d_6_output.data->data,conv2d_9_output.data->data});
  forward_eltwise_integer_INT8(&eltwise_10_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(10, 1, { eltwise_10_output.data->data});
}
void forward_lite_eltwise_integer_INT8_eltwise_17(_stai_network_context* net_ctx)
{
  conv2d_13_output_array.data = AI_PTR(net_ctx->_activations[1] + 0);
  conv2d_13_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 0);
  conv2d_16_output_array.data = AI_PTR(net_ctx->_activations[1] + 28672);
  conv2d_16_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 28672);
  eltwise_17_output_array.data = AI_PTR(net_ctx->_activations[1] + 4096);
  eltwise_17_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 4096);
  _STAI_NETWORK_EVENT_NODE_START_CB(17, 2, { conv2d_13_output.data->data,conv2d_16_output.data->data});
  forward_eltwise_integer_INT8(&eltwise_17_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(17, 1, { eltwise_17_output.data->data});
}
void forward_lite_eltwise_integer_INT8_eltwise_21(_stai_network_context* net_ctx)
{
  eltwise_17_output_array.data = AI_PTR(net_ctx->_activations[1] + 4096);
  eltwise_17_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 4096);
  conv2d_20_output_array.data = AI_PTR(net_ctx->_activations[1] + 0);
  conv2d_20_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 0);
  eltwise_21_output_array.data = AI_PTR(net_ctx->_activations[1] + 8192);
  eltwise_21_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 8192);
  _STAI_NETWORK_EVENT_NODE_START_CB(21, 2, { eltwise_17_output.data->data,conv2d_20_output.data->data});
  forward_eltwise_integer_INT8(&eltwise_21_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(21, 1, { eltwise_21_output.data->data});
}
void forward_lite_eltwise_integer_INT8_eltwise_28(_stai_network_context* net_ctx)
{
  conv2d_24_output_array.data = AI_PTR(net_ctx->_activations[1] + 0);
  conv2d_24_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 0);
  conv2d_27_output_array.data = AI_PTR(net_ctx->_activations[1] + 10752);
  conv2d_27_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 10752);
  eltwise_28_output_array.data = AI_PTR(net_ctx->_activations[1] + 1536);
  eltwise_28_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 1536);
  _STAI_NETWORK_EVENT_NODE_START_CB(28, 2, { conv2d_24_output.data->data,conv2d_27_output.data->data});
  forward_eltwise_integer_INT8(&eltwise_28_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(28, 1, { eltwise_28_output.data->data});
}
void forward_lite_eltwise_integer_INT8_eltwise_32(_stai_network_context* net_ctx)
{
  eltwise_28_output_array.data = AI_PTR(net_ctx->_activations[1] + 1536);
  eltwise_28_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 1536);
  conv2d_31_output_array.data = AI_PTR(net_ctx->_activations[1] + 0);
  conv2d_31_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 0);
  eltwise_32_output_array.data = AI_PTR(net_ctx->_activations[1] + 3072);
  eltwise_32_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 3072);
  _STAI_NETWORK_EVENT_NODE_START_CB(32, 2, { eltwise_28_output.data->data,conv2d_31_output.data->data});
  forward_eltwise_integer_INT8(&eltwise_32_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(32, 1, { eltwise_32_output.data->data});
}
void forward_lite_eltwise_integer_INT8_eltwise_36(_stai_network_context* net_ctx)
{
  eltwise_32_output_array.data = AI_PTR(net_ctx->_activations[1] + 3072);
  eltwise_32_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 3072);
  conv2d_35_output_array.data = AI_PTR(net_ctx->_activations[1] + 0);
  conv2d_35_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 0);
  eltwise_36_output_array.data = AI_PTR(net_ctx->_activations[1] + 1536);
  eltwise_36_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 1536);
  _STAI_NETWORK_EVENT_NODE_START_CB(36, 2, { eltwise_32_output.data->data,conv2d_35_output.data->data});
  forward_eltwise_integer_INT8(&eltwise_36_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(36, 1, { eltwise_36_output.data->data});
}
void forward_lite_eltwise_integer_INT8_eltwise_43(_stai_network_context* net_ctx)
{
  conv2d_39_output_array.data = AI_PTR(net_ctx->_activations[1] + 9216);
  conv2d_39_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 9216);
  conv2d_42_output_array.data = AI_PTR(net_ctx->_activations[1] + 0);
  conv2d_42_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 0);
  eltwise_43_output_array.data = AI_PTR(net_ctx->_activations[1] + 2048);
  eltwise_43_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 2048);
  _STAI_NETWORK_EVENT_NODE_START_CB(43, 2, { conv2d_39_output.data->data,conv2d_42_output.data->data});
  forward_eltwise_integer_INT8(&eltwise_43_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(43, 1, { eltwise_43_output.data->data});
}
void forward_lite_eltwise_integer_INT8_eltwise_47(_stai_network_context* net_ctx)
{
  eltwise_43_output_array.data = AI_PTR(net_ctx->_activations[1] + 2048);
  eltwise_43_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 2048);
  conv2d_46_output_array.data = AI_PTR(net_ctx->_activations[1] + 0);
  conv2d_46_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 0);
  eltwise_47_output_array.data = AI_PTR(net_ctx->_activations[1] + 4096);
  eltwise_47_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 4096);
  _STAI_NETWORK_EVENT_NODE_START_CB(47, 2, { eltwise_43_output.data->data,conv2d_46_output.data->data});
  forward_eltwise_integer_INT8(&eltwise_47_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(47, 1, { eltwise_47_output.data->data});
}
void forward_lite_eltwise_integer_INT8_eltwise_54(_stai_network_context* net_ctx)
{
  conv2d_50_output_array.data = AI_PTR(net_ctx->_activations[1] + 3072);
  conv2d_50_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 3072);
  conv2d_53_output_array.data = AI_PTR(net_ctx->_activations[1] + 0);
  conv2d_53_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 0);
  eltwise_54_output_array.data = AI_PTR(net_ctx->_activations[1] + 896);
  eltwise_54_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 896);
  _STAI_NETWORK_EVENT_NODE_START_CB(54, 2, { conv2d_50_output.data->data,conv2d_53_output.data->data});
  forward_eltwise_integer_INT8(&eltwise_54_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(54, 1, { eltwise_54_output.data->data});
}
void forward_lite_eltwise_integer_INT8_eltwise_58(_stai_network_context* net_ctx)
{
  eltwise_54_output_array.data = AI_PTR(net_ctx->_activations[1] + 896);
  eltwise_54_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 896);
  conv2d_57_output_array.data = AI_PTR(net_ctx->_activations[1] + 0);
  conv2d_57_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 0);
  eltwise_58_output_array.data = AI_PTR(net_ctx->_activations[1] + 1792);
  eltwise_58_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 1792);
  _STAI_NETWORK_EVENT_NODE_START_CB(58, 2, { eltwise_54_output.data->data,conv2d_57_output.data->data});
  forward_eltwise_integer_INT8(&eltwise_58_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(58, 1, { eltwise_58_output.data->data});
}
void forward_lite_ap_integer_INT8_pool_63(_stai_network_context* net_ctx)
{
  conv2d_62_output_array.data = AI_PTR(net_ctx->_activations[1] + 7168);
  conv2d_62_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 7168);
  pool_63_output_array.data = AI_PTR(net_ctx->_activations[1] + 0);
  pool_63_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 0);
  _STAI_NETWORK_EVENT_NODE_START_CB(63, 1, { conv2d_62_output.data->data});
  forward_ap_integer_INT8(&pool_63_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(63, 1, { pool_63_output.data->data});
}
void forward_lite_dense_integer_SSSA_ch_gemm_64(_stai_network_context* net_ctx)
{
  pool_63_output_array.data = AI_PTR(net_ctx->_activations[1] + 0);
  pool_63_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 0);
  gemm_64_weights_array.data = AI_PTR(net_ctx->_weights[0] + 410224);
  gemm_64_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 410224);
  gemm_64_bias_array.data = AI_PTR(net_ctx->_weights[0] + 414064);
  gemm_64_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 414064);
  gemm_64_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  gemm_64_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  gemm_64_output_array.data = AI_PTR(net_ctx->_activations[1] + 1280);
  gemm_64_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 1280);
  _STAI_NETWORK_EVENT_NODE_START_CB(64, 1, { pool_63_output.data->data});
  forward_dense_integer_SSSA_ch(&gemm_64_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(64, 1, { gemm_64_output.data->data});
}

/*****************************************************************************/


static const ai_u32 serving_default_input_layer_50_0_conversion_t_in_0_shape_h_w_ch_d_prod_const_u32 = 16384;
static const ai_u8 serving_default_input_layer_50_0_conversion_t_in_0_fmt_zero_const_u8 = 0;
static const ai_i8 serving_default_input_layer_50_0_conversion_t_out_0_fmt_zero_const_s8 = -128;

static const ai_u16 conv2d_0_t_in_0_shape_w_const_u16 = 128;
static const ai_u16 conv2d_0_t_in_0_shape_h_const_u16 = 128;
static const ai_u16 conv2d_0_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_0_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_0_t_in_0_shape_ch_const_u16 = 1;
static const ai_u16 conv2d_0_t_out_0_shape_ch_const_u16 = 3;
static const ai_i8 conv2d_0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_0_t_in_0_fmt_scale_const_f32 = 0.9999597668647766f;
static const ai_float conv2d_0_t_out_0_fmt_scale_const_f32 = 0.7065966129302979f;
static const ai_float conv2d_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.005581282544881105f, 0.005441753193736076f, 0.005208952818065882f);
static const ai_layer_format_type conv2d_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_u16 conv2d_1_t_in_0_shape_w_const_u16 = 128;
static const ai_u16 conv2d_1_t_out_0_shape_ch_const_u16 = 16;
static const ai_u16 conv2d_1_t_weight_0_shape_w_const_u16 = 3;
static const ai_i32 conv2d_1_l_pad_W_0_const_s32 = 0;
static const ai_u16 conv2d_1_l_stride_0_const_u16 = 2;
static const ai_i8 conv2d_1_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_1_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_1_t_in_0_fmt_scale_const_f32 = 0.7065966129302979f;
static const ai_float conv2d_1_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_1_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(6.327265145955607e-05f, 8.34153761388734e-05f, 0.0011886466527357697f, 4.978112588815975e-09f, 3.580329575925134e-05f, 0.0012534332927316427f, 4.101470274520125e-09f, 3.937008052901092e-09f, 3.937008052901092e-09f, 3.937008052901092e-09f, 3.937008052901092e-09f, 0.00015948756481520832f, 3.937008052901092e-09f, 3.937008052901092e-09f, 0.0018339015077799559f, 0.0003036952402908355f);
static const ai_layer_format_type conv2d_1_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;
static const ai_u16 conv2d_1_t_out_0_shape_w_const_u16 = 64;

static const ai_i8 conv2d_2_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 conv2d_2_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 conv2d_2_pad_before_t_in_0_shape_h_const_u32 = 64;

static const ai_u16 conv2d_2_t_in_0_shape_w_const_u16 = 66;
static const ai_u16 conv2d_2_t_in_0_shape_h_const_u16 = 66;
static const ai_u16 conv2d_2_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 conv2d_2_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_2_l_stride_0_const_u16 = 1;
static const ai_i8 conv2d_2_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_2_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_2_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_2_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_2_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.007765833754092455f, 0.006568482611328363f, 0.009751880541443825f, 0.3139938414096832f, 0.012767176143825054f, 0.0034916626755148172f, 0.06294555217027664f, 0.336553156375885f, 0.6555811762809753f, 0.01133005041629076f, 0.33615633845329285f, 0.002337902318686247f, 0.3032977879047394f, 0.01352775003761053f, 0.003026610240340233f, 0.00039661306072957814f);
static const ai_u16 conv2d_2_t_out_0_shape_w_const_u16 = 64;
static const ai_u16 conv2d_2_t_out_0_shape_h_const_u16 = 64;

static const ai_u16 conv2d_3_t_in_0_shape_w_const_u16 = 64;
static const ai_u16 conv2d_3_t_in_0_shape_h_const_u16 = 64;
static const ai_u16 conv2d_3_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_3_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_3_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 conv2d_3_t_out_0_shape_ch_const_u16 = 8;
static const ai_i8 conv2d_3_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_3_t_out_0_fmt_zero_const_s8 = 15;
static const ai_float conv2d_3_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_3_t_out_0_fmt_scale_const_f32 = 0.49765023589134216f;
static const ai_float conv2d_3_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.07086090743541718f, 0.0411166176199913f, 0.1094738095998764f, 0.040958184748888016f, 0.13487502932548523f, 0.0443316213786602f, 0.05463417246937752f, 0.06543125212192535f);
static const ai_layer_format_type conv2d_3_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_u16 conv2d_4_t_in_0_shape_w_const_u16 = 64;
static const ai_u16 conv2d_4_t_in_0_shape_h_const_u16 = 64;
static const ai_u16 conv2d_4_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_4_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_4_t_in_0_shape_ch_const_u16 = 8;
static const ai_u16 conv2d_4_t_out_0_shape_ch_const_u16 = 48;
static const ai_i8 conv2d_4_t_in_0_fmt_zero_const_s8 = 15;
static const ai_i8 conv2d_4_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_4_t_in_0_fmt_scale_const_f32 = 0.49765023589134216f;
static const ai_float conv2d_4_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_4_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(3.937008052901092e-09f, 0.0018650174606591463f, 0.002467399463057518f, 0.004407557658851147f, 0.001009872299619019f, 0.0016130133299157023f, 0.003430536948144436f, 0.004916246514767408f, 0.0018056510016322136f, 0.003741270862519741f, 0.0016336599364876747f, 0.0009356859372928739f, 0.00043060327880084515f, 0.0010982508538290858f, 0.001444572233594954f, 0.0023890496231615543f, 3.937008052901092e-09f, 0.0008354817400686443f, 0.0025725746527314186f, 0.0006017734413035214f, 0.001299391034990549f, 0.0004909429117105901f, 0.0006408154149539769f, 0.002121045021340251f, 0.000419632502598688f, 3.937008052901092e-09f, 0.0010252564679831266f, 0.001950184116140008f, 0.0006682341336272657f, 0.0008638820727355778f, 0.0018555426504462957f, 0.00231275986880064f, 1.1603746408184179e-08f, 0.0010747001506388187f, 0.0009617966134101152f, 0.00306100957095623f, 0.004689422901719809f, 0.0020922094117850065f, 0.006156226620078087f, 0.0005438823136501014f, 0.0005127672338858247f, 0.0006811395287513733f, 0.0015732607571408153f, 0.0007729104836471379f, 0.0009502642205916345f, 0.0005629805964417756f, 0.0012323901755735278f, 0.0016020878683775663f);
static const ai_layer_format_type conv2d_4_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_i8 conv2d_5_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 conv2d_5_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 conv2d_5_pad_before_t_in_0_shape_h_const_u32 = 64;

static const ai_u16 conv2d_5_t_in_0_shape_w_const_u16 = 66;
static const ai_u16 conv2d_5_t_in_0_shape_h_const_u16 = 66;
static const ai_u16 conv2d_5_t_in_0_shape_ch_const_u16 = 48;
static const ai_u16 conv2d_5_l_stride_1_const_u16 = 2;
static const ai_u16 conv2d_5_l_stride_0_const_u16 = 2;
static const ai_i8 conv2d_5_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_5_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_5_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_5_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_5_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.035710372030735016f, 0.003916418179869652f, 0.010873735882341862f, 0.001945069758221507f, 0.0030989870429039f, 0.0016564760589972138f, 0.0023100494872778654f, 0.0007840690668672323f, 0.02271745540201664f, 0.005398370325565338f, 0.008072246797382832f, 0.025720471516251564f, 0.018936077132821083f, 0.003027360187843442f, 0.004975865129381418f, 0.005220936611294746f, 0.002608107402920723f, 0.007126083597540855f, 0.005303915124386549f, 0.0027275218162685633f, 0.03368391469120979f, 0.006959512829780579f, 0.014178247191011906f, 0.005819086451083422f, 0.008951066993176937f, 0.008054054342210293f, 0.003683566814288497f, 0.007897513918578625f, 0.011610870249569416f, 0.02099589630961418f, 0.005434401798993349f, 0.010740650817751884f, 0.0039700609631836414f, 0.003265669569373131f, 0.002398113487288356f, 0.006623946595937014f, 0.0012400195701047778f, 0.00225838297046721f, 0.0014233543770387769f, 0.009441457688808441f, 0.010708979330956936f, 0.011478462256491184f, 0.004012025892734528f, 0.006937436293810606f, 0.048822831362485886f, 0.010490297339856625f, 0.014442041516304016f, 0.03089837171137333f);
static const ai_u16 conv2d_5_t_out_0_shape_w_const_u16 = 32;
static const ai_u16 conv2d_5_t_out_0_shape_h_const_u16 = 32;

static const ai_u16 conv2d_6_t_in_0_shape_w_const_u16 = 32;
static const ai_u16 conv2d_6_t_in_0_shape_h_const_u16 = 32;
static const ai_u16 conv2d_6_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_6_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_6_t_in_0_shape_ch_const_u16 = 48;
static const ai_u16 conv2d_6_t_out_0_shape_ch_const_u16 = 8;
static const ai_i8 conv2d_6_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_6_t_out_0_fmt_zero_const_s8 = 6;
static const ai_float conv2d_6_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_6_t_out_0_fmt_scale_const_f32 = 0.3467442989349365f;
static const ai_float conv2d_6_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.03796317055821419f, 0.020367320626974106f, 0.02534296177327633f, 0.03725607320666313f, 0.026529958471655846f, 0.02793160453438759f, 0.04065617173910141f, 0.0364021360874176f);
static const ai_layer_format_type conv2d_6_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_u16 conv2d_7_t_in_0_shape_w_const_u16 = 32;
static const ai_u16 conv2d_7_t_in_0_shape_h_const_u16 = 32;
static const ai_u16 conv2d_7_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_7_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_7_t_in_0_shape_ch_const_u16 = 8;
static const ai_u16 conv2d_7_t_out_0_shape_ch_const_u16 = 48;
static const ai_i8 conv2d_7_t_in_0_fmt_zero_const_s8 = 6;
static const ai_i8 conv2d_7_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_7_t_in_0_fmt_scale_const_f32 = 0.3467442989349365f;
static const ai_float conv2d_7_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_7_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.00046547825331799686f, 0.001321733696386218f, 3.937008052901092e-09f, 0.0012031854130327702f, 0.0009467217023484409f, 0.0006667110719718039f, 0.0009155574953183532f, 0.0006947867223061621f, 0.0007088600541464984f, 0.0010892597492784262f, 0.00018628746329341084f, 0.0018592013511806726f, 0.000272601522738114f, 0.0022102808579802513f, 0.001031161518767476f, 0.0005558539996854961f, 0.009015623480081558f, 0.004324482288211584f, 0.002058144425973296f, 0.0007064250530675054f, 0.0008913109195418656f, 0.0008359251660294831f, 0.0011017987271770835f, 0.0039812191389501095f, 0.0008642742759548128f, 0.0016223954735323787f, 0.0011656215647235513f, 0.0012752535985782743f, 0.0014486400177702308f, 0.0003680038789752871f, 0.0006635201862081885f, 0.004648115951567888f, 0.0007607336156070232f, 0.0015305940760299563f, 0.0013298318954184651f, 0.0006210440769791603f, 0.0016095987521111965f, 0.0008878568769432604f, 0.0006580399931408465f, 0.003594886278733611f, 0.0032172431237995625f, 3.937008052901092e-09f, 0.0005972753278911114f, 0.0004935032338835299f, 0.002208285266533494f, 0.0012768080923706293f, 0.004434442147612572f, 0.0010230077896267176f);
static const ai_layer_format_type conv2d_7_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_i8 conv2d_8_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 conv2d_8_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 conv2d_8_pad_before_t_in_0_shape_h_const_u32 = 32;

static const ai_u16 conv2d_8_t_in_0_shape_w_const_u16 = 34;
static const ai_u16 conv2d_8_t_in_0_shape_h_const_u16 = 34;
static const ai_u16 conv2d_8_t_in_0_shape_ch_const_u16 = 48;
static const ai_u16 conv2d_8_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_8_l_stride_0_const_u16 = 1;
static const ai_i8 conv2d_8_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_8_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_8_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_8_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_8_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.006884897127747536f, 0.003019406460225582f, 0.0005144455353729427f, 0.0104289585724473f, 0.00642318045720458f, 0.0076599023304879665f, 0.007652889937162399f, 0.005504148080945015f, 0.01092545222491026f, 0.004372493363916874f, 0.008550282567739487f, 0.006696389988064766f, 0.008483068086206913f, 0.006025705020874739f, 0.004992887377738953f, 0.012694078497588634f, 0.008249483071267605f, 0.0022448122035712004f, 0.005977529566735029f, 0.0034220204688608646f, 0.0035029170103371143f, 0.010076485574245453f, 0.003664252581074834f, 0.0074100736528635025f, 0.010796292684972286f, 0.005260025151073933f, 0.0016131956363096833f, 0.0068164062686264515f, 0.005658530164510012f, 0.00878010131418705f, 0.004498063120990992f, 0.001971463207155466f, 0.009343169629573822f, 0.003921471070498228f, 0.007251047063618898f, 0.007405971176922321f, 0.002359343459829688f, 0.008278326131403446f, 0.007021954283118248f, 0.0033301254734396935f, 0.005758683197200298f, 0.11410766839981079f, 0.006251608487218618f, 0.011035761795938015f, 0.004451119340956211f, 0.005096548702567816f, 0.0015510939992964268f, 0.00404542637988925f);
static const ai_u16 conv2d_8_t_out_0_shape_w_const_u16 = 32;
static const ai_u16 conv2d_8_t_out_0_shape_h_const_u16 = 32;

static const ai_u16 conv2d_9_t_in_0_shape_w_const_u16 = 32;
static const ai_u16 conv2d_9_t_in_0_shape_h_const_u16 = 32;
static const ai_u16 conv2d_9_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_9_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_9_t_in_0_shape_ch_const_u16 = 48;
static const ai_u16 conv2d_9_t_out_0_shape_ch_const_u16 = 8;
static const ai_i8 conv2d_9_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_9_t_out_0_fmt_zero_const_s8 = -14;
static const ai_float conv2d_9_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_9_t_out_0_fmt_scale_const_f32 = 0.6685433387756348f;
static const ai_float conv2d_9_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.024408262223005295f, 0.0344914086163044f, 0.022277740761637688f, 0.027780089527368546f, 0.03738660737872124f, 0.02647893875837326f, 0.06192346289753914f, 0.059042852371931076f);
static const ai_layer_format_type conv2d_9_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 conv2d_11_t_in_0_shape_w_const_u16 = 32;
static const ai_u16 conv2d_11_t_in_0_shape_h_const_u16 = 32;
static const ai_u16 conv2d_11_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_11_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_11_t_in_0_shape_ch_const_u16 = 8;
static const ai_u16 conv2d_11_t_out_0_shape_ch_const_u16 = 48;
static const ai_i8 conv2d_11_t_in_0_fmt_zero_const_s8 = -7;
static const ai_i8 conv2d_11_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_11_t_in_0_fmt_scale_const_f32 = 0.6306323409080505f;
static const ai_float conv2d_11_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_11_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0005719534819945693f, 0.0010056821629405022f, 0.0008275563595816493f, 0.0008950107148848474f, 0.0009519734885543585f, 0.00045791573938913643f, 9.283958206651732e-05f, 0.0002563111193012446f, 0.00031294452492147684f, 0.0007337350980378687f, 0.0029937289655208588f, 0.0003891394007951021f, 0.0003627708356361836f, 0.0005408644210547209f, 0.00045486673479899764f, 0.002465285127982497f, 0.0023815385065972805f, 0.0004731094231829047f, 0.00038158471579663455f, 0.00031846092315390706f, 0.0004948233836330473f, 0.002670405898243189f, 0.00039556011324748397f, 0.00027312382007949054f, 0.0005529536865651608f, 0.0028180498629808426f, 0.000419903575675562f, 0.0010670892661437392f, 0.0008000757661648095f, 0.0003411344368942082f, 0.0006963199703022838f, 0.0003698203945532441f, 0.0020643752068281174f, 0.00038125229184515774f, 0.00034192181192338467f, 0.00030034640803933144f, 0.002790815196931362f, 0.0011047697626054287f, 0.0005338009796105325f, 0.0012278292560949922f, 0.0023505031131207943f, 0.0009459410212002695f, 0.001589559717103839f, 0.0004839354078285396f, 0.0008886067662388086f, 0.00013585663691628724f, 0.0022407365031540394f, 0.0006672308081761003f);
static const ai_layer_format_type conv2d_11_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_i8 conv2d_12_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 conv2d_12_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 conv2d_12_pad_before_t_in_0_shape_h_const_u32 = 32;

static const ai_u16 conv2d_12_t_in_0_shape_w_const_u16 = 34;
static const ai_u16 conv2d_12_t_in_0_shape_h_const_u16 = 34;
static const ai_u16 conv2d_12_t_in_0_shape_ch_const_u16 = 48;
static const ai_u16 conv2d_12_l_stride_1_const_u16 = 2;
static const ai_u16 conv2d_12_l_stride_0_const_u16 = 2;
static const ai_i8 conv2d_12_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_12_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_12_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_12_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_12_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0025376772973686457f, 0.0029538129456341267f, 0.0023426145780831575f, 0.003665994619950652f, 0.0018968203803524375f, 0.0055236960761249065f, 0.01491483487188816f, 0.006983896717429161f, 0.005051139276474714f, 0.003455430967733264f, 0.0023610424250364304f, 0.005856684409081936f, 0.006103167310357094f, 0.0028958420734852552f, 0.005484973080456257f, 0.0013093661982566118f, 0.0009291677852161229f, 0.0027660939376801252f, 0.005437226500362158f, 0.008528618142008781f, 0.0033753658644855022f, 0.0012755641946569085f, 0.006122005637735128f, 0.006618829909712076f, 0.004598835948854685f, 0.0011619431897997856f, 0.003953698091208935f, 0.002283641370013356f, 0.005016769282519817f, 0.006001177243888378f, 0.0032753944396972656f, 0.0049770791083574295f, 0.0020989691838622093f, 0.005003744270652533f, 0.00326438806951046f, 0.003924403805285692f, 0.0008321739151142538f, 0.004113285336643457f, 0.004218562971800566f, 0.0023415004834532738f, 0.001444341498427093f, 0.002247904660180211f, 0.0016050200210884213f, 0.004625930450856686f, 0.0029857875779271126f, 0.014283763244748116f, 0.002304328139871359f, 0.003760767634958029f);
static const ai_u16 conv2d_12_t_out_0_shape_w_const_u16 = 16;
static const ai_u16 conv2d_12_t_out_0_shape_h_const_u16 = 16;

static const ai_u16 conv2d_13_t_in_0_shape_w_const_u16 = 16;
static const ai_u16 conv2d_13_t_in_0_shape_h_const_u16 = 16;
static const ai_u16 conv2d_13_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_13_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_13_t_in_0_shape_ch_const_u16 = 48;
static const ai_u16 conv2d_13_t_out_0_shape_ch_const_u16 = 16;
static const ai_i8 conv2d_13_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_13_t_out_0_fmt_zero_const_s8 = 0;
static const ai_float conv2d_13_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_13_t_out_0_fmt_scale_const_f32 = 0.29803141951560974f;
static const ai_float conv2d_13_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.03443755581974983f, 0.029835017397999763f, 0.03763090446591377f, 0.03934776410460472f, 0.035458050668239594f, 0.05154938995838165f, 0.0399877168238163f, 0.039672501385211945f, 0.024239331483840942f, 0.028379781171679497f, 0.03657436743378639f, 0.043294794857501984f, 0.03428686782717705f, 0.037004660815000534f, 0.019969260320067406f, 0.027326496317982674f);
static const ai_layer_format_type conv2d_13_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_u16 conv2d_14_t_in_0_shape_w_const_u16 = 16;
static const ai_u16 conv2d_14_t_in_0_shape_h_const_u16 = 16;
static const ai_u16 conv2d_14_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_14_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_14_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 conv2d_14_t_out_0_shape_ch_const_u16 = 96;
static const ai_i8 conv2d_14_t_in_0_fmt_zero_const_s8 = 0;
static const ai_i8 conv2d_14_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_14_t_in_0_fmt_scale_const_f32 = 0.29803141951560974f;
static const ai_float conv2d_14_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_14_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0010162737453356385f, 0.0029664435423910618f, 0.0008732352289371192f, 0.001767966547049582f, 0.0008188571664504707f, 0.0007114830077625811f, 0.002015496836975217f, 0.0015792850172147155f, 0.0007483321824111044f, 0.0006354820798151195f, 0.0015571698313578963f, 0.0004559821973089129f, 0.0014610118232667446f, 0.0011121483985334635f, 0.0014945251168683171f, 0.0008060400723479688f, 0.001305935438722372f, 0.0015555054415017366f, 0.0016210557660087943f, 0.0010137626668438315f, 0.00037479042657651007f, 0.0007762496825307608f, 0.0005115527892485261f, 0.000848126714117825f, 0.000408238178351894f, 0.0019635262433439493f, 0.0010795268462970853f, 0.0015729888109490275f, 0.0020375470630824566f, 0.0011186012998223305f, 0.001122362446039915f, 0.0013030122499912977f, 0.0009140998590737581f, 0.0004171431064605713f, 0.0007464355439879f, 0.0009978270391002297f, 0.0009237845079042017f, 0.0010409350506961346f, 0.000785124662797898f, 0.0010463724611327052f, 0.0014663387555629015f, 0.0005393149913288653f, 0.0007879461045376956f, 0.0006936541758477688f, 0.0009585056686773896f, 0.00048795880866236985f, 0.0003281448152847588f, 0.0018804752035066485f, 0.0003971307596657425f, 0.0010993892792612314f, 0.0014611847000196576f, 0.0007790697272866964f, 0.0008490614127367735f, 0.00034332158975303173f, 0.0011636706767603755f, 0.0009202170185744762f, 0.0009202077053487301f, 0.0013423647033050656f, 0.0041366321966052055f, 0.0006758786621503532f, 0.0007822082261554897f, 0.0009478482534177601f, 0.0008838344365358353f, 0.0012735576601698995f, 0.0005343748489394784f, 0.00266256183385849f, 0.0009097380097955465f, 0.0004780329472851008f, 0.0008213183027692139f, 0.000457324436865747f, 0.0008082857821136713f, 0.0007171743782237172f, 0.0006025762413628399f, 0.00042960038990713656f, 0.0010385502828285098f, 0.0010934536112472415f, 0.0004564168630167842f, 0.0016249320469796658f, 0.0006242306553758681f, 0.0010748625500127673f, 0.00044701359001919627f, 0.0007228536996990442f, 0.0005758388433605433f, 0.0015667262487113476f, 0.0018054465763270855f, 0.0007489986601285636f, 0.0010136731434613466f, 0.0006008682539686561f, 0.0010194632923230529f, 0.0009971791878342628f, 0.004384189378470182f, 0.002340341452509165f, 0.0007048010593280196f, 0.0018012786749750376f, 0.0006570964469574392f, 0.0008630469674244523f);
static const ai_layer_format_type conv2d_14_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_i8 conv2d_15_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 conv2d_15_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 conv2d_15_pad_before_t_in_0_shape_h_const_u32 = 16;

static const ai_u16 conv2d_15_t_in_0_shape_w_const_u16 = 18;
static const ai_u16 conv2d_15_t_in_0_shape_h_const_u16 = 18;
static const ai_u16 conv2d_15_t_in_0_shape_ch_const_u16 = 96;
static const ai_u16 conv2d_15_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_15_l_stride_0_const_u16 = 1;
static const ai_i8 conv2d_15_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_15_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_15_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_15_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_15_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.007762870751321316f, 0.013328425586223602f, 0.008728258311748505f, 0.011465253308415413f, 0.007084859535098076f, 0.0050886389799416065f, 0.010489416308701038f, 0.011382410302758217f, 0.00588560476899147f, 0.011089668609201908f, 0.0017129319021478295f, 0.02474098838865757f, 0.005206048022955656f, 0.026687754318118095f, 0.010327610187232494f, 0.010233127512037754f, 0.0074376752600073814f, 0.00814130250364542f, 0.018027838319540024f, 0.006599274463951588f, 0.017145749181509018f, 0.013450569473206997f, 0.01054032426327467f, 0.00616350956261158f, 0.055195290595293045f, 0.004230308346450329f, 0.0077001661993563175f, 0.0067642261274158955f, 0.0017777810571715236f, 0.018467199057340622f, 0.007857504300773144f, 0.009901592507958412f, 0.003757554106414318f, 0.01733035035431385f, 0.011655839160084724f, 0.017604978755116463f, 0.007098576053977013f, 0.013899528421461582f, 0.005975336767733097f, 0.001952475868165493f, 0.004789021331816912f, 0.014934114180505276f, 0.011017515324056149f, 0.01097116619348526f, 0.006032606586813927f, 0.014911748468875885f, 0.018396111205220222f, 0.005687673110514879f, 0.022197743877768517f, 0.030169088393449783f, 0.004847953096032143f, 0.00673460029065609f, 0.008918892592191696f, 0.011748306453227997f, 0.010519050993025303f, 0.0043118540197610855f, 0.005196289159357548f, 0.004470765590667725f, 0.007011083420366049f, 0.007131258957087994f, 0.007052444852888584f, 0.004099512007087469f, 0.004306276328861713f, 0.004252347629517317f, 0.0062219444662332535f, 0.003304914338514209f, 0.005839036777615547f, 0.007963799871504307f, 0.014609905891120434f, 0.020361721515655518f, 0.00793894100934267f, 0.006981309503316879f, 0.006859078537672758f, 0.014963028021156788f, 0.0052134795114398f, 0.00915705319494009f, 0.010986527428030968f, 0.0053376262076199055f, 0.00829761940985918f, 0.00476465467363596f, 0.007154989056289196f, 0.014116163365542889f, 0.0060634370893239975f, 0.046127382665872574f, 0.011527697555720806f, 0.004760188516229391f, 0.01202340517193079f, 0.007957888767123222f, 0.004803525749593973f, 0.005545887164771557f, 0.0013205395080149174f, 0.008541093207895756f, 0.00710340915247798f, 0.006189015693962574f, 0.004237677436321974f, 0.008943460881710052f);
static const ai_u16 conv2d_15_t_out_0_shape_w_const_u16 = 16;
static const ai_u16 conv2d_15_t_out_0_shape_h_const_u16 = 16;

static const ai_u16 conv2d_16_t_in_0_shape_w_const_u16 = 16;
static const ai_u16 conv2d_16_t_in_0_shape_h_const_u16 = 16;
static const ai_u16 conv2d_16_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_16_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_16_t_in_0_shape_ch_const_u16 = 96;
static const ai_u16 conv2d_16_t_out_0_shape_ch_const_u16 = 16;
static const ai_i8 conv2d_16_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_16_t_out_0_fmt_zero_const_s8 = 5;
static const ai_float conv2d_16_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_16_t_out_0_fmt_scale_const_f32 = 0.3349054753780365f;
static const ai_float conv2d_16_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.017254767939448357f, 0.01701357774436474f, 0.021070118993520737f, 0.012501359917223454f, 0.01765737496316433f, 0.023372920230031013f, 0.014737548306584358f, 0.02664966695010662f, 0.012865865603089333f, 0.013439172878861427f, 0.01385547872632742f, 0.00829460471868515f, 0.009885718114674091f, 0.010864201933145523f, 0.00960324052721262f, 0.009659125469624996f);
static const ai_layer_format_type conv2d_16_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 conv2d_18_t_in_0_shape_w_const_u16 = 16;
static const ai_u16 conv2d_18_t_in_0_shape_h_const_u16 = 16;
static const ai_u16 conv2d_18_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_18_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_18_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 conv2d_18_t_out_0_shape_ch_const_u16 = 96;
static const ai_i8 conv2d_18_t_in_0_fmt_zero_const_s8 = 5;
static const ai_i8 conv2d_18_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_18_t_in_0_fmt_scale_const_f32 = 0.3349054753780365f;
static const ai_float conv2d_18_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_18_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.00026084340061061084f, 0.00030048922053538263f, 0.00048167697968892753f, 0.000381648336770013f, 0.0007306633051484823f, 0.0005341155338101089f, 0.0005639364826492965f, 0.001154723926447332f, 0.0008244728087447584f, 0.00025739939883351326f, 0.0019127874402329326f, 0.0006398605182766914f, 0.0007554655312560499f, 0.0005338318296708167f, 0.0027368981391191483f, 0.00034115262678824365f, 0.0005420513916760683f, 0.00041566023719497025f, 0.000632894691079855f, 0.0006360216648317873f, 0.001817470882087946f, 0.00039469028706662357f, 0.0003212439769413322f, 0.00036820321111008525f, 0.0006117462180554867f, 0.000535125145688653f, 0.0006422249134629965f, 0.0003185788227710873f, 0.0005746716633439064f, 0.0016148019349202514f, 0.0007661085692234337f, 0.0005470506730489433f, 0.0005460830288939178f, 0.0005640836898237467f, 0.0010260473936796188f, 0.0005439479136839509f, 0.001160222920589149f, 0.0008951208437792957f, 0.00021038760314695537f, 0.0007745458278805017f, 0.0008264235220849514f, 0.0007714227540418506f, 0.00033657628227956593f, 0.0006431695655919611f, 0.0018003071891143918f, 0.0006817933754064143f, 0.0007720824214629829f, 0.0011647057253867388f, 0.0006697488133795559f, 0.00048532223445363343f, 0.0005378909991122782f, 0.0009120817994698882f, 0.0008751248242333531f, 0.0025556895416229963f, 0.0007448925753124058f, 0.0007183977286331356f, 0.0006057547288946807f, 0.0003239140787627548f, 0.0014626823831349611f, 0.0004553562612272799f, 0.0006637676269747317f, 0.00026493743644095957f, 0.00035853387089446187f, 0.00044955569319427013f, 0.0014290662948042154f, 0.0011239692103117704f, 0.0022747148759663105f, 0.0005815755575895309f, 0.0007421464542858303f, 0.001400500419549644f, 0.0010720377322286367f, 0.0005793694872409105f, 0.0010480997152626514f, 0.0012329781893640757f, 0.0005461497348733246f, 0.0008061245898716152f, 0.0009482466266490519f, 0.00066617620177567f, 0.0011900373501703143f, 0.000806343974545598f, 0.00042260054033249617f, 0.0005872935871593654f, 0.0018208092078566551f, 0.0005387493292801082f, 0.0006730371969752014f, 0.0008030450553633273f, 0.0005289423279464245f, 0.0005335368332453072f, 0.0007678091642446816f, 0.0010289587080478668f, 0.000592404801864177f, 0.0007840503240004182f, 0.0015873099910095334f, 0.0007038527401164174f, 0.0014114526566118002f, 0.0008693183190189302f);
static const ai_layer_format_type conv2d_18_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_i8 conv2d_19_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 conv2d_19_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 conv2d_19_pad_before_t_in_0_shape_h_const_u32 = 16;

static const ai_u16 conv2d_19_t_in_0_shape_w_const_u16 = 18;
static const ai_u16 conv2d_19_t_in_0_shape_h_const_u16 = 18;
static const ai_u16 conv2d_19_t_in_0_shape_ch_const_u16 = 96;
static const ai_u16 conv2d_19_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_19_l_stride_0_const_u16 = 1;
static const ai_i8 conv2d_19_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_19_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_19_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_19_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_19_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.02400190755724907f, 0.013135924935340881f, 0.0066948276944458485f, 0.009747509844601154f, 0.0196759644895792f, 0.010168026201426983f, 0.007955871522426605f, 0.002731945598497987f, 0.004572100937366486f, 0.014729511924088001f, 0.005078259389847517f, 0.014650456607341766f, 0.005044566001743078f, 0.010004992596805096f, 0.0009196224855259061f, 0.012161275371909142f, 0.011327057145535946f, 0.008476772345602512f, 0.004715967923402786f, 0.007274257019162178f, 0.012109791859984398f, 0.0224367193877697f, 0.009727012366056442f, 0.006505997851490974f, 0.0061027235351502895f, 0.011564296670258045f, 0.005962518975138664f, 0.025429604575037956f, 0.01683705858886242f, 0.01052701473236084f, 0.0063219815492630005f, 0.008922675624489784f, 0.01800384931266308f, 0.010747366584837437f, 0.0033295259345322847f, 0.006250045727938414f, 0.0033347811549901962f, 0.013295711018145084f, 0.011298686265945435f, 0.00827973335981369f, 0.011235787533223629f, 0.002411814173683524f, 0.008382320404052734f, 0.011974072083830833f, 0.0022066673263907433f, 0.005201139021664858f, 0.011795315891504288f, 0.007218632847070694f, 0.014007148332893848f, 0.005811585113406181f, 0.0021443995647132397f, 0.017434701323509216f, 0.004978215787559748f, 0.0030971821397542953f, 0.006418777164071798f, 0.0039834631606936455f, 0.005849156063050032f, 0.005203488748520613f, 0.0016133495373651385f, 0.01087619923055172f, 0.015666775405406952f, 0.040122952312231064f, 0.01187303476035595f, 0.006400808226317167f, 0.005361572839319706f, 0.004521073307842016f, 0.0022565824910998344f, 0.010664261877536774f, 0.007114659063518047f, 0.00224397168494761f, 0.0070768254809081554f, 0.007407618220895529f, 0.0017567439936101437f, 0.009046260267496109f, 0.011978121474385262f, 0.011732393875718117f, 0.009134394116699696f, 0.00589791452512145f, 0.004948048386722803f, 0.004892411641776562f, 0.010797480121254921f, 0.016850998625159264f, 0.0015424239682033658f, 0.02601717971265316f, 0.0032904648687690496f, 0.004834375809878111f, 0.013598152436316013f, 0.006593174301087856f, 0.006742959842085838f, 0.005390811711549759f, 0.01357237994670868f, 0.007985827513039112f, 0.008020548149943352f, 0.007959234528243542f, 0.0019169426523149014f, 0.006166733801364899f);
static const ai_u16 conv2d_19_t_out_0_shape_w_const_u16 = 16;
static const ai_u16 conv2d_19_t_out_0_shape_h_const_u16 = 16;

static const ai_u16 conv2d_20_t_in_0_shape_w_const_u16 = 16;
static const ai_u16 conv2d_20_t_in_0_shape_h_const_u16 = 16;
static const ai_u16 conv2d_20_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_20_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_20_t_in_0_shape_ch_const_u16 = 96;
static const ai_u16 conv2d_20_t_out_0_shape_ch_const_u16 = 16;
static const ai_i8 conv2d_20_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_20_t_out_0_fmt_zero_const_s8 = -13;
static const ai_float conv2d_20_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_20_t_out_0_fmt_scale_const_f32 = 0.38780033588409424f;
static const ai_float conv2d_20_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.009663505479693413f, 0.0195898599922657f, 0.03513135761022568f, 0.0053163873963057995f, 0.01982731558382511f, 0.019584303721785545f, 0.007970995269715786f, 0.02290934883058071f, 0.013894198462367058f, 0.01820821315050125f, 0.010290296748280525f, 0.008139285258948803f, 0.01355320867151022f, 0.013998416252434254f, 0.009121579118072987f, 0.008299061097204685f);
static const ai_layer_format_type conv2d_20_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 conv2d_22_t_in_0_shape_w_const_u16 = 16;
static const ai_u16 conv2d_22_t_in_0_shape_h_const_u16 = 16;
static const ai_u16 conv2d_22_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_22_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_22_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 conv2d_22_t_out_0_shape_ch_const_u16 = 96;
static const ai_i8 conv2d_22_t_in_0_fmt_zero_const_s8 = -13;
static const ai_i8 conv2d_22_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_22_t_in_0_fmt_scale_const_f32 = 0.38780033588409424f;
static const ai_float conv2d_22_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_22_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.00041123625123873353f, 0.0015513107646256685f, 0.00111137586645782f, 0.0013994473265483975f, 0.00048730956041254103f, 0.0006391665083356202f, 0.0003636158362496644f, 0.0006033367244526744f, 0.0006783268763683736f, 0.0006883165333420038f, 0.0005916160880587995f, 0.000426749000325799f, 0.0013148060534149408f, 0.0008258695597760379f, 0.0018080638255923986f, 0.001252767164260149f, 0.0009788094321265817f, 0.0005043858545832336f, 0.0010693254880607128f, 0.0003822668513748795f, 0.0002758959890343249f, 0.0006942971958778799f, 0.0006799197872169316f, 0.00040194817120209336f, 0.000932197377551347f, 0.00043293775524944067f, 0.0005572756635956466f, 0.00048813363537192345f, 5.412245445768349e-05f, 0.0005790695431642234f, 0.001450180890969932f, 0.0008941214182414114f, 0.0008433503098785877f, 0.0008553880616091192f, 0.00048560582217760384f, 0.0009443542221561074f, 0.001037282170727849f, 0.0006556555745191872f, 0.0006695279735140502f, 0.0015549692325294018f, 0.00042434188071638346f, 0.0009742433903738856f, 0.0011913832277059555f, 0.001251006149686873f, 0.000606559740845114f, 0.0005764784291386604f, 0.0005625189514830709f, 0.00037783506559208035f, 0.0013812872348353267f, 0.0005983267910778522f, 0.0005882024415768683f, 0.001154795871116221f, 0.004363485146313906f, 0.001277006114833057f, 0.0006314222700893879f, 0.0005342630320228636f, 0.0018175313016399741f, 0.00036894541699439287f, 0.00040142732905223966f, 0.001170691684819758f, 0.0010673817014321685f, 0.0011828296119347215f, 0.0011311209527775645f, 0.0010072281584143639f, 0.0004900287603959441f, 0.00028069529798813164f, 0.0009878044947981834f, 0.002267297124490142f, 0.0008282200433313847f, 0.0006293837795965374f, 0.001137033337727189f, 0.0013246225425973535f, 0.0006584048387594521f, 0.0004590564058162272f, 0.000449689308879897f, 0.0010725604370236397f, 0.000804235867690295f, 0.0006027822964824736f, 0.0009112976840697229f, 0.00025701444246806204f, 0.0004530267615336925f, 0.0006809410988353193f, 0.0005541619029827416f, 0.0011644853511825204f, 0.0017899039667099714f, 0.000633115996606648f, 0.0004344526387285441f, 0.0005374255124479532f, 0.00037003966281190515f, 0.0005767710390500724f, 0.0005493176868185401f, 0.0012930232333019376f, 0.0011394965695217252f, 0.0010580545058473945f, 0.0010131864110007882f, 0.0008783223456703126f);
static const ai_layer_format_type conv2d_22_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_i8 conv2d_23_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 conv2d_23_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 conv2d_23_pad_before_t_in_0_shape_h_const_u32 = 16;

static const ai_u16 conv2d_23_t_in_0_shape_w_const_u16 = 18;
static const ai_u16 conv2d_23_t_in_0_shape_h_const_u16 = 18;
static const ai_u16 conv2d_23_t_in_0_shape_ch_const_u16 = 96;
static const ai_u16 conv2d_23_l_stride_1_const_u16 = 2;
static const ai_u16 conv2d_23_l_stride_0_const_u16 = 2;
static const ai_i8 conv2d_23_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_23_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_23_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_23_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_23_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.002549567958340049f, 0.00139905558899045f, 0.00391472689807415f, 0.002421654760837555f, 0.006237370427697897f, 0.002201283583417535f, 0.004423544276505709f, 0.0070763444527983665f, 0.005370126571506262f, 0.002563230926170945f, 0.004471141379326582f, 0.004384256899356842f, 0.0015143989585340023f, 0.003083754563704133f, 0.002230045385658741f, 0.0027135214768350124f, 0.003252472961321473f, 0.004433130379766226f, 0.0022218506783246994f, 0.002675699070096016f, 0.006999681703746319f, 0.00238110963255167f, 0.0027040746062994003f, 0.01165329571813345f, 0.00293497322127223f, 0.003503142623230815f, 0.003852690337225795f, 0.004196421708911657f, 0.030263785272836685f, 0.0036434391513466835f, 0.0010646408190950751f, 0.0016783184837549925f, 0.006877928506582975f, 0.0017205155454576015f, 0.003247082931920886f, 0.00322857778519392f, 0.0013796697603538632f, 0.002433507004752755f, 0.004700516350567341f, 0.0013072651345282793f, 0.0037743146531283855f, 0.0011215847916901112f, 0.0023280614987015724f, 0.0017205195035785437f, 0.003921460825949907f, 0.0022484459914267063f, 0.0034002342727035284f, 0.0022939336486160755f, 0.002096648560836911f, 0.0031144076492637396f, 0.0035372350830584764f, 0.00288348994217813f, 0.0013647687155753374f, 0.004116357304155827f, 0.004378734156489372f, 0.001694042468443513f, 0.0013611996546387672f, 0.003489976515993476f, 0.00236209062859416f, 0.00232002348639071f, 0.002062917221337557f, 0.002316222060471773f, 0.001031917636282742f, 0.0034209738951176405f, 0.003620811505243182f, 0.009431092999875546f, 0.0013082735240459442f, 0.0012101450702175498f, 0.0018150025280192494f, 0.0022263797000050545f, 0.004269593860954046f, 0.0027808474842458963f, 0.002886551432311535f, 0.002774882595986128f, 0.00510899955406785f, 0.0023619697894901037f, 0.0026614118833094835f, 0.0020981875713914633f, 0.0023762129712849855f, 0.0045297956094145775f, 0.0031880324240773916f, 0.0028763669542968273f, 0.004489317536354065f, 0.0009817880345508456f, 0.003096304601058364f, 0.0028759657870978117f, 0.002155148424208164f, 0.0037949238903820515f, 0.003116184612736106f, 0.002814670093357563f, 0.01088736578822136f, 0.0036487476900219917f, 0.0015441726427525282f, 0.0027573518455028534f, 0.0020679819863289595f, 0.00270692165941f);
static const ai_u16 conv2d_23_t_out_0_shape_w_const_u16 = 8;
static const ai_u16 conv2d_23_t_out_0_shape_h_const_u16 = 8;

static const ai_u16 conv2d_24_t_in_0_shape_w_const_u16 = 8;
static const ai_u16 conv2d_24_t_in_0_shape_h_const_u16 = 8;
static const ai_u16 conv2d_24_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_24_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_24_t_in_0_shape_ch_const_u16 = 96;
static const ai_u16 conv2d_24_t_out_0_shape_ch_const_u16 = 24;
static const ai_i8 conv2d_24_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_24_t_out_0_fmt_zero_const_s8 = 12;
static const ai_float conv2d_24_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_24_t_out_0_fmt_scale_const_f32 = 0.2426532357931137f;
static const ai_float conv2d_24_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.018454307690262794f, 0.013570522889494896f, 0.013568855822086334f, 0.022837383672595024f, 0.01644561067223549f, 0.02001037634909153f, 0.014993350021541119f, 0.01602194644510746f, 0.015028106980025768f, 0.021361196413636208f, 0.022845715284347534f, 0.01511739008128643f, 0.023612016811966896f, 0.02557484246790409f, 0.021969806402921677f, 0.023671340197324753f, 0.014808681793510914f, 0.018468881025910378f, 0.01556017342954874f, 0.015157250687479973f, 0.014209887944161892f, 0.0173113401979208f, 0.02046327479183674f, 0.01601163111627102f);
static const ai_layer_format_type conv2d_24_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_u16 conv2d_25_t_in_0_shape_w_const_u16 = 8;
static const ai_u16 conv2d_25_t_in_0_shape_h_const_u16 = 8;
static const ai_u16 conv2d_25_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_25_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_25_t_in_0_shape_ch_const_u16 = 24;
static const ai_u16 conv2d_25_t_out_0_shape_ch_const_u16 = 144;
static const ai_i8 conv2d_25_t_in_0_fmt_zero_const_s8 = 12;
static const ai_i8 conv2d_25_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_25_t_in_0_fmt_scale_const_f32 = 0.2426532357931137f;
static const ai_float conv2d_25_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_25_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0007105523836798966f, 0.0007541501545347273f, 0.0008073350763879716f, 0.0005036639631725848f, 0.0006440632860176265f, 0.0004897373728454113f, 0.0006464743637479842f, 0.0010705814929679036f, 0.001014089211821556f, 0.0006492444663308561f, 0.0004638964310288429f, 0.000626000517513603f, 0.0009791654301807284f, 0.0021062202285975218f, 0.0003585437953006476f, 0.0002677920274436474f, 0.00042552160448394716f, 0.00034734030487015843f, 0.0005494524375535548f, 0.00043194284080527723f, 0.0005962115828879178f, 0.0004954428877681494f, 0.0007190278265625238f, 0.0012775094946846366f, 0.00034183330717496574f, 0.000723912613466382f, 0.0009964981582015753f, 0.0010383009212091565f, 0.0009948068764060736f, 0.0012224772945046425f, 0.0006150217959657311f, 0.0003499222220852971f, 9.672240412328392e-05f, 0.00041619795956648886f, 0.0009519330342300236f, 0.0009083267650566995f, 0.0009249958675354719f, 0.0002459823153913021f, 0.0006696826312690973f, 0.0003608840925153345f, 0.0003995107254013419f, 0.0006811351631768048f, 0.0006431224755942822f, 0.0002327482361579314f, 0.0005153430392965674f, 0.0010906917741522193f, 0.0003899318107869476f, 0.0008697187877260149f, 0.001014271634630859f, 0.0004999815719202161f, 0.0005902127013541758f, 0.00068667036248371f, 0.0006889419164508581f, 0.0007219024701043963f, 0.0008884998969733715f, 0.0003469188523013145f, 0.0005123062874190509f, 0.00047911799629218876f, 0.0006288212607614696f, 0.00015963423356879503f, 0.00045602384489029646f, 8.111229544738308e-05f, 0.0012366385199129581f, 0.0008004889241419733f, 0.0005162339075468481f, 0.0006961744511500001f, 0.0006998226745054126f, 0.0011905921855941415f, 0.0004380815662443638f, 0.0005610231892205775f, 0.0009293724433518946f, 0.0006545937503688037f, 0.0008676649886183441f, 0.00037012790562584996f, 0.0010850983671844006f, 0.0017955132061615586f, 0.0007051032735034823f, 0.0009264251566492021f, 0.00041673966916278005f, 0.0008427357533946633f, 0.00039254376315511763f, 0.0009235924226231873f, 0.000505247269757092f, 0.0008742987411096692f, 0.0021599296014755964f, 0.0009686759440228343f, 0.00024108827346935868f, 0.0006262906244955957f, 0.0008161510922946036f, 0.0009847808396443725f, 0.0004604354326147586f, 0.0010715187527239323f, 0.0014779954217374325f, 0.00047540481318719685f, 0.0008080452098511159f, 0.0005959536647424102f, 0.0008058393141254783f, 0.0008218797738663852f, 0.0009109779493883252f, 0.0006272279424592853f, 0.00032033445313572884f, 0.0005446605500765145f, 0.0007656757952645421f, 0.0006873482489027083f, 0.00017446039419155568f, 0.0003206153050996363f, 0.0005456926883198321f, 0.0003206280234735459f, 0.0008735224837437272f, 0.0004851129779126495f, 0.0002829308796208352f, 0.0009223738452419639f, 0.0004888072144240141f, 0.0009089059894904494f, 0.00053045415552333f, 0.0010398890590295196f, 0.00017303448112215847f, 0.0008099762490019202f, 0.0007788133225403726f, 0.0007573424372822046f, 0.0011380707146599889f, 0.0007718095439486206f, 0.0006688992725685239f, 0.00038210838101804256f, 0.00025096486206166446f, 0.000941393431276083f, 0.0006653663003817201f, 0.0005358748021535575f, 0.00042201438918709755f, 0.0007358721923083067f, 0.00015760204405523837f, 0.0005824893596582115f, 0.0006157092284411192f, 0.0006471298402175307f, 0.0006409670459106565f, 0.00048815630725584924f, 0.0006511576357297599f, 0.0004655986849684268f, 0.0003909188380930573f, 0.00018106844800058752f, 0.0008046372677199543f, 0.00044476287439465523f, 0.0006180995842441916f, 0.0003297535586170852f);
static const ai_layer_format_type conv2d_25_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_i8 conv2d_26_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 conv2d_26_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 conv2d_26_pad_before_t_in_0_shape_h_const_u32 = 8;

static const ai_u16 conv2d_26_t_in_0_shape_w_const_u16 = 10;
static const ai_u16 conv2d_26_t_in_0_shape_h_const_u16 = 10;
static const ai_u16 conv2d_26_t_in_0_shape_ch_const_u16 = 144;
static const ai_u16 conv2d_26_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_26_l_stride_0_const_u16 = 1;
static const ai_i8 conv2d_26_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_26_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_26_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_26_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_26_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.007794192060828209f, 0.008436715230345726f, 0.00467414828017354f, 0.027584146708250046f, 0.015171268954873085f, 0.019524361938238144f, 0.007514600642025471f, 0.0042030224576592445f, 0.010719319805502892f, 0.00869869627058506f, 0.004910783376544714f, 0.020456800237298012f, 0.011660614050924778f, 0.004069725517183542f, 0.01728016696870327f, 0.013427730649709702f, 0.009895650669932365f, 0.012598820962011814f, 0.004943403881043196f, 0.013434343039989471f, 0.015250741504132748f, 0.007333256304264069f, 0.006262002978473902f, 0.009920166805386543f, 0.01602001301944256f, 0.005290877539664507f, 0.009971018880605698f, 0.009534835815429688f, 0.005103416740894318f, 0.0028098151087760925f, 0.021626977249979973f, 0.021349124610424042f, 0.05592174828052521f, 0.00970417633652687f, 0.003975023049861193f, 0.008410719223320484f, 0.01199012529104948f, 0.009249706752598286f, 0.005033038090914488f, 0.01320382859557867f, 0.01804267056286335f, 0.008796673268079758f, 0.006968972738832235f, 0.02472248487174511f, 0.012053260579705238f, 0.014210216701030731f, 0.014095737598836422f, 0.006157392170280218f, 0.008985842578113079f, 0.010254396125674248f, 0.017573583871126175f, 0.016651062294840813f, 0.019201509654521942f, 0.011088121682405472f, 0.01208265870809555f, 0.01675656996667385f, 0.008177869953215122f, 0.015244974754750729f, 0.011344206519424915f, 0.05785655975341797f, 0.020642386749386787f, 0.019516615197062492f, 0.009468219242990017f, 0.00837445817887783f, 0.05365965887904167f, 0.007877930998802185f, 0.016846904531121254f, 0.023826222866773605f, 0.025784343481063843f, 0.013264778070151806f, 0.00542928883805871f, 0.00462646409869194f, 0.007594220340251923f, 0.012098669074475765f, 0.0073529137298464775f, 0.00532534159719944f, 0.006455735769122839f, 0.021144526079297066f, 0.008242162875831127f, 0.003630902851000428f, 0.005218637175858021f, 0.014332082122564316f, 0.009062957018613815f, 0.012297394685447216f, 0.00272059952840209f, 0.0019696850795298815f, 0.017064552754163742f, 0.007630669511854649f, 0.006040857173502445f, 0.0065182712860405445f, 0.0073560732416808605f, 0.004364428576081991f, 0.0027568426448851824f, 0.010340668261051178f, 0.00650552986189723f, 0.007252441253513098f, 0.00831511989235878f, 0.006579551845788956f, 0.005934529006481171f, 0.005398952402174473f, 0.008096923120319843f, 0.009931100532412529f, 0.010707049630582333f, 0.009174402803182602f, 0.0374639630317688f, 0.020387498661875725f, 0.008333971723914146f, 0.016595803201198578f, 0.006922981236129999f, 0.009809747338294983f, 0.011950228363275528f, 0.0038515026681125164f, 0.008105049841105938f, 0.007164849899709225f, 0.006777268368750811f, 0.01597822643816471f, 0.012275457382202148f, 0.01758190430700779f, 0.0028997568879276514f, 0.006483474280685186f, 0.014085335657000542f, 0.007623522076755762f, 0.0032207402400672436f, 0.04259425029158592f, 0.014886711724102497f, 0.01150562334805727f, 0.0074521140195429325f, 0.0039332276210188866f, 0.011654664762318134f, 0.009093795903027058f, 0.04330677166581154f, 0.018253423273563385f, 0.016185130923986435f, 0.007243943400681019f, 0.005534629803150892f, 0.008214649744331837f, 0.006545407697558403f, 0.005437979940325022f, 0.016626380383968353f, 0.011450531892478466f, 0.0073425741866230965f, 0.009270400740206242f, 0.014569575898349285f, 0.015914980322122574f);
static const ai_u16 conv2d_26_t_out_0_shape_w_const_u16 = 8;
static const ai_u16 conv2d_26_t_out_0_shape_h_const_u16 = 8;

static const ai_u16 conv2d_27_t_in_0_shape_w_const_u16 = 8;
static const ai_u16 conv2d_27_t_in_0_shape_h_const_u16 = 8;
static const ai_u16 conv2d_27_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_27_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_27_t_in_0_shape_ch_const_u16 = 144;
static const ai_u16 conv2d_27_t_out_0_shape_ch_const_u16 = 24;
static const ai_i8 conv2d_27_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_27_t_out_0_fmt_zero_const_s8 = 6;
static const ai_float conv2d_27_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_27_t_out_0_fmt_scale_const_f32 = 0.24610859155654907f;
static const ai_float conv2d_27_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.004155958537012339f, 0.006485664285719395f, 0.014786618761718273f, 0.00779110100120306f, 0.005144608672708273f, 0.011005230247974396f, 0.005456649698317051f, 0.030735507607460022f, 0.005491230171173811f, 0.009400466457009315f, 0.005728844087570906f, 0.0033955485559999943f, 0.005068729631602764f, 0.007713088765740395f, 0.002997956471517682f, 0.005016022361814976f, 0.005636023357510567f, 0.007913242094218731f, 0.0039779325015842915f, 0.019588572904467583f, 0.006034798454493284f, 0.004149922169744968f, 0.01076267659664154f, 0.003702575108036399f);
static const ai_layer_format_type conv2d_27_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 conv2d_29_t_in_0_shape_w_const_u16 = 8;
static const ai_u16 conv2d_29_t_in_0_shape_h_const_u16 = 8;
static const ai_u16 conv2d_29_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_29_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_29_t_in_0_shape_ch_const_u16 = 24;
static const ai_u16 conv2d_29_t_out_0_shape_ch_const_u16 = 144;
static const ai_i8 conv2d_29_t_in_0_fmt_zero_const_s8 = 6;
static const ai_i8 conv2d_29_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_29_t_in_0_fmt_scale_const_f32 = 0.24610859155654907f;
static const ai_float conv2d_29_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_29_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.00023752645938657224f, 0.0007522468804381788f, 0.0008905639406293631f, 0.0008215337875299156f, 0.0007976029883138835f, 0.0001698947889963165f, 0.0006374260410666466f, 0.00019268241885583848f, 0.0005132902879267931f, 0.00014921645924914628f, 0.0004662801220547408f, 0.00025795691180974245f, 0.0007241216953843832f, 0.0003350060142111033f, 0.0006195154855959117f, 0.0003555833827704191f, 0.00033124530455097556f, 0.00117817847058177f, 0.00031311428756453097f, 0.0003085553471464664f, 0.00046482915058732033f, 0.0005510219489224255f, 0.0006358686950989068f, 0.0006456256378442049f, 0.0005530021153390408f, 0.000504926429130137f, 0.00040405092295259237f, 0.0005117089604027569f, 0.00035524950362741947f, 0.0003617105830926448f, 0.0004993894835934043f, 0.0005334385787136853f, 0.0006945712957531214f, 0.0005301524652168155f, 0.0008038352825678885f, 0.0008734627044759691f, 0.0007410816033370793f, 0.0006780405528843403f, 0.000274274789262563f, 0.0005416207714006305f, 0.0005905079306103289f, 0.0005039166426286101f, 0.0007204936118796468f, 0.0001958064822247252f, 0.0006882371380925179f, 8.318177424371243e-05f, 0.00048542904551140964f, 0.0003140552435070276f, 0.0006416952819563448f, 0.0005377862835302949f, 0.000691408000420779f, 0.000588564551435411f, 0.0006006177281960845f, 0.00017420656513422728f, 0.0005714066792279482f, 0.0007214085781015456f, 0.0004885373055003583f, 0.0004880581109318882f, 0.00045963129377923906f, 0.0006219004280865192f, 0.00043835933320224285f, 0.00023439903452526778f, 0.0006213454762473702f, 0.0005452249897643924f, 0.0005765511305071414f, 0.0004092252056580037f, 0.00037246313877403736f, 0.0005912281922064722f, 0.0011066652368754148f, 0.0007076006149873137f, 0.0006577011081390083f, 0.0009616670431569219f, 0.0005636602872982621f, 0.0002976534888148308f, 0.0004594514612108469f, 0.0007924995734356344f, 0.0005132514634169638f, 0.0002659260935615748f, 0.0002467904123477638f, 0.0005029825260862708f, 0.00038994450005702674f, 0.0010894116712734103f, 0.00033598707523196936f, 0.00021071365335956216f, 0.0003988003882113844f, 0.000519102206453681f, 0.0006304744747467339f, 0.0008361320942640305f, 0.0005918389651924372f, 0.00033758278004825115f, 0.000449751183623448f, 0.0007590972236357629f, 0.00045772697194479406f, 0.0003767930611502379f, 0.0018023529555648565f, 0.0006478331633843482f, 0.0006528613157570362f, 0.00016821034660097212f, 0.0008035427308641374f, 0.00039963473682291806f, 0.00035328525700606406f, 0.0004898097249679267f, 0.0006312998593784869f, 0.000274202146101743f, 0.0006065061897970736f, 0.0005468913004733622f, 0.0006390802445821464f, 0.0007050288841128349f, 0.00042563219903968275f, 0.0005958523834124207f, 0.0003124887007288635f, 0.0002645201457198709f, 0.0004413124406710267f, 0.0008088533650152385f, 0.0004440530319698155f, 0.0002685567014850676f, 0.0009893279056996107f, 0.00028416054556146264f, 0.0004285433969926089f, 0.00031840972951613367f, 0.0007620221585966647f, 0.0008572004153393209f, 0.00048741017235442996f, 0.0006623787339776754f, 0.00047388955135829747f, 0.00035687797935679555f, 0.0007345956983044744f, 0.000760856841225177f, 0.0005711697158403695f, 0.001090484787710011f, 0.0005146755720488727f, 0.0004992077010683715f, 0.0004401893529575318f, 0.0005706304800696671f, 0.00037115084705874324f, 0.00037321465788409114f, 0.0009851144859567285f, 0.0010521110380068421f, 0.001298959250561893f, 0.0006975276046432555f, 0.000432248052675277f, 0.00023817340843379498f, 0.0003587417595554143f, 0.0005559977726079524f);
static const ai_layer_format_type conv2d_29_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_i8 conv2d_30_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 conv2d_30_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 conv2d_30_pad_before_t_in_0_shape_h_const_u32 = 8;

static const ai_u16 conv2d_30_t_in_0_shape_w_const_u16 = 10;
static const ai_u16 conv2d_30_t_in_0_shape_h_const_u16 = 10;
static const ai_u16 conv2d_30_t_in_0_shape_ch_const_u16 = 144;
static const ai_u16 conv2d_30_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_30_l_stride_0_const_u16 = 1;
static const ai_i8 conv2d_30_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_30_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_30_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_30_t_out_0_fmt_scale_const_f32 = 0.023016922175884247f;
static const ai_float conv2d_30_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.015731118619441986f, 0.010752095840871334f, 0.00435028038918972f, 0.005034246016293764f, 0.0037049732636660337f, 0.02008788473904133f, 0.009324098937213421f, 0.013907336629927158f, 0.00662632891908288f, 0.01590479537844658f, 0.004852066747844219f, 0.022447964176535606f, 0.018659118562936783f, 0.00431732228025794f, 0.007317479234188795f, 0.012635566294193268f, 0.028017565608024597f, 0.014960414730012417f, 0.02337077260017395f, 0.021480588242411613f, 0.02244972623884678f, 0.007826002314686775f, 0.010555589571595192f, 0.01246658619493246f, 0.007321011740714312f, 0.01034014392644167f, 0.013907753862440586f, 0.011864462867379189f, 0.0164662953466177f, 0.006184070836752653f, 0.007937876507639885f, 0.007991574704647064f, 0.007783540524542332f, 0.005465841386467218f, 0.004370920360088348f, 0.008836607448756695f, 0.008456282317638397f, 0.016551002860069275f, 0.017826374620199203f, 0.015143447555601597f, 0.012941068038344383f, 0.009419096633791924f, 0.008783301338553429f, 0.013030857779085636f, 0.005404502619057894f, 0.008333136327564716f, 0.006074761506170034f, 0.012829750776290894f, 0.01271272636950016f, 0.01676774024963379f, 0.006252172403037548f, 0.013886469416320324f, 0.006636447738856077f, 0.021080711856484413f, 0.0035499120131134987f, 0.009094309993088245f, 0.006155255250632763f, 0.021566513925790787f, 0.0059910244308412075f, 0.006449885666370392f, 0.007889652624726295f, 0.010095945559442043f, 0.004127253778278828f, 0.004716179799288511f, 0.022059405222535133f, 0.011144719086587429f, 0.005292279180139303f, 0.005645003169775009f, 0.005751556251198053f, 0.004303032532334328f, 0.0076997168362140656f, 0.004248560406267643f, 0.01390551682561636f, 0.010137252509593964f, 0.006049427203834057f, 0.00308991689234972f, 0.022626040503382683f, 0.006652584299445152f, 0.014319048263132572f, 0.00741293141618371f, 0.010616538114845753f, 0.008754064328968525f, 0.016424402594566345f, 0.009250015020370483f, 0.014637124724686146f, 0.007238361518830061f, 0.008878909982740879f, 0.006836762651801109f, 0.002214512089267373f, 0.011622310616075993f, 0.017898494377732277f, 0.004893120843917131f, 0.006453795358538628f, 0.005812793038785458f, 0.00956627819687128f, 0.0029550199396908283f, 0.009817808866500854f, 0.014850676991045475f, 0.008262314833700657f, 0.009332582354545593f, 0.013964655809104443f, 0.014828674495220184f, 0.07604683935642242f, 0.03232801333069801f, 0.006677483208477497f, 0.0036181407049298286f, 0.008022806607186794f, 0.008104914799332619f, 0.008661852218210697f, 0.009764742106199265f, 0.010649415664374828f, 0.010043317452073097f, 0.01328849047422409f, 0.006745939143002033f, 0.010474931448698044f, 0.012664321810007095f, 0.0053259339183568954f, 0.009631422348320484f, 0.007435587700456381f, 0.02751179225742817f, 0.019760599359869957f, 0.011016004718840122f, 0.006118184421211481f, 0.017050208523869514f, 0.02597883902490139f, 0.018338631838560104f, 0.0035940129309892654f, 0.005758306942880154f, 0.0058397892862558365f, 0.005701291840523481f, 0.009502876549959183f, 0.006709815468639135f, 0.0035920783411711454f, 0.006264890544116497f, 0.009607726708054543f, 0.022382939234375954f, 0.007074645720422268f, 0.007441000547260046f, 0.01658046618103981f, 0.006355264689773321f, 0.009510613046586514f, 0.014616456814110279f, 0.01698189042508602f, 0.006086356937885284f);
static const ai_u16 conv2d_30_t_out_0_shape_w_const_u16 = 8;
static const ai_u16 conv2d_30_t_out_0_shape_h_const_u16 = 8;

static const ai_u16 conv2d_31_t_in_0_shape_w_const_u16 = 8;
static const ai_u16 conv2d_31_t_in_0_shape_h_const_u16 = 8;
static const ai_u16 conv2d_31_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_31_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_31_t_in_0_shape_ch_const_u16 = 144;
static const ai_u16 conv2d_31_t_out_0_shape_ch_const_u16 = 24;
static const ai_i8 conv2d_31_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_31_t_out_0_fmt_zero_const_s8 = 3;
static const ai_float conv2d_31_t_in_0_fmt_scale_const_f32 = 0.023016922175884247f;
static const ai_float conv2d_31_t_out_0_fmt_scale_const_f32 = 0.24313980340957642f;
static const ai_float conv2d_31_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.006341969594359398f, 0.006100571248680353f, 0.018907619640231133f, 0.0059804487973451614f, 0.008640679530799389f, 0.0080850999802351f, 0.007138325832784176f, 0.034332480281591415f, 0.0046830326318740845f, 0.008722152560949326f, 0.005375401582568884f, 0.00378624745644629f, 0.00931895337998867f, 0.014123690314590931f, 0.005497133359313011f, 0.004778894130140543f, 0.006770513951778412f, 0.004582847468554974f, 0.00852253194898367f, 0.022568218410015106f, 0.006447512656450272f, 0.005599990487098694f, 0.009468034841120243f, 0.006076797842979431f);
static const ai_layer_format_type conv2d_31_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 conv2d_33_t_in_0_shape_w_const_u16 = 8;
static const ai_u16 conv2d_33_t_in_0_shape_h_const_u16 = 8;
static const ai_u16 conv2d_33_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_33_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_33_t_in_0_shape_ch_const_u16 = 24;
static const ai_u16 conv2d_33_t_out_0_shape_ch_const_u16 = 144;
static const ai_i8 conv2d_33_t_in_0_fmt_zero_const_s8 = 3;
static const ai_i8 conv2d_33_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_33_t_in_0_fmt_scale_const_f32 = 0.24313980340957642f;
static const ai_float conv2d_33_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_33_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0007160999812185764f, 0.0001951057492988184f, 0.0007769004441797733f, 0.0009348143357783556f, 0.0009536735014989972f, 0.0011753839207813144f, 0.0006910046213306487f, 0.0004976732307113707f, 0.0004838868335355073f, 0.0007715574465692043f, 0.00044130656169727445f, 0.0005604818579740822f, 0.0010807542130351067f, 0.0003154945734422654f, 0.00036577193532139063f, 0.000622906198259443f, 0.0002239631867269054f, 0.0005357743939384818f, 0.0006638811319135129f, 0.000890479248482734f, 0.0004938417114317417f, 0.0009536347351968288f, 0.0008609644719399512f, 0.0002343497471883893f, 0.00043172016739845276f, 0.0008532467181794345f, 0.0008105122251436114f, 0.000520744186360389f, 0.0006856920081190765f, 0.0003809613408520818f, 0.00017122268036473542f, 0.0004942843806929886f, 0.0004552272439468652f, 0.0004430821572896093f, 0.0003416725085116923f, 0.0003116291482001543f, 0.0008482529665343463f, 0.0005663303891196847f, 0.0004890526179224253f, 0.0004689120687544346f, 0.0010457906173542142f, 0.0002597823622636497f, 0.0005417601205408573f, 0.0003216537297703326f, 0.0010353026445955038f, 0.0014084825525060296f, 0.000923836138099432f, 0.0008010743767954409f, 0.0007022008066996932f, 0.00047310395166277885f, 0.00027159691671840847f, 0.0008137602708302438f, 0.0006360244587995112f, 0.0003805635205935687f, 0.0005005095736123621f, 0.00021556900173891336f, 0.0005065655568614602f, 0.0006087585934437811f, 0.0003037180285900831f, 0.0009003516170196235f, 0.0006859253044240177f, 0.0004141975659877062f, 0.000417973380535841f, 0.0005914026987738907f, 0.0007381775649264455f, 0.000751777901314199f, 0.0007669442566111684f, 0.0003281592798884958f, 0.0006253571482375264f, 0.0006642092484980822f, 0.0008164020837284625f, 0.0003247002314310521f, 0.0005778843769803643f, 0.00041731770033948123f, 0.00021761050447821617f, 0.000748810765799135f, 0.0004655805532820523f, 0.0003824183077085763f, 0.00035016503534279764f, 0.0013426049845293164f, 0.0006520707975141704f, 0.00046043848851695657f, 0.0009178492473438382f, 0.00036854229983873665f, 0.0010834963759407401f, 0.00033366828574799f, 0.0005916615482419729f, 0.0006184944650158286f, 0.0003636333567555994f, 0.0002646766370162368f, 0.00023553127539344132f, 0.0011088571045547724f, 0.00046222269884310663f, 0.0008007939904928207f, 0.0006749663734808564f, 0.0006870689685456455f, 0.000516414234880358f, 0.0006079774466343224f, 0.0010518078925088048f, 0.0004737878043670207f, 0.0011109086917713284f, 0.000709584157448262f, 0.0008634963887743652f, 0.0009149442194029689f, 0.0007908964180387557f, 0.0004270813660696149f, 0.0004156042996328324f, 0.0005491369520314038f, 0.00020852430316153914f, 0.0006485982448793948f, 0.0007944138487800956f, 0.0003851985966321081f, 0.0007788600632920861f, 0.0005353520391508937f, 0.0008314743754453957f, 0.0006970813265070319f, 0.0007041210774332285f, 0.001337600639089942f, 0.000679900636896491f, 0.000286472262814641f, 0.0005580928409472108f, 0.0007214808138087392f, 0.0003065479686483741f, 0.0005254897405393422f, 0.00018908896890934557f, 0.0006186600658111274f, 0.00019249055185355246f, 0.0009474102407693863f, 0.0006695865886285901f, 0.0003598004986997694f, 0.00048551539657637477f, 0.00041659027920104563f, 0.0005896864458918571f, 0.0007826162036508322f, 0.0005386705743148923f, 0.00022830137459095567f, 0.0006617768085561693f, 0.0004345212655607611f, 0.0008064923458732665f, 0.0008867840515449643f, 0.0007072597509250045f, 0.0003329204919282347f, 0.0002139879943570122f, 0.0003857555566355586f);
static const ai_layer_format_type conv2d_33_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_i8 conv2d_34_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 conv2d_34_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 conv2d_34_pad_before_t_in_0_shape_h_const_u32 = 8;

static const ai_u16 conv2d_34_t_in_0_shape_w_const_u16 = 10;
static const ai_u16 conv2d_34_t_in_0_shape_h_const_u16 = 10;
static const ai_u16 conv2d_34_t_in_0_shape_ch_const_u16 = 144;
static const ai_u16 conv2d_34_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_34_l_stride_0_const_u16 = 1;
static const ai_i8 conv2d_34_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_34_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_34_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_34_t_out_0_fmt_scale_const_f32 = 0.02275237813591957f;
static const ai_float conv2d_34_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0034110157284885645f, 0.0051077441312372684f, 0.0054244715720415115f, 0.00613446393981576f, 0.00957790482789278f, 0.004476518835872412f, 0.006489350460469723f, 0.006284929346293211f, 0.008257939480245113f, 0.004356666933745146f, 0.01088064257055521f, 0.005827197339385748f, 0.005362679250538349f, 0.010261612012982368f, 0.015601269900798798f, 0.005484582390636206f, 0.018473271280527115f, 0.017725180834531784f, 0.014865149743855f, 0.0103768864646554f, 0.004373083356767893f, 0.009915406815707684f, 0.005421442911028862f, 0.013219134882092476f, 0.016713857650756836f, 0.016777433454990387f, 0.008880477398633957f, 0.011874276213347912f, 0.007169422693550587f, 0.0076013365760445595f, 0.026325803250074387f, 0.008884957991540432f, 0.00679813651368022f, 0.006592625752091408f, 0.014911768957972527f, 0.012436462566256523f, 0.0033187505323439837f, 0.007713541854172945f, 0.012884164229035378f, 0.017856691032648087f, 0.005482753738760948f, 0.012124503962695599f, 0.0032599428668618202f, 0.007893191650509834f, 0.004753112327307463f, 0.004533807747066021f, 0.010149613954126835f, 0.004044028930366039f, 0.008317760191857815f, 0.007020303513854742f, 0.009130280464887619f, 0.008093088865280151f, 0.01006886176764965f, 0.005439106374979019f, 0.006101307924836874f, 0.0177965946495533f, 0.006179161369800568f, 0.006123699713498354f, 0.01409006305038929f, 0.006729619111865759f, 0.002640136983245611f, 0.010833929292857647f, 0.00790848396718502f, 0.004970062058418989f, 0.0052903140895068645f, 0.003615680616348982f, 0.007740001194179058f, 0.008926594629883766f, 0.004350848030298948f, 0.005838657729327679f, 0.0075368694961071014f, 0.0161907821893692f, 0.006558020133525133f, 0.010221915319561958f, 0.024141836911439896f, 0.027249084785580635f, 0.006547039374709129f, 0.00786138791590929f, 0.029389677569270134f, 0.0027088127098977566f, 0.02050594612956047f, 0.014538371004164219f, 0.007027200888842344f, 0.010461301542818546f, 0.009109601378440857f, 0.01960465870797634f, 0.004307050257921219f, 0.006938941311091185f, 0.007346496917307377f, 0.015723297372460365f, 0.010186688043177128f, 0.007677788380533457f, 0.005091102793812752f, 0.004354835022240877f, 0.005725049413740635f, 0.0025066728703677654f, 0.003773174714297056f, 0.015178519301116467f, 0.023651860654354095f, 0.009753190912306309f, 0.01284489780664444f, 0.004798774607479572f, 0.006954851094633341f, 0.014658904634416103f, 0.008853712119162083f, 0.01076570525765419f, 0.013456699438393116f, 0.008423879742622375f, 0.026738161221146584f, 0.005662521347403526f, 0.005138432141393423f, 0.006587872747331858f, 0.004664739593863487f, 0.005552768241614103f, 0.01014093216508627f, 0.008621729910373688f, 0.01008395291864872f, 0.0027426702436059713f, 0.006632955279201269f, 0.012045545503497124f, 0.012626906856894493f, 0.006368538364768028f, 0.01652330905199051f, 0.007953081279993057f, 0.04403131082653999f, 0.011403138749301434f, 0.019457994028925896f, 0.003121386980637908f, 0.008139582350850105f, 0.012496860697865486f, 0.00822436809539795f, 0.00794962141662836f, 0.004753215704113245f, 0.009818261489272118f, 0.010973969474434853f, 0.00683286739513278f, 0.009291806258261204f, 0.010515101253986359f, 0.002782453317195177f, 0.004953277762979269f, 0.005686849821358919f, 0.021199945360422134f, 0.014142051339149475f, 0.007893175818026066f);
static const ai_u16 conv2d_34_t_out_0_shape_w_const_u16 = 8;
static const ai_u16 conv2d_34_t_out_0_shape_h_const_u16 = 8;

static const ai_u16 conv2d_35_t_in_0_shape_w_const_u16 = 8;
static const ai_u16 conv2d_35_t_in_0_shape_h_const_u16 = 8;
static const ai_u16 conv2d_35_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_35_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_35_t_in_0_shape_ch_const_u16 = 144;
static const ai_u16 conv2d_35_t_out_0_shape_ch_const_u16 = 24;
static const ai_i8 conv2d_35_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_35_t_out_0_fmt_zero_const_s8 = 13;
static const ai_float conv2d_35_t_in_0_fmt_scale_const_f32 = 0.02275237813591957f;
static const ai_float conv2d_35_t_out_0_fmt_scale_const_f32 = 0.25250938534736633f;
static const ai_float conv2d_35_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.01078756246715784f, 0.005137486383318901f, 0.018025267869234085f, 0.010156507603824139f, 0.007379297632724047f, 0.007418329827487469f, 0.0041476162150502205f, 0.023296784609556198f, 0.00449300417676568f, 0.00673102168366313f, 0.005922472104430199f, 0.0034223697148263454f, 0.005391332786530256f, 0.008849930949509144f, 0.0037091353442519903f, 0.004003279842436314f, 0.0046779196709394455f, 0.01303610298782587f, 0.005948367994278669f, 0.023196972906589508f, 0.004912721924483776f, 0.00422916654497385f, 0.00621072156354785f, 0.004469404462724924f);
static const ai_layer_format_type conv2d_35_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 conv2d_37_t_in_0_shape_w_const_u16 = 8;
static const ai_u16 conv2d_37_t_in_0_shape_h_const_u16 = 8;
static const ai_u16 conv2d_37_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_37_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_37_t_in_0_shape_ch_const_u16 = 24;
static const ai_u16 conv2d_37_t_out_0_shape_ch_const_u16 = 144;
static const ai_i8 conv2d_37_t_in_0_fmt_zero_const_s8 = 13;
static const ai_i8 conv2d_37_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_37_t_in_0_fmt_scale_const_f32 = 0.25250938534736633f;
static const ai_float conv2d_37_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_37_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0004757078713737428f, 0.000953798764385283f, 0.00034351192880421877f, 0.0008226004429161549f, 0.00043555544107221067f, 0.0007545510306954384f, 0.0005792046431452036f, 0.0008597548585385084f, 0.0008085602894425392f, 0.0008339688065461814f, 0.00038542470429092646f, 0.0014341895002871752f, 0.0006115249707363546f, 0.0006655064062215388f, 0.0006809292826801538f, 0.0007736555999144912f, 0.0009123843046836555f, 0.0003483411273919046f, 0.000672399764880538f, 0.0005408581346273422f, 0.0005734666483476758f, 0.0003475928970146924f, 0.00047761277528479695f, 0.0008819036884233356f, 0.00041672514635138214f, 0.0009212992736138403f, 0.0006001620786264539f, 0.0009088910301215947f, 0.0008356209727935493f, 0.0009443238377571106f, 0.000583346001803875f, 0.0015694168396294117f, 0.0009021644364111125f, 0.0007737020496279001f, 0.0008154890965670347f, 0.0007997581269592047f, 0.0011524045839905739f, 0.0008895736536942422f, 0.0003786876914091408f, 0.0007561339880339801f, 0.0004657481622416526f, 0.0007242426509037614f, 0.0008408124558627605f, 0.0007031176355667412f, 0.0011587589979171753f, 0.0009252296295017004f, 0.0005571271758526564f, 0.000508263474330306f, 0.0007819252205081284f, 0.0004826712538488209f, 0.0006083907792344689f, 0.0006102344486862421f, 0.0009988864185288548f, 0.0006019524880684912f, 0.0007058184128254652f, 0.0009455786203034222f, 0.0008678220910951495f, 0.0008436585776507854f, 0.0009170809644274414f, 0.0005776155740022659f, 0.0003710189776029438f, 0.0006459889700636268f, 0.0008066282607614994f, 0.0007310971268452704f, 0.0005964736919850111f, 0.0007068641716614366f, 0.0005583686870522797f, 0.00022310401254799217f, 0.0004322310269344598f, 0.0008601426379755139f, 0.0007380659808404744f, 0.0007217677193693817f, 0.00043511699186638f, 0.0005394822801463306f, 0.000614385528024286f, 0.0009457459673285484f, 0.00028962851502001286f, 0.0007342931348830462f, 0.000631020637229085f, 0.000723672506865114f, 0.00034893397241830826f, 0.0006697074859403074f, 0.0006982820923440158f, 0.00026637327391654253f, 0.00037195958429947495f, 0.0012109605595469475f, 0.0005595787079073489f, 0.0009054634138010442f, 0.0006998584140092134f, 0.0006639202474616468f, 0.0007553324685432017f, 0.0007345068734139204f, 0.0007963312673382461f, 0.0010521858930587769f, 0.0004762151511386037f, 0.0007601127144880593f, 0.0009852746734395623f, 0.0016899063484743237f, 0.0006015175604261458f, 0.0005954049411229789f, 0.0005343171651475132f, 0.0006363129359669983f, 0.0003083153860643506f, 0.0012181428028270602f, 0.00047233194345608354f, 0.0009314916678704321f, 0.0010005354415625334f, 0.0006121034384705126f, 0.00040259474189952016f, 0.001065065385773778f, 0.0005953708896413445f, 0.0007054120651446283f, 0.0006268942379392684f, 0.0014800118515267968f, 0.0006408910849131644f, 0.0016677622916176915f, 0.0006799099501222372f, 0.0007107665878720582f, 0.0007486604736186564f, 0.0005606826161965728f, 0.0004584827402140945f, 0.0012123632477596402f, 0.0007451135898008943f, 0.000324057909892872f, 0.0008676765719428658f, 0.000489170546643436f, 0.0008257607114501297f, 0.000810083991382271f, 0.0006394197698682547f, 0.0007308933418244123f, 0.0006580583867616951f, 0.000883128319401294f, 0.0005121704307384789f, 0.00047659725532867014f, 0.00033058144617825747f, 0.0008766785613261163f, 0.0012227697297930717f, 0.0012000638525933027f, 0.0005820670048706234f, 0.0006929931696504354f, 0.000619754078797996f, 0.0008661817410029471f, 0.0007207801099866629f, 0.0011397687485441566f);
static const ai_layer_format_type conv2d_37_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_i8 conv2d_38_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 conv2d_38_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 conv2d_38_pad_before_t_in_0_shape_h_const_u32 = 8;

static const ai_u16 conv2d_38_t_in_0_shape_w_const_u16 = 10;
static const ai_u16 conv2d_38_t_in_0_shape_h_const_u16 = 10;
static const ai_u16 conv2d_38_t_in_0_shape_ch_const_u16 = 144;
static const ai_u16 conv2d_38_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_38_l_stride_0_const_u16 = 1;
static const ai_i8 conv2d_38_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_38_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_38_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_38_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_38_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0029088316950947046f, 0.005452578421682119f, 0.007229106035083532f, 0.0034475200809538364f, 0.007199072744697332f, 0.004156865645200014f, 0.010691668838262558f, 0.003053162479773164f, 0.0066151730716228485f, 0.004766929429024458f, 0.010760899633169174f, 0.0030202148482203484f, 0.003138410858809948f, 0.008350487798452377f, 0.009129693731665611f, 0.0069356332533061504f, 0.004495324566960335f, 0.0028864946216344833f, 0.003569596679881215f, 0.003401939757168293f, 0.009436056949198246f, 0.008417706936597824f, 0.01109654176980257f, 0.004104851745069027f, 0.0103194210678339f, 0.0036458426620811224f, 0.0035361892078071833f, 0.009383713826537132f, 0.007224848493933678f, 0.0034807089250534773f, 0.0019327390473335981f, 0.004186796490103006f, 0.006599560379981995f, 0.003429972566664219f, 0.0015237957704812288f, 0.0037648617289960384f, 0.00407056137919426f, 0.004154197406023741f, 0.005422509275376797f, 0.005832992028445005f, 0.005558942444622517f, 0.004417522344738245f, 0.007106566336005926f, 0.00793582946062088f, 0.0060185897164046764f, 0.0022908851969987154f, 0.0026811505667865276f, 0.010679906234145164f, 0.00471990741789341f, 0.006289034150540829f, 0.002195464912801981f, 0.008583685383200645f, 0.0034554197918623686f, 0.008611173368990421f, 0.002018021186813712f, 0.006469584070146084f, 0.008748206309974194f, 0.006229123566299677f, 0.0016104135429486632f, 0.007292645517736673f, 0.010034958831965923f, 0.0027703894302248955f, 0.0059451088309288025f, 0.005915198475122452f, 0.004831141792237759f, 0.0024763131514191628f, 0.003935002721846104f, 0.011112535372376442f, 0.006015089340507984f, 0.007065055426210165f, 0.007405157200992107f, 0.009633729234337807f, 0.003410073695704341f, 0.004849362187087536f, 0.005277921911329031f, 0.002667047083377838f, 0.010644139721989632f, 0.005626879632472992f, 0.006373699754476547f, 0.005103608127683401f, 0.012384391389787197f, 0.007816552184522152f, 0.003279465716332197f, 0.014767042361199856f, 0.006644273642450571f, 0.0044424086809158325f, 0.0033769330475479364f, 0.0030066913459450006f, 0.007030232809484005f, 0.007656462024897337f, 0.0015295306220650673f, 0.0024873174261301756f, 0.0022516846656799316f, 0.005928270984441042f, 0.008700065314769745f, 0.006535189691931009f, 0.003117612563073635f, 0.005168521776795387f, 0.002860386623069644f, 0.006614404264837503f, 0.01066584512591362f, 0.005330100189894438f, 0.009337466210126877f, 0.006766572594642639f, 0.0038003523368388414f, 0.005535965785384178f, 0.0026431637816131115f, 0.00595463952049613f, 0.00675246911123395f, 0.001756581594236195f, 0.009466147981584072f, 0.0041357288137078285f, 0.005371161736547947f, 0.005841433070600033f, 0.008716009557247162f, 0.0023789287079125643f, 0.004280673805624247f, 0.004890017211437225f, 0.009852929040789604f, 0.0026447775308042765f, 0.021495746448636055f, 0.002173250075429678f, 0.0020791334100067616f, 0.013543662615120411f, 0.005744590424001217f, 0.003832706483080983f, 0.0070698861964046955f, 0.00502279307693243f, 0.0033469691406935453f, 0.005803660023957491f, 0.0038213201332837343f, 0.00330251082777977f, 0.008880868554115295f, 0.0070794676430523396f, 0.01342832576483488f, 0.007702144794166088f, 0.004146822728216648f, 0.004691540729254484f, 0.004404551815241575f, 0.007928460836410522f, 0.010662248358130455f, 0.003881118493154645f, 0.005879927892237902f, 0.0016610914608463645f);
static const ai_u16 conv2d_38_t_out_0_shape_w_const_u16 = 8;
static const ai_u16 conv2d_38_t_out_0_shape_h_const_u16 = 8;

static const ai_u16 conv2d_39_t_in_0_shape_w_const_u16 = 8;
static const ai_u16 conv2d_39_t_in_0_shape_h_const_u16 = 8;
static const ai_u16 conv2d_39_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_39_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_39_t_in_0_shape_ch_const_u16 = 144;
static const ai_u16 conv2d_39_t_out_0_shape_ch_const_u16 = 32;
static const ai_i8 conv2d_39_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_39_t_out_0_fmt_zero_const_s8 = 3;
static const ai_float conv2d_39_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_39_t_out_0_fmt_scale_const_f32 = 0.18943023681640625f;
static const ai_float conv2d_39_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.014322489500045776f, 0.015525104478001595f, 0.01251438818871975f, 0.011899356730282307f, 0.014431418851017952f, 0.014111441560089588f, 0.016880201175808907f, 0.009776610881090164f, 0.010870921425521374f, 0.015352851711213589f, 0.011449906975030899f, 0.011545208282768726f, 0.009914547204971313f, 0.014938612468540668f, 0.01087538804858923f, 0.014422993175685406f, 0.01265542022883892f, 0.013993287459015846f, 0.011702893301844597f, 0.010395647026598454f, 0.018757613375782967f, 0.022031525149941444f, 0.011373171582818031f, 0.010407108813524246f, 0.014381767250597477f, 0.026070259511470795f, 0.011511002667248249f, 0.010211440734565258f, 0.010874995961785316f, 0.014546309597790241f, 0.01837996579706669f, 0.014187062159180641f);
static const ai_layer_format_type conv2d_39_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_u16 conv2d_40_t_in_0_shape_w_const_u16 = 8;
static const ai_u16 conv2d_40_t_in_0_shape_h_const_u16 = 8;
static const ai_u16 conv2d_40_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_40_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_40_t_in_0_shape_ch_const_u16 = 32;
static const ai_u16 conv2d_40_t_out_0_shape_ch_const_u16 = 192;
static const ai_i8 conv2d_40_t_in_0_fmt_zero_const_s8 = 3;
static const ai_i8 conv2d_40_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_40_t_in_0_fmt_scale_const_f32 = 0.18943023681640625f;
static const ai_float conv2d_40_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_40_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.00038172403583303094f, 0.002161920303478837f, 0.0009455140097998083f, 0.00046413333620876074f, 0.0008515334338881075f, 0.0012448796769604087f, 0.0005650506936945021f, 0.0007952276500873268f, 0.00042574413237161934f, 0.0010666653979569674f, 0.00198134733363986f, 0.0004465171368792653f, 0.0004629300383385271f, 0.00035430476418696344f, 0.0007268344634212554f, 0.0005959725822322071f, 0.0008708425448276103f, 0.00100004265550524f, 0.0008582925074733794f, 0.0008277902379631996f, 0.0009803250432014465f, 0.0006772768101654947f, 0.0010649964679032564f, 0.0004870568518526852f, 0.0020304813515394926f, 0.00043728190939873457f, 0.0008517056703567505f, 0.0006989003159105778f, 0.0008004556293599308f, 0.0010963958920910954f, 0.000412756169680506f, 0.0004539082874543965f, 0.0011272559640929103f, 0.002074891934171319f, 0.0007067460683174431f, 0.0009650757419876754f, 0.0015948553336784244f, 0.0004118025244679302f, 0.0005244840285740793f, 0.00134925520978868f, 0.0008918942767195404f, 0.0008412592578679323f, 0.0012526930077001452f, 0.0008805738179944456f, 0.0006880656001158059f, 0.0012136843288317323f, 0.0008393579046241939f, 0.0009525419445708394f, 0.0009095699060708284f, 0.000616895908024162f, 0.0009298411896452308f, 0.000805664574727416f, 0.00044894410530105233f, 0.0008940048282966018f, 0.0006968112429603934f, 0.0008007535943761468f, 0.0006058284197933972f, 0.0008733377326279879f, 0.00044077177881263196f, 0.0013049424160271883f, 0.0006166680832393467f, 0.0008198400610126555f, 0.0009732670150697231f, 0.0008631988894194365f, 0.0009530560346320271f, 0.0010314019164070487f, 0.0009229673305526376f, 0.0006712721078656614f, 0.0011051585897803307f, 0.000697474810294807f, 0.0017090708715841174f, 0.001025042263790965f, 0.0013634870992973447f, 0.0005469952011480927f, 0.00043239115620963275f, 0.0008600034634582698f, 0.000990265398286283f, 0.0005903035053052008f, 0.0013805784983560443f, 0.0009928869549185038f, 0.00126259692478925f, 0.00040171126602217555f, 0.0014763722429051995f, 0.0004894858575426042f, 0.00032889717840589583f, 0.0009431747603230178f, 0.0007900097989477217f, 0.00030828022863715887f, 0.00029709789669141173f, 0.000874600256793201f, 0.00033682482899166644f, 0.0005937559180893004f, 0.0004323683388065547f, 0.00043707064469344914f, 0.0008010099409148097f, 0.0006761528202332556f, 0.0010967865819111466f, 0.0013907805550843477f, 0.0006518197478726506f, 0.0006377061363309622f, 0.0005842153914272785f, 0.00037078908644616604f, 0.001346180448308587f, 0.00043588923290371895f, 0.001262724632397294f, 0.0006931194802746177f, 0.0005016782088205218f, 0.0005221787141636014f, 0.0003468973736744374f, 0.00032790747354738414f, 0.0009534501004964113f, 0.0006428085616789758f, 0.0006246478296816349f, 0.0007554757758043706f, 0.000623160507529974f, 0.00034242606488987803f, 0.0009826294844970107f, 0.00029866970726288855f, 0.0009740320383571088f, 0.0008122441940940917f, 0.0004310040094424039f, 0.0009469295619055629f, 0.0006345466244965792f, 0.00035788831883110106f, 0.0010446824599057436f, 0.0008322698995471001f, 0.0008528410689905286f, 0.001041343784891069f, 0.00044742401223629713f, 0.0002260627952637151f, 0.0010054651647806168f, 0.0011452723992988467f, 0.0003714156919158995f, 0.0004523219831753522f, 0.00036206585355103016f, 0.0008245458011515439f, 0.0012606981908902526f, 0.0003308311279397458f, 0.0005752422148361802f, 0.0007791052921675146f, 0.001216848730109632f, 0.0006732997135259211f, 0.00038964927080087364f, 0.0015802365960553288f, 0.0009918567957356572f, 0.0008842445677146316f, 0.0004004451911896467f, 0.0010634768987074494f, 0.0006963032647036016f, 0.0009744440903887153f, 0.0005145993200130761f, 0.0026019755750894547f, 0.000929477799218148f, 0.000554899568669498f, 0.0009548312518745661f, 0.0003912810352630913f, 0.0013762355083599687f, 0.0004677706747315824f, 0.0012688541319221258f, 0.001608936581760645f, 0.0008296191808767617f, 0.0007852010312490165f, 0.0010292096994817257f, 0.0008499869145452976f, 0.0006487631471827626f, 0.0008837314089760184f, 0.0017433813773095608f, 0.0008272206760011613f, 0.0009909480577334762f, 0.0010522513184696436f, 0.0007343502948060632f, 0.001132718869484961f, 0.0006075018900446594f, 0.0009486785274930298f, 0.0011065149446949363f, 0.0010686272289603949f, 0.0007608605665154755f, 0.0009068332146853209f, 0.001162458793260157f, 0.0007335427217185497f, 0.0017922999104484916f, 0.0019835729617625475f, 0.000743882090318948f, 0.0008056327933445573f, 0.0005122091388329864f, 0.0008243421325460076f, 0.000970042310655117f, 0.0003117425658274442f, 0.0004201672854833305f, 0.0011218460276722908f, 0.0013537263730540872f, 0.00030709075508639216f);
static const ai_layer_format_type conv2d_40_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_i8 conv2d_41_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 conv2d_41_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 conv2d_41_pad_before_t_in_0_shape_h_const_u32 = 8;

static const ai_u16 conv2d_41_t_in_0_shape_w_const_u16 = 10;
static const ai_u16 conv2d_41_t_in_0_shape_h_const_u16 = 10;
static const ai_u16 conv2d_41_t_in_0_shape_ch_const_u16 = 192;
static const ai_u16 conv2d_41_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_41_l_stride_0_const_u16 = 1;
static const ai_i8 conv2d_41_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_41_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_41_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_41_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_41_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.047915033996105194f, 0.0038134432397782803f, 0.002362822415307164f, 0.0072767408564686775f, 0.013975515961647034f, 0.004593339748680592f, 0.008004533126950264f, 0.004811355844140053f, 0.011021774262189865f, 0.008365901187062263f, 0.00261850212700665f, 0.009821232408285141f, 0.01276312954723835f, 0.018517332151532173f, 0.003631079802289605f, 0.006945777218788862f, 0.01778685860335827f, 0.0027224624063819647f, 0.007625275291502476f, 0.010480716824531555f, 0.018519455567002296f, 0.0055174678564071655f, 0.004863161593675613f, 0.008463793434202671f, 0.003802406368777156f, 0.004408413078635931f, 0.009449242614209652f, 0.010222133249044418f, 0.005722944159060717f, 0.003339048707857728f, 0.012061240151524544f, 0.006383118685334921f, 0.0024685803800821304f, 0.003710322780534625f, 0.01138501986861229f, 0.007473267614841461f, 0.0026388911064714193f, 0.006254102103412151f, 0.010105317458510399f, 0.00951226707547903f, 0.014122837223112583f, 0.002466063480824232f, 0.0017631122609600425f, 0.004449530970305204f, 0.01153065450489521f, 0.00226450152695179f, 0.0073132822290062904f, 0.006278433836996555f, 0.01508370228111744f, 0.002878755796700716f, 0.006774640176445246f, 0.013431605882942677f, 0.006220012437552214f, 0.006564689800143242f, 0.009053552523255348f, 0.006242710165679455f, 0.003527663415297866f, 0.009706283919513226f, 0.007508106995373964f, 0.022241082042455673f, 0.01263931393623352f, 0.004136191215366125f, 0.010296806693077087f, 0.008065705187618732f, 0.012292281724512577f, 0.003890953492373228f, 0.0018509442452341318f, 0.008988240733742714f, 0.011675829999148846f, 0.006092315074056387f, 0.010145454667508602f, 0.0054938895627856255f, 0.011924528516829014f, 0.00848439708352089f, 0.009876796044409275f, 0.006304754875600338f, 0.005492737982422113f, 0.0030892810318619013f, 0.009897416457533836f, 0.010390620678663254f, 0.0020618673879653215f, 0.017200928181409836f, 0.004292714409530163f, 0.01325144898146391f, 0.012195061892271042f, 0.00644836388528347f, 0.002105421619489789f, 0.02490432932972908f, 0.020369982346892357f, 0.0031680043321102858f, 0.015014396980404854f, 0.0066875373013317585f, 0.03949872776865959f, 0.008561168797314167f, 0.007547030225396156f, 0.014667253009974957f, 0.0013729785569012165f, 0.0018176506273448467f, 0.008284901268780231f, 0.003149664029479027f, 0.0043856557458639145f, 0.008924681693315506f, 0.006699428893625736f, 0.014624349772930145f, 0.007089512888342142f, 0.006652092561125755f, 0.01582745648920536f, 0.00788924191147089f, 0.014046215452253819f, 0.006062397733330727f, 0.011225211434066296f, 0.00811492558568716f, 0.0045223357155919075f, 0.005965971387922764f, 0.014449582435190678f, 0.018693478778004646f, 0.014922204427421093f, 0.014464545994997025f, 0.010741934180259705f, 0.010673596523702145f, 0.013886420987546444f, 0.007655677385628223f, 0.012569163925945759f, 0.012056559324264526f, 0.004873408004641533f, 0.004283882211893797f, 0.010721994563937187f, 0.008562172763049603f, 0.00802276749163866f, 0.02452669106423855f, 0.0027213608846068382f, 0.0063649932853877544f, 0.01264403946697712f, 0.020186040550470352f, 0.014548487961292267f, 0.005929307546466589f, 0.003747709561139345f, 0.015430239960551262f, 0.014340861700475216f, 0.013774440623819828f, 0.007032488007098436f, 0.010387002490460873f, 0.015903454273939133f, 0.004369121044874191f, 0.008747836574912071f, 0.00642187986522913f, 0.01353574637323618f, 0.010569339618086815f, 0.0053544421680271626f, 0.012328422628343105f, 0.006079023703932762f, 0.0026775600854307413f, 0.0071546114049851894f, 0.009755516424775124f, 0.009758517146110535f, 0.009824180975556374f, 0.005470401607453823f, 0.007968232966959476f, 0.006950179114937782f, 0.004841638263314962f, 0.0029689972288906574f, 0.005449261516332626f, 0.013878234662115574f, 0.006109501700848341f, 0.012473647482693195f, 0.009607155807316303f, 0.0017660321900621057f, 0.02163826860487461f, 0.003390420461073518f, 0.012231813743710518f, 0.00798412412405014f, 0.0019581657834351063f, 0.011950535699725151f, 0.008698760531842709f, 0.005342318210750818f, 0.0019424684578552842f, 0.004905062261968851f, 0.009433185681700706f, 0.01312195509672165f, 0.013718205504119396f, 0.004205968230962753f, 0.006020813714712858f, 0.011777481995522976f, 0.010336533188819885f, 0.007404983509331942f, 0.010799271985888481f, 0.005165042821317911f, 0.009993663057684898f, 0.010078923776745796f, 0.004079178906977177f, 0.001955566927790642f, 0.01599954627454281f);
static const ai_u16 conv2d_41_t_out_0_shape_w_const_u16 = 8;
static const ai_u16 conv2d_41_t_out_0_shape_h_const_u16 = 8;

static const ai_u16 conv2d_42_t_in_0_shape_w_const_u16 = 8;
static const ai_u16 conv2d_42_t_in_0_shape_h_const_u16 = 8;
static const ai_u16 conv2d_42_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_42_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_42_t_in_0_shape_ch_const_u16 = 192;
static const ai_u16 conv2d_42_t_out_0_shape_ch_const_u16 = 32;
static const ai_i8 conv2d_42_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_42_t_out_0_fmt_zero_const_s8 = 2;
static const ai_float conv2d_42_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_42_t_out_0_fmt_scale_const_f32 = 0.2056785374879837f;
static const ai_float conv2d_42_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0038000489585101604f, 0.004853623919188976f, 0.006696934346109629f, 0.005116305314004421f, 0.004776571411639452f, 0.004980467259883881f, 0.005413434933871031f, 0.012731516733765602f, 0.014915239065885544f, 0.016779443249106407f, 0.00459276745095849f, 0.01218355167657137f, 0.003994752187281847f, 0.016217561438679695f, 0.0057852487079799175f, 0.0035674688406288624f, 0.014587299898266792f, 0.0044539072550833225f, 0.007771949749439955f, 0.0055874926038086414f, 0.006527863908559084f, 0.01808486320078373f, 0.011767234653234482f, 0.01400213036686182f, 0.013659259304404259f, 0.011949623934924603f, 0.014541985467076302f, 0.0054342360235750675f, 0.004583045374602079f, 0.004561969079077244f, 0.009746971540153027f, 0.0050291200168430805f);
static const ai_layer_format_type conv2d_42_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 conv2d_44_t_in_0_shape_w_const_u16 = 8;
static const ai_u16 conv2d_44_t_in_0_shape_h_const_u16 = 8;
static const ai_u16 conv2d_44_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_44_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_44_t_in_0_shape_ch_const_u16 = 32;
static const ai_u16 conv2d_44_t_out_0_shape_ch_const_u16 = 192;
static const ai_i8 conv2d_44_t_in_0_fmt_zero_const_s8 = 2;
static const ai_i8 conv2d_44_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_44_t_in_0_fmt_scale_const_f32 = 0.2056785374879837f;
static const ai_float conv2d_44_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_44_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.00037110617267899215f, 0.00037759909173473716f, 0.0003928501682821661f, 0.0005658713052980602f, 0.00027804001001641154f, 0.0004850662953685969f, 0.0005802793893963099f, 0.0008433636394329369f, 0.0006233950844034553f, 0.000677552365232259f, 0.0005858851945959032f, 0.0007134162005968392f, 0.0006448394851759076f, 0.0004129699955228716f, 0.0007826681830920279f, 0.00033565994817763567f, 0.0008392601157538593f, 0.000567311595659703f, 0.0008833014871925116f, 0.0008296099840663373f, 0.00028887236840091646f, 0.000357604818418622f, 0.0007874754955992103f, 0.0006257464992813766f, 0.0001874161825980991f, 0.0004594937199726701f, 0.0007090900326147676f, 0.0005986341857351363f, 0.0006199835916049778f, 0.0007255654199980199f, 0.0006634089513681829f, 0.0008935226942412555f, 0.00033791971509344876f, 0.0002991177898366004f, 0.00042909313924610615f, 0.0011976197129115462f, 0.0003411930229049176f, 0.0006298520602285862f, 0.0005904489662498236f, 0.0005589528009295464f, 0.00041463246452622116f, 0.0008112526847980917f, 0.0008749791886657476f, 0.000223514114622958f, 0.0007390572573058307f, 0.0006022355519235134f, 0.0003590495325624943f, 0.0006864413735456765f, 0.0008661382016725838f, 0.001117308042012155f, 0.0008658372680656612f, 0.0007748212083242834f, 0.0003521011967677623f, 0.0012754489434882998f, 0.0007015009759925306f, 0.0007116346387192607f, 0.0005731228739023209f, 0.00025715550873428583f, 0.0004570815362967551f, 0.0006431391229853034f, 0.0005621258751489222f, 0.0007649270119145513f, 0.0006737125804647803f, 0.0008748785476200283f, 0.000536341336555779f, 0.00041564146522432566f, 0.0007023293292149901f, 0.0005600801669061184f, 0.0005769013077951968f, 0.0011335022281855345f, 0.0003183357184752822f, 0.00023427770065609366f, 0.0009267895948141813f, 0.0004633271601051092f, 0.00011646393249975517f, 0.0002255189319839701f, 0.00025306924362666905f, 0.00038361118640750647f, 0.000747012032661587f, 0.0002877652004826814f, 0.0005781603395007551f, 0.0005170811200514436f, 0.00040959694888442755f, 0.0006051245145499706f, 0.0005946543533354998f, 0.0007368147489614785f, 0.0010898445034399629f, 0.00011204027396161109f, 0.0008394558099098504f, 0.0008266515214927495f, 0.001034031854942441f, 0.0008164448663592339f, 0.0005425082636065781f, 0.0008852784521877766f, 0.00043690865277312696f, 0.0007626680890098214f, 0.0010002953931689262f, 0.0006769669707864523f, 0.00030749032157473266f, 0.0006995106232352555f, 0.000842287961859256f, 0.00025683874264359474f, 0.0010850470280274749f, 0.0007759408908896148f, 0.0009563227649778128f, 0.000361328711733222f, 0.0009507393115200102f, 0.0004470687417779118f, 0.0006390271591953933f, 0.0007512540323659778f, 0.00047951945452950895f, 0.0008305794908665121f, 0.0006747277802787721f, 0.0006838754052296281f, 0.0007734603132121265f, 0.0009075169800780714f, 0.00025298300897702575f, 0.0008786403341218829f, 0.0008163560996763408f, 0.0010122215608134866f, 0.0004097712517250329f, 0.0007373602129518986f, 0.0008165543549694121f, 0.0008499708492308855f, 0.0006018925341777503f, 0.0007969319703988731f, 0.0002884897403419018f, 0.0014642655150964856f, 0.00022868401720188558f, 0.001224182778969407f, 0.0007585129933431745f, 0.0009466973133385181f, 0.0006871104706078768f, 0.0003552507550921291f, 0.0002998716081492603f, 0.00028435891726985574f, 0.000550148484762758f, 0.00081667210906744f, 0.0007409878890030086f, 0.0005769061972387135f, 0.0003262813843321055f, 0.00042271538404747844f, 0.0009067536448128521f, 0.0006113068084232509f, 0.0004199895483907312f, 0.0009072546963579953f, 0.0005571729852817953f, 0.0008846139535307884f, 0.0008513773791491985f, 0.0007309494540095329f, 0.0012928829528391361f, 0.00040789623744785786f, 0.0010694318916648626f, 0.0009983406635001302f, 0.0010273390216752887f, 0.0008167350897565484f, 0.001060417271219194f, 0.0009096526191569865f, 0.0006605490925721824f, 0.0003272062458563596f, 0.0006341924890875816f, 0.00020579806005116552f, 0.0008148850174620748f, 0.0007347468053922057f, 0.0003682092938106507f, 0.0006129803950898349f, 0.000713762070517987f, 0.0011883844854310155f, 0.0006352493073791265f, 0.00050981534877792f, 0.0009221452055498958f, 0.0005137619446031749f, 0.0003871112421620637f, 0.0008745933882892132f, 0.0007048173574730754f, 0.0004686887259595096f, 0.0003251937741879374f, 0.0007864977815188468f, 0.0009979309979826212f, 0.0005246744258329272f, 0.0005680756876245141f, 0.0008985430467873812f, 0.0009352932102046907f, 0.00043765324517153203f, 0.0007906498503871262f, 0.00033326857374049723f, 0.00037826120387762785f, 0.0008715977892279625f, 0.0007619022508151829f, 0.0005901753320358694f, 0.0004881070926785469f, 0.0008771988213993609f);
static const ai_layer_format_type conv2d_44_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_i8 conv2d_45_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 conv2d_45_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 conv2d_45_pad_before_t_in_0_shape_h_const_u32 = 8;

static const ai_u16 conv2d_45_t_in_0_shape_w_const_u16 = 10;
static const ai_u16 conv2d_45_t_in_0_shape_h_const_u16 = 10;
static const ai_u16 conv2d_45_t_in_0_shape_ch_const_u16 = 192;
static const ai_u16 conv2d_45_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_45_l_stride_0_const_u16 = 1;
static const ai_i8 conv2d_45_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_45_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_45_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_45_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_45_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.007600768003612757f, 0.010200188495218754f, 0.02074613980948925f, 0.015436011366546154f, 0.01013992540538311f, 0.009946856647729874f, 0.012846787460148335f, 0.0015081744641065598f, 0.009695990942418575f, 0.008384977467358112f, 0.0032747245859354734f, 0.010878277942538261f, 0.005093404091894627f, 0.008959535509347916f, 0.011762042529881f, 0.012064752168953419f, 0.002534909173846245f, 0.009945577010512352f, 0.011330008506774902f, 0.012934951111674309f, 0.010256820358335972f, 0.006691381800919771f, 0.006436995696276426f, 0.0035383948124945164f, 0.010531600564718246f, 0.00856176670640707f, 0.015015415847301483f, 0.007822122424840927f, 0.0059274896048009396f, 0.01320339273661375f, 0.00640148064121604f, 0.0022904935758560896f, 0.007168070878833532f, 0.020711710676550865f, 0.009849496185779572f, 0.006958788260817528f, 0.011162802577018738f, 0.015965066850185394f, 0.0018298374488949776f, 0.006637078244239092f, 0.004514029249548912f, 0.0076752919703722f, 0.0038872421719133854f, 0.0047662039287388325f, 0.005595498718321323f, 0.004601909313350916f, 0.014841144904494286f, 0.012793084606528282f, 0.005274664144963026f, 0.005197357852011919f, 0.004680439364165068f, 0.005132061429321766f, 0.008080574683845043f, 0.008063895627856255f, 0.010532533749938011f, 0.005102963652461767f, 0.008128153160214424f, 0.014068060554564f, 0.012387891300022602f, 0.0018290983280166984f, 0.0023613804951310158f, 0.0075727603398263454f, 0.008306358009576797f, 0.0034616768825799227f, 0.010217009112238884f, 0.003895319765433669f, 0.009859333746135235f, 0.005252011585980654f, 0.013484383001923561f, 0.002998975571244955f, 0.01545647345483303f, 0.02249668538570404f, 0.0038081773091107607f, 0.008649380877614021f, 0.03047618828713894f, 0.008539616130292416f, 0.013106856495141983f, 0.0052205100655555725f, 0.0016561198281124234f, 0.01143617369234562f, 0.009395812638103962f, 0.005640173796564341f, 0.00781615637242794f, 0.010257111862301826f, 0.012803700752556324f, 0.010381147265434265f, 0.004081982187926769f, 0.05741387978196144f, 0.012648990377783775f, 0.006020114757120609f, 0.0029420212376862764f, 0.004009835887700319f, 0.0076269772835075855f, 0.002887632232159376f, 0.006598558276891708f, 0.007060650736093521f, 0.007498437073081732f, 0.015595180913805962f, 0.020201286301016808f, 0.02062942646443844f, 0.005409939680248499f, 0.02321936935186386f, 0.018252823501825333f, 0.006554222200065851f, 0.0043597957119345665f, 0.022477345541119576f, 0.011729404330253601f, 0.010346480645239353f, 0.007245644927024841f, 0.006157455500215292f, 0.005178205203264952f, 0.004790567327290773f, 0.008173488080501556f, 0.003903405973687768f, 0.01016185898333788f, 0.0035296364221721888f, 0.011842546053230762f, 0.005755073390901089f, 0.0015115321148186922f, 0.009982510469853878f, 0.005557296331971884f, 0.011109567247331142f, 0.004080996382981539f, 0.005721770226955414f, 0.003930282313376665f, 0.008251328021287918f, 0.008612673729658127f, 0.0029806147795170546f, 0.01000242680311203f, 0.006191782653331757f, 0.0067543457262218f, 0.0028412211686372757f, 0.005200044251978397f, 0.010543635115027428f, 0.01644127443432808f, 0.013865687884390354f, 0.007869484834372997f, 0.007439229637384415f, 0.002763776108622551f, 0.002470563631504774f, 0.004168894607573748f, 0.005800299812108278f, 0.005175556521862745f, 0.005283739417791367f, 0.025189347565174103f, 0.0076495627872645855f, 0.005339794792234898f, 0.006366930902004242f, 0.004435562528669834f, 0.0035004320088773966f, 0.004673704970628023f, 0.011301307938992977f, 0.0029393548611551523f, 0.004689796362072229f, 0.00765716889873147f, 0.011941814795136452f, 0.010402930900454521f, 0.005816259421408176f, 0.006703963968902826f, 0.02092680148780346f, 0.006262888200581074f, 0.011139465495944023f, 0.00728713721036911f, 0.008980578742921352f, 0.00634017214179039f, 0.0041359239257872105f, 0.0057831802405416965f, 0.015564247965812683f, 0.006768843159079552f, 0.009231246076524258f, 0.006784857250750065f, 0.018716266378760338f, 0.007620875257998705f, 0.007217269856482744f, 0.0030345511622726917f, 0.006402276922017336f, 0.01320772897452116f, 0.007532955147325993f, 0.003089157398790121f, 0.005355288274586201f, 0.0035292976535856724f, 0.005422742571681738f, 0.00784717220813036f, 0.00770968571305275f, 0.0065351324155926704f, 0.01605169102549553f, 0.007360448595136404f, 0.004297652281820774f, 0.008601474575698376f, 0.005843927152454853f, 0.003610003739595413f, 0.0029575831722468138f);
static const ai_u16 conv2d_45_t_out_0_shape_w_const_u16 = 8;
static const ai_u16 conv2d_45_t_out_0_shape_h_const_u16 = 8;

static const ai_u16 conv2d_46_t_in_0_shape_w_const_u16 = 8;
static const ai_u16 conv2d_46_t_in_0_shape_h_const_u16 = 8;
static const ai_u16 conv2d_46_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_46_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_46_t_in_0_shape_ch_const_u16 = 192;
static const ai_u16 conv2d_46_t_out_0_shape_ch_const_u16 = 32;
static const ai_i8 conv2d_46_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_46_t_out_0_fmt_zero_const_s8 = -20;
static const ai_float conv2d_46_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_46_t_out_0_fmt_scale_const_f32 = 0.30521687865257263f;
static const ai_float conv2d_46_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.005626226309686899f, 0.0052769603207707405f, 0.0109220240265131f, 0.00642929645255208f, 0.00700730737298727f, 0.009944505989551544f, 0.0065011135302484035f, 0.01195314060896635f, 0.013198708184063435f, 0.01891976408660412f, 0.0059166569262743f, 0.015671562403440475f, 0.005430274643003941f, 0.015838783234357834f, 0.006160880904644728f, 0.004464946687221527f, 0.02340042218565941f, 0.004964014515280724f, 0.0102187879383564f, 0.005531864706426859f, 0.007593442220240831f, 0.02004915475845337f, 0.019276779145002365f, 0.012901573441922665f, 0.01550170686095953f, 0.013982126489281654f, 0.012188824824988842f, 0.006877700332552195f, 0.005068191327154636f, 0.007060067728161812f, 0.00855337455868721f, 0.00871992763131857f);
static const ai_layer_format_type conv2d_46_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 conv2d_48_t_in_0_shape_w_const_u16 = 8;
static const ai_u16 conv2d_48_t_in_0_shape_h_const_u16 = 8;
static const ai_u16 conv2d_48_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_48_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_48_t_in_0_shape_ch_const_u16 = 32;
static const ai_u16 conv2d_48_t_out_0_shape_ch_const_u16 = 192;
static const ai_i8 conv2d_48_t_in_0_fmt_zero_const_s8 = -20;
static const ai_i8 conv2d_48_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_48_t_in_0_fmt_scale_const_f32 = 0.30521687865257263f;
static const ai_float conv2d_48_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_48_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0009074999834410846f, 0.0009287375723943114f, 0.0009477526764385402f, 0.0013425161596387625f, 0.0008388166315853596f, 0.0006657897029072046f, 0.0006159830954857171f, 0.0007921401993371546f, 0.001194702461361885f, 0.0007782391621731222f, 0.0006867058691568673f, 0.0005265821819193661f, 0.0009250703151337802f, 0.0007134911138564348f, 0.0013341398444026709f, 0.0004720118595287204f, 0.000489438243675977f, 0.0005176997510716319f, 0.000744607939850539f, 0.0008941517444327474f, 0.0007793604163452983f, 0.00042348692659288645f, 0.001260145683772862f, 0.0008949954644776881f, 0.0013272828655317426f, 0.0008667981601320207f, 0.0007700725109316409f, 0.0006838677800260484f, 0.0006369443726725876f, 0.0008954066433943808f, 0.0006034030811861157f, 0.0008029175805859268f, 0.0007076485780999064f, 0.0007552038296125829f, 0.0003377036191523075f, 0.0007935580797493458f, 0.0006175604648888111f, 0.0012091490207239985f, 0.0006208852864801884f, 0.0008404745603911579f, 0.0009001063299365342f, 0.0004967633285559714f, 0.0008415616466663778f, 0.0014262680197134614f, 0.000632250914350152f, 0.0006178512703627348f, 0.0006553244893439114f, 0.0006226160912774503f, 0.00012000973947579041f, 0.0006530866958200932f, 0.0005002411780878901f, 0.000495903252158314f, 0.0006838692934252322f, 0.0004763993201777339f, 0.0012927642092108727f, 0.0005863985861651599f, 0.0005833847681060433f, 0.0006094538257457316f, 0.0008790424908511341f, 0.0009465256007388234f, 0.001086617005057633f, 0.0007927544065751135f, 0.001059112255461514f, 0.0005538521218113601f, 0.0003463702159933746f, 0.0007946405676193535f, 0.0005668639787472785f, 0.0010161312529817224f, 0.0009988897945731878f, 0.0007125734118744731f, 0.000951494206674397f, 0.0006919243023730814f, 0.0011545034358277917f, 0.000531677738763392f, 0.0004737365816254169f, 0.0006727861473336816f, 0.0009193331934511662f, 0.0011851395247504115f, 0.0003775811637751758f, 0.0007046690443530679f, 0.0010190659668296576f, 0.0005751879070885479f, 0.0011231533717364073f, 0.001026925747282803f, 0.0005149002536199987f, 0.0007823779014870524f, 0.000589642848353833f, 0.0006639783969148993f, 0.0005455143400467932f, 0.0007240324630402029f, 0.00048498803516849875f, 0.0007544290856458247f, 0.000834308099001646f, 0.0008604527683928609f, 0.0006150687113404274f, 0.000823965878225863f, 0.00045660289470106363f, 0.0006310812314040959f, 0.0006376015953719616f, 0.000531151017639786f, 0.0006189178675413132f, 0.0006497287540696561f, 0.0007575939525850117f, 0.0006682039820589125f, 0.000740394985768944f, 0.0011506914161145687f, 0.0010470862034708261f, 0.0007356939604505897f, 0.0007718912675045431f, 0.0007879328913986683f, 0.0007583606638945639f, 0.0007658362737856805f, 0.0008868178119882941f, 0.0006749582826159894f, 0.0006611176650039852f, 0.0007453542202711105f, 0.0007884710212238133f, 0.0005504986038431525f, 0.0008258649613708258f, 0.000749017926864326f, 0.0011978150578215718f, 0.0010382075561210513f, 0.0012699553044512868f, 0.0010714303934946656f, 0.0010210267500951886f, 0.0006942623876966536f, 0.0007227372261695564f, 0.0007938190829008818f, 0.0005879528471268713f, 0.0009208702831529081f, 0.001094916369765997f, 0.0005406269920058548f, 0.0017178674461320043f, 0.0005587952327914536f, 0.000777953362558037f, 0.0016906220698729157f, 0.00043907645158469677f, 0.0008059490937739611f, 0.000820557470433414f, 0.0007517549674957991f, 0.0008788626291789114f, 0.0005748272524215281f, 0.000720708747394383f, 0.0007102666422724724f, 0.0009021374862641096f, 0.0010933780577033758f, 0.0007028476102277637f, 0.0010880230693146586f, 0.0008151173824444413f, 0.0006770468899048865f, 0.0006930981180630624f, 0.0005099010304547846f, 0.0007293191738426685f, 0.0005271646659821272f, 0.0009482831228524446f, 0.002111696405336261f, 0.0006573712453246117f, 0.0006706955027766526f, 0.0008525209850631654f, 0.00042591875535435975f, 0.000808832875918597f, 0.0011523234425112605f, 0.0009139651083387434f, 0.0005941016133874655f, 0.0009624023223295808f, 0.001489221234805882f, 0.0008615981787443161f, 0.00041864061495289207f, 0.0009435699903406203f, 0.00014640852168668061f, 0.001085435738787055f, 0.0007863142527639866f, 0.0011416145134717226f, 0.0009398690308444202f, 0.00041756045538932085f, 0.0005736990715377033f, 0.0007215686491690576f, 0.0005167924100533128f, 0.0006183612858876586f, 0.0005834249895997345f, 0.0009035824914462864f, 0.0011762625072151423f, 0.0009411703795194626f, 0.000506156065966934f, 0.00014447810826823115f, 0.0010910391574725509f, 0.0005745055968873203f, 0.0008679491584189236f, 0.0008121835417114198f, 0.0006837145192548633f, 0.000964558043051511f, 0.0007599749369546771f);
static const ai_layer_format_type conv2d_48_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_i8 conv2d_49_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 conv2d_49_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 conv2d_49_pad_before_t_in_0_shape_h_const_u32 = 8;

static const ai_u16 conv2d_49_t_in_0_shape_w_const_u16 = 10;
static const ai_u16 conv2d_49_t_in_0_shape_h_const_u16 = 10;
static const ai_u16 conv2d_49_t_in_0_shape_ch_const_u16 = 192;
static const ai_u16 conv2d_49_l_stride_1_const_u16 = 2;
static const ai_u16 conv2d_49_l_stride_0_const_u16 = 2;
static const ai_i8 conv2d_49_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_49_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_49_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_49_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_49_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.004151937551796436f, 0.0033385646529495716f, 0.003347759135067463f, 0.003122643567621708f, 0.0037664971314370632f, 0.005106151103973389f, 0.0036135069094598293f, 0.005695530213415623f, 0.0029629941564053297f, 0.002286297967657447f, 0.002878297818824649f, 0.0037699691019952297f, 0.0069530075415968895f, 0.004071996547281742f, 0.00316197844222188f, 0.0043418556451797485f, 0.004970289766788483f, 0.0029725362546741962f, 0.0018977818544954062f, 0.006429985631257296f, 0.005650474689900875f, 0.005271905101835728f, 0.0019534113816916943f, 0.004532749764621258f, 0.005232305731624365f, 0.0046759420074522495f, 0.005804776679724455f, 0.004295785445719957f, 0.004875910002738237f, 0.002801358699798584f, 0.0028988406993448734f, 0.004585250746458769f, 0.005375096574425697f, 0.0028006499633193016f, 0.005957331508398056f, 0.006822820752859116f, 0.0036744950339198112f, 0.004483610857278109f, 0.006957183592021465f, 0.0018262399826198816f, 0.004425249062478542f, 0.003621483687311411f, 0.0034517564345151186f, 0.005317294038832188f, 0.00351717765443027f, 0.005441602319478989f, 0.004125318024307489f, 0.0065428148955106735f, 0.019595857709646225f, 0.002748193684965372f, 0.004422339610755444f, 0.0034383023157715797f, 0.0028312120120972395f, 0.004834120161831379f, 0.0029583917930722237f, 0.005127027165144682f, 0.006207569502294064f, 0.003850710578262806f, 0.0038031728472560644f, 0.002316487953066826f, 0.0036166838835924864f, 0.0056345341727137566f, 0.0038888994604349136f, 0.005032807122915983f, 0.003580981632694602f, 0.0023665667977184057f, 0.0030152134131640196f, 0.006606963463127613f, 0.0037960107438266277f, 0.00211793789640069f, 0.004704812541604042f, 0.0026378731708973646f, 0.012583892792463303f, 0.006041315849870443f, 0.004216080065816641f, 0.0034189815632998943f, 0.0037189312279224396f, 0.003920261282473803f, 0.006252574734389782f, 0.0037998633924871683f, 0.0046848817728459835f, 0.0027859779074788094f, 0.004812197759747505f, 0.009990638121962547f, 0.004464227240532637f, 0.003469084622338414f, 0.0049322075210511684f, 0.003633153857663274f, 0.004466994199901819f, 0.004907372873276472f, 0.007042237091809511f, 0.005653268191963434f, 0.005380771588534117f, 0.005069200415164232f, 0.003457383718341589f, 0.003308717394247651f, 0.0036414675414562225f, 0.00820503756403923f, 0.003813675604760647f, 0.003701964858919382f, 0.004145442973822355f, 0.003950303420424461f, 0.003375128610059619f, 0.004773514810949564f, 0.005005518440157175f, 0.007604033686220646f, 0.0071319337002933025f, 0.004056995268911123f, 0.0021945827174931765f, 0.003916106652468443f, 0.007022556848824024f, 0.004073841497302055f, 0.004618489183485508f, 0.004986957181245089f, 0.006456953007727861f, 0.003012872999534011f, 0.0037770867347717285f, 0.004016779828816652f, 0.0040743290446698666f, 0.007909775711596012f, 0.005947952624410391f, 0.0038743256591260433f, 0.00617446331307292f, 0.0037467596121132374f, 0.0020980299450457096f, 0.002713001798838377f, 0.002215059008449316f, 0.003087254473939538f, 0.0038487701676785946f, 0.00476030632853508f, 0.002472628839313984f, 0.004446073900908232f, 0.0027519860304892063f, 0.0027543979231268167f, 0.008644893765449524f, 0.0016290991334244609f, 0.005139938555657864f, 0.002765666926279664f, 0.0034556433092802763f, 0.0038790905382484198f, 0.0038856850005686283f, 0.003075322834774852f, 0.0029946041759103537f, 0.004345928784459829f, 0.003807992907240987f, 0.005992666352540255f, 0.002804657444357872f, 0.004530178848654032f, 0.004352070391178131f, 0.003334006294608116f, 0.0033500732388347387f, 0.006721665151417255f, 0.0021440060809254646f, 0.004214216955006123f, 0.0037446627393364906f, 0.0028458537999540567f, 0.005737410392612219f, 0.005693692248314619f, 0.002834920072928071f, 0.002789468737319112f, 0.0058811130002141f, 0.0018912078812718391f, 0.0033511922229081392f, 0.002186276949942112f, 0.006383605767041445f, 0.004892251919955015f, 0.003126508789137006f, 0.005666711833328009f, 0.0015953098190948367f, 0.007235340774059296f, 0.003775332123041153f, 0.0026544330175966024f, 0.0036675380542874336f, 0.0029062428511679173f, 0.0026601245626807213f, 0.0036641848273575306f, 0.004437262192368507f, 0.005304181948304176f, 0.003385520074516535f, 0.005623801611363888f, 0.0029425218235701323f, 0.00564608396962285f, 0.005814606789499521f, 0.004506393801420927f, 0.020551005378365517f, 0.0056862845085561275f, 0.004401439800858498f, 0.0028833611868321896f, 0.004508123733103275f, 0.0036246117670089006f, 0.002679555444046855f, 0.006618042942136526f);
static const ai_u16 conv2d_49_t_out_0_shape_w_const_u16 = 4;
static const ai_u16 conv2d_49_t_out_0_shape_h_const_u16 = 4;

static const ai_u16 conv2d_50_t_in_0_shape_w_const_u16 = 4;
static const ai_u16 conv2d_50_t_in_0_shape_h_const_u16 = 4;
static const ai_u16 conv2d_50_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_50_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_50_t_in_0_shape_ch_const_u16 = 192;
static const ai_u16 conv2d_50_t_out_0_shape_ch_const_u16 = 56;
static const ai_i8 conv2d_50_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_50_t_out_0_fmt_zero_const_s8 = 3;
static const ai_float conv2d_50_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_50_t_out_0_fmt_scale_const_f32 = 0.16475240886211395f;
static const ai_float conv2d_50_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.010281436145305634f, 0.007061775773763657f, 0.012028403580188751f, 0.0071714832447469234f, 0.00918346457183361f, 0.010199032723903656f, 0.009924613870680332f, 0.0071434201672673225f, 0.009504430927336216f, 0.007515275850892067f, 0.007161491084843874f, 0.008184997364878654f, 0.00823455024510622f, 0.011875181458890438f, 0.010369367897510529f, 0.009487582370638847f, 0.009586328640580177f, 0.009234528057277203f, 0.013057872653007507f, 0.010940375737845898f, 0.011106549762189388f, 0.00964626856148243f, 0.006779937073588371f, 0.01100378017872572f, 0.010201235301792622f, 0.009784507565200329f, 0.011573957279324532f, 0.009969978593289852f, 0.009182707406580448f, 0.009935261681675911f, 0.00925978273153305f, 0.00994813721626997f, 0.009659795090556145f, 0.01002842653542757f, 0.012335870414972305f, 0.011027908883988857f, 0.009942284785211086f, 0.007269782945513725f, 0.009705247357487679f, 0.01058467011898756f, 0.01292054820805788f, 0.008909734897315502f, 0.011358649469912052f, 0.010048587806522846f, 0.012206753715872765f, 0.007930989377200603f, 0.007068187929689884f, 0.009506694972515106f, 0.006603793706744909f, 0.009498209692537785f, 0.007685256190598011f, 0.009245290420949459f, 0.008533968590199947f, 0.010549161583185196f, 0.010263956151902676f, 0.005713467486202717f);
static const ai_layer_format_type conv2d_50_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_u16 conv2d_51_t_in_0_shape_w_const_u16 = 4;
static const ai_u16 conv2d_51_t_in_0_shape_h_const_u16 = 4;
static const ai_u16 conv2d_51_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_51_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_51_t_in_0_shape_ch_const_u16 = 56;
static const ai_u16 conv2d_51_t_out_0_shape_ch_const_u16 = 336;
static const ai_i8 conv2d_51_t_in_0_fmt_zero_const_s8 = 3;
static const ai_i8 conv2d_51_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_51_t_in_0_fmt_scale_const_f32 = 0.16475240886211395f;
static const ai_float conv2d_51_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_51_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.00044825999066233635f, 0.0011642136378213763f, 0.0009410465718246996f, 0.000605667766649276f, 0.0006773641216568649f, 0.0009327296284027398f, 0.000628564739599824f, 0.0006718182703480124f, 0.0006324269925244153f, 0.0006194591405801475f, 0.0006237942725419998f, 0.0004123267426621169f, 0.000706620397977531f, 0.00037852663081139326f, 0.0007786129717715085f, 0.0006512033869512379f, 0.0009579177713021636f, 0.0012174961157143116f, 0.0007871814887039363f, 0.0009572397684678435f, 0.0008540789713151753f, 0.0009895390830934048f, 0.000455674366094172f, 0.00012640649219974875f, 0.0003543448983691633f, 0.0008151807705871761f, 0.0010815595742315054f, 0.0004611745534930378f, 0.00023598606640007347f, 0.0005568126798607409f, 0.00032081606332212687f, 0.0004028694238513708f, 0.0008509763283655047f, 0.0004097599594388157f, 0.0007938020280562341f, 0.00029115533106960356f, 0.0012903183232992887f, 0.0005982354050502181f, 0.0006851960206404328f, 0.0012651487486436963f, 0.0005905277212150395f, 0.0005784571985714138f, 0.00041319229057990015f, 0.0009997658198699355f, 0.000349161826306954f, 0.0008685500943101943f, 0.0007509042043238878f, 0.00048098518163897097f, 0.001347633427940309f, 0.000455840170616284f, 0.000727443490177393f, 0.0008078916580416262f, 0.0004067147383466363f, 0.0009125325013883412f, 0.0005165190668776631f, 0.0006152105052024126f, 0.0005005149869248271f, 0.00034606625558808446f, 0.0006363179418258369f, 0.0011003636755049229f, 0.0005774159799329937f, 0.0011127956677228212f, 0.0005023376434110105f, 0.00035663385642692447f, 0.0004477532347664237f, 0.0005979659617878497f, 0.000496996333822608f, 0.00033817693474702537f, 0.0008850889862515032f, 0.000669760222081095f, 0.0004342063330113888f, 0.0006445274339057505f, 0.0006465344340540469f, 0.00048035310464911163f, 0.000964443024713546f, 0.0005684051429852843f, 0.0011639335425570607f, 0.0004940198268741369f, 0.0003964314528275281f, 0.0005612724926322699f, 0.0005712800193578005f, 0.0008821499650366604f, 0.001183264539577067f, 0.0006438989657908678f, 0.00042471851338632405f, 0.0007707138429395854f, 0.0005471695913001895f, 0.0006093095289543271f, 0.0006750752800144255f, 0.0004029124975204468f, 0.0010833899723365903f, 0.0011223710607737303f, 0.0002997207338921726f, 0.0004633212520275265f, 0.001796600641682744f, 0.0008736451272852719f, 0.0006381258717738092f, 0.0008959748665802181f, 0.00019050690752919763f, 0.0008726021042093635f, 0.0011406041448935866f, 0.00043920971802435815f, 0.00036785355769097805f, 0.0006489089573733509f, 0.0010510089341551065f, 0.0010369764640927315f, 0.00044016659376211464f, 0.0008413844625465572f, 0.0011305372463539243f, 0.000978735159151256f, 0.0010204979917034507f, 0.000664224149659276f, 0.0005972488434053957f, 0.0007549990550614893f, 0.0007841248880140483f, 0.0005815979093313217f, 0.0002173227840103209f, 0.0008393999887630343f, 0.0011950235348194838f, 0.000980796874500811f, 0.0007427262025885284f, 0.0005784805398434401f, 0.0005515788798220456f, 0.0011400418588891625f, 0.00043558856123127043f, 0.0004887359100393951f, 0.0011708408128470182f, 0.0003251057642046362f, 0.0014451427850872278f, 0.0005458518862724304f, 0.0009838734986260533f, 0.0005974298692308366f, 0.0008797086193226278f, 0.0004352414980530739f, 0.0009490912780165672f, 0.0009528228547424078f, 0.000329795730067417f, 0.0006871468503959477f, 0.0005606160848401487f, 0.00035635402309708297f, 0.00041961640818044543f, 0.0005059178802184761f, 0.0008834192412905395f, 0.0004781230818480253f, 0.0022307264152914286f, 0.0008148649940267205f, 0.0006176663446240127f, 0.0009845845634117723f, 0.0004169525345787406f, 0.0012753010960295796f, 0.0008546730969101191f, 0.0006233166204765439f, 0.0008376279147341847f, 0.0006916582933627069f, 0.0005486226873472333f, 0.00033047955366782844f, 0.0008096709498204291f, 0.0008734441362321377f, 0.0009464858449064195f, 0.00050998420920223f, 0.0006177598843351007f, 0.0008121091523207724f, 0.00036892513162456453f, 0.0006098963785916567f, 0.0005412111640907824f, 0.000939507910516113f, 0.0004909745184704661f, 0.00031223654514178634f, 0.0003694756014738232f, 0.0007723679882474244f, 0.0014070785837247968f, 0.0006606708629988134f, 0.00021896586986258626f, 0.0004647972818929702f, 0.0003874623798765242f, 0.00028646961436606944f, 0.0006741040851920843f, 0.000912003219127655f, 0.0005164205795153975f, 0.0009720126399770379f, 0.0009618933545425534f, 0.00041486541158519685f, 0.0009273874456994236f, 0.0013063476653769612f, 0.0014392600860446692f, 0.0009392350330017507f, 0.001049202517606318f, 0.00046333266072906554f, 0.0007139096269384027f, 0.00034571648575365543f, 0.00036515199462883174f, 0.0011400784133002162f, 0.0006957820151001215f, 0.0006627689581364393f, 0.0013115809997543693f, 0.000904320040717721f, 0.00057618273422122f, 0.0008012913749553263f, 0.0006114593707025051f, 0.0010117841884493828f, 0.0005975094973109663f, 0.001121600391343236f, 0.0008800107170827687f, 0.0005334655288606882f, 0.0010250062914565206f, 0.0008843672112561762f, 0.00045969136408530176f, 0.0016924735391512513f, 0.0006465685437433422f, 0.0007504537934437394f, 0.0009725689305923879f, 0.0008735593874007463f, 0.0005616164999082685f, 0.00048493503709323704f, 0.001771134091541171f, 0.0004135690105613321f, 0.00017595684039406478f, 0.0007913032895885408f, 0.0007137941429391503f, 0.0006453472306020558f, 0.0009266321430914104f, 0.0005285173538140953f, 0.0009120645700022578f, 0.0007790780509822071f, 0.0006082748295739293f, 0.000772983068600297f, 0.0004998038057237864f, 0.0010251164203509688f, 0.0007810536772012711f, 0.0005836302880197763f, 0.00023667362984269857f, 0.0012062725145369768f, 0.00021206779638305306f, 0.0013272756477817893f, 0.00048719465848989785f, 0.0008231713436543941f, 0.0005465224967338145f, 0.0006363531574606895f, 0.0008723042556084692f, 0.0005895735812373459f, 0.0019938002806156874f, 0.000710318679921329f, 0.000578982406295836f, 0.0006914955447427928f, 0.0006421448779292405f, 0.0008766051614657044f, 0.0011703912168741226f, 0.00033279749914072454f, 0.0007189547177404165f, 0.0010944540845230222f, 0.0008877989021129906f, 0.0009751729085110128f, 0.0006671606679446995f, 0.00018987389921676368f, 0.0010910517303273082f, 0.0005245396168902516f, 0.0003662334056571126f, 0.0008838934008963406f, 0.0010294470703229308f, 0.0007213670760393143f, 0.000612496689427644f, 0.0007579532102681696f, 0.001108491444028914f, 0.0012109592789784074f, 0.0004565923591144383f, 0.0004855668521486223f, 0.0009391602361574769f, 0.000993675203062594f, 0.0008101222338154912f, 0.0006105855572968721f, 0.00032878172351047397f, 0.0007031267741695046f, 0.0003823492443189025f, 0.0011958752293139696f, 0.00037750304909422994f, 0.0005188081413507462f, 0.0012224508682265878f, 0.0010248384205624461f, 0.0003329248575028032f, 0.0006962487823329866f, 0.00024219324404839426f, 0.0010778657160699368f, 0.000583233661018312f, 0.0010225418955087662f, 0.0008096836390905082f, 0.0005661947652697563f, 0.0005853896145708859f, 0.0003265842969994992f, 0.0007997584762051702f, 0.0008056539227254689f, 0.0005901926197111607f, 0.0006218130583874881f, 0.0005895327194593847f, 0.00045165084884501994f, 0.0007842826307751238f, 0.0006286163115873933f, 0.0006385906599462032f, 0.0006239438662305474f, 0.0011386172845959663f, 0.0005061296396888793f, 0.0009758895030245185f, 0.0005902900011278689f, 0.0008381895022466779f, 0.000742904026992619f, 0.0005953305517323315f, 0.0005177536513656378f, 0.0008352791774086654f, 0.0007023174548521638f, 0.000380455021513626f, 0.0012827628524973989f, 0.0005397821078076959f, 0.0011278301244601607f, 0.0002649759117048234f, 0.00033532045199535787f, 0.0004595395002979785f, 0.0010770746739581227f, 0.0005937920068390667f, 0.0005264391074888408f, 0.0005696064908988774f, 0.0013676607050001621f, 0.000394448492443189f, 0.0005710349651053548f, 0.0005909496103413403f, 0.00043108678073622286f, 0.0007806639187037945f, 0.00047312697279267013f, 0.00039876747177913785f, 0.0005054281791672111f, 0.0014289134414866567f, 0.0015027908375486732f, 0.0007408689125441015f, 0.0004895517486147583f, 0.0011302933562546968f, 0.001987895229831338f, 0.0012430172646418214f, 0.00044652342330664396f);
static const ai_layer_format_type conv2d_51_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_i8 conv2d_52_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 conv2d_52_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 conv2d_52_pad_before_t_in_0_shape_h_const_u32 = 4;

static const ai_u16 conv2d_52_t_in_0_shape_w_const_u16 = 6;
static const ai_u16 conv2d_52_t_in_0_shape_h_const_u16 = 6;
static const ai_u16 conv2d_52_t_in_0_shape_ch_const_u16 = 336;
static const ai_u16 conv2d_52_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_52_l_stride_0_const_u16 = 1;
static const ai_i8 conv2d_52_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_52_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_52_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_52_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_52_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.004628154449164867f, 0.011535124853253365f, 0.006502679083496332f, 0.012569321319460869f, 0.0053933290764689445f, 0.005576349329203367f, 0.004242741968482733f, 0.009559247642755508f, 0.00851348415017128f, 0.008523239754140377f, 0.009331763722002506f, 0.008408453315496445f, 0.0022381767630577087f, 0.01013009250164032f, 0.006025723647326231f, 0.005468170158565044f, 0.0087682344019413f, 0.0046331435441970825f, 0.006436644122004509f, 0.011537213809788227f, 0.0053404695354402065f, 0.0068432181142270565f, 0.008430219255387783f, 0.012922858819365501f, 0.011911321431398392f, 0.011101942509412766f, 0.008295778185129166f, 0.004653755109757185f, 0.013543208129703999f, 0.006679717451334f, 0.0077141751535236835f, 0.011229291558265686f, 0.005496395751833916f, 0.02015669085085392f, 0.009204058907926083f, 0.016246527433395386f, 0.013719664886593819f, 0.007749255746603012f, 0.008035986684262753f, 0.006398375146090984f, 0.015849299728870392f, 0.01263506431132555f, 0.007656200788915157f, 0.010282539762556553f, 0.02552805282175541f, 0.012818142771720886f, 0.006976408883929253f, 0.007818296551704407f, 0.003516765311360359f, 0.011100847274065018f, 0.007252950221300125f, 0.0029536844231188297f, 0.01062504667788744f, 0.005429512821137905f, 0.007964096963405609f, 0.007993226870894432f, 0.006191831547766924f, 0.00784080196171999f, 0.005471883807331324f, 0.011160398833453655f, 0.012529195286333561f, 0.007051174063235521f, 0.014774016104638577f, 0.009781568311154842f, 0.01715080998837948f, 0.0032769751269370317f, 0.00987975113093853f, 0.009531884454190731f, 0.0037373676896095276f, 0.011628136970102787f, 0.013391085900366306f, 0.006210405845195055f, 0.0038866563700139523f, 0.016377460211515427f, 0.007387673016637564f, 0.007769371848553419f, 0.008236613124608994f, 0.003909994848072529f, 0.007789057679474354f, 0.007131070364266634f, 0.019378090277314186f, 0.016299141570925713f, 0.009960402734577656f, 0.007499671075493097f, 0.012107601389288902f, 0.004195597022771835f, 0.0037587599363178015f, 0.006895198952406645f, 0.008771415799856186f, 0.013985870406031609f, 0.012152780778706074f, 0.010066863149404526f, 0.005083825904875994f, 0.016676094383001328f, 0.00812321063131094f, 0.012552781961858273f, 0.008235032670199871f, 0.016546785831451416f, 0.018639447167515755f, 0.003956171218305826f, 0.004150827415287495f, 0.012515670619904995f, 0.010309853591024876f, 0.0047398763708770275f, 0.00554656283929944f, 0.0059293401427567005f, 0.005864303093403578f, 0.014308176934719086f, 0.009569576010107994f, 0.004601854830980301f, 0.011165757663547993f, 0.007659945171326399f, 0.007194373290985823f, 0.0024621516931802034f, 0.006252483930438757f, 0.020305631682276726f, 0.018453164026141167f, 0.0049769994802773f, 0.007273141294717789f, 0.010507266037166119f, 0.008180242031812668f, 0.007953022606670856f, 0.0061988141387701035f, 0.004284595604985952f, 0.005509188864380121f, 0.010883808135986328f, 0.013428838923573494f, 0.020420392975211143f, 0.0046419487334787846f, 0.009308312088251114f, 0.007627523969858885f, 0.00626870384439826f, 0.0037440573796629906f, 0.011992072686553001f, 0.008711096830666065f, 0.012084584683179855f, 0.019577879458665848f, 0.006716820411384106f, 0.006133443210273981f, 0.010524621233344078f, 0.008128203451633453f, 0.010441992431879044f, 0.003140921937301755f, 0.007046698126941919f, 0.004981297999620438f, 0.004022940527647734f, 0.0071224551647901535f, 0.007139093708246946f, 0.010980445891618729f, 0.004098022356629372f, 0.0069285244680941105f, 0.005096734967082739f, 0.0035594794899225235f, 0.004622926004230976f, 0.011093458160758018f, 0.012623445130884647f, 0.00968779157847166f, 0.008545209653675556f, 0.009434887208044529f, 0.007228972390294075f, 0.009687441401183605f, 0.006473583169281483f, 0.009406644850969315f, 0.004576040897518396f, 0.006264278199523687f, 0.009618342854082584f, 0.008364380337297916f, 0.010604484006762505f, 0.008774211630225182f, 0.006262478418648243f, 0.005369951017200947f, 0.009242826141417027f, 0.014824158512055874f, 0.0071431552059948444f, 0.009408622980117798f, 0.013762927614152431f, 0.006869478151202202f, 0.0032038032077252865f, 0.018890168517827988f, 0.013199794106185436f, 0.005446071270853281f, 0.01251031644642353f, 0.0029503488913178444f, 0.008944456465542316f, 0.014990903437137604f, 0.0090787997469306f, 0.006052864249795675f, 0.004822189919650555f, 0.004910229705274105f, 0.01245963852852583f, 0.14870969951152802f, 0.00912479218095541f, 0.009918808937072754f, 0.008867049589753151f, 0.006363054737448692f, 0.006842134520411491f, 0.005310289096087217f, 0.004051185213029385f, 0.013332737609744072f, 0.008430536836385727f, 0.007265217136591673f, 0.008501134812831879f, 0.004526799079030752f, 0.008211089298129082f, 0.0038408483378589153f, 0.00595575338229537f, 0.009218614548444748f, 0.0030716184992343187f, 0.008714959025382996f, 0.004992387723177671f, 0.002926279790699482f, 0.011752641759812832f, 0.005690343212336302f, 0.011915966868400574f, 0.010174732655286789f, 0.008686669170856476f, 0.017161868512630463f, 0.0029791181441396475f, 0.009463557042181492f, 0.005511696916073561f, 0.006015199236571789f, 0.006048427429050207f, 0.010766582563519478f, 0.006636111065745354f, 0.004482747055590153f, 0.005304331425577402f, 0.007592425215989351f, 0.006765221245586872f, 0.004893651697784662f, 0.008091268129646778f, 0.010541511699557304f, 0.004062822554260492f, 0.015513482503592968f, 0.0032401704229414463f, 0.009975855238735676f, 0.005915302317589521f, 0.008434003219008446f, 0.006667507346719503f, 0.0027013930957764387f, 0.00737789086997509f, 0.020227566361427307f, 0.008027767762541771f, 0.007146418560296297f, 0.0054656267166137695f, 0.005954436492174864f, 0.014502231031656265f, 0.0063061220571398735f, 0.008308690041303635f, 0.010533178225159645f, 0.012207476422190666f, 0.004851114004850388f, 0.0034363986924290657f, 0.00913435872644186f, 0.012427770532667637f, 0.006533895153552294f, 0.007701460737735033f, 0.01524067297577858f, 0.002261006971821189f, 0.004630180075764656f, 0.012559461407363415f, 0.006041319575160742f, 0.006682413164526224f, 0.004632461350411177f, 0.002382436767220497f, 0.007484474219381809f, 0.027517782524228096f, 0.007455592509359121f, 0.005870001390576363f, 0.014459706842899323f, 0.008727981708943844f, 0.01647518388926983f, 0.006422894075512886f, 0.008099399507045746f, 0.0081778634339571f, 0.00818475428968668f, 0.005627011880278587f, 0.019721612334251404f, 0.012282857671380043f, 0.008636643178761005f, 0.007642013020813465f, 0.014378884807229042f, 0.011721259914338589f, 0.007435936480760574f, 0.017214413732290268f, 0.007086270954459906f, 0.008214099332690239f, 0.010377145372331142f, 0.010539264418184757f, 0.006404609885066748f, 0.018398988991975784f, 0.009802854619920254f, 0.008568069897592068f, 0.011962486431002617f, 0.00699147954583168f, 0.006923488341271877f, 0.008272630162537098f, 0.005789593327790499f, 0.007491086144000292f, 0.014057276770472527f, 0.009096195921301842f, 0.004946839530020952f, 0.0067634666338562965f, 0.0060166893526911736f, 0.0069874911569058895f, 0.006029783748090267f, 0.005486002657562494f, 0.008500269614160061f, 0.005431459750980139f, 0.00685877026990056f, 0.005321615841239691f, 0.010388502851128578f, 0.004778783302754164f, 0.014380269683897495f, 0.009201345033943653f, 0.00829935260117054f, 0.01891063153743744f, 0.00658186012879014f, 0.012321842834353447f, 0.007649477571249008f, 0.004767449572682381f, 0.016405876725912094f, 0.012384163215756416f, 0.010997344739735126f, 0.006202017422765493f, 0.018180113285779953f, 0.012162845581769943f, 0.01101410947740078f, 0.008749200962483883f, 0.005851848516613245f, 0.006507862824946642f, 0.007690165191888809f, 0.006586248986423016f, 0.00980906281620264f, 0.004022571723908186f, 0.012828168459236622f, 0.018524469807744026f);
static const ai_u16 conv2d_52_t_out_0_shape_w_const_u16 = 4;
static const ai_u16 conv2d_52_t_out_0_shape_h_const_u16 = 4;

static const ai_u16 conv2d_53_t_in_0_shape_w_const_u16 = 4;
static const ai_u16 conv2d_53_t_in_0_shape_h_const_u16 = 4;
static const ai_u16 conv2d_53_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_53_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_53_t_in_0_shape_ch_const_u16 = 336;
static const ai_u16 conv2d_53_t_out_0_shape_ch_const_u16 = 56;
static const ai_i8 conv2d_53_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_53_t_out_0_fmt_zero_const_s8 = 6;
static const ai_float conv2d_53_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_53_t_out_0_fmt_scale_const_f32 = 0.17767465114593506f;
static const ai_float conv2d_53_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.01015836838632822f, 0.007388358004391193f, 0.004511324688792229f, 0.011556360870599747f, 0.008014175109565258f, 0.006627054885029793f, 0.010126211680471897f, 0.008505587466061115f, 0.0029768471140414476f, 0.013440201990306377f, 0.006333487574011087f, 0.005159991793334484f, 0.0070519535802304745f, 0.006374745164066553f, 0.003678342094644904f, 0.004796857945621014f, 0.01071846578270197f, 0.0036466217134147882f, 0.010628443211317062f, 0.0058467998169362545f, 0.007209511008113623f, 0.0067808665335178375f, 0.01549393031746149f, 0.009792527183890343f, 0.00789734162390232f, 0.006382064428180456f, 0.005017675459384918f, 0.0031013363040983677f, 0.006361016072332859f, 0.003365609096363187f, 0.004054299555718899f, 0.0033861997071653605f, 0.005535010248422623f, 0.00440916046500206f, 0.010393014177680016f, 0.0053988974541425705f, 0.0058441078290343285f, 0.004421250894665718f, 0.008422506973147392f, 0.0037191598676145077f, 0.006713381968438625f, 0.004261479247361422f, 0.0054794070310890675f, 0.004965090658515692f, 0.0029735113494098186f, 0.005267253145575523f, 0.010973923839628696f, 0.00380347459577024f, 0.002808596473187208f, 0.005215256009250879f, 0.0044618695974349976f, 0.006915505044162273f, 0.005162748508155346f, 0.010177127085626125f, 0.005905458237975836f, 0.005112797487527132f);
static const ai_layer_format_type conv2d_53_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 conv2d_55_t_in_0_shape_w_const_u16 = 4;
static const ai_u16 conv2d_55_t_in_0_shape_h_const_u16 = 4;
static const ai_u16 conv2d_55_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_55_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_55_t_in_0_shape_ch_const_u16 = 56;
static const ai_u16 conv2d_55_t_out_0_shape_ch_const_u16 = 336;
static const ai_i8 conv2d_55_t_in_0_fmt_zero_const_s8 = 6;
static const ai_i8 conv2d_55_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_55_t_in_0_fmt_scale_const_f32 = 0.17767465114593506f;
static const ai_float conv2d_55_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_55_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0007493348093703389f, 0.0005196940037421882f, 0.0007188020972535014f, 0.00028859046869911253f, 0.0007436881423927844f, 0.0003041705349460244f, 0.00024156052677426487f, 0.0007896453607827425f, 0.000691032619215548f, 0.0007815567078068852f, 0.00037848629290238023f, 0.000393687398172915f, 0.000478447531349957f, 0.00020032364409416914f, 0.00034003431210294366f, 0.00040867822826839983f, 0.0007542050443589687f, 0.0006094976561143994f, 0.0006643757806159556f, 0.0010587518336251378f, 0.0007930323481559753f, 0.0007160333916544914f, 0.0006062376778572798f, 0.0009515737765468657f, 0.0004879937623627484f, 0.002598920837044716f, 0.0005509700276888907f, 0.0007727635093033314f, 0.0005260392208583653f, 0.00016586952551733702f, 0.0003780704864766449f, 0.0007581846439279616f, 0.0009868753841146827f, 0.0002322886575711891f, 0.000494304287713021f, 0.00039094631210900843f, 0.0009750067256391048f, 0.0009487807401455939f, 0.0005265322979539633f, 0.0009813809301704168f, 0.000683792750351131f, 0.0007333579706028104f, 0.000604171771556139f, 0.0005220327875576913f, 0.0005867609870620072f, 0.0004167052102275193f, 0.00043105773511342704f, 0.000961327226832509f, 0.0006109629175625741f, 0.0004013344587292522f, 0.0002708157990127802f, 0.00038401587517000735f, 0.00036078423727303743f, 0.0005163567839190364f, 0.0005193998222239316f, 0.0004324991605244577f, 0.00020731668337248266f, 0.0005472421762533486f, 0.000494346662890166f, 0.0008630536613054574f, 0.0007818886660970747f, 0.00041856488678604364f, 0.0006570016266778111f, 0.0007017719326540828f, 0.0007643562857992947f, 0.0005319046322256327f, 0.00041286865598522127f, 0.0007932139560580254f, 0.0008888066513463855f, 0.00027293607126921415f, 0.00038270934601314366f, 0.0004112226888537407f, 0.000445873592980206f, 0.0007197661907412112f, 0.0006255098269321024f, 0.0006477631977759302f, 0.0006961051840335131f, 0.0011120685376226902f, 0.00025228707818314433f, 0.0006819324335083365f, 0.0007062210352160037f, 0.0007220837287604809f, 0.0008764297235757113f, 0.0008648891234770417f, 0.000791581638623029f, 0.000477276393212378f, 0.0005599118885584176f, 0.0011090938933193684f, 0.000841387314721942f, 0.0004444123769644648f, 0.0005801518564112484f, 0.0006447206251323223f, 0.0004887770628556609f, 0.000675529008731246f, 0.0005763123626820743f, 0.0010016622254624963f, 0.00040776116657070816f, 0.0004970795707777143f, 0.00033266705577261746f, 0.0005675627617165446f, 0.0002638692385517061f, 0.0004019693878944963f, 0.000388128450140357f, 0.00040785406599752605f, 0.00033741581137292087f, 0.0007191365584731102f, 0.000744533899705857f, 0.0006850712234154344f, 0.00092798174591735f, 0.0005509158363565803f, 0.0005227153887972236f, 0.0006181620992720127f, 0.0009010797366499901f, 0.0019529558485373855f, 0.0006342798005789518f, 0.0005158133571967483f, 0.0005829140427522361f, 0.0005711380508728325f, 0.0008509548497386277f, 0.00043355522211641073f, 0.0008232576074078679f, 0.0010132803581655025f, 0.000501588627230376f, 0.0006796909729018807f, 0.00034290298935957253f, 0.0006457970594055951f, 0.00038471355219371617f, 0.00025547112454660237f, 0.0006888096686452627f, 0.0005102430004626513f, 0.0013855041470378637f, 0.0004264019662514329f, 0.0006845356547273695f, 0.00039869837928563356f, 0.0006751407054252923f, 0.0005559730343520641f, 0.00031362802837975323f, 0.0005066522862762213f, 0.0002174643741454929f, 0.0008324747323058546f, 0.0004933513118885458f, 0.0003900117299053818f, 0.0007848528330214322f, 0.0007466195384040475f, 0.000412620633142069f, 0.0006383876316249371f, 0.0005793605814687908f, 0.0007810425013303757f, 0.00044596681254915893f, 0.0008419265504926443f, 0.0007421823684126139f, 0.0004544727853499353f, 0.0005891809705644846f, 0.00079117133282125f, 0.00016700031119398773f, 0.0008575921528972685f, 0.0011465911520645022f, 0.000726867641787976f, 0.00022870476823300123f, 0.0005973911611363292f, 0.0004902119399048388f, 0.0005252622650004923f, 0.0003838328702840954f, 0.0004434737202245742f, 0.0009559460449963808f, 0.0011332465801388025f, 0.0007489167037419975f, 0.001712657161988318f, 0.0003372046339791268f, 0.0004728824715130031f, 0.0008742443169467151f, 0.0007896577008068562f, 0.0010263480944558978f, 0.0007479620398953557f, 0.0004646169545594603f, 0.0009565676446072757f, 0.0006700747180730104f, 0.00032800674671307206f, 0.0004311103839427233f, 0.0005964052979834378f, 0.00047870088019408286f, 0.0005267887027002871f, 0.0007612842600792646f, 0.0006706162239424884f, 0.0008299347828142345f, 0.0006160279153846204f, 0.0010428371606394649f, 0.0006682865205220878f, 0.0009166211821138859f, 0.0006533152773045003f, 0.0009511529933661222f, 0.0014222830068320036f, 0.0006287038559094071f, 0.0008414170006290078f, 0.00116275018081069f, 0.0003976526204496622f, 0.0013027162058278918f, 0.0006579500040970743f, 0.0005376460612751544f, 0.00035187765024602413f, 0.00019923744548577815f, 0.0009423125884495676f, 0.0009020256693474948f, 0.0004888190305791795f, 0.0006302557303570211f, 0.0006908646901138127f, 0.0007868662942200899f, 0.00034914127900265157f, 0.0005796446348540485f, 0.0003719658707268536f, 0.0007551657618023455f, 0.0008096923120319843f, 0.0003221159568056464f, 0.0008014762424863875f, 0.0005410780431702733f, 0.000562593515496701f, 0.00021389572066254914f, 0.0002569076023064554f, 0.0005863713449798524f, 0.0006606417591683567f, 0.000580138701479882f, 0.00022829881345387548f, 0.0004961416125297546f, 0.0005090815830044448f, 0.0005764481029473245f, 0.0003469575894996524f, 0.0005822990788146853f, 0.0007848212844692171f, 0.0008011515601538122f, 0.0006844039889983833f, 0.0009549499372951686f, 0.0006435947143472731f, 0.0012227364350110292f, 0.0007410892867483199f, 0.0007598247611895204f, 0.00052199128549546f, 0.0007019425975158811f, 0.0003818753466475755f, 0.0007928077247925103f, 0.0002885300782509148f, 0.00019440293544903398f, 0.0008906454895623028f, 0.0009818548569455743f, 0.0010341857559978962f, 0.0006817860994488001f, 0.0009507507784292102f, 0.0008774829911999404f, 0.0010173105401918292f, 0.0005182806053198874f, 0.0008149521891027689f, 0.0005588095518760383f, 0.0005931365303695202f, 0.0009623301448300481f, 0.000897068704944104f, 0.0009637949406169355f, 0.0007399580790661275f, 0.00033297567279078066f, 0.0008666684734635055f, 0.0009289124864153564f, 0.0007542234379798174f, 0.0005039963871240616f, 0.0006299842498265207f, 0.00043585459934547544f, 0.000588748138397932f, 0.0008243701886385679f, 0.0003307878796476871f, 0.000809738296084106f, 0.0010321836452931166f, 0.0007078943308442831f, 0.0007271142676472664f, 0.00037703453563153744f, 0.0006699643563479185f, 0.0012830441119149327f, 0.0003094259009230882f, 0.0010537139605730772f, 0.0006341546541079879f, 0.0011668987572193146f, 0.0007073067245073617f, 0.0014366479590535164f, 0.0006819049012847245f, 0.0006128178210929036f, 0.0009191290591843426f, 0.0009722317336127162f, 0.00038135721115395427f, 0.0008091056370176375f, 0.0002968739136122167f, 0.0008877722430042922f, 0.0006143229547888041f, 0.0012635994935408235f, 0.0008189420332200825f, 0.0005760400672443211f, 0.0011309117544442415f, 0.0007456314633600414f, 0.0006466814666055143f, 0.0006933489348739386f, 0.0009102389449253678f, 0.0006710710003972054f, 0.0005473442724905908f, 0.000756074849050492f, 0.0004377725417725742f, 0.0005831819726154208f, 0.0005015573697164655f, 0.0006063126493245363f, 0.0006512165418826044f, 0.0007308251224458218f, 0.000395468290662393f, 0.001251167617738247f, 0.0005817312630824745f, 0.001186721958220005f, 0.0007996174972504377f, 0.0007987632416188717f, 0.00042080844286829233f, 0.000939448713324964f, 0.0006768496823497117f, 0.0015347272856160998f, 0.0006870916113257408f, 0.0007163104019127786f, 0.00108776253182441f, 0.0006304630660451949f, 0.00020220417354721576f, 0.0005082159768790007f, 0.0006544211646541953f, 0.0009602974751032889f, 0.00025070359697565436f, 0.0006003237213008106f, 0.0006831690552644432f, 0.0006229319842532277f, 0.0005173728568479419f, 0.00043430516961961985f, 0.0005993570084683597f, 0.0005069100297987461f, 0.00032458489295095205f, 0.0006684381514787674f, 0.0008172053494490683f, 0.0005396787310019135f, 0.0007956197950989008f);
static const ai_layer_format_type conv2d_55_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_i8 conv2d_56_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 conv2d_56_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 conv2d_56_pad_before_t_in_0_shape_h_const_u32 = 4;

static const ai_u16 conv2d_56_t_in_0_shape_w_const_u16 = 6;
static const ai_u16 conv2d_56_t_in_0_shape_h_const_u16 = 6;
static const ai_u16 conv2d_56_t_in_0_shape_ch_const_u16 = 336;
static const ai_u16 conv2d_56_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_56_l_stride_0_const_u16 = 1;
static const ai_i8 conv2d_56_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_56_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_56_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_56_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_56_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.01686757244169712f, 0.011492895893752575f, 0.002655399264767766f, 0.008799474686384201f, 0.005973667372018099f, 0.007857349701225758f, 0.014117172919213772f, 0.006857092957943678f, 0.005701316986232996f, 0.006515955086797476f, 0.009151102043688297f, 0.005498358979821205f, 0.006494041997939348f, 0.010219006799161434f, 0.015127413906157017f, 0.01131477765738964f, 0.004884808789938688f, 0.0043594688177108765f, 0.004002391826361418f, 0.003040578216314316f, 0.0034413430839776993f, 0.024902431294322014f, 0.006779329385608435f, 0.0074699511751532555f, 0.0015876174438744783f, 0.0018660767236724496f, 0.006926861125975847f, 0.004853245336562395f, 0.016802288591861725f, 0.018069155514240265f, 0.007412986364215612f, 0.00430770730599761f, 0.0012559763854369521f, 0.005953085608780384f, 0.006443217396736145f, 0.010789156891405582f, 0.007742462679743767f, 0.0054811653681099415f, 0.006612389348447323f, 0.008358817547559738f, 0.04603688046336174f, 0.005261423531919718f, 0.01694520376622677f, 0.007017189636826515f, 0.011214321479201317f, 0.008077912963926792f, 0.009551354683935642f, 0.005857869517058134f, 0.0029192573856562376f, 0.007869272492825985f, 0.017891013994812965f, 0.006908379960805178f, 0.007805296219885349f, 0.009681275114417076f, 0.003831734647974372f, 0.007466619834303856f, 0.08909378945827484f, 0.012570429593324661f, 0.013736558146774769f, 0.004411235451698303f, 0.0020592613145709038f, 0.017869982868433f, 0.014072125777602196f, 0.003049378516152501f, 0.0048075271770358086f, 0.011206045746803284f, 0.004119706340134144f, 0.008562711998820305f, 0.005968939978629351f, 0.009363164193928242f, 0.010769025422632694f, 0.008563968352973461f, 0.005227537825703621f, 0.003706315066665411f, 0.0076974225230515f, 0.010199592448771f, 0.007171996403485537f, 0.005258625373244286f, 0.010219833813607693f, 0.012160316109657288f, 0.00988283846527338f, 0.010087007656693459f, 0.005704123992472887f, 0.00515281967818737f, 0.0036457288078963757f, 0.00481722317636013f, 0.01545439288020134f, 0.0015670886496081948f, 0.00426427973434329f, 0.01124395802617073f, 0.003708507167175412f, 0.004743129946291447f, 0.00800363626331091f, 0.002456743735820055f, 0.02136884070932865f, 0.006209593266248703f, 0.002481034491211176f, 0.01361122913658619f, 0.018395420163869858f, 0.002695282455533743f, 0.015231087803840637f, 0.003534941468387842f, 0.010952914133667946f, 0.020659945905208588f, 0.02412501908838749f, 0.004260817542672157f, 0.007174099329859018f, 0.004945114720612764f, 0.005923996213823557f, 0.010834471322596073f, 0.006194174289703369f, 0.004936435725539923f, 0.005064703058451414f, 0.003756144316866994f, 0.0062819719314575195f, 0.012609259225428104f, 0.008069953881204128f, 0.0038830654229968786f, 0.011179916560649872f, 0.005233765114098787f, 0.008820542134344578f, 0.0023724129423499107f, 0.0031986283138394356f, 0.007797254715114832f, 0.003699246095493436f, 0.008841089904308319f, 0.003170610638335347f, 0.01299388986080885f, 0.011962120421230793f, 0.011715075932443142f, 0.011717002838850021f, 0.01214927714318037f, 0.017841704189777374f, 0.039187442511320114f, 0.004135298077017069f, 0.005809493828564882f, 0.005286624655127525f, 0.008228986524045467f, 0.022830046713352203f, 0.0035171634517610073f, 0.010068490169942379f, 0.01931079849600792f, 0.006220608483999968f, 0.005093998275697231f, 0.015764445066452026f, 0.007624870166182518f, 0.01520135160535574f, 0.008444384671747684f, 0.008420208469033241f, 0.0036351094022393227f, 0.005389292724430561f, 0.015825524926185608f, 0.003973860759288073f, 0.007610764820128679f, 0.0059377942234277725f, 0.00922865979373455f, 0.0073463208973407745f, 0.007896380499005318f, 0.005514281336218119f, 0.013719976879656315f, 0.005559919402003288f, 0.0074794176034629345f, 0.022544393315911293f, 0.008911588229238987f, 0.004309115465730429f, 0.003299419302493334f, 0.010074527002871037f, 0.005055269226431847f, 0.01029340922832489f, 0.005831598304212093f, 0.005005904007703066f, 0.010964245535433292f, 0.0021359932143241167f, 0.007222707848995924f, 0.0040189353749156f, 0.008329278789460659f, 0.008766287937760353f, 0.0045030866749584675f, 0.00733932526782155f, 0.007312541827559471f, 0.008852771483361721f, 0.009053302928805351f, 0.011584987863898277f, 0.003786468179896474f, 0.006506409961730242f, 0.0028525153174996376f, 0.0059553938917815685f, 0.007201561238616705f, 0.004256071522831917f, 0.012212404049932957f, 0.006945245899260044f, 0.004447693005204201f, 0.0035325693897902966f, 0.0036386779975146055f, 0.007344122510403395f, 0.0061479113064706326f, 0.007654436863958836f, 0.00735490070655942f, 0.016804136335849762f, 0.024147098883986473f, 0.021157706156373024f, 0.007644542958587408f, 0.00982507411390543f, 0.006888733711093664f, 0.014870372600853443f, 0.0042616999708116055f, 0.0026401937939226627f, 0.009656166657805443f, 0.009050490334630013f, 0.009110595099627972f, 0.0029413150623440742f, 0.003741653636097908f, 0.006940919440239668f, 0.004020914901047945f, 0.011472114361822605f, 0.004644415806978941f, 0.014082123525440693f, 0.01876676455140114f, 0.007163326721638441f, 0.008689218200743198f, 0.008724751882255077f, 0.012218122370541096f, 0.01265793852508068f, 0.017391525208950043f, 0.012570057064294815f, 0.012687341310083866f, 0.022499222308397293f, 0.006290028803050518f, 0.005010496359318495f, 0.002442233031615615f, 0.0024972830433398485f, 0.010346982628107071f, 0.011582870967686176f, 0.007158585824072361f, 0.012684374116361141f, 0.0030032554641366005f, 0.007997618056833744f, 0.004979553632438183f, 0.004782778676599264f, 0.010254567489027977f, 0.01637839339673519f, 0.004567031282931566f, 0.0026363267097622156f, 0.006290348246693611f, 0.007457330357283354f, 0.009786147624254227f, 0.007055561989545822f, 0.006555637810379267f, 0.00530538335442543f, 0.010152864269912243f, 0.015418476425111294f, 0.0038681465666741133f, 0.01795678399503231f, 0.006057641003280878f, 0.00442194240167737f, 0.011801245622336864f, 0.005422444548457861f, 0.001932537299580872f, 0.004114137031137943f, 0.008425462990999222f, 0.01231291051954031f, 0.008745336905121803f, 0.017358850687742233f, 0.007222154643386602f, 0.006229652091860771f, 0.013538362458348274f, 0.0063897897489368916f, 0.0031490158289670944f, 0.005237455479800701f, 0.00821109488606453f, 0.008131635375320911f, 0.006198244169354439f, 0.002868971088901162f, 0.036906272172927856f, 0.00750403618440032f, 0.004456760827451944f, 0.008485809899866581f, 0.008434155024588108f, 0.0118143605068326f, 0.0027126602362841368f, 0.006672977935522795f, 0.007158032618463039f, 0.0036809612065553665f, 0.005915433634072542f, 0.006051320116966963f, 0.017300929874181747f, 0.00833970122039318f, 0.005830447189509869f, 0.005571987479925156f, 0.004004800226539373f, 0.002360263606533408f, 0.005244604777544737f, 0.013455952517688274f, 0.011325566098093987f, 0.0030635881703346968f, 0.005195066798478365f, 0.00684822304174304f, 0.010159548372030258f, 0.011246767826378345f, 0.003043979872018099f, 0.0038460330106317997f, 0.019026074558496475f, 0.010682063177227974f, 0.008063127286732197f, 0.005663852673023939f, 0.01015937514603138f, 0.004913011100143194f, 0.007433410733938217f, 0.00547076715156436f, 0.005251027178019285f, 0.0069175357930362225f, 0.004321443848311901f, 0.006236946675926447f, 0.006572099402546883f, 0.006662572734057903f, 0.01479770801961422f, 0.0025652735494077206f, 0.005406641401350498f, 0.008557220920920372f, 0.0069611952640116215f, 0.010289620608091354f, 0.016755256801843643f, 0.00566169572994113f, 0.01947435550391674f, 0.00481907045468688f, 0.004656963050365448f, 0.005091933533549309f, 0.011006991378962994f, 0.006862742360681295f, 0.0028988136909902096f, 0.007866556756198406f, 0.009931186214089394f, 0.010587743483483791f, 0.005613866727799177f, 0.004375047516077757f, 0.013971741311252117f);
static const ai_u16 conv2d_56_t_out_0_shape_w_const_u16 = 4;
static const ai_u16 conv2d_56_t_out_0_shape_h_const_u16 = 4;

static const ai_u16 conv2d_57_t_in_0_shape_w_const_u16 = 4;
static const ai_u16 conv2d_57_t_in_0_shape_h_const_u16 = 4;
static const ai_u16 conv2d_57_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_57_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_57_t_in_0_shape_ch_const_u16 = 336;
static const ai_u16 conv2d_57_t_out_0_shape_ch_const_u16 = 56;
static const ai_i8 conv2d_57_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_57_t_out_0_fmt_zero_const_s8 = 3;
static const ai_float conv2d_57_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_57_t_out_0_fmt_scale_const_f32 = 0.18914462625980377f;
static const ai_float conv2d_57_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.023781390860676765f, 0.009605860337615013f, 0.00841265358030796f, 0.013286236673593521f, 0.010947673581540585f, 0.010860847309231758f, 0.011816775426268578f, 0.01883324235677719f, 0.0072615426033735275f, 0.02317364327609539f, 0.009967454709112644f, 0.007577128708362579f, 0.012669543735682964f, 0.009854014962911606f, 0.00648745009675622f, 0.0072866701520979404f, 0.011997411027550697f, 0.007019481156021357f, 0.013415509834885597f, 0.006952280644327402f, 0.00604540528729558f, 0.006492130924016237f, 0.022771330550312996f, 0.01629270613193512f, 0.016247283667325974f, 0.009961215779185295f, 0.008049603551626205f, 0.008005337789654732f, 0.009004967287182808f, 0.0048598372377455235f, 0.005479383748024702f, 0.009290218353271484f, 0.010180302895605564f, 0.008047942072153091f, 0.007352830842137337f, 0.008485847152769566f, 0.011537102051079273f, 0.007324527483433485f, 0.012322356924414635f, 0.010121756233274937f, 0.009076359681785107f, 0.008892916142940521f, 0.0079997219145298f, 0.00675575016066432f, 0.008015480823814869f, 0.008694766089320183f, 0.02181498520076275f, 0.010609512217342854f, 0.0052122087217867374f, 0.007619200274348259f, 0.007015484385192394f, 0.008308285847306252f, 0.006728225853294134f, 0.009447029791772366f, 0.006832617800682783f, 0.010057305917143822f);
static const ai_layer_format_type conv2d_57_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 conv2d_59_t_in_0_shape_w_const_u16 = 4;
static const ai_u16 conv2d_59_t_in_0_shape_h_const_u16 = 4;
static const ai_u16 conv2d_59_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_59_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_59_t_in_0_shape_ch_const_u16 = 56;
static const ai_u16 conv2d_59_t_out_0_shape_ch_const_u16 = 336;
static const ai_i8 conv2d_59_t_in_0_fmt_zero_const_s8 = 3;
static const ai_i8 conv2d_59_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_59_t_in_0_fmt_scale_const_f32 = 0.18914462625980377f;
static const ai_float conv2d_59_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_59_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.00046792603097856045f, 0.0006814998923800886f, 0.0004830939578823745f, 0.0004093440074939281f, 0.0007796632125973701f, 0.0003631480794865638f, 0.0004716349649243057f, 0.0005750585696659982f, 0.0007162179099395871f, 0.0010007524397224188f, 0.0006492114043794572f, 0.0007559185032732785f, 0.0006828524055890739f, 0.0005396753549575806f, 0.0008044319692999125f, 0.0004098308563698083f, 0.0011051385663449764f, 0.0006478599971160293f, 0.0005253437557257712f, 0.0011187064228579402f, 0.0008391535375267267f, 0.0006528340163640678f, 0.0006886956980451941f, 0.0005263494094833732f, 0.0010480614146217704f, 0.0004049306153319776f, 0.0005066710291430354f, 0.000685746141243726f, 0.0006871125660836697f, 0.0005438817897811532f, 0.0006033382378518581f, 0.0011016492499038577f, 0.0005636918940581381f, 0.000542762572877109f, 0.0011060109827667475f, 0.0004479244234971702f, 0.0005668305093422532f, 0.0008508081082254648f, 0.0005511492490768433f, 0.00058434356469661f, 0.0004918245831504464f, 0.0007379076560027897f, 0.0008067604503594339f, 0.0006934645934961736f, 0.0007861730991862714f, 0.0004611418698914349f, 0.0005490048206411302f, 0.0010466809617355466f, 0.0008308697724714875f, 0.0003883296449203044f, 0.0005539198755286634f, 0.0006984144565649331f, 0.0010109504219144583f, 0.0005301847704686224f, 0.0006019000429660082f, 0.0004939874052070081f, 0.0006586762610822916f, 0.0006854012026451528f, 0.0005189941730350256f, 0.0005551979993470013f, 0.0004311323573347181f, 0.00038703414611518383f, 0.0006137493764981627f, 0.00042230129474774003f, 0.0006914559053257108f, 0.0004557896463666111f, 0.001106272335164249f, 0.0007284222519956529f, 0.00048017824883572757f, 0.0008573568193241954f, 0.0009247505804523826f, 0.0007848883979022503f, 0.0004794458218384534f, 0.000506633601617068f, 0.0008669968810863793f, 0.0007272864459082484f, 0.0004572288889903575f, 0.0008747208630666137f, 0.0009383985307067633f, 0.0006931571988388896f, 0.0006927162175998092f, 0.0004919399507343769f, 0.0013544041430577636f, 0.0008393348543904722f, 0.0005471009644679725f, 0.000785523559898138f, 0.0005763867520727217f, 0.0005564913735724986f, 0.0009792015189304948f, 0.0007080384530127048f, 0.0005473055643960834f, 0.0004387018270790577f, 0.0003709586744662374f, 0.0008784614037722349f, 0.0005947459139861166f, 0.0005122324801050127f, 0.0005024833371862769f, 0.00037310441257432103f, 0.0006525840726681054f, 0.000602202897425741f, 0.0004197311354801059f, 0.001026045880280435f, 0.0006208755658008158f, 0.0006422997685149312f, 0.0005879917298443615f, 0.0008221147581934929f, 0.0005262286867946386f, 0.0006135112489573658f, 0.0004931749426759779f, 0.0006328175077214837f, 0.00043155805906280875f, 0.0004668206092901528f, 0.0006990637048147619f, 0.0005433812621049583f, 0.0007884405204094946f, 0.0012858407571911812f, 0.000516878382768482f, 0.0007528415881097317f, 0.0010787162464112043f, 0.0006438323762267828f, 0.0007675742381252348f, 0.0004696898686233908f, 0.0008979024714790285f, 0.00040049018571153283f, 0.0009260662482120097f, 0.00036285148235037923f, 0.0007462715147994459f, 0.0008734752191230655f, 0.00043691927567124367f, 0.0003993778664153069f, 0.00046115019358694553f, 0.000488429213874042f, 0.00043377079418860376f, 0.0006724799750372767f, 0.0006745013524778187f, 0.0007809235248714685f, 0.0007651928463019431f, 0.0006833744118921459f, 0.0007725745090283453f, 0.0005835015326738358f, 0.0006440472789108753f, 0.00044647749746218324f, 0.00047074948088265955f, 0.0005391348386183381f, 0.0007992455502972007f, 0.000904515094589442f, 0.0008653413970023394f, 0.000855540856719017f, 0.0009065206977538764f, 0.0005978964036330581f, 0.0004852288111578673f, 0.0007667815661989152f, 0.000637880468275398f, 0.0005841958918608725f, 0.0005884644924663007f, 0.0006252197781577706f, 0.0004650054615922272f, 0.0010551391169428825f, 0.0007682110881432891f, 0.0004878010367974639f, 0.0006162964273244143f, 0.0011788932606577873f, 0.0008173095411621034f, 0.0006196111207827926f, 0.0007215308141894639f, 0.0006212961161509156f, 0.0004628438618965447f, 0.0004959887592121959f, 0.0008235700661316514f, 0.0008322871290147305f, 0.0005535196396522224f, 0.0008356444886885583f, 0.0007181804394349456f, 0.000816581305116415f, 0.0004553348699118942f, 0.0008483049459755421f, 0.0006133748102001846f, 0.000454646855359897f, 0.0004948711139149964f, 0.0006635197205469012f, 0.0007864172221161425f, 0.00045054458314552903f, 0.0004276479303371161f, 0.0006897960556671023f, 0.0007204673020169139f, 0.0006413753144443035f, 0.0005131615907885134f, 0.00045724370284006f, 0.00037586240796372294f, 0.0005847598658874631f, 0.0006637463229708374f, 0.000824932474642992f, 0.0005914451903663576f, 0.0007751274388283491f, 0.0004516849876381457f, 0.0007392023107968271f, 0.0005787167465314269f, 0.0007724377210251987f, 0.0008496216032654047f, 0.0006832938524894416f, 0.0005552920047193766f, 0.0007201368571259081f, 0.0005191797390580177f, 0.0007590267923660576f, 0.000631786126177758f, 0.0011053590569645166f, 0.0005751415737904608f, 0.00043276805081404746f, 0.0005365514080040157f, 0.0009336277144029737f, 0.0004959759535267949f, 0.0005722732748836279f, 0.0007381709874607623f, 0.0006648845155723393f, 0.0006092377589084208f, 0.0006894891266711056f, 0.0006617751787416637f, 0.0007947483099997044f, 0.0005181862507015467f, 0.0007579645607620478f, 0.0007559616933576763f, 0.0004682192229665816f, 0.001200065715238452f, 0.0005376300541684031f, 0.0005882153636775911f, 0.0004888325347565114f, 0.0005575805553235114f, 0.00045889848843216896f, 0.0006826600292697549f, 0.0009144538780674338f, 0.000782779126893729f, 0.0008248162339441478f, 0.0007437539170496166f, 0.0005074426881037652f, 0.0007147184223867953f, 0.0006522109033539891f, 0.0005779247148893774f, 0.0005738300969824195f, 0.001109905424527824f, 0.0005924245342612267f, 0.0004594753263518214f, 0.0005439103697426617f, 0.0005150873330421746f, 0.00130735884886235f, 0.0006625761743634939f, 0.0007145063718780875f, 0.0009132205741479993f, 0.0006235251785255969f, 0.0008854496409185231f, 0.0006372873322106898f, 0.0005692436825484037f, 0.0007152430480346084f, 0.0010046121897175908f, 0.0005281231133267283f, 0.00047394950524903834f, 0.0005292933783493936f, 0.0006770757609046996f, 0.0006670830771327019f, 0.0006729012820869684f, 0.0006133440765552223f, 0.0006573670543730259f, 0.0006168358377180994f, 0.0009074510890059173f, 0.0004165477876085788f, 0.0006691417074762285f, 0.0004703031445387751f, 0.0006179133779369295f, 0.0007367677753791213f, 0.00024498984566889703f, 0.0007971000741235912f, 0.0007112277671694756f, 0.0008710392285138369f, 0.0006930596427991986f, 0.0005101200076751411f, 0.0005379506037570536f, 0.0009253656025975943f, 0.0005849020672030747f, 0.000659920449834317f, 0.0009787982562556863f, 0.0007181101245805621f, 0.0005708975368179381f, 0.00040576609899289906f, 0.0006280582747422159f, 0.0006485217018052936f, 0.0008159152348525822f, 0.0005729307304136455f, 0.0008329391712322831f, 0.0006301231333054602f, 0.0006467723287642002f, 0.0010436950251460075f, 0.0007738540880382061f, 0.000518637418281287f, 0.0006065204506739974f, 0.00044440230703912675f, 0.0005563348531723022f, 0.00048025810974650085f, 0.0008030875469557941f, 0.0007841244805604219f, 0.0005169826908968389f, 0.0004587358271237463f, 0.0007642582058906555f, 0.00041386825614608824f, 0.0006987690576352179f, 0.00041355445864610374f, 0.0006667140405625105f, 0.0004875832237303257f, 0.0005110391066409647f, 0.000579972576815635f, 0.0007701972499489784f, 0.0007761353626847267f, 0.0007429347606375813f, 0.000602301093749702f, 0.0005265740328468382f, 0.000532406906131655f, 0.0008141219732351601f, 0.0006836677202954888f, 0.000671752670314163f, 0.0013608889421448112f, 0.00047588456072844565f, 0.0007394786807708442f, 0.0006041149608790874f, 0.0009479373693466187f, 0.0006248769350349903f, 0.0004477751499507576f, 0.0009006204199977219f, 0.0009076354908756912f, 0.0005373713793233037f, 0.00075713632395491f, 0.0005724714719690382f, 0.0006618553888984025f, 0.0011086446465924382f, 0.0004614016506820917f, 0.0007330305525101721f, 0.001273171161301434f, 0.0004658592806663364f, 0.00039567932253703475f);
static const ai_layer_format_type conv2d_59_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_i8 conv2d_60_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 conv2d_60_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 conv2d_60_pad_before_t_in_0_shape_h_const_u32 = 4;

static const ai_u16 conv2d_60_t_in_0_shape_w_const_u16 = 6;
static const ai_u16 conv2d_60_t_in_0_shape_h_const_u16 = 6;
static const ai_u16 conv2d_60_t_in_0_shape_ch_const_u16 = 336;
static const ai_u16 conv2d_60_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_60_l_stride_0_const_u16 = 1;
static const ai_i8 conv2d_60_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_60_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_60_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_60_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_60_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.015764586627483368f, 0.005636846646666527f, 0.005331544671207666f, 0.006076293531805277f, 0.013459615409374237f, 0.007064109668135643f, 0.007815818302333355f, 0.008611644618213177f, 0.007114015985280275f, 0.0045434110797941685f, 0.008837953209877014f, 0.0071854786947369576f, 0.004631737247109413f, 0.006813076790422201f, 0.004237784538418055f, 0.005162215791642666f, 0.010162333026528358f, 0.003381918417289853f, 0.011920359916985035f, 0.007496966049075127f, 0.009865139611065388f, 0.0058168512769043446f, 0.0037996002938598394f, 0.00566921615973115f, 0.003948785364627838f, 0.007250715047121048f, 0.0040642921812832355f, 0.004220656119287014f, 0.005261720158159733f, 0.0048885042779147625f, 0.006662197411060333f, 0.006102654617279768f, 0.010248957201838493f, 0.005521126091480255f, 0.00447503849864006f, 0.008083367720246315f, 0.004869870841503143f, 0.004459110554307699f, 0.00495819840580225f, 0.009581767953932285f, 0.010939463041722775f, 0.004308678675442934f, 0.0044652363285422325f, 0.007792473770678043f, 0.00947009027004242f, 0.006567628588527441f, 0.0065567647106945515f, 0.004555325955152512f, 0.004317303188145161f, 0.006815674714744091f, 0.0035713370889425278f, 0.005824636202305555f, 0.006945984438061714f, 0.006615123711526394f, 0.0034856670536100864f, 0.006084267981350422f, 0.0058673713356256485f, 0.0036152449902147055f, 0.004504561424255371f, 0.0049917446449398994f, 0.005494586657732725f, 0.0041086324490606785f, 0.0043928176164627075f, 0.006813204847276211f, 0.006128824315965176f, 0.011089028790593147f, 0.006158169358968735f, 0.004736754111945629f, 0.00817527063190937f, 0.011699914932250977f, 0.006190908141434193f, 0.004951003473252058f, 0.0050679706037044525f, 0.01094868965446949f, 0.005692485254257917f, 0.004663741681724787f, 0.0057353791780769825f, 0.006462449207901955f, 0.004718783777207136f, 0.004547559190541506f, 0.0050310902297496796f, 0.005041030701249838f, 0.004084354266524315f, 0.004668469075113535f, 0.005228664726018906f, 0.005720547866076231f, 0.00788097269833088f, 0.005058207083493471f, 0.0051520224660634995f, 0.010166026651859283f, 0.005239798221737146f, 0.00765694584697485f, 0.009495268575847149f, 0.004382966086268425f, 0.0037846833001822233f, 0.005764639936387539f, 0.006560942158102989f, 0.004002715460956097f, 0.0075467596761882305f, 0.01023664977401495f, 0.007042792625725269f, 0.003183581866323948f, 0.008127563633024693f, 0.005443289410322905f, 0.007921389304101467f, 0.0058053117245435715f, 0.006794627755880356f, 0.00570431724190712f, 0.007924208417534828f, 0.005611533299088478f, 0.00424416596069932f, 0.004829393699765205f, 0.007694034371525049f, 0.0053033786825835705f, 0.00489985104650259f, 0.0068271527998149395f, 0.004499772563576698f, 0.005864930804818869f, 0.0038453054148703814f, 0.004707389045506716f, 0.004934675991535187f, 0.008820801973342896f, 0.009129001758992672f, 0.0052519808523356915f, 0.003750921692699194f, 0.004525152500718832f, 0.010735826566815376f, 0.010145661421120167f, 0.009531133808195591f, 0.004844382870942354f, 0.007259958889335394f, 0.004256764426827431f, 0.00523092458024621f, 0.004188541788607836f, 0.00598328560590744f, 0.004594915080815554f, 0.006147789303213358f, 0.006418529432266951f, 0.00797139760106802f, 0.006083792541176081f, 0.0043502855114638805f, 0.004324019420892f, 0.008477470837533474f, 0.0035079277586191893f, 0.0036828555166721344f, 0.014425947330892086f, 0.005328969098627567f, 0.005770780611783266f, 0.005846915766596794f, 0.005491699557751417f, 0.0077131460420787334f, 0.007804627995938063f, 0.007672091946005821f, 0.0057360511273145676f, 0.005452989600598812f, 0.005941866431385279f, 0.005860983859747648f, 0.002983382437378168f, 0.0059477477334439754f, 0.009981521405279636f, 0.0033427653834223747f, 0.0066559589467942715f, 0.005651991814374924f, 0.006838698405772448f, 0.004625989124178886f, 0.006193771492689848f, 0.008695860393345356f, 0.005526568274945021f, 0.0035051468294113874f, 0.004822548478841782f, 0.01009995024651289f, 0.0039029212202876806f, 0.004414085298776627f, 0.003555861534550786f, 0.004597150254994631f, 0.004344105254858732f, 0.00416142912581563f, 0.005163012072443962f, 0.006471080705523491f, 0.007722472306340933f, 0.004765904974192381f, 0.003479946870356798f, 0.004538319073617458f, 0.006743285804986954f, 0.010389266535639763f, 0.00571806263178587f, 0.006762050092220306f, 0.005692192818969488f, 0.005652329418808222f, 0.005883296485990286f, 0.003452572040259838f, 0.005694171879440546f, 0.005849416833370924f, 0.006005567032843828f, 0.010934495367109776f, 0.0067868540063500404f, 0.01196125615388155f, 0.0055985357612371445f, 0.003129805438220501f, 0.0044633327051997185f, 0.012330159544944763f, 0.0064278822392225266f, 0.004677420482039452f, 0.0052194781601428986f, 0.008269814774394035f, 0.004420318640768528f, 0.00828801654279232f, 0.004714331589639187f, 0.0111932922154665f, 0.005621728487312794f, 0.004793595056980848f, 0.010748916305601597f, 0.006538888905197382f, 0.006027253344655037f, 0.004901271779090166f, 0.005951417610049248f, 0.003731918754056096f, 0.0055730403400957584f, 0.04049016535282135f, 0.005758360959589481f, 0.006409265100955963f, 0.00688416650518775f, 0.006381165701895952f, 0.004793628118932247f, 0.006955461110919714f, 0.005639941897243261f, 0.008776815608143806f, 0.007707573473453522f, 0.0066101690754294395f, 0.004623098764568567f, 0.004655209835618734f, 0.00698295421898365f, 0.009816652163863182f, 0.007278761360794306f, 0.010083687491714954f, 0.008860512636601925f, 0.006573017220944166f, 0.005483126267790794f, 0.008590539917349815f, 0.008145804516971111f, 0.008832843042910099f, 0.007622479926794767f, 0.004941077902913094f, 0.0037728869356215f, 0.005058146547526121f, 0.0041775181889534f, 0.009726754389703274f, 0.005610071588307619f, 0.005041747819632292f, 0.00436293613165617f, 0.006756930612027645f, 0.004944507963955402f, 0.006463770288974047f, 0.007526445668190718f, 0.005935361608862877f, 0.0039380700327456f, 0.005740981083363295f, 0.00823741964995861f, 0.008245901204645634f, 0.01006676722317934f, 0.018001582473516464f, 0.005584652069956064f, 0.004883243702352047f, 0.006060307379812002f, 0.004734255839139223f, 0.007161385379731655f, 0.007075899746268988f, 0.00782120879739523f, 0.007777557708323002f, 0.005526474677026272f, 0.006216784939169884f, 0.002921500476077199f, 0.006582488771528006f, 0.010689705610275269f, 0.01369080226868391f, 0.005744739901274443f, 0.009085201658308506f, 0.004921842832118273f, 0.007823608815670013f, 0.006757022347301245f, 0.004791254177689552f, 0.008667088113725185f, 0.0065170046873390675f, 0.004120993427932262f, 0.005880534648895264f, 0.004587259609252214f, 0.0057657333090901375f, 0.007220389787107706f, 0.008626587688922882f, 0.002331517403945327f, 0.006528445053845644f, 0.0051641035825014114f, 0.006452054716646671f, 0.007153596729040146f, 0.006145553197711706f, 0.009989136829972267f, 0.006390286609530449f, 0.007114906329661608f, 0.01274612545967102f, 0.007755611091852188f, 0.006695274729281664f, 0.009893820621073246f, 0.0072384728118777275f, 0.009063507430255413f, 0.004222029820084572f, 0.022747570648789406f, 0.005477419123053551f, 0.009041093289852142f, 0.00951121561229229f, 0.007397280540317297f, 0.009413591586053371f, 0.006537439301609993f, 0.004998765420168638f, 0.007091972976922989f, 0.0058045946061611176f, 0.010210000909864902f, 0.004998161923140287f, 0.004175070207566023f, 0.007064045872539282f, 0.0066633629612624645f, 0.007004522252827883f, 0.005948630627244711f, 0.007234582677483559f, 0.005327833816409111f, 0.006097778212279081f, 0.003385043004527688f, 0.0059534888714551926f, 0.005622272379696369f, 0.00609980383887887f, 0.007854796946048737f, 0.006209786981344223f, 0.006333174649626017f, 0.004635569639503956f, 0.004723618738353252f, 0.005985632073134184f, 0.007181123364716768f);
static const ai_u16 conv2d_60_t_out_0_shape_w_const_u16 = 4;
static const ai_u16 conv2d_60_t_out_0_shape_h_const_u16 = 4;

static const ai_u16 conv2d_61_t_in_0_shape_w_const_u16 = 4;
static const ai_u16 conv2d_61_t_in_0_shape_h_const_u16 = 4;
static const ai_u16 conv2d_61_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_61_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_61_t_in_0_shape_ch_const_u16 = 336;
static const ai_u16 conv2d_61_t_out_0_shape_ch_const_u16 = 112;
static const ai_i8 conv2d_61_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_61_t_out_0_fmt_zero_const_s8 = 0;
static const ai_float conv2d_61_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_61_t_out_0_fmt_scale_const_f32 = 0.1278703808784485f;
static const ai_float conv2d_61_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.006019478663802147f, 0.0032522028777748346f, 0.0037573580630123615f, 0.0040496778674423695f, 0.0039513809606432915f, 0.005648597609251738f, 0.0056136613711714745f, 0.005988452583551407f, 0.003640820737928152f, 0.003614276647567749f, 0.005209231283515692f, 0.004961807280778885f, 0.005637699272483587f, 0.006297605112195015f, 0.006450273562222719f, 0.003596471156924963f, 0.005439599975943565f, 0.004859547596424818f, 0.0029524886049330235f, 0.00550048379227519f, 0.005139267537742853f, 0.0032146135345101357f, 0.0052671777084469795f, 0.006528620142489672f, 0.00571103859692812f, 0.004580709617584944f, 0.004456648137420416f, 0.003837368218228221f, 0.0037182625383138657f, 0.005748293828219175f, 0.006871488876640797f, 0.0041889906860888f, 0.004297453444451094f, 0.0039809285663068295f, 0.005239258520305157f, 0.0034181552473455667f, 0.004011455923318863f, 0.005484584253281355f, 0.006045572925359011f, 0.004496037028729916f, 0.005865756887942553f, 0.004635464865714312f, 0.0035086374264210463f, 0.004582412075251341f, 0.006786249577999115f, 0.005084783770143986f, 0.005344083067029715f, 0.00447522709146142f, 0.00519379461184144f, 0.003874731482937932f, 0.0059938617050647736f, 0.00451415590941906f, 0.005428605247288942f, 0.005217156372964382f, 0.0047362628392875195f, 0.004249874036759138f, 0.004688753746449947f, 0.0038153487257659435f, 0.005882323253899813f, 0.004591907374560833f, 0.0062292227521538734f, 0.003289753571152687f, 0.005676412954926491f, 0.004372057504951954f, 0.004610881675034761f, 0.003663511946797371f, 0.0036452198401093483f, 0.006091563031077385f, 0.003358241403475404f, 0.0075092073529958725f, 0.002256827661767602f, 0.005904046352952719f, 0.004674172028899193f, 0.005595406051725149f, 0.004210051614791155f, 0.0054535046219825745f, 0.0036746214609593153f, 0.004931648727506399f, 0.004859529435634613f, 0.004861530382186174f, 0.004106985405087471f, 0.00353036867454648f, 0.004766448400914669f, 0.00485074520111084f, 0.006497349590063095f, 0.0033897405955940485f, 0.005168022587895393f, 0.004382043145596981f, 0.0036531277000904083f, 0.005540516227483749f, 0.004004358313977718f, 0.0037087518721818924f, 0.005238881800323725f, 0.004725857172161341f, 0.00483714509755373f, 0.005512262228876352f, 0.0058406637981534f, 0.004377026576548815f, 0.00585705554112792f, 0.004187308251857758f, 0.0038813950959593058f, 0.007486451417207718f, 0.00396446930244565f, 0.004119426477700472f, 0.00416837353259325f, 0.005946015473455191f, 0.00453505152836442f, 0.00395464152097702f, 0.005573442671447992f, 0.0055824522860348225f, 0.008567522279918194f, 0.006199242547154427f);
static const ai_layer_format_type conv2d_61_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_u16 conv2d_62_t_in_0_shape_w_const_u16 = 4;
static const ai_u16 conv2d_62_t_in_0_shape_h_const_u16 = 4;
static const ai_u16 conv2d_62_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_62_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_62_t_in_0_shape_ch_const_u16 = 112;
static const ai_u16 conv2d_62_t_out_0_shape_ch_const_u16 = 1280;
static const ai_i8 conv2d_62_t_in_0_fmt_zero_const_s8 = 0;
static const ai_i8 conv2d_62_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_62_t_in_0_fmt_scale_const_f32 = 0.1278703808784485f;
static const ai_float conv2d_62_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_62_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0030403779819607735f, 0.0034556749742478132f, 0.004182478412985802f, 0.0038632003124803305f, 0.003635064698755741f, 0.0031684476416558027f, 0.0028295726515352726f, 0.0033609920646995306f, 0.002282656729221344f, 0.004501049872487783f, 0.0030674925073981285f, 0.0029385953675955534f, 0.004133297596126795f, 0.0035128993913531303f, 0.004094758536666632f, 0.00290244841016829f, 0.004036190453916788f, 0.0026262300089001656f, 0.0034158218186348677f, 0.0027032478246837854f, 0.00503814360126853f, 0.00302455248311162f, 0.00269905524328351f, 0.0035458351485431194f, 0.003398847533389926f, 0.0033664279617369175f, 0.0027836540248245f, 0.0027730611618608236f, 0.0024409324396401644f, 0.005224744323641062f, 0.004870313685387373f, 0.0026312547270208597f, 0.003739690175279975f, 0.0034600626677274704f, 0.0037643564864993095f, 0.002719238167628646f, 0.0037892614491283894f, 0.003862452693283558f, 0.002644084393978119f, 0.0032401462085545063f, 0.003109352895990014f, 0.004723788239061832f, 0.0024865572340786457f, 0.005582199897617102f, 0.0024232775904238224f, 0.006085287779569626f, 0.003148901741951704f, 0.0031799483112990856f, 0.0029106028378009796f, 0.004029940348118544f, 0.004134939052164555f, 0.004330634139478207f, 0.003730821656063199f, 0.004064241424202919f, 0.0036439872346818447f, 0.0031290575861930847f, 0.002556036226451397f, 0.003130499040707946f, 0.0030146250501275063f, 0.002987487940117717f, 0.004784656688570976f, 0.0037343627773225307f, 0.003068857826292515f, 0.0033585878554731607f, 0.0024139846209436655f, 0.004602449014782906f, 0.0040206932462751865f, 0.0026055725757032633f, 0.003787853755056858f, 0.004634898621588945f, 0.0035213876981288195f, 0.003135544015094638f, 0.00407783966511488f, 0.0026281934697180986f, 0.0028819111175835133f, 0.003635777859017253f, 0.00515322620049119f, 0.00524388812482357f, 0.0022374573163688183f, 0.002960158744826913f, 0.003782230196520686f, 0.0029543666169047356f, 0.0032703771721571684f, 0.002757309703156352f, 0.002897168742492795f, 0.0031586058903485537f, 0.00247016828507185f, 0.003409871133044362f, 0.003965746145695448f, 0.0031216193456202745f, 0.004128752741962671f, 0.002189537975937128f, 0.002633885247632861f, 0.004661308601498604f, 0.002936368575319648f, 0.004777788650244474f, 0.003370670834556222f, 0.0032428218983113766f, 0.0035989524330943823f, 0.0032774547580629587f, 0.0040552434511482716f, 0.0032726433128118515f, 0.0031993465963751078f, 0.0035269022919237614f, 0.0029581887647509575f, 0.004227939993143082f, 0.00317014055326581f, 0.0038431002758443356f, 0.002269082237035036f, 0.0033751782029867172f, 0.0043828776106238365f, 0.004548471886664629f, 0.0038453927263617516f, 0.0031726432498544455f, 0.0033180667087435722f, 0.002975290175527334f, 0.005027998238801956f, 0.0034610345028340816f, 0.005802519619464874f, 0.0027463717851787806f, 0.00305161252617836f, 0.004202357959002256f, 0.004041297826915979f, 0.003801973769441247f, 0.0025450570974498987f, 0.003217078046873212f, 0.0036025820299983025f, 0.003395884996280074f, 0.003930435050278902f, 0.0038910838775336742f, 0.0040536923334002495f, 0.002741089556366205f, 0.004272850695997477f, 0.0037778527475893497f, 0.0031527408864349127f, 0.0026350400876253843f, 0.0026743384078145027f, 0.004312172066420317f, 0.0028176680207252502f, 0.004040084779262543f, 0.003720224369317293f, 0.004355640150606632f, 0.00286306394264102f, 0.002761057810857892f, 0.003929374739527702f, 0.005550800357013941f, 0.003311852226033807f, 0.002442486584186554f, 0.004485982935875654f, 0.0041374028660357f, 0.0030806316062808037f, 0.0032645203173160553f, 0.002907251939177513f, 0.002657852601259947f, 0.004022025503218174f, 0.0024163946509361267f, 0.003973199054598808f, 0.0034525319933891296f, 0.0029908441938459873f, 0.0032797781750559807f, 0.0027983328327536583f, 0.00403562281280756f, 0.0030361430253833532f, 0.0037897664587944746f, 0.0027697787154465914f, 0.0032172424253076315f, 0.0042394669726490974f, 0.0023350717965513468f, 0.0037106815725564957f, 0.004409160930663347f, 0.0034315260127186775f, 0.00452993530780077f, 0.0030934547539800406f, 0.004545578267425299f, 0.002453095279633999f, 0.002944797044619918f, 0.002581910928711295f, 0.0032170037738978863f, 0.0029857880435884f, 0.0024720304645597935f, 0.002811314072459936f, 0.0026051951572299004f, 0.004052897449582815f, 0.003640363924205303f, 0.0032082151155918837f, 0.002841189270839095f, 0.003586655715480447f, 0.0038643369916826487f, 0.0023263164330273867f, 0.0032894073519855738f, 0.002554586622864008f, 0.0034141710493713617f, 0.0035476430784910917f, 0.0037619860377162695f, 0.003200216917321086f, 0.0033215065486729145f, 0.003923012875020504f, 0.0024925637990236282f, 0.0036955317482352257f, 0.002859105123206973f, 0.003345650853589177f, 0.003755135927349329f, 0.00277877738699317f, 0.002641101134940982f, 0.0038457836490124464f, 0.00482366094365716f, 0.0036008297465741634f, 0.0036431991029530764f, 0.003027437487617135f, 0.003949329257011414f, 0.003362159477546811f, 0.004422157537192106f, 0.003359200432896614f, 0.0032880117651075125f, 0.004875050857663155f, 0.0032176338136196136f, 0.002768936101347208f, 0.003605050966143608f, 0.0030260183848440647f, 0.0037125153467059135f, 0.0035505329724401236f, 0.0028344879392534494f, 0.003725710790604353f, 0.0031257783994078636f, 0.0025833507534116507f, 0.0031882948242127895f, 0.003323629731312394f, 0.004149911925196648f, 0.0029678603168576956f, 0.002748321508988738f, 0.003920421004295349f, 0.0034741717390716076f, 0.0028480126056820154f, 0.0029949515592306852f, 0.004021612927317619f, 0.0017104733269661665f, 0.0028143853414803743f, 0.0030418443493545055f, 0.003426291747018695f, 0.003276390954852104f, 0.004472922999411821f, 0.00285640568472445f, 0.003531186142936349f, 0.004611685872077942f, 0.004540976136922836f, 0.003173160832375288f, 0.003714947961270809f, 0.003544944804161787f, 0.0028092768043279648f, 0.002423255704343319f, 0.00397363118827343f, 0.0028347414918243885f, 0.003371838480234146f, 0.0037198010832071304f, 0.0032269302755594254f, 0.0030556530691683292f, 0.002944813808426261f, 0.004400735255330801f, 0.004063377156853676f, 0.0037800660356879234f, 0.003132285550236702f, 0.003739413805305958f, 0.004837892483919859f, 0.0029767549131065607f, 0.004634351935237646f, 0.004196129273623228f, 0.004516026470810175f, 0.003099254099652171f, 0.0032759697642177343f, 0.003389124758541584f, 0.0031737962272018194f, 0.0030015427619218826f, 0.004712805617600679f, 0.002919053193181753f, 0.0039723943918943405f, 0.0034924547653645277f, 0.003104105358943343f, 0.004012672230601311f, 0.003300923155620694f, 0.003686170792207122f, 0.003226070897653699f, 0.0030093269888311625f, 0.0032977829687297344f, 0.0035733813419938087f, 0.0033401085529476404f, 0.002840920817106962f, 0.003704297123476863f, 0.0028144815005362034f, 0.004012798424810171f, 0.004254214931279421f, 0.0032094833441078663f, 0.0028458177112042904f, 0.002483922988176346f, 0.003550346242263913f, 0.0041175298392772675f, 0.004808509256690741f, 0.003089574631303549f, 0.003949637059122324f, 0.0034486432559788227f, 0.002649603644385934f, 0.003051471197977662f, 0.002771018771454692f, 0.003616534871980548f, 0.003902308875694871f, 0.0036986316554248333f, 0.00281161954626441f, 0.00335879297927022f, 0.0031221271492540836f, 0.0033867040183395147f, 0.004518660716712475f, 0.0035605914890766144f, 0.0032461783848702908f, 0.0037314221262931824f, 0.003154157428070903f, 0.0034192604944109917f, 0.0039226398803293705f, 0.003507952205836773f, 0.005019066389650106f, 0.003116527572274208f, 0.00543389143422246f, 0.0032688446808606386f, 0.004646982066333294f, 0.0029570315964519978f, 0.0023324768990278244f, 0.0035750751849263906f, 0.0036619300954043865f, 0.003823802573606372f, 0.0033362770918756723f, 0.0038414921145886183f, 0.002504711737856269f, 0.0038521750830113888f, 0.0031403996981680393f, 0.004895993508398533f, 0.005809265188872814f, 0.003923782147467136f, 0.002497355453670025f, 0.0036980605218559504f, 0.003898787312209606f, 0.0032284113112837076f, 0.002677689539268613f, 0.005055170506238937f, 0.0034940934274345636f, 0.0038684329483658075f, 0.0024172235280275345f, 0.0042624641209840775f, 0.004528976045548916f, 0.004663198720663786f, 0.002818254055455327f, 0.00293536763638258f, 0.003158688312396407f, 0.003373368876054883f, 0.004853148013353348f, 0.003296524053439498f, 0.002906135283410549f, 0.0035154856741428375f, 0.0027775210328400135f, 0.0036050640046596527f, 0.002927565248683095f, 0.005065003409981728f, 0.00311169121414423f, 0.0029340903274714947f, 0.0031389326322823763f, 0.0027832724153995514f, 0.002038027159869671f, 0.0026677369605749846f, 0.0036353531759232283f, 0.0028102281503379345f, 0.0031730730552226305f, 0.0026201887521892786f, 0.003704173257574439f, 0.004057115409523249f, 0.004345565102994442f, 0.003599552670493722f, 0.002398296259343624f, 0.003732673591002822f, 0.0029043718241155148f, 0.0021142440382391214f, 0.0034357868134975433f, 0.003488235641270876f, 0.003033366287127137f, 0.0032646984327584505f, 0.0038961691316217184f, 0.0031882680486887693f, 0.002849495504051447f, 0.0032394793815910816f, 0.0034114313311874866f, 0.0027123724576085806f, 0.0025500815827399492f, 0.002616375219076872f, 0.003087184391915798f, 0.0032344216015189886f, 0.00495381373912096f, 0.004039438907057047f, 0.003733773482963443f, 0.0034021937754005194f, 0.002915806369856f, 0.004350569099187851f, 0.0037306370213627815f, 0.0024662036448717117f, 0.0036480179987847805f, 0.0034052475821226835f, 0.003362911520525813f, 0.0033183484338223934f, 0.0033212085254490376f, 0.0031203608959913254f, 0.0037365765310823917f, 0.0029341690242290497f, 0.0032202554866671562f, 0.0029532681219279766f, 0.002805736381560564f, 0.003916732035577297f, 0.0029045583214610815f, 0.002626861212775111f, 0.0035244510509073734f, 0.0038813313003629446f, 0.0029116221703588963f, 0.00442263251170516f, 0.002833795500919223f, 0.004507123492658138f, 0.003066660836338997f, 0.0034304449800401926f, 0.0034576398320496082f, 0.003494233125820756f, 0.003059916663914919f, 0.0028690847102552652f, 0.002835138700902462f, 0.003373943269252777f, 0.003655623411759734f, 0.003473586170002818f, 0.003535703755915165f, 0.003503478365018964f, 0.0034328715410083532f, 0.004652346484363079f, 0.0027018953114748f, 0.003134009428322315f, 0.003917412832379341f, 0.0027353926561772823f, 0.0036949547939002514f, 0.003980973269790411f, 0.0029545940924435854f, 0.0035881297662854195f, 0.0042111738584935665f, 0.002885665511712432f, 0.003961661830544472f, 0.0037417400162667036f, 0.0029739614110440016f, 0.003247915767133236f, 0.003206221852451563f, 0.003279216354712844f, 0.003437465988099575f, 0.002963670762255788f, 0.0033818709198385477f, 0.003207857022061944f, 0.003980821464210749f, 0.003226482542231679f, 0.0035669468343257904f, 0.0036573591642081738f, 0.0038317369762808084f, 0.002816832158714533f, 0.003986187744885683f, 0.005317761097103357f, 0.003391844220459461f, 0.002887079957872629f, 0.004072209354490042f, 0.0025442098267376423f, 0.0030926812905818224f, 0.0028981699142605066f, 0.0034467491786926985f, 0.0029593713115900755f, 0.0026245182380080223f, 0.003291897941380739f, 0.003353777341544628f, 0.003079095622524619f, 0.002844385104253888f, 0.002793088788166642f, 0.004126495216041803f, 0.003493744181469083f, 0.004502879921346903f, 0.004125668667256832f, 0.002990109845995903f, 0.0028783089946955442f, 0.004033347126096487f, 0.004291430115699768f, 0.0039345137774944305f, 0.0033336090855300426f, 0.0032411653082817793f, 0.003401537425816059f, 0.0033830630127340555f, 0.0034584468230605125f, 0.002796315820887685f, 0.0031948918476700783f, 0.003658569883555174f, 0.004062097519636154f, 0.0031321258284151554f, 0.004241653718054295f, 0.0025565102696418762f, 0.003423016518354416f, 0.0035493632312864065f, 0.004296921659260988f, 0.004007302224636078f, 0.004086768254637718f, 0.003630327759310603f, 0.0045476751402020454f, 0.0030372566543519497f, 0.003221308346837759f, 0.0022643941920250654f, 0.0038942506071180105f, 0.003700530854985118f, 0.002518211957067251f, 0.0030933357775211334f, 0.003043834352865815f, 0.0038880561478435993f, 0.003653720486909151f, 0.003971012309193611f, 0.0033663909416645765f, 0.0029531391337513924f, 0.0040885997004806995f, 0.003583352081477642f, 0.003969251178205013f, 0.002612743992358446f, 0.003739726496860385f, 0.003085720119997859f, 0.002805272350087762f, 0.0031004641205072403f, 0.0031242985278367996f, 0.0037455405108630657f, 0.0031921775080263615f, 0.0033170294482260942f, 0.002475402317941189f, 0.0031905677169561386f, 0.0032186186872422695f, 0.004002960864454508f, 0.003272158792242408f, 0.0030649318359792233f, 0.003189947223290801f, 0.003082960844039917f, 0.004403321072459221f, 0.003227665089070797f, 0.002774744527414441f, 0.003404806600883603f, 0.0031432274263352156f, 0.0032430971041321754f, 0.005051658023148775f, 0.003206866793334484f, 0.0034860458690673113f, 0.003112025326117873f, 0.0030335502233356237f, 0.003830910427495837f, 0.0031500626355409622f, 0.0037005338817834854f, 0.002820499474182725f, 0.0033070130739361048f, 0.0038198153488337994f, 0.002905339002609253f, 0.002954340074211359f, 0.003473709337413311f, 0.003024670761078596f, 0.0035160090774297714f, 0.0024202156346291304f, 0.0025844420306384563f, 0.0033313476014882326f, 0.004459451884031296f, 0.003181596053764224f, 0.0036595077253878117f, 0.0028953147120773792f, 0.004080237355083227f, 0.0032565337605774403f, 0.002907714108005166f, 0.0034827962517738342f, 0.004436756018549204f, 0.0035805057268589735f, 0.004144827835261822f, 0.0028862946201115847f, 0.0034157023765146732f, 0.0036447651218622923f, 0.004104746505618095f, 0.003619672730565071f, 0.0038297127466648817f, 0.0039000867400318384f, 0.0021711853332817554f, 0.002477840753272176f, 0.0030753067694604397f, 0.0028085194062441587f, 0.0033555796835571527f, 0.003822724102064967f, 0.0028284331783652306f, 0.0030216933228075504f, 0.0038029022980481386f, 0.002831609919667244f, 0.0024721857625991106f, 0.0041216891258955f, 0.003159405430778861f, 0.003681645728647709f, 0.0031726539600640535f, 0.003029582090675831f, 0.003021618351340294f, 0.0030861948616802692f, 0.002725201426073909f, 0.002835104474797845f, 0.003912085201591253f, 0.0029971522744745016f, 0.004394317977130413f, 0.0042119454592466354f, 0.0037531028501689434f, 0.0034786786418408155f, 0.003237274009734392f, 0.00294685666449368f, 0.003075552172958851f, 0.0036979306023567915f, 0.004280906170606613f, 0.0031014445703476667f, 0.003079051151871681f, 0.0045326873660087585f, 0.0032437145709991455f, 0.002941176760941744f, 0.004331361036747694f, 0.0035092162434011698f, 0.002748348517343402f, 0.0037314328365027905f, 0.004752238281071186f, 0.002320114756003022f, 0.003519816789776087f, 0.004168112762272358f, 0.004196331836283207f, 0.0027356971986591816f, 0.004147978499531746f, 0.0037778401747345924f, 0.0031124334782361984f, 0.003362813265994191f, 0.0034588328562676907f, 0.003346972865983844f, 0.004282740876078606f, 0.003196493722498417f, 0.0030021294951438904f, 0.003870350541546941f, 0.0032876296900212765f, 0.0030156676657497883f, 0.003968209493905306f, 0.003233362687751651f, 0.0033370275050401688f, 0.004383997060358524f, 0.0029922877438366413f, 0.0037803887389600277f, 0.0032939810771495104f, 0.0029306095093488693f, 0.0034114436712116003f, 0.004743734374642372f, 0.0035481201484799385f, 0.003292594337835908f, 0.0023397551849484444f, 0.0025174126494675875f, 0.003019348019734025f, 0.0035677903797477484f, 0.004186106380075216f, 0.003096856875345111f, 0.003915409091860056f, 0.002780783222988248f, 0.003191713010892272f, 0.003404526738449931f, 0.005250038579106331f, 0.003160850377753377f, 0.003414303995668888f, 0.0034945239312946796f, 0.003476435085758567f, 0.0036271752323955297f, 0.0032140053808689117f, 0.0028676544316112995f, 0.0049440497532486916f, 0.00417271489277482f, 0.003165662754327059f, 0.003585458267480135f, 0.005288818385452032f, 0.00409907940775156f, 0.003171591553837061f, 0.003053938737139106f, 0.0025478487368673086f, 0.002735655987635255f, 0.005000503733754158f, 0.002515744883567095f, 0.002878674538806081f, 0.003997000865638256f, 0.004563917405903339f, 0.0021703944075852633f, 0.003000990953296423f, 0.0029675334226340055f, 0.003538788529112935f, 0.0022801090963184834f, 0.0036579545121639967f, 0.002761055016890168f, 0.003457077546045184f, 0.0037969532422721386f, 0.0030033402144908905f, 0.003601805306971073f, 0.0027311791200190783f, 0.003339652670547366f, 0.0039992062374949455f, 0.002837333595380187f, 0.0032053482718765736f, 0.004695482086390257f, 0.0043586124666035175f, 0.003377980552613735f, 0.0037180553190410137f, 0.002991194138303399f, 0.0034095162991434336f, 0.003028678009286523f, 0.003965799231082201f, 0.003317673457786441f, 0.004191921558231115f, 0.00359566742554307f, 0.003546847030520439f, 0.004086144268512726f, 0.0025138803757727146f, 0.0047786845825612545f, 0.0024983373004943132f, 0.0037820287980139256f, 0.0033624402713030577f, 0.0026756555307656527f, 0.0035451038274914026f, 0.004234550520777702f, 0.0027862684801220894f, 0.0030799272935837507f, 0.003793864045292139f, 0.0031864128541201353f, 0.004005795810371637f, 0.002660044701769948f, 0.00330475065857172f, 0.0023987472523003817f, 0.004793006461113691f, 0.0032967764418572187f, 0.0033297103364020586f, 0.004439106676727533f, 0.00566335441544652f, 0.003654588246718049f, 0.0035475355107337236f, 0.0021061061415821314f, 0.0020803846418857574f, 0.0031555795576423407f, 0.0026720210444182158f, 0.0028238219674676657f, 0.0022714766673743725f, 0.0024119783192873f, 0.003852158784866333f, 0.002922090934589505f, 0.0029532320331782103f, 0.003155055455863476f, 0.002803320763632655f, 0.0025080398190766573f, 0.003497061552479863f, 0.0031162595842033625f, 0.003987239208072424f, 0.00350839551538229f, 0.003515691263601184f, 0.003205129411071539f, 0.004838809836655855f, 0.0033793319016695023f, 0.002903172979131341f, 0.00319785182364285f, 0.0032325689680874348f, 0.004492191597819328f, 0.003355864668264985f, 0.0034194164909422398f, 0.0037341597490012646f, 0.003358625341206789f, 0.003537762677296996f, 0.0032998104579746723f, 0.0036432526540011168f, 0.0029293254483491182f, 0.003287835046648979f, 0.0032893558964133263f, 0.002595112891867757f, 0.003707737661898136f, 0.003590453416109085f, 0.0032200636342167854f, 0.0030852877534925938f, 0.0040801819413900375f, 0.002726249396800995f, 0.0035488554276525974f, 0.005215225275605917f, 0.002371701644733548f, 0.003766725305467844f, 0.0036091639194637537f, 0.003102949820458889f, 0.0029815956950187683f, 0.002753249369561672f, 0.0029806476086378098f, 0.0034862004686146975f, 0.0033153558615595102f, 0.004852125886827707f, 0.004325474612414837f, 0.0035246191546320915f, 0.0040490408428013325f, 0.0033264956437051296f, 0.0032053126487880945f, 0.0036398316733539104f, 0.00323485117405653f, 0.0033818676602095366f, 0.004029812291264534f, 0.003391121979802847f, 0.0027535385452210903f, 0.002500014379620552f, 0.0033076046966016293f, 0.003988972865045071f, 0.003231781767681241f, 0.002922387793660164f, 0.0036162298638373613f, 0.003482961794361472f, 0.0032491120509803295f, 0.0033549759536981583f, 0.003029115265235305f, 0.0034845375921577215f, 0.0033501586876809597f, 0.0025622632820159197f, 0.004001094494014978f, 0.0035642702132463455f, 0.0029968474991619587f, 0.0033240325283259153f, 0.004109309520572424f, 0.0025690270122140646f, 0.003725918475538492f, 0.0038259292487055063f, 0.004176809918135405f, 0.004135642666369677f, 0.003237203462049365f, 0.004026079550385475f, 0.0024542934261262417f, 0.004082977306097746f, 0.0046365028247237206f, 0.004394953604787588f, 0.002785259857773781f, 0.003956659696996212f, 0.0036425692960619926f, 0.0035840687341988087f, 0.0035639542620629072f, 0.004554868210107088f, 0.003299026982858777f, 0.003039994742721319f, 0.0029191148933023214f, 0.004090667236596346f, 0.0035121222026646137f, 0.0024014324881136417f, 0.0037189065478742123f, 0.003126525552943349f, 0.0033236127346754074f, 0.002722564386203885f, 0.0033349525183439255f, 0.0037483794149011374f, 0.004473926033824682f, 0.004327414557337761f, 0.004401634447276592f, 0.0029269899241626263f, 0.0037142918445169926f, 0.004607285372912884f, 0.0031544279772788286f, 0.003509935922920704f, 0.0033635550644248724f, 0.002526034601032734f, 0.002957486780360341f, 0.003143804147839546f, 0.002757533686235547f, 0.0030028491746634245f, 0.0031538773328065872f, 0.004093052353709936f, 0.0031842838507145643f, 0.004428515676409006f, 0.004046045709401369f, 0.0036356050986796618f, 0.003978265915066004f, 0.0032085287384688854f, 0.003269795561209321f, 0.0038123857229948044f, 0.003441612934693694f, 0.0038648766931146383f, 0.0023215769324451685f, 0.0038241627626121044f, 0.002983103273436427f, 0.0031762912403792143f, 0.004835054744035006f, 0.0024931668303906918f, 0.003750442061573267f, 0.004045190289616585f, 0.0028487215749919415f, 0.00399117823690176f, 0.0026413905434310436f, 0.0037333043292164803f, 0.003480566432699561f, 0.0028966956306248903f, 0.00408373586833477f, 0.0033819072414189577f, 0.003926573786884546f, 0.0029243247117847204f, 0.0034432592801749706f, 0.003432725789025426f, 0.002439961303025484f, 0.0033993397373706102f, 0.003677454311400652f, 0.0034650464076548815f, 0.003420148743316531f, 0.0030167349614202976f, 0.002754098968580365f, 0.0031966762617230415f, 0.004573910962790251f, 0.002805363154038787f, 0.0032232345547527075f, 0.0029378575272858143f, 0.0031950543634593487f, 0.0029745756182819605f, 0.0031985961832106113f, 0.0032668353524059057f, 0.0044241915456950665f, 0.0032774736173450947f, 0.002485846169292927f, 0.002886346774175763f, 0.003280696924775839f, 0.0038889036513864994f, 0.00316671677865088f, 0.0033035106025636196f, 0.0028532135765999556f, 0.002896337304264307f, 0.0037605753168463707f, 0.003834021044895053f, 0.003632542211562395f, 0.0027265746612101793f, 0.0025337832048535347f, 0.002801830880343914f, 0.0032127462327480316f, 0.003567640669643879f, 0.003084008814767003f, 0.004338419530540705f, 0.002696861745789647f, 0.004430986475199461f, 0.002732666442170739f, 0.0028128609992563725f, 0.0020259812008589506f, 0.002977810800075531f, 0.0036109250504523516f, 0.0037119376938790083f, 0.0030199671164155006f, 0.00292566092684865f, 0.0028208340518176556f, 0.004154546651989222f, 0.0033950405195355415f, 0.003263808786869049f, 0.0033124410547316074f, 0.0032646856270730495f, 0.0028978639747947454f, 0.0036781481467187405f, 0.0029319599270820618f, 0.004607713781297207f, 0.0029899151995778084f, 0.0036736982874572277f, 0.004631036892533302f, 0.0024755781050771475f, 0.0020564564038068056f, 0.0035879691131412983f, 0.0030522432643920183f, 0.00359576684422791f, 0.00324401562102139f, 0.0037152976728975773f, 0.0029976163059473038f, 0.002807775977998972f, 0.0045072645880281925f, 0.0029139663092792034f, 0.0037099614273756742f, 0.0035460549406707287f, 0.0020821676589548588f, 0.002881815191358328f, 0.0032670381478965282f, 0.003202517982572317f, 0.0022990452125668526f, 0.003972548525780439f, 0.003186774207279086f, 0.003185480600222945f, 0.002485788892954588f, 0.004279306624084711f, 0.002665827516466379f, 0.0035847926046699286f, 0.0029719758313149214f, 0.003436423372477293f, 0.0024321014061570168f, 0.00424299668520689f, 0.004171170759946108f, 0.0032184224110096693f, 0.0029247684869915247f, 0.004460644908249378f, 0.0029902919195592403f, 0.0028861945029348135f, 0.0038552640471607447f, 0.002665526932105422f, 0.002861520042642951f, 0.002993745729327202f, 0.002683053258806467f, 0.004349599126726389f, 0.002919984282925725f, 0.0037669125013053417f, 0.00324672507122159f, 0.0046578021720051765f, 0.0036950756330043077f, 0.0032206035684794188f, 0.005846401210874319f, 0.0028153955936431885f, 0.0025745041202753782f, 0.0034747396130114794f, 0.0024914704263210297f, 0.003428218886256218f, 0.0028371578082442284f, 0.0028632536996155977f, 0.003059093840420246f, 0.0037377430126070976f, 0.003020016010850668f, 0.0049297017976641655f, 0.004534287378191948f, 0.0028044464997947216f, 0.0034357013646513224f, 0.003789325011894107f, 0.003601046046242118f, 0.0030439526308327913f, 0.0029589240439236164f, 0.0038858691696077585f, 0.003424478927627206f, 0.00345599465072155f, 0.0035918059293180704f, 0.0026107821613550186f, 0.003785767825320363f, 0.002263343892991543f, 0.003425242379307747f, 0.0036002695560455322f, 0.0027481326833367348f, 0.003106087911874056f, 0.0021857854444533587f, 0.0036176114808768034f, 0.0031761564314365387f, 0.0028185811825096607f, 0.0035869337152689695f, 0.002854711376130581f, 0.0032206717878580093f, 0.0029153935611248016f, 0.005226887296885252f, 0.0027939751744270325f, 0.00233809114433825f, 0.003124136244878173f, 0.0034836281556636095f, 0.002895053941756487f, 0.0031773250084370375f, 0.0034155449829995632f, 0.0032466549891978502f, 0.0038561192341148853f, 0.005499626509845257f, 0.00331287388689816f, 0.0029455586336553097f, 0.003280641045421362f, 0.003580452874302864f, 0.004232828505337238f, 0.0027796528302133083f, 0.0034577338956296444f, 0.0021180547773838043f, 0.0030231375712901354f, 0.0029571473132818937f, 0.0031705296132713556f, 0.0034487575758248568f, 0.0032157658133655787f, 0.003243077080696821f, 0.0036676228046417236f, 0.002725809346884489f, 0.003721012733876705f, 0.003394423285499215f, 0.003589499741792679f, 0.0034231164027005434f, 0.003242793958634138f, 0.003433220088481903f, 0.002067904220893979f, 0.0033111812081187963f, 0.0029443134553730488f, 0.0038813967257738113f, 0.0032814834266901016f, 0.003922369331121445f, 0.003426442388445139f, 0.00293751317076385f, 0.0028470356483012438f, 0.003314556088298559f, 0.0032822133507579565f, 0.0029065508861094713f, 0.003813358023762703f, 0.0036045797169208527f, 0.002706753322854638f, 0.004553733393549919f, 0.0028621649835258722f, 0.002810039557516575f, 0.003131522098556161f, 0.002457616850733757f, 0.00520439725369215f, 0.0028233621269464493f, 0.003610387211665511f, 0.004140770994126797f, 0.0034326764289289713f, 0.0030641083139926195f, 0.003223274601623416f, 0.0030944522004574537f, 0.004118409939110279f, 0.0033426331356167793f, 0.0026928517036139965f, 0.0029137267265468836f, 0.003448651870712638f, 0.0025739860720932484f, 0.003905459074303508f, 0.0026986082084476948f, 0.003581263357773423f, 0.0031608922872692347f, 0.003437352366745472f, 0.002987668616697192f, 0.004447998013347387f, 0.005213129334151745f, 0.003712933976203203f, 0.0036746871192008257f, 0.0025970127899199724f, 0.0038606461603194475f, 0.004627428948879242f, 0.003238004632294178f, 0.004478733520954847f, 0.005025829654186964f, 0.0037817435804754496f, 0.0035021970979869366f, 0.0028945524245500565f, 0.003962589427828789f, 0.004543771035969257f, 0.0029480024240911007f, 0.004158353433012962f, 0.004605329129844904f, 0.004979478660970926f, 0.0021574932616204023f, 0.0034999563358724117f, 0.002995280548930168f, 0.004296680446714163f, 0.0028781043365597725f, 0.003867479506880045f, 0.0018423334695398808f, 0.004621624946594238f, 0.0029552571941167116f, 0.003027637954801321f, 0.005502117332071066f, 0.0021529081277549267f, 0.003743671113625169f, 0.0035517150536179543f, 0.0033644521608948708f, 0.0033170594833791256f, 0.003778135636821389f, 0.004578047897666693f, 0.0032008029520511627f, 0.003017503535374999f, 0.003252914408221841f, 0.002103472128510475f, 0.003895179135724902f, 0.006006318144500256f, 0.003193892538547516f, 0.0038154805079102516f, 0.0037448019720613956f, 0.004262095782905817f, 0.003594309091567993f, 0.00561391469091177f, 0.0035793278366327286f, 0.0035146446898579597f, 0.0032114009372889996f, 0.004055689554661512f, 0.004189318045973778f, 0.003842263715341687f, 0.002349945018067956f, 0.0032343631610274315f, 0.0037029890809208155f, 0.003635851899161935f, 0.0026103288400918245f, 0.002795839449390769f, 0.003004438243806362f, 0.003484777407720685f, 0.006441272329539061f, 0.004942821804434061f, 0.004458156414330006f, 0.004038222134113312f, 0.0035233995877206326f, 0.0033998943399637938f, 0.0029703739564865828f, 0.0032373794820159674f, 0.0045597972348332405f, 0.004006240516901016f, 0.004170189145952463f, 0.004389590118080378f, 0.003476739628240466f, 0.003963812720030546f, 0.0040079704485833645f, 0.004147136118263006f, 0.004247632343322039f, 0.005239271093159914f, 0.0030187934171408415f, 0.0034798390697687864f, 0.004100068937987089f, 0.004816458094865084f, 0.003060861025005579f, 0.004593059420585632f, 0.00307724392041564f, 0.0033171256072819233f, 0.004345996305346489f, 0.0035185576416552067f, 0.003962812479585409f, 0.003075422253459692f, 0.004174557980149984f, 0.0030762816313654184f, 0.0029807339888066053f, 0.003346322337165475f, 0.0033120515290647745f, 0.003165798494592309f, 0.003836281131953001f, 0.003420966910198331f, 0.004324416629970074f, 0.0027135033160448074f, 0.0028167434502393007f, 0.00286254845559597f, 0.003776869270950556f, 0.004250402096658945f, 0.004265028052031994f, 0.004149927757680416f, 0.0036660374607890844f, 0.0030074752867221832f, 0.0026996706146746874f, 0.0035726234782487154f, 0.004012850113213062f, 0.004102449864149094f, 0.0028941279742866755f, 0.0030045046005398035f, 0.0033384303096681833f, 0.003178525483235717f, 0.002445133402943611f, 0.0027339854277670383f, 0.0027017269749194384f, 0.004902417305856943f, 0.006217334885150194f, 0.003910827450454235f, 0.0032332094851881266f, 0.004140589386224747f, 0.0031889225356280804f, 0.003691992489621043f, 0.002774754771962762f, 0.003845750354230404f, 0.0036165646743029356f, 0.003846538718789816f, 0.003408136311918497f, 0.0035747999791055918f, 0.0035513807088136673f, 0.0026613050140440464f, 0.0029465181287378073f, 0.0035784088540822268f, 0.004829094745218754f, 0.004003751557320356f, 0.003756314981728792f, 0.0024522908497601748f, 0.0031530875712633133f, 0.0030694708693772554f, 0.002775745000690222f, 0.002557991072535515f, 0.003152392106130719f, 0.0031853346154093742f, 0.003967129159718752f, 0.002348894951865077f, 0.0031626049894839525f, 0.002972076181322336f, 0.0038275676779448986f, 0.003276343457400799f, 0.004269400145858526f, 0.0026504837442189455f, 0.003051087958738208f, 0.0026967511512339115f, 0.0035800973419100046f, 0.00427220156416297f, 0.002603215863928199f, 0.0030055230017751455f, 0.0036595866549760103f, 0.0033190916292369366f, 0.0031237294897437096f, 0.0032204734161496162f, 0.0036758400965481997f, 0.003263141494244337f, 0.00288361101411283f, 0.003829939989373088f);
static const ai_layer_format_type conv2d_62_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;



static const ai_u32 nl_65_t_in_0_shape_ch_prod_const_u32 = 3;

static const ai_u32 conversion_66_t_out_0_shape_h_w_ch_d_prod_const_u32 = 3;
static const ai_float conversion_66_t_in_0_fmt_scale_const_f32 = 0.00390625f;
static const ai_i8 conversion_66_t_in_0_fmt_zero_const_s8 = -128;
STAI_API_ENTRY
stai_return_code stai_network_run(
  stai_network* network,
  const stai_run_mode mode)
{
   STAI_UNUSED(mode)
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_ACTIVATIONS) != STAI_FLAG_ACTIVATIONS,
        STAI_ERROR_NETWORK_INVALID_ACTIVATIONS_PTR, net_ctx->_return_code)

  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_INPUTS) != STAI_FLAG_INPUTS,
                  STAI_ERROR_NETWORK_INVALID_IN_PTR, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_OUTPUTS) != STAI_FLAG_OUTPUTS,
                  STAI_ERROR_NETWORK_INVALID_OUT_PTR, net_ctx->_return_code)

  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_WEIGHTS) != STAI_FLAG_WEIGHTS,
                  STAI_ERROR_NETWORK_INVALID_WEIGHTS_PTR, net_ctx->_return_code)


  /* LITE_KERNEL_SECTION BEGIN serving_default_input_layer_50_0_conversion */
  {
      const ai_u8* serving_default_input_layer_50_0_conversion_t_in_0_ptr_const_u8 = (ai_u8*)(net_ctx->_inputs[0] + 0);
    ai_i8* serving_default_input_layer_50_0_conversion_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[1] + 41280);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(0, 1, {(stai_ptr) serving_default_input_layer_50_0_conversion_t_in_0_ptr_const_u8});
    
  forward_lite_node_convert_integer_iu8os8(serving_default_input_layer_50_0_conversion_t_in_0_ptr_const_u8, serving_default_input_layer_50_0_conversion_t_out_0_ptr_s8, serving_default_input_layer_50_0_conversion_t_in_0_shape_h_w_ch_d_prod_const_u32, 1.0f, serving_default_input_layer_50_0_conversion_t_in_0_fmt_zero_const_u8, serving_default_input_layer_50_0_conversion_t_out_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(0, 1, {(stai_ptr) serving_default_input_layer_50_0_conversion_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END serving_default_input_layer_50_0_conversion */
  /* LITE_KERNEL_SECTION BEGIN conv2d_0 */
  {
      const ai_i8* conv2d_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[1] + 41280);
    const ai_i8* conv2d_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 0);
    const ai_i32* conv2d_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 4);
    ai_i8* conv2d_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[1] + 8128);
    ai_i16* conv2d_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 5184);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(0, 1, {(stai_ptr) conv2d_0_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_0_t_in_0_ptr_const_s8, conv2d_0_t_in_0_shape_w_const_u16, conv2d_0_t_in_0_shape_h_const_u16, conv2d_0_l_stride_1_const_u16, conv2d_0_l_stride_0_const_u16, conv2d_0_t_in_0_shape_ch_const_u16, conv2d_0_t_weight_0_ptr_const_s8, conv2d_0_t_out_0_shape_ch_const_u16, conv2d_0_t_weight_1_ptr_const_s32, conv2d_0_t_in_0_fmt_zero_const_s8, conv2d_0_t_out_0_fmt_zero_const_s8, conv2d_0_t_in_0_fmt_scale_const_f32, conv2d_0_t_out_0_fmt_scale_const_f32, conv2d_0_t_weight_0_fmt_scale_const_f32, conv2d_0_l_out_ch_format_const_layer_format_type, conv2d_0_t_out_0_ptr_s8, 1, 34, conv2d_0_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(0, 1, {(stai_ptr) conv2d_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_0 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_1 */
  {
      const ai_i8* conv2d_1_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[1] + 8128);
    const ai_i8* conv2d_1_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 16);
    const ai_i32* conv2d_1_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 448);
    ai_i8* conv2d_1_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 5184);
    ai_i16* conv2d_1_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 3988);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(1, 1, {(stai_ptr) conv2d_1_t_in_0_ptr_const_s8});
    
  forward_lite_conv2d_rgb_sssa8_ch(conv2d_1_t_in_0_ptr_const_s8, conv2d_1_t_in_0_shape_w_const_u16, conv2d_1_t_weight_0_ptr_const_s8, conv2d_1_t_out_0_shape_ch_const_u16, conv2d_1_t_weight_0_shape_w_const_u16, conv2d_1_l_pad_W_0_const_s32, conv2d_1_l_stride_0_const_u16, conv2d_1_t_weight_1_ptr_const_s32, conv2d_1_t_in_0_fmt_zero_const_s8, conv2d_1_t_out_0_fmt_zero_const_s8, conv2d_1_t_in_0_fmt_scale_const_f32, conv2d_1_t_out_0_fmt_scale_const_f32, conv2d_1_t_weight_0_fmt_scale_const_f32, conv2d_1_l_out_ch_format_const_layer_format_type, conv2d_1_t_out_0_ptr_s8, conv2d_1_t_out_0_shape_w_const_u16, 1196, conv2d_1_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(1, 1, {(stai_ptr) conv2d_1_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_1 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_2_pad_before */
  {
      const ai_ptr conv2d_2_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 5184);
    ai_ptr conv2d_2_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 1024);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(2, 1, {(stai_ptr) conv2d_2_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(conv2d_2_pad_before_t_in_0_ptr_const_ptr, conv2d_2_pad_before_t_out_0_ptr_ptr, (ai_handle)(conv2d_2_pad_before_v_pad_constant_value_const_s8), conv2d_2_pad_before_t_in_0_fmt_bitsize_const_s16, conv2d_2_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(1024), (ai_i32)(1056), (ai_i32)(1056), (ai_i32)(16), (ai_i32)(16));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(2, 1, {(stai_ptr) conv2d_2_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END conv2d_2_pad_before */
  /* LITE_KERNEL_SECTION BEGIN conv2d_2 */
  {
      const ai_i8* conv2d_2_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 1024);
    const ai_i8* conv2d_2_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 512);
    const ai_i32* conv2d_2_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 656);
    ai_i8* conv2d_2_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 0);
    ai_i16* conv2d_2_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 70720);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(2, 1, {(stai_ptr) conv2d_2_t_in_0_ptr_const_s8});
    
  forward_lite_dw_3x3_sssa8_ch(conv2d_2_t_in_0_ptr_const_s8, conv2d_2_t_in_0_shape_w_const_u16, conv2d_2_t_in_0_shape_h_const_u16, conv2d_2_t_in_0_shape_ch_const_u16, conv2d_2_t_weight_0_ptr_const_s8, conv2d_2_l_stride_1_const_u16, conv2d_2_l_stride_0_const_u16, conv2d_2_t_weight_1_ptr_const_s32, conv2d_2_t_in_0_fmt_zero_const_s8, conv2d_2_t_out_0_fmt_zero_const_s8, conv2d_2_t_in_0_fmt_scale_const_f32, conv2d_2_t_out_0_fmt_scale_const_f32, conv2d_2_t_weight_0_fmt_scale_const_f32, conv2d_2_t_out_0_ptr_s8, conv2d_2_t_out_0_shape_w_const_u16, conv2d_2_t_out_0_shape_h_const_u16, 0, 593, conv2d_2_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(2, 1, {(stai_ptr) conv2d_2_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_2 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_3 */
  {
      const ai_i8* conv2d_3_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 0);
    const ai_i8* conv2d_3_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 720);
    const ai_i32* conv2d_3_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 848);
    ai_i8* conv2d_3_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[1] + 8128);
    ai_i16* conv2d_3_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 65536);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(3, 1, {(stai_ptr) conv2d_3_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_3_t_in_0_ptr_const_s8, conv2d_3_t_in_0_shape_w_const_u16, conv2d_3_t_in_0_shape_h_const_u16, conv2d_3_l_stride_1_const_u16, conv2d_3_l_stride_0_const_u16, conv2d_3_t_in_0_shape_ch_const_u16, conv2d_3_t_weight_0_ptr_const_s8, conv2d_3_t_out_0_shape_ch_const_u16, conv2d_3_t_weight_1_ptr_const_s32, conv2d_3_t_in_0_fmt_zero_const_s8, conv2d_3_t_out_0_fmt_zero_const_s8, conv2d_3_t_in_0_fmt_scale_const_f32, conv2d_3_t_out_0_fmt_scale_const_f32, conv2d_3_t_weight_0_fmt_scale_const_f32, conv2d_3_l_out_ch_format_const_layer_format_type, conv2d_3_t_out_0_ptr_s8, 1, 144, conv2d_3_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(3, 1, {(stai_ptr) conv2d_3_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_3 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_4 */
  {
      const ai_i8* conv2d_4_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[1] + 8128);
    const ai_i8* conv2d_4_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 880);
    const ai_i32* conv2d_4_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 1264);
    ai_i8* conv2d_4_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[2] + 12480);
    ai_i16* conv2d_4_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(4, 1, {(stai_ptr) conv2d_4_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_4_t_in_0_ptr_const_s8, conv2d_4_t_in_0_shape_w_const_u16, conv2d_4_t_in_0_shape_h_const_u16, conv2d_4_l_stride_1_const_u16, conv2d_4_l_stride_0_const_u16, conv2d_4_t_in_0_shape_ch_const_u16, conv2d_4_t_weight_0_ptr_const_s8, conv2d_4_t_out_0_shape_ch_const_u16, conv2d_4_t_weight_1_ptr_const_s32, conv2d_4_t_in_0_fmt_zero_const_s8, conv2d_4_t_out_0_fmt_zero_const_s8, conv2d_4_t_in_0_fmt_scale_const_f32, conv2d_4_t_out_0_fmt_scale_const_f32, conv2d_4_t_weight_0_fmt_scale_const_f32, conv2d_4_l_out_ch_format_const_layer_format_type, conv2d_4_t_out_0_ptr_s8, 1, 512, conv2d_4_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(4, 1, {(stai_ptr) conv2d_4_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_4 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_5_pad_before */
  {
      const ai_ptr conv2d_5_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[2] + 12480);
    ai_ptr conv2d_5_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[2] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(5, 1, {(stai_ptr) conv2d_5_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(conv2d_5_pad_before_t_in_0_ptr_const_ptr, conv2d_5_pad_before_t_out_0_ptr_ptr, (ai_handle)(conv2d_5_pad_before_v_pad_constant_value_const_s8), conv2d_5_pad_before_t_in_0_fmt_bitsize_const_s16, conv2d_5_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(3072), (ai_i32)(0), (ai_i32)(6336), (ai_i32)(0), (ai_i32)(96));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(5, 1, {(stai_ptr) conv2d_5_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END conv2d_5_pad_before */
  /* LITE_KERNEL_SECTION BEGIN conv2d_5 */
  {
      const ai_i8* conv2d_5_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[2] + 0);
    const ai_i8* conv2d_5_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 1456);
    const ai_i32* conv2d_5_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 1888);
    ai_i8* conv2d_5_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[1] + 8128);
    ai_i16* conv2d_5_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(5, 1, {(stai_ptr) conv2d_5_t_in_0_ptr_const_s8});
    
  forward_lite_dw_3x3_sssa8_ch(conv2d_5_t_in_0_ptr_const_s8, conv2d_5_t_in_0_shape_w_const_u16, conv2d_5_t_in_0_shape_h_const_u16, conv2d_5_t_in_0_shape_ch_const_u16, conv2d_5_t_weight_0_ptr_const_s8, conv2d_5_l_stride_1_const_u16, conv2d_5_l_stride_0_const_u16, conv2d_5_t_weight_1_ptr_const_s32, conv2d_5_t_in_0_fmt_zero_const_s8, conv2d_5_t_out_0_fmt_zero_const_s8, conv2d_5_t_in_0_fmt_scale_const_f32, conv2d_5_t_out_0_fmt_scale_const_f32, conv2d_5_t_weight_0_fmt_scale_const_f32, conv2d_5_t_out_0_ptr_s8, conv2d_5_t_out_0_shape_w_const_u16, conv2d_5_t_out_0_shape_h_const_u16, 0, 1777, conv2d_5_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(5, 1, {(stai_ptr) conv2d_5_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_5 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_6 */
  {
      const ai_i8* conv2d_6_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[1] + 8128);
    const ai_i8* conv2d_6_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 2080);
    const ai_i32* conv2d_6_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 2464);
    ai_i8* conv2d_6_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[1] + 57280);
    ai_i16* conv2d_6_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(6, 1, {(stai_ptr) conv2d_6_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_6_t_in_0_ptr_const_s8, conv2d_6_t_in_0_shape_w_const_u16, conv2d_6_t_in_0_shape_h_const_u16, conv2d_6_l_stride_1_const_u16, conv2d_6_l_stride_0_const_u16, conv2d_6_t_in_0_shape_ch_const_u16, conv2d_6_t_weight_0_ptr_const_s8, conv2d_6_t_out_0_shape_ch_const_u16, conv2d_6_t_weight_1_ptr_const_s32, conv2d_6_t_in_0_fmt_zero_const_s8, conv2d_6_t_out_0_fmt_zero_const_s8, conv2d_6_t_in_0_fmt_scale_const_f32, conv2d_6_t_out_0_fmt_scale_const_f32, conv2d_6_t_weight_0_fmt_scale_const_f32, conv2d_6_l_out_ch_format_const_layer_format_type, conv2d_6_t_out_0_ptr_s8, 1, 272, conv2d_6_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(6, 1, {(stai_ptr) conv2d_6_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_6 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_7 */
  {
      const ai_i8* conv2d_7_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[1] + 57280);
    const ai_i8* conv2d_7_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 2496);
    const ai_i32* conv2d_7_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 2880);
    ai_i8* conv2d_7_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[1] + 8128);
    ai_i16* conv2d_7_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(7, 1, {(stai_ptr) conv2d_7_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_7_t_in_0_ptr_const_s8, conv2d_7_t_in_0_shape_w_const_u16, conv2d_7_t_in_0_shape_h_const_u16, conv2d_7_l_stride_1_const_u16, conv2d_7_l_stride_0_const_u16, conv2d_7_t_in_0_shape_ch_const_u16, conv2d_7_t_weight_0_ptr_const_s8, conv2d_7_t_out_0_shape_ch_const_u16, conv2d_7_t_weight_1_ptr_const_s32, conv2d_7_t_in_0_fmt_zero_const_s8, conv2d_7_t_out_0_fmt_zero_const_s8, conv2d_7_t_in_0_fmt_scale_const_f32, conv2d_7_t_out_0_fmt_scale_const_f32, conv2d_7_t_weight_0_fmt_scale_const_f32, conv2d_7_l_out_ch_format_const_layer_format_type, conv2d_7_t_out_0_ptr_s8, 1, 512, conv2d_7_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(7, 1, {(stai_ptr) conv2d_7_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_7 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_8_pad_before */
  {
      const ai_ptr conv2d_8_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[1] + 8128);
    ai_ptr conv2d_8_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[1] + 1792);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(8, 1, {(stai_ptr) conv2d_8_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(conv2d_8_pad_before_t_in_0_ptr_const_ptr, conv2d_8_pad_before_t_out_0_ptr_ptr, (ai_handle)(conv2d_8_pad_before_v_pad_constant_value_const_s8), conv2d_8_pad_before_t_in_0_fmt_bitsize_const_s16, conv2d_8_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(1536), (ai_i32)(1632), (ai_i32)(1632), (ai_i32)(48), (ai_i32)(48));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(8, 1, {(stai_ptr) conv2d_8_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END conv2d_8_pad_before */
  /* LITE_KERNEL_SECTION BEGIN conv2d_8 */
  {
      const ai_i8* conv2d_8_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[1] + 1792);
    const ai_i8* conv2d_8_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 3072);
    const ai_i32* conv2d_8_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 3504);
    ai_i8* conv2d_8_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[1] + 256);
    ai_i16* conv2d_8_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(8, 1, {(stai_ptr) conv2d_8_t_in_0_ptr_const_s8});
    
  forward_lite_dw_3x3_sssa8_ch(conv2d_8_t_in_0_ptr_const_s8, conv2d_8_t_in_0_shape_w_const_u16, conv2d_8_t_in_0_shape_h_const_u16, conv2d_8_t_in_0_shape_ch_const_u16, conv2d_8_t_weight_0_ptr_const_s8, conv2d_8_l_stride_1_const_u16, conv2d_8_l_stride_0_const_u16, conv2d_8_t_weight_1_ptr_const_s32, conv2d_8_t_in_0_fmt_zero_const_s8, conv2d_8_t_out_0_fmt_zero_const_s8, conv2d_8_t_in_0_fmt_scale_const_f32, conv2d_8_t_out_0_fmt_scale_const_f32, conv2d_8_t_weight_0_fmt_scale_const_f32, conv2d_8_t_out_0_ptr_s8, conv2d_8_t_out_0_shape_w_const_u16, conv2d_8_t_out_0_shape_h_const_u16, 0, 1777, conv2d_8_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(8, 1, {(stai_ptr) conv2d_8_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_8 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_9 */
  {
      const ai_i8* conv2d_9_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[1] + 256);
    const ai_i8* conv2d_9_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 3696);
    const ai_i32* conv2d_9_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 4080);
    ai_i8* conv2d_9_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[1] + 0);
    ai_i16* conv2d_9_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(9, 1, {(stai_ptr) conv2d_9_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_9_t_in_0_ptr_const_s8, conv2d_9_t_in_0_shape_w_const_u16, conv2d_9_t_in_0_shape_h_const_u16, conv2d_9_l_stride_1_const_u16, conv2d_9_l_stride_0_const_u16, conv2d_9_t_in_0_shape_ch_const_u16, conv2d_9_t_weight_0_ptr_const_s8, conv2d_9_t_out_0_shape_ch_const_u16, conv2d_9_t_weight_1_ptr_const_s32, conv2d_9_t_in_0_fmt_zero_const_s8, conv2d_9_t_out_0_fmt_zero_const_s8, conv2d_9_t_in_0_fmt_scale_const_f32, conv2d_9_t_out_0_fmt_scale_const_f32, conv2d_9_t_weight_0_fmt_scale_const_f32, conv2d_9_l_out_ch_format_const_layer_format_type, conv2d_9_t_out_0_ptr_s8, 1, 272, conv2d_9_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(9, 1, {(stai_ptr) conv2d_9_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_9 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_10 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_10(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_10 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_11 */
  {
      const ai_i8* conv2d_11_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[1] + 8192);
    const ai_i8* conv2d_11_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 4112);
    const ai_i32* conv2d_11_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 4496);
    ai_i8* conv2d_11_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[1] + 16384);
    ai_i16* conv2d_11_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(11, 1, {(stai_ptr) conv2d_11_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_11_t_in_0_ptr_const_s8, conv2d_11_t_in_0_shape_w_const_u16, conv2d_11_t_in_0_shape_h_const_u16, conv2d_11_l_stride_1_const_u16, conv2d_11_l_stride_0_const_u16, conv2d_11_t_in_0_shape_ch_const_u16, conv2d_11_t_weight_0_ptr_const_s8, conv2d_11_t_out_0_shape_ch_const_u16, conv2d_11_t_weight_1_ptr_const_s32, conv2d_11_t_in_0_fmt_zero_const_s8, conv2d_11_t_out_0_fmt_zero_const_s8, conv2d_11_t_in_0_fmt_scale_const_f32, conv2d_11_t_out_0_fmt_scale_const_f32, conv2d_11_t_weight_0_fmt_scale_const_f32, conv2d_11_l_out_ch_format_const_layer_format_type, conv2d_11_t_out_0_ptr_s8, 1, 512, conv2d_11_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(11, 1, {(stai_ptr) conv2d_11_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_11 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_12_pad_before */
  {
      const ai_ptr conv2d_12_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[1] + 16384);
    ai_ptr conv2d_12_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[1] + 10048);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(12, 1, {(stai_ptr) conv2d_12_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(conv2d_12_pad_before_t_in_0_ptr_const_ptr, conv2d_12_pad_before_t_out_0_ptr_ptr, (ai_handle)(conv2d_12_pad_before_v_pad_constant_value_const_s8), conv2d_12_pad_before_t_in_0_fmt_bitsize_const_s16, conv2d_12_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(1536), (ai_i32)(0), (ai_i32)(3264), (ai_i32)(0), (ai_i32)(96));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(12, 1, {(stai_ptr) conv2d_12_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END conv2d_12_pad_before */
  /* LITE_KERNEL_SECTION BEGIN conv2d_12 */
  {
      const ai_i8* conv2d_12_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[1] + 10048);
    const ai_i8* conv2d_12_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 4688);
    const ai_i32* conv2d_12_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 5120);
    ai_i8* conv2d_12_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[1] + 9280);
    ai_i16* conv2d_12_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(12, 1, {(stai_ptr) conv2d_12_t_in_0_ptr_const_s8});
    
  forward_lite_dw_3x3_sssa8_ch(conv2d_12_t_in_0_ptr_const_s8, conv2d_12_t_in_0_shape_w_const_u16, conv2d_12_t_in_0_shape_h_const_u16, conv2d_12_t_in_0_shape_ch_const_u16, conv2d_12_t_weight_0_ptr_const_s8, conv2d_12_l_stride_1_const_u16, conv2d_12_l_stride_0_const_u16, conv2d_12_t_weight_1_ptr_const_s32, conv2d_12_t_in_0_fmt_zero_const_s8, conv2d_12_t_out_0_fmt_zero_const_s8, conv2d_12_t_in_0_fmt_scale_const_f32, conv2d_12_t_out_0_fmt_scale_const_f32, conv2d_12_t_weight_0_fmt_scale_const_f32, conv2d_12_t_out_0_ptr_s8, conv2d_12_t_out_0_shape_w_const_u16, conv2d_12_t_out_0_shape_h_const_u16, 0, 1777, conv2d_12_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(12, 1, {(stai_ptr) conv2d_12_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_12 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_13 */
  {
      const ai_i8* conv2d_13_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[1] + 9280);
    const ai_i8* conv2d_13_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 5312);
    const ai_i32* conv2d_13_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 6080);
    ai_i8* conv2d_13_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[1] + 0);
    ai_i16* conv2d_13_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(13, 1, {(stai_ptr) conv2d_13_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_13_t_in_0_ptr_const_s8, conv2d_13_t_in_0_shape_w_const_u16, conv2d_13_t_in_0_shape_h_const_u16, conv2d_13_l_stride_1_const_u16, conv2d_13_l_stride_0_const_u16, conv2d_13_t_in_0_shape_ch_const_u16, conv2d_13_t_weight_0_ptr_const_s8, conv2d_13_t_out_0_shape_ch_const_u16, conv2d_13_t_weight_1_ptr_const_s32, conv2d_13_t_in_0_fmt_zero_const_s8, conv2d_13_t_out_0_fmt_zero_const_s8, conv2d_13_t_in_0_fmt_scale_const_f32, conv2d_13_t_out_0_fmt_scale_const_f32, conv2d_13_t_weight_0_fmt_scale_const_f32, conv2d_13_l_out_ch_format_const_layer_format_type, conv2d_13_t_out_0_ptr_s8, 1, 352, conv2d_13_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(13, 1, {(stai_ptr) conv2d_13_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_13 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_14 */
  {
      const ai_i8* conv2d_14_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[1] + 0);
    const ai_i8* conv2d_14_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 6144);
    const ai_i32* conv2d_14_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 7680);
    ai_i8* conv2d_14_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[1] + 4096);
    ai_i16* conv2d_14_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(14, 1, {(stai_ptr) conv2d_14_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_14_t_in_0_ptr_const_s8, conv2d_14_t_in_0_shape_w_const_u16, conv2d_14_t_in_0_shape_h_const_u16, conv2d_14_l_stride_1_const_u16, conv2d_14_l_stride_0_const_u16, conv2d_14_t_in_0_shape_ch_const_u16, conv2d_14_t_weight_0_ptr_const_s8, conv2d_14_t_out_0_shape_ch_const_u16, conv2d_14_t_weight_1_ptr_const_s32, conv2d_14_t_in_0_fmt_zero_const_s8, conv2d_14_t_out_0_fmt_zero_const_s8, conv2d_14_t_in_0_fmt_scale_const_f32, conv2d_14_t_out_0_fmt_scale_const_f32, conv2d_14_t_weight_0_fmt_scale_const_f32, conv2d_14_l_out_ch_format_const_layer_format_type, conv2d_14_t_out_0_ptr_s8, 1, 1024, conv2d_14_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(14, 1, {(stai_ptr) conv2d_14_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_14 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_15_pad_before */
  {
      const ai_ptr conv2d_15_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[1] + 4096);
    ai_ptr conv2d_15_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[1] + 28672);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(15, 1, {(stai_ptr) conv2d_15_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(conv2d_15_pad_before_t_in_0_ptr_const_ptr, conv2d_15_pad_before_t_out_0_ptr_ptr, (ai_handle)(conv2d_15_pad_before_v_pad_constant_value_const_s8), conv2d_15_pad_before_t_in_0_fmt_bitsize_const_s16, conv2d_15_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(1536), (ai_i32)(1728), (ai_i32)(1728), (ai_i32)(96), (ai_i32)(96));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(15, 1, {(stai_ptr) conv2d_15_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END conv2d_15_pad_before */
  /* LITE_KERNEL_SECTION BEGIN conv2d_15 */
  {
      const ai_i8* conv2d_15_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[1] + 28672);
    const ai_i8* conv2d_15_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 8064);
    const ai_i32* conv2d_15_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 8928);
    ai_i8* conv2d_15_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[1] + 4096);
    ai_i16* conv2d_15_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(15, 1, {(stai_ptr) conv2d_15_t_in_0_ptr_const_s8});
    
  forward_lite_dw_3x3_sssa8_ch(conv2d_15_t_in_0_ptr_const_s8, conv2d_15_t_in_0_shape_w_const_u16, conv2d_15_t_in_0_shape_h_const_u16, conv2d_15_t_in_0_shape_ch_const_u16, conv2d_15_t_weight_0_ptr_const_s8, conv2d_15_l_stride_1_const_u16, conv2d_15_l_stride_0_const_u16, conv2d_15_t_weight_1_ptr_const_s32, conv2d_15_t_in_0_fmt_zero_const_s8, conv2d_15_t_out_0_fmt_zero_const_s8, conv2d_15_t_in_0_fmt_scale_const_f32, conv2d_15_t_out_0_fmt_scale_const_f32, conv2d_15_t_weight_0_fmt_scale_const_f32, conv2d_15_t_out_0_ptr_s8, conv2d_15_t_out_0_shape_w_const_u16, conv2d_15_t_out_0_shape_h_const_u16, 0, 3553, conv2d_15_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(15, 1, {(stai_ptr) conv2d_15_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_15 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_16 */
  {
      const ai_i8* conv2d_16_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[1] + 4096);
    const ai_i8* conv2d_16_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 9312);
    const ai_i32* conv2d_16_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 10848);
    ai_i8* conv2d_16_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[1] + 28672);
    ai_i16* conv2d_16_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(16, 1, {(stai_ptr) conv2d_16_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_16_t_in_0_ptr_const_s8, conv2d_16_t_in_0_shape_w_const_u16, conv2d_16_t_in_0_shape_h_const_u16, conv2d_16_l_stride_1_const_u16, conv2d_16_l_stride_0_const_u16, conv2d_16_t_in_0_shape_ch_const_u16, conv2d_16_t_weight_0_ptr_const_s8, conv2d_16_t_out_0_shape_ch_const_u16, conv2d_16_t_weight_1_ptr_const_s32, conv2d_16_t_in_0_fmt_zero_const_s8, conv2d_16_t_out_0_fmt_zero_const_s8, conv2d_16_t_in_0_fmt_scale_const_f32, conv2d_16_t_out_0_fmt_scale_const_f32, conv2d_16_t_weight_0_fmt_scale_const_f32, conv2d_16_l_out_ch_format_const_layer_format_type, conv2d_16_t_out_0_ptr_s8, 1, 544, conv2d_16_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(16, 1, {(stai_ptr) conv2d_16_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_16 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_17 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_17(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_17 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_18 */
  {
      const ai_i8* conv2d_18_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[1] + 4096);
    const ai_i8* conv2d_18_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 10912);
    const ai_i32* conv2d_18_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 12448);
    ai_i8* conv2d_18_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[1] + 8192);
    ai_i16* conv2d_18_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(18, 1, {(stai_ptr) conv2d_18_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_18_t_in_0_ptr_const_s8, conv2d_18_t_in_0_shape_w_const_u16, conv2d_18_t_in_0_shape_h_const_u16, conv2d_18_l_stride_1_const_u16, conv2d_18_l_stride_0_const_u16, conv2d_18_t_in_0_shape_ch_const_u16, conv2d_18_t_weight_0_ptr_const_s8, conv2d_18_t_out_0_shape_ch_const_u16, conv2d_18_t_weight_1_ptr_const_s32, conv2d_18_t_in_0_fmt_zero_const_s8, conv2d_18_t_out_0_fmt_zero_const_s8, conv2d_18_t_in_0_fmt_scale_const_f32, conv2d_18_t_out_0_fmt_scale_const_f32, conv2d_18_t_weight_0_fmt_scale_const_f32, conv2d_18_l_out_ch_format_const_layer_format_type, conv2d_18_t_out_0_ptr_s8, 1, 1024, conv2d_18_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(18, 1, {(stai_ptr) conv2d_18_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_18 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_19_pad_before */
  {
      const ai_ptr conv2d_19_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[1] + 8192);
    ai_ptr conv2d_19_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[1] + 32768);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(19, 1, {(stai_ptr) conv2d_19_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(conv2d_19_pad_before_t_in_0_ptr_const_ptr, conv2d_19_pad_before_t_out_0_ptr_ptr, (ai_handle)(conv2d_19_pad_before_v_pad_constant_value_const_s8), conv2d_19_pad_before_t_in_0_fmt_bitsize_const_s16, conv2d_19_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(1536), (ai_i32)(1728), (ai_i32)(1728), (ai_i32)(96), (ai_i32)(96));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(19, 1, {(stai_ptr) conv2d_19_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END conv2d_19_pad_before */
  /* LITE_KERNEL_SECTION BEGIN conv2d_19 */
  {
      const ai_i8* conv2d_19_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[1] + 32768);
    const ai_i8* conv2d_19_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 12832);
    const ai_i32* conv2d_19_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 13696);
    ai_i8* conv2d_19_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[1] + 8192);
    ai_i16* conv2d_19_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(19, 1, {(stai_ptr) conv2d_19_t_in_0_ptr_const_s8});
    
  forward_lite_dw_3x3_sssa8_ch(conv2d_19_t_in_0_ptr_const_s8, conv2d_19_t_in_0_shape_w_const_u16, conv2d_19_t_in_0_shape_h_const_u16, conv2d_19_t_in_0_shape_ch_const_u16, conv2d_19_t_weight_0_ptr_const_s8, conv2d_19_l_stride_1_const_u16, conv2d_19_l_stride_0_const_u16, conv2d_19_t_weight_1_ptr_const_s32, conv2d_19_t_in_0_fmt_zero_const_s8, conv2d_19_t_out_0_fmt_zero_const_s8, conv2d_19_t_in_0_fmt_scale_const_f32, conv2d_19_t_out_0_fmt_scale_const_f32, conv2d_19_t_weight_0_fmt_scale_const_f32, conv2d_19_t_out_0_ptr_s8, conv2d_19_t_out_0_shape_w_const_u16, conv2d_19_t_out_0_shape_h_const_u16, 0, 3553, conv2d_19_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(19, 1, {(stai_ptr) conv2d_19_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_19 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_20 */
  {
      const ai_i8* conv2d_20_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[1] + 8192);
    const ai_i8* conv2d_20_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 14080);
    const ai_i32* conv2d_20_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 15616);
    ai_i8* conv2d_20_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[1] + 0);
    ai_i16* conv2d_20_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(20, 1, {(stai_ptr) conv2d_20_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_20_t_in_0_ptr_const_s8, conv2d_20_t_in_0_shape_w_const_u16, conv2d_20_t_in_0_shape_h_const_u16, conv2d_20_l_stride_1_const_u16, conv2d_20_l_stride_0_const_u16, conv2d_20_t_in_0_shape_ch_const_u16, conv2d_20_t_weight_0_ptr_const_s8, conv2d_20_t_out_0_shape_ch_const_u16, conv2d_20_t_weight_1_ptr_const_s32, conv2d_20_t_in_0_fmt_zero_const_s8, conv2d_20_t_out_0_fmt_zero_const_s8, conv2d_20_t_in_0_fmt_scale_const_f32, conv2d_20_t_out_0_fmt_scale_const_f32, conv2d_20_t_weight_0_fmt_scale_const_f32, conv2d_20_l_out_ch_format_const_layer_format_type, conv2d_20_t_out_0_ptr_s8, 1, 544, conv2d_20_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(20, 1, {(stai_ptr) conv2d_20_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_20 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_21 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_21(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_21 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_22 */
  {
      const ai_i8* conv2d_22_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[1] + 8192);
    const ai_i8* conv2d_22_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 15680);
    const ai_i32* conv2d_22_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 17216);
    ai_i8* conv2d_22_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[1] + 12288);
    ai_i16* conv2d_22_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(22, 1, {(stai_ptr) conv2d_22_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_22_t_in_0_ptr_const_s8, conv2d_22_t_in_0_shape_w_const_u16, conv2d_22_t_in_0_shape_h_const_u16, conv2d_22_l_stride_1_const_u16, conv2d_22_l_stride_0_const_u16, conv2d_22_t_in_0_shape_ch_const_u16, conv2d_22_t_weight_0_ptr_const_s8, conv2d_22_t_out_0_shape_ch_const_u16, conv2d_22_t_weight_1_ptr_const_s32, conv2d_22_t_in_0_fmt_zero_const_s8, conv2d_22_t_out_0_fmt_zero_const_s8, conv2d_22_t_in_0_fmt_scale_const_f32, conv2d_22_t_out_0_fmt_scale_const_f32, conv2d_22_t_weight_0_fmt_scale_const_f32, conv2d_22_l_out_ch_format_const_layer_format_type, conv2d_22_t_out_0_ptr_s8, 1, 1024, conv2d_22_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(22, 1, {(stai_ptr) conv2d_22_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_22 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_23_pad_before */
  {
      const ai_ptr conv2d_23_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[1] + 12288);
    ai_ptr conv2d_23_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[1] + 5760);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(23, 1, {(stai_ptr) conv2d_23_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(conv2d_23_pad_before_t_in_0_ptr_const_ptr, conv2d_23_pad_before_t_out_0_ptr_ptr, (ai_handle)(conv2d_23_pad_before_v_pad_constant_value_const_s8), conv2d_23_pad_before_t_in_0_fmt_bitsize_const_s16, conv2d_23_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(1536), (ai_i32)(0), (ai_i32)(3456), (ai_i32)(0), (ai_i32)(192));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(23, 1, {(stai_ptr) conv2d_23_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END conv2d_23_pad_before */
  /* LITE_KERNEL_SECTION BEGIN conv2d_23 */
  {
      const ai_i8* conv2d_23_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[1] + 5760);
    const ai_i8* conv2d_23_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 17600);
    const ai_i32* conv2d_23_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 18464);
    ai_i8* conv2d_23_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[1] + 36864);
    ai_i16* conv2d_23_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(23, 1, {(stai_ptr) conv2d_23_t_in_0_ptr_const_s8});
    
  forward_lite_dw_3x3_sssa8_ch(conv2d_23_t_in_0_ptr_const_s8, conv2d_23_t_in_0_shape_w_const_u16, conv2d_23_t_in_0_shape_h_const_u16, conv2d_23_t_in_0_shape_ch_const_u16, conv2d_23_t_weight_0_ptr_const_s8, conv2d_23_l_stride_1_const_u16, conv2d_23_l_stride_0_const_u16, conv2d_23_t_weight_1_ptr_const_s32, conv2d_23_t_in_0_fmt_zero_const_s8, conv2d_23_t_out_0_fmt_zero_const_s8, conv2d_23_t_in_0_fmt_scale_const_f32, conv2d_23_t_out_0_fmt_scale_const_f32, conv2d_23_t_weight_0_fmt_scale_const_f32, conv2d_23_t_out_0_ptr_s8, conv2d_23_t_out_0_shape_w_const_u16, conv2d_23_t_out_0_shape_h_const_u16, 0, 3553, conv2d_23_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(23, 1, {(stai_ptr) conv2d_23_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_23 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_24 */
  {
      const ai_i8* conv2d_24_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[1] + 36864);
    const ai_i8* conv2d_24_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 18848);
    const ai_i32* conv2d_24_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 21152);
    ai_i8* conv2d_24_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[1] + 0);
    ai_i16* conv2d_24_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(24, 1, {(stai_ptr) conv2d_24_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_24_t_in_0_ptr_const_s8, conv2d_24_t_in_0_shape_w_const_u16, conv2d_24_t_in_0_shape_h_const_u16, conv2d_24_l_stride_1_const_u16, conv2d_24_l_stride_0_const_u16, conv2d_24_t_in_0_shape_ch_const_u16, conv2d_24_t_weight_0_ptr_const_s8, conv2d_24_t_out_0_shape_ch_const_u16, conv2d_24_t_weight_1_ptr_const_s32, conv2d_24_t_in_0_fmt_zero_const_s8, conv2d_24_t_out_0_fmt_zero_const_s8, conv2d_24_t_in_0_fmt_scale_const_f32, conv2d_24_t_out_0_fmt_scale_const_f32, conv2d_24_t_weight_0_fmt_scale_const_f32, conv2d_24_l_out_ch_format_const_layer_format_type, conv2d_24_t_out_0_ptr_s8, 1, 624, conv2d_24_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(24, 1, {(stai_ptr) conv2d_24_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_24 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_25 */
  {
      const ai_i8* conv2d_25_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[1] + 0);
    const ai_i8* conv2d_25_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 21248);
    const ai_i32* conv2d_25_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 24704);
    ai_i8* conv2d_25_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[1] + 1536);
    ai_i16* conv2d_25_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(25, 1, {(stai_ptr) conv2d_25_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_25_t_in_0_ptr_const_s8, conv2d_25_t_in_0_shape_w_const_u16, conv2d_25_t_in_0_shape_h_const_u16, conv2d_25_l_stride_1_const_u16, conv2d_25_l_stride_0_const_u16, conv2d_25_t_in_0_shape_ch_const_u16, conv2d_25_t_weight_0_ptr_const_s8, conv2d_25_t_out_0_shape_ch_const_u16, conv2d_25_t_weight_1_ptr_const_s32, conv2d_25_t_in_0_fmt_zero_const_s8, conv2d_25_t_out_0_fmt_zero_const_s8, conv2d_25_t_in_0_fmt_scale_const_f32, conv2d_25_t_out_0_fmt_scale_const_f32, conv2d_25_t_weight_0_fmt_scale_const_f32, conv2d_25_l_out_ch_format_const_layer_format_type, conv2d_25_t_out_0_ptr_s8, 1, 1536, conv2d_25_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(25, 1, {(stai_ptr) conv2d_25_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_25 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_26_pad_before */
  {
      const ai_ptr conv2d_26_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[1] + 1536);
    ai_ptr conv2d_26_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[1] + 10752);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(26, 1, {(stai_ptr) conv2d_26_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(conv2d_26_pad_before_t_in_0_ptr_const_ptr, conv2d_26_pad_before_t_out_0_ptr_ptr, (ai_handle)(conv2d_26_pad_before_v_pad_constant_value_const_s8), conv2d_26_pad_before_t_in_0_fmt_bitsize_const_s16, conv2d_26_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(1152), (ai_i32)(1440), (ai_i32)(1440), (ai_i32)(144), (ai_i32)(144));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(26, 1, {(stai_ptr) conv2d_26_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END conv2d_26_pad_before */
  /* LITE_KERNEL_SECTION BEGIN conv2d_26 */
  {
      const ai_i8* conv2d_26_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[1] + 10752);
    const ai_i8* conv2d_26_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 25280);
    const ai_i32* conv2d_26_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 26576);
    ai_i8* conv2d_26_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[1] + 1536);
    ai_i16* conv2d_26_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(26, 1, {(stai_ptr) conv2d_26_t_in_0_ptr_const_s8});
    
  forward_lite_dw_3x3_sssa8_ch(conv2d_26_t_in_0_ptr_const_s8, conv2d_26_t_in_0_shape_w_const_u16, conv2d_26_t_in_0_shape_h_const_u16, conv2d_26_t_in_0_shape_ch_const_u16, conv2d_26_t_weight_0_ptr_const_s8, conv2d_26_l_stride_1_const_u16, conv2d_26_l_stride_0_const_u16, conv2d_26_t_weight_1_ptr_const_s32, conv2d_26_t_in_0_fmt_zero_const_s8, conv2d_26_t_out_0_fmt_zero_const_s8, conv2d_26_t_in_0_fmt_scale_const_f32, conv2d_26_t_out_0_fmt_scale_const_f32, conv2d_26_t_weight_0_fmt_scale_const_f32, conv2d_26_t_out_0_ptr_s8, conv2d_26_t_out_0_shape_w_const_u16, conv2d_26_t_out_0_shape_h_const_u16, 0, 5329, conv2d_26_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(26, 1, {(stai_ptr) conv2d_26_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_26 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_27 */
  {
      const ai_i8* conv2d_27_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[1] + 1536);
    const ai_i8* conv2d_27_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 27152);
    const ai_i32* conv2d_27_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 30608);
    ai_i8* conv2d_27_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[1] + 10752);
    ai_i16* conv2d_27_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(27, 1, {(stai_ptr) conv2d_27_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_27_t_in_0_ptr_const_s8, conv2d_27_t_in_0_shape_w_const_u16, conv2d_27_t_in_0_shape_h_const_u16, conv2d_27_l_stride_1_const_u16, conv2d_27_l_stride_0_const_u16, conv2d_27_t_in_0_shape_ch_const_u16, conv2d_27_t_weight_0_ptr_const_s8, conv2d_27_t_out_0_shape_ch_const_u16, conv2d_27_t_weight_1_ptr_const_s32, conv2d_27_t_in_0_fmt_zero_const_s8, conv2d_27_t_out_0_fmt_zero_const_s8, conv2d_27_t_in_0_fmt_scale_const_f32, conv2d_27_t_out_0_fmt_scale_const_f32, conv2d_27_t_weight_0_fmt_scale_const_f32, conv2d_27_l_out_ch_format_const_layer_format_type, conv2d_27_t_out_0_ptr_s8, 1, 816, conv2d_27_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(27, 1, {(stai_ptr) conv2d_27_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_27 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_28 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_28(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_28 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_29 */
  {
      const ai_i8* conv2d_29_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[1] + 1536);
    const ai_i8* conv2d_29_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 30704);
    const ai_i32* conv2d_29_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 34160);
    ai_i8* conv2d_29_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[1] + 3072);
    ai_i16* conv2d_29_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(29, 1, {(stai_ptr) conv2d_29_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_29_t_in_0_ptr_const_s8, conv2d_29_t_in_0_shape_w_const_u16, conv2d_29_t_in_0_shape_h_const_u16, conv2d_29_l_stride_1_const_u16, conv2d_29_l_stride_0_const_u16, conv2d_29_t_in_0_shape_ch_const_u16, conv2d_29_t_weight_0_ptr_const_s8, conv2d_29_t_out_0_shape_ch_const_u16, conv2d_29_t_weight_1_ptr_const_s32, conv2d_29_t_in_0_fmt_zero_const_s8, conv2d_29_t_out_0_fmt_zero_const_s8, conv2d_29_t_in_0_fmt_scale_const_f32, conv2d_29_t_out_0_fmt_scale_const_f32, conv2d_29_t_weight_0_fmt_scale_const_f32, conv2d_29_l_out_ch_format_const_layer_format_type, conv2d_29_t_out_0_ptr_s8, 1, 1536, conv2d_29_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(29, 1, {(stai_ptr) conv2d_29_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_29 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_30_pad_before */
  {
      const ai_ptr conv2d_30_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[1] + 3072);
    ai_ptr conv2d_30_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[1] + 12288);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(30, 1, {(stai_ptr) conv2d_30_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(conv2d_30_pad_before_t_in_0_ptr_const_ptr, conv2d_30_pad_before_t_out_0_ptr_ptr, (ai_handle)(conv2d_30_pad_before_v_pad_constant_value_const_s8), conv2d_30_pad_before_t_in_0_fmt_bitsize_const_s16, conv2d_30_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(1152), (ai_i32)(1440), (ai_i32)(1440), (ai_i32)(144), (ai_i32)(144));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(30, 1, {(stai_ptr) conv2d_30_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END conv2d_30_pad_before */
  /* LITE_KERNEL_SECTION BEGIN conv2d_30 */
  {
      const ai_i8* conv2d_30_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[1] + 12288);
    const ai_i8* conv2d_30_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 34736);
    const ai_i32* conv2d_30_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 36032);
    ai_i8* conv2d_30_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[1] + 3072);
    ai_i16* conv2d_30_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(30, 1, {(stai_ptr) conv2d_30_t_in_0_ptr_const_s8});
    
  forward_lite_dw_3x3_sssa8_ch(conv2d_30_t_in_0_ptr_const_s8, conv2d_30_t_in_0_shape_w_const_u16, conv2d_30_t_in_0_shape_h_const_u16, conv2d_30_t_in_0_shape_ch_const_u16, conv2d_30_t_weight_0_ptr_const_s8, conv2d_30_l_stride_1_const_u16, conv2d_30_l_stride_0_const_u16, conv2d_30_t_weight_1_ptr_const_s32, conv2d_30_t_in_0_fmt_zero_const_s8, conv2d_30_t_out_0_fmt_zero_const_s8, conv2d_30_t_in_0_fmt_scale_const_f32, conv2d_30_t_out_0_fmt_scale_const_f32, conv2d_30_t_weight_0_fmt_scale_const_f32, conv2d_30_t_out_0_ptr_s8, conv2d_30_t_out_0_shape_w_const_u16, conv2d_30_t_out_0_shape_h_const_u16, 0, 5329, conv2d_30_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(30, 1, {(stai_ptr) conv2d_30_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_30 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_31 */
  {
      const ai_i8* conv2d_31_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[1] + 3072);
    const ai_i8* conv2d_31_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 36608);
    const ai_i32* conv2d_31_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 40064);
    ai_i8* conv2d_31_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[1] + 0);
    ai_i16* conv2d_31_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(31, 1, {(stai_ptr) conv2d_31_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_31_t_in_0_ptr_const_s8, conv2d_31_t_in_0_shape_w_const_u16, conv2d_31_t_in_0_shape_h_const_u16, conv2d_31_l_stride_1_const_u16, conv2d_31_l_stride_0_const_u16, conv2d_31_t_in_0_shape_ch_const_u16, conv2d_31_t_weight_0_ptr_const_s8, conv2d_31_t_out_0_shape_ch_const_u16, conv2d_31_t_weight_1_ptr_const_s32, conv2d_31_t_in_0_fmt_zero_const_s8, conv2d_31_t_out_0_fmt_zero_const_s8, conv2d_31_t_in_0_fmt_scale_const_f32, conv2d_31_t_out_0_fmt_scale_const_f32, conv2d_31_t_weight_0_fmt_scale_const_f32, conv2d_31_l_out_ch_format_const_layer_format_type, conv2d_31_t_out_0_ptr_s8, 1, 816, conv2d_31_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(31, 1, {(stai_ptr) conv2d_31_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_31 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_32 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_32(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_32 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_33 */
  {
      const ai_i8* conv2d_33_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[1] + 3072);
    const ai_i8* conv2d_33_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 40160);
    const ai_i32* conv2d_33_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 43616);
    ai_i8* conv2d_33_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[1] + 4608);
    ai_i16* conv2d_33_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(33, 1, {(stai_ptr) conv2d_33_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_33_t_in_0_ptr_const_s8, conv2d_33_t_in_0_shape_w_const_u16, conv2d_33_t_in_0_shape_h_const_u16, conv2d_33_l_stride_1_const_u16, conv2d_33_l_stride_0_const_u16, conv2d_33_t_in_0_shape_ch_const_u16, conv2d_33_t_weight_0_ptr_const_s8, conv2d_33_t_out_0_shape_ch_const_u16, conv2d_33_t_weight_1_ptr_const_s32, conv2d_33_t_in_0_fmt_zero_const_s8, conv2d_33_t_out_0_fmt_zero_const_s8, conv2d_33_t_in_0_fmt_scale_const_f32, conv2d_33_t_out_0_fmt_scale_const_f32, conv2d_33_t_weight_0_fmt_scale_const_f32, conv2d_33_l_out_ch_format_const_layer_format_type, conv2d_33_t_out_0_ptr_s8, 1, 1536, conv2d_33_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(33, 1, {(stai_ptr) conv2d_33_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_33 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_34_pad_before */
  {
      const ai_ptr conv2d_34_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[1] + 4608);
    ai_ptr conv2d_34_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[1] + 13824);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(34, 1, {(stai_ptr) conv2d_34_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(conv2d_34_pad_before_t_in_0_ptr_const_ptr, conv2d_34_pad_before_t_out_0_ptr_ptr, (ai_handle)(conv2d_34_pad_before_v_pad_constant_value_const_s8), conv2d_34_pad_before_t_in_0_fmt_bitsize_const_s16, conv2d_34_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(1152), (ai_i32)(1440), (ai_i32)(1440), (ai_i32)(144), (ai_i32)(144));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(34, 1, {(stai_ptr) conv2d_34_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END conv2d_34_pad_before */
  /* LITE_KERNEL_SECTION BEGIN conv2d_34 */
  {
      const ai_i8* conv2d_34_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[1] + 13824);
    const ai_i8* conv2d_34_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 44192);
    const ai_i32* conv2d_34_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 45488);
    ai_i8* conv2d_34_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[1] + 4608);
    ai_i16* conv2d_34_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(34, 1, {(stai_ptr) conv2d_34_t_in_0_ptr_const_s8});
    
  forward_lite_dw_3x3_sssa8_ch(conv2d_34_t_in_0_ptr_const_s8, conv2d_34_t_in_0_shape_w_const_u16, conv2d_34_t_in_0_shape_h_const_u16, conv2d_34_t_in_0_shape_ch_const_u16, conv2d_34_t_weight_0_ptr_const_s8, conv2d_34_l_stride_1_const_u16, conv2d_34_l_stride_0_const_u16, conv2d_34_t_weight_1_ptr_const_s32, conv2d_34_t_in_0_fmt_zero_const_s8, conv2d_34_t_out_0_fmt_zero_const_s8, conv2d_34_t_in_0_fmt_scale_const_f32, conv2d_34_t_out_0_fmt_scale_const_f32, conv2d_34_t_weight_0_fmt_scale_const_f32, conv2d_34_t_out_0_ptr_s8, conv2d_34_t_out_0_shape_w_const_u16, conv2d_34_t_out_0_shape_h_const_u16, 0, 5329, conv2d_34_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(34, 1, {(stai_ptr) conv2d_34_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_34 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_35 */
  {
      const ai_i8* conv2d_35_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[1] + 4608);
    const ai_i8* conv2d_35_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 46064);
    const ai_i32* conv2d_35_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 49520);
    ai_i8* conv2d_35_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[1] + 0);
    ai_i16* conv2d_35_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(35, 1, {(stai_ptr) conv2d_35_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_35_t_in_0_ptr_const_s8, conv2d_35_t_in_0_shape_w_const_u16, conv2d_35_t_in_0_shape_h_const_u16, conv2d_35_l_stride_1_const_u16, conv2d_35_l_stride_0_const_u16, conv2d_35_t_in_0_shape_ch_const_u16, conv2d_35_t_weight_0_ptr_const_s8, conv2d_35_t_out_0_shape_ch_const_u16, conv2d_35_t_weight_1_ptr_const_s32, conv2d_35_t_in_0_fmt_zero_const_s8, conv2d_35_t_out_0_fmt_zero_const_s8, conv2d_35_t_in_0_fmt_scale_const_f32, conv2d_35_t_out_0_fmt_scale_const_f32, conv2d_35_t_weight_0_fmt_scale_const_f32, conv2d_35_l_out_ch_format_const_layer_format_type, conv2d_35_t_out_0_ptr_s8, 1, 816, conv2d_35_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(35, 1, {(stai_ptr) conv2d_35_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_35 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_36 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_36(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_36 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_37 */
  {
      const ai_i8* conv2d_37_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[1] + 1536);
    const ai_i8* conv2d_37_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 49616);
    const ai_i32* conv2d_37_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 53072);
    ai_i8* conv2d_37_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[1] + 3072);
    ai_i16* conv2d_37_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(37, 1, {(stai_ptr) conv2d_37_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_37_t_in_0_ptr_const_s8, conv2d_37_t_in_0_shape_w_const_u16, conv2d_37_t_in_0_shape_h_const_u16, conv2d_37_l_stride_1_const_u16, conv2d_37_l_stride_0_const_u16, conv2d_37_t_in_0_shape_ch_const_u16, conv2d_37_t_weight_0_ptr_const_s8, conv2d_37_t_out_0_shape_ch_const_u16, conv2d_37_t_weight_1_ptr_const_s32, conv2d_37_t_in_0_fmt_zero_const_s8, conv2d_37_t_out_0_fmt_zero_const_s8, conv2d_37_t_in_0_fmt_scale_const_f32, conv2d_37_t_out_0_fmt_scale_const_f32, conv2d_37_t_weight_0_fmt_scale_const_f32, conv2d_37_l_out_ch_format_const_layer_format_type, conv2d_37_t_out_0_ptr_s8, 1, 1536, conv2d_37_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(37, 1, {(stai_ptr) conv2d_37_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_37 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_38_pad_before */
  {
      const ai_ptr conv2d_38_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[1] + 3072);
    ai_ptr conv2d_38_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[1] + 12288);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(38, 1, {(stai_ptr) conv2d_38_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(conv2d_38_pad_before_t_in_0_ptr_const_ptr, conv2d_38_pad_before_t_out_0_ptr_ptr, (ai_handle)(conv2d_38_pad_before_v_pad_constant_value_const_s8), conv2d_38_pad_before_t_in_0_fmt_bitsize_const_s16, conv2d_38_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(1152), (ai_i32)(1440), (ai_i32)(1440), (ai_i32)(144), (ai_i32)(144));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(38, 1, {(stai_ptr) conv2d_38_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END conv2d_38_pad_before */
  /* LITE_KERNEL_SECTION BEGIN conv2d_38 */
  {
      const ai_i8* conv2d_38_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[1] + 12288);
    const ai_i8* conv2d_38_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 53648);
    const ai_i32* conv2d_38_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 54944);
    ai_i8* conv2d_38_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[1] + 0);
    ai_i16* conv2d_38_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(38, 1, {(stai_ptr) conv2d_38_t_in_0_ptr_const_s8});
    
  forward_lite_dw_3x3_sssa8_ch(conv2d_38_t_in_0_ptr_const_s8, conv2d_38_t_in_0_shape_w_const_u16, conv2d_38_t_in_0_shape_h_const_u16, conv2d_38_t_in_0_shape_ch_const_u16, conv2d_38_t_weight_0_ptr_const_s8, conv2d_38_l_stride_1_const_u16, conv2d_38_l_stride_0_const_u16, conv2d_38_t_weight_1_ptr_const_s32, conv2d_38_t_in_0_fmt_zero_const_s8, conv2d_38_t_out_0_fmt_zero_const_s8, conv2d_38_t_in_0_fmt_scale_const_f32, conv2d_38_t_out_0_fmt_scale_const_f32, conv2d_38_t_weight_0_fmt_scale_const_f32, conv2d_38_t_out_0_ptr_s8, conv2d_38_t_out_0_shape_w_const_u16, conv2d_38_t_out_0_shape_h_const_u16, 0, 5329, conv2d_38_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(38, 1, {(stai_ptr) conv2d_38_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_38 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_39 */
  {
      const ai_i8* conv2d_39_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[1] + 0);
    const ai_i8* conv2d_39_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 55520);
    const ai_i32* conv2d_39_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 60128);
    ai_i8* conv2d_39_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[1] + 9216);
    ai_i16* conv2d_39_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(39, 1, {(stai_ptr) conv2d_39_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_39_t_in_0_ptr_const_s8, conv2d_39_t_in_0_shape_w_const_u16, conv2d_39_t_in_0_shape_h_const_u16, conv2d_39_l_stride_1_const_u16, conv2d_39_l_stride_0_const_u16, conv2d_39_t_in_0_shape_ch_const_u16, conv2d_39_t_weight_0_ptr_const_s8, conv2d_39_t_out_0_shape_ch_const_u16, conv2d_39_t_weight_1_ptr_const_s32, conv2d_39_t_in_0_fmt_zero_const_s8, conv2d_39_t_out_0_fmt_zero_const_s8, conv2d_39_t_in_0_fmt_scale_const_f32, conv2d_39_t_out_0_fmt_scale_const_f32, conv2d_39_t_weight_0_fmt_scale_const_f32, conv2d_39_l_out_ch_format_const_layer_format_type, conv2d_39_t_out_0_ptr_s8, 1, 896, conv2d_39_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(39, 1, {(stai_ptr) conv2d_39_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_39 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_40 */
  {
      const ai_i8* conv2d_40_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[1] + 9216);
    const ai_i8* conv2d_40_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 60256);
    const ai_i32* conv2d_40_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 66400);
    ai_i8* conv2d_40_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[1] + 11264);
    ai_i16* conv2d_40_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(40, 1, {(stai_ptr) conv2d_40_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_40_t_in_0_ptr_const_s8, conv2d_40_t_in_0_shape_w_const_u16, conv2d_40_t_in_0_shape_h_const_u16, conv2d_40_l_stride_1_const_u16, conv2d_40_l_stride_0_const_u16, conv2d_40_t_in_0_shape_ch_const_u16, conv2d_40_t_weight_0_ptr_const_s8, conv2d_40_t_out_0_shape_ch_const_u16, conv2d_40_t_weight_1_ptr_const_s32, conv2d_40_t_in_0_fmt_zero_const_s8, conv2d_40_t_out_0_fmt_zero_const_s8, conv2d_40_t_in_0_fmt_scale_const_f32, conv2d_40_t_out_0_fmt_scale_const_f32, conv2d_40_t_weight_0_fmt_scale_const_f32, conv2d_40_l_out_ch_format_const_layer_format_type, conv2d_40_t_out_0_ptr_s8, 1, 2048, conv2d_40_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(40, 1, {(stai_ptr) conv2d_40_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_40 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_41_pad_before */
  {
      const ai_ptr conv2d_41_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[1] + 11264);
    ai_ptr conv2d_41_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[1] + 23552);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(41, 1, {(stai_ptr) conv2d_41_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(conv2d_41_pad_before_t_in_0_ptr_const_ptr, conv2d_41_pad_before_t_out_0_ptr_ptr, (ai_handle)(conv2d_41_pad_before_v_pad_constant_value_const_s8), conv2d_41_pad_before_t_in_0_fmt_bitsize_const_s16, conv2d_41_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(1536), (ai_i32)(1920), (ai_i32)(1920), (ai_i32)(192), (ai_i32)(192));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(41, 1, {(stai_ptr) conv2d_41_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END conv2d_41_pad_before */
  /* LITE_KERNEL_SECTION BEGIN conv2d_41 */
  {
      const ai_i8* conv2d_41_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[1] + 23552);
    const ai_i8* conv2d_41_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 67168);
    const ai_i32* conv2d_41_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 68896);
    ai_i8* conv2d_41_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[1] + 11264);
    ai_i16* conv2d_41_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(41, 1, {(stai_ptr) conv2d_41_t_in_0_ptr_const_s8});
    
  forward_lite_dw_3x3_sssa8_ch(conv2d_41_t_in_0_ptr_const_s8, conv2d_41_t_in_0_shape_w_const_u16, conv2d_41_t_in_0_shape_h_const_u16, conv2d_41_t_in_0_shape_ch_const_u16, conv2d_41_t_weight_0_ptr_const_s8, conv2d_41_l_stride_1_const_u16, conv2d_41_l_stride_0_const_u16, conv2d_41_t_weight_1_ptr_const_s32, conv2d_41_t_in_0_fmt_zero_const_s8, conv2d_41_t_out_0_fmt_zero_const_s8, conv2d_41_t_in_0_fmt_scale_const_f32, conv2d_41_t_out_0_fmt_scale_const_f32, conv2d_41_t_weight_0_fmt_scale_const_f32, conv2d_41_t_out_0_ptr_s8, conv2d_41_t_out_0_shape_w_const_u16, conv2d_41_t_out_0_shape_h_const_u16, 0, 7105, conv2d_41_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(41, 1, {(stai_ptr) conv2d_41_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_41 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_42 */
  {
      const ai_i8* conv2d_42_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[1] + 11264);
    const ai_i8* conv2d_42_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 69664);
    const ai_i32* conv2d_42_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 75808);
    ai_i8* conv2d_42_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[1] + 0);
    ai_i16* conv2d_42_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(42, 1, {(stai_ptr) conv2d_42_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_42_t_in_0_ptr_const_s8, conv2d_42_t_in_0_shape_w_const_u16, conv2d_42_t_in_0_shape_h_const_u16, conv2d_42_l_stride_1_const_u16, conv2d_42_l_stride_0_const_u16, conv2d_42_t_in_0_shape_ch_const_u16, conv2d_42_t_weight_0_ptr_const_s8, conv2d_42_t_out_0_shape_ch_const_u16, conv2d_42_t_weight_1_ptr_const_s32, conv2d_42_t_in_0_fmt_zero_const_s8, conv2d_42_t_out_0_fmt_zero_const_s8, conv2d_42_t_in_0_fmt_scale_const_f32, conv2d_42_t_out_0_fmt_scale_const_f32, conv2d_42_t_weight_0_fmt_scale_const_f32, conv2d_42_l_out_ch_format_const_layer_format_type, conv2d_42_t_out_0_ptr_s8, 1, 1088, conv2d_42_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(42, 1, {(stai_ptr) conv2d_42_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_42 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_43 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_43(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_43 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_44 */
  {
      const ai_i8* conv2d_44_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[1] + 2048);
    const ai_i8* conv2d_44_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 75936);
    const ai_i32* conv2d_44_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 82080);
    ai_i8* conv2d_44_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[1] + 4096);
    ai_i16* conv2d_44_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(44, 1, {(stai_ptr) conv2d_44_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_44_t_in_0_ptr_const_s8, conv2d_44_t_in_0_shape_w_const_u16, conv2d_44_t_in_0_shape_h_const_u16, conv2d_44_l_stride_1_const_u16, conv2d_44_l_stride_0_const_u16, conv2d_44_t_in_0_shape_ch_const_u16, conv2d_44_t_weight_0_ptr_const_s8, conv2d_44_t_out_0_shape_ch_const_u16, conv2d_44_t_weight_1_ptr_const_s32, conv2d_44_t_in_0_fmt_zero_const_s8, conv2d_44_t_out_0_fmt_zero_const_s8, conv2d_44_t_in_0_fmt_scale_const_f32, conv2d_44_t_out_0_fmt_scale_const_f32, conv2d_44_t_weight_0_fmt_scale_const_f32, conv2d_44_l_out_ch_format_const_layer_format_type, conv2d_44_t_out_0_ptr_s8, 1, 2048, conv2d_44_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(44, 1, {(stai_ptr) conv2d_44_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_44 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_45_pad_before */
  {
      const ai_ptr conv2d_45_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[1] + 4096);
    ai_ptr conv2d_45_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[1] + 16384);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(45, 1, {(stai_ptr) conv2d_45_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(conv2d_45_pad_before_t_in_0_ptr_const_ptr, conv2d_45_pad_before_t_out_0_ptr_ptr, (ai_handle)(conv2d_45_pad_before_v_pad_constant_value_const_s8), conv2d_45_pad_before_t_in_0_fmt_bitsize_const_s16, conv2d_45_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(1536), (ai_i32)(1920), (ai_i32)(1920), (ai_i32)(192), (ai_i32)(192));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(45, 1, {(stai_ptr) conv2d_45_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END conv2d_45_pad_before */
  /* LITE_KERNEL_SECTION BEGIN conv2d_45 */
  {
      const ai_i8* conv2d_45_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[1] + 16384);
    const ai_i8* conv2d_45_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 82848);
    const ai_i32* conv2d_45_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 84576);
    ai_i8* conv2d_45_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[1] + 4096);
    ai_i16* conv2d_45_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(45, 1, {(stai_ptr) conv2d_45_t_in_0_ptr_const_s8});
    
  forward_lite_dw_3x3_sssa8_ch(conv2d_45_t_in_0_ptr_const_s8, conv2d_45_t_in_0_shape_w_const_u16, conv2d_45_t_in_0_shape_h_const_u16, conv2d_45_t_in_0_shape_ch_const_u16, conv2d_45_t_weight_0_ptr_const_s8, conv2d_45_l_stride_1_const_u16, conv2d_45_l_stride_0_const_u16, conv2d_45_t_weight_1_ptr_const_s32, conv2d_45_t_in_0_fmt_zero_const_s8, conv2d_45_t_out_0_fmt_zero_const_s8, conv2d_45_t_in_0_fmt_scale_const_f32, conv2d_45_t_out_0_fmt_scale_const_f32, conv2d_45_t_weight_0_fmt_scale_const_f32, conv2d_45_t_out_0_ptr_s8, conv2d_45_t_out_0_shape_w_const_u16, conv2d_45_t_out_0_shape_h_const_u16, 0, 7105, conv2d_45_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(45, 1, {(stai_ptr) conv2d_45_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_45 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_46 */
  {
      const ai_i8* conv2d_46_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[1] + 4096);
    const ai_i8* conv2d_46_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 85344);
    const ai_i32* conv2d_46_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 91488);
    ai_i8* conv2d_46_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[1] + 0);
    ai_i16* conv2d_46_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(46, 1, {(stai_ptr) conv2d_46_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_46_t_in_0_ptr_const_s8, conv2d_46_t_in_0_shape_w_const_u16, conv2d_46_t_in_0_shape_h_const_u16, conv2d_46_l_stride_1_const_u16, conv2d_46_l_stride_0_const_u16, conv2d_46_t_in_0_shape_ch_const_u16, conv2d_46_t_weight_0_ptr_const_s8, conv2d_46_t_out_0_shape_ch_const_u16, conv2d_46_t_weight_1_ptr_const_s32, conv2d_46_t_in_0_fmt_zero_const_s8, conv2d_46_t_out_0_fmt_zero_const_s8, conv2d_46_t_in_0_fmt_scale_const_f32, conv2d_46_t_out_0_fmt_scale_const_f32, conv2d_46_t_weight_0_fmt_scale_const_f32, conv2d_46_l_out_ch_format_const_layer_format_type, conv2d_46_t_out_0_ptr_s8, 1, 1088, conv2d_46_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(46, 1, {(stai_ptr) conv2d_46_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_46 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_47 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_47(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_47 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_48 */
  {
      const ai_i8* conv2d_48_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[1] + 4096);
    const ai_i8* conv2d_48_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 91616);
    const ai_i32* conv2d_48_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 97760);
    ai_i8* conv2d_48_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[1] + 6144);
    ai_i16* conv2d_48_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(48, 1, {(stai_ptr) conv2d_48_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_48_t_in_0_ptr_const_s8, conv2d_48_t_in_0_shape_w_const_u16, conv2d_48_t_in_0_shape_h_const_u16, conv2d_48_l_stride_1_const_u16, conv2d_48_l_stride_0_const_u16, conv2d_48_t_in_0_shape_ch_const_u16, conv2d_48_t_weight_0_ptr_const_s8, conv2d_48_t_out_0_shape_ch_const_u16, conv2d_48_t_weight_1_ptr_const_s32, conv2d_48_t_in_0_fmt_zero_const_s8, conv2d_48_t_out_0_fmt_zero_const_s8, conv2d_48_t_in_0_fmt_scale_const_f32, conv2d_48_t_out_0_fmt_scale_const_f32, conv2d_48_t_weight_0_fmt_scale_const_f32, conv2d_48_l_out_ch_format_const_layer_format_type, conv2d_48_t_out_0_ptr_s8, 1, 2048, conv2d_48_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(48, 1, {(stai_ptr) conv2d_48_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_48 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_49_pad_before */
  {
      const ai_ptr conv2d_49_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[1] + 6144);
    ai_ptr conv2d_49_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[1] + 18432);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(49, 1, {(stai_ptr) conv2d_49_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(conv2d_49_pad_before_t_in_0_ptr_const_ptr, conv2d_49_pad_before_t_out_0_ptr_ptr, (ai_handle)(conv2d_49_pad_before_v_pad_constant_value_const_s8), conv2d_49_pad_before_t_in_0_fmt_bitsize_const_s16, conv2d_49_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(1536), (ai_i32)(0), (ai_i32)(3840), (ai_i32)(0), (ai_i32)(384));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(49, 1, {(stai_ptr) conv2d_49_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END conv2d_49_pad_before */
  /* LITE_KERNEL_SECTION BEGIN conv2d_49 */
  {
      const ai_i8* conv2d_49_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[1] + 18432);
    const ai_i8* conv2d_49_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 98528);
    const ai_i32* conv2d_49_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 100256);
    ai_i8* conv2d_49_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[1] + 0);
    ai_i16* conv2d_49_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(49, 1, {(stai_ptr) conv2d_49_t_in_0_ptr_const_s8});
    
  forward_lite_dw_3x3_sssa8_ch(conv2d_49_t_in_0_ptr_const_s8, conv2d_49_t_in_0_shape_w_const_u16, conv2d_49_t_in_0_shape_h_const_u16, conv2d_49_t_in_0_shape_ch_const_u16, conv2d_49_t_weight_0_ptr_const_s8, conv2d_49_l_stride_1_const_u16, conv2d_49_l_stride_0_const_u16, conv2d_49_t_weight_1_ptr_const_s32, conv2d_49_t_in_0_fmt_zero_const_s8, conv2d_49_t_out_0_fmt_zero_const_s8, conv2d_49_t_in_0_fmt_scale_const_f32, conv2d_49_t_out_0_fmt_scale_const_f32, conv2d_49_t_weight_0_fmt_scale_const_f32, conv2d_49_t_out_0_ptr_s8, conv2d_49_t_out_0_shape_w_const_u16, conv2d_49_t_out_0_shape_h_const_u16, 0, 7105, conv2d_49_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(49, 1, {(stai_ptr) conv2d_49_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_49 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_50 */
  {
      const ai_i8* conv2d_50_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[1] + 0);
    const ai_i8* conv2d_50_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 101024);
    const ai_i32* conv2d_50_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 111776);
    ai_i8* conv2d_50_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[1] + 3072);
    ai_i16* conv2d_50_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(50, 1, {(stai_ptr) conv2d_50_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_50_t_in_0_ptr_const_s8, conv2d_50_t_in_0_shape_w_const_u16, conv2d_50_t_in_0_shape_h_const_u16, conv2d_50_l_stride_1_const_u16, conv2d_50_l_stride_0_const_u16, conv2d_50_t_in_0_shape_ch_const_u16, conv2d_50_t_weight_0_ptr_const_s8, conv2d_50_t_out_0_shape_ch_const_u16, conv2d_50_t_weight_1_ptr_const_s32, conv2d_50_t_in_0_fmt_zero_const_s8, conv2d_50_t_out_0_fmt_zero_const_s8, conv2d_50_t_in_0_fmt_scale_const_f32, conv2d_50_t_out_0_fmt_scale_const_f32, conv2d_50_t_weight_0_fmt_scale_const_f32, conv2d_50_l_out_ch_format_const_layer_format_type, conv2d_50_t_out_0_ptr_s8, 1, 1328, conv2d_50_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(50, 1, {(stai_ptr) conv2d_50_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_50 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_51 */
  {
      const ai_i8* conv2d_51_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[1] + 3072);
    const ai_i8* conv2d_51_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 112000);
    const ai_i32* conv2d_51_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 130816);
    ai_i8* conv2d_51_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[1] + 3968);
    ai_i16* conv2d_51_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(51, 1, {(stai_ptr) conv2d_51_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_51_t_in_0_ptr_const_s8, conv2d_51_t_in_0_shape_w_const_u16, conv2d_51_t_in_0_shape_h_const_u16, conv2d_51_l_stride_1_const_u16, conv2d_51_l_stride_0_const_u16, conv2d_51_t_in_0_shape_ch_const_u16, conv2d_51_t_weight_0_ptr_const_s8, conv2d_51_t_out_0_shape_ch_const_u16, conv2d_51_t_weight_1_ptr_const_s32, conv2d_51_t_in_0_fmt_zero_const_s8, conv2d_51_t_out_0_fmt_zero_const_s8, conv2d_51_t_in_0_fmt_scale_const_f32, conv2d_51_t_out_0_fmt_scale_const_f32, conv2d_51_t_weight_0_fmt_scale_const_f32, conv2d_51_l_out_ch_format_const_layer_format_type, conv2d_51_t_out_0_ptr_s8, 1, 3584, conv2d_51_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(51, 1, {(stai_ptr) conv2d_51_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_51 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_52_pad_before */
  {
      const ai_ptr conv2d_52_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[1] + 3968);
    ai_ptr conv2d_52_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[1] + 9344);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(52, 1, {(stai_ptr) conv2d_52_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(conv2d_52_pad_before_t_in_0_ptr_const_ptr, conv2d_52_pad_before_t_out_0_ptr_ptr, (ai_handle)(conv2d_52_pad_before_v_pad_constant_value_const_s8), conv2d_52_pad_before_t_in_0_fmt_bitsize_const_s16, conv2d_52_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(1344), (ai_i32)(2016), (ai_i32)(2016), (ai_i32)(336), (ai_i32)(336));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(52, 1, {(stai_ptr) conv2d_52_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END conv2d_52_pad_before */
  /* LITE_KERNEL_SECTION BEGIN conv2d_52 */
  {
      const ai_i8* conv2d_52_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[1] + 9344);
    const ai_i8* conv2d_52_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 132160);
    const ai_i32* conv2d_52_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 135184);
    ai_i8* conv2d_52_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[1] + 3968);
    ai_i16* conv2d_52_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(52, 1, {(stai_ptr) conv2d_52_t_in_0_ptr_const_s8});
    
  forward_lite_dw_3x3_sssa8_ch(conv2d_52_t_in_0_ptr_const_s8, conv2d_52_t_in_0_shape_w_const_u16, conv2d_52_t_in_0_shape_h_const_u16, conv2d_52_t_in_0_shape_ch_const_u16, conv2d_52_t_weight_0_ptr_const_s8, conv2d_52_l_stride_1_const_u16, conv2d_52_l_stride_0_const_u16, conv2d_52_t_weight_1_ptr_const_s32, conv2d_52_t_in_0_fmt_zero_const_s8, conv2d_52_t_out_0_fmt_zero_const_s8, conv2d_52_t_in_0_fmt_scale_const_f32, conv2d_52_t_out_0_fmt_scale_const_f32, conv2d_52_t_weight_0_fmt_scale_const_f32, conv2d_52_t_out_0_ptr_s8, conv2d_52_t_out_0_shape_w_const_u16, conv2d_52_t_out_0_shape_h_const_u16, 0, 12433, conv2d_52_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(52, 1, {(stai_ptr) conv2d_52_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_52 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_53 */
  {
      const ai_i8* conv2d_53_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[1] + 3968);
    const ai_i8* conv2d_53_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 136528);
    const ai_i32* conv2d_53_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 155344);
    ai_i8* conv2d_53_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[1] + 0);
    ai_i16* conv2d_53_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(53, 1, {(stai_ptr) conv2d_53_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_53_t_in_0_ptr_const_s8, conv2d_53_t_in_0_shape_w_const_u16, conv2d_53_t_in_0_shape_h_const_u16, conv2d_53_l_stride_1_const_u16, conv2d_53_l_stride_0_const_u16, conv2d_53_t_in_0_shape_ch_const_u16, conv2d_53_t_weight_0_ptr_const_s8, conv2d_53_t_out_0_shape_ch_const_u16, conv2d_53_t_weight_1_ptr_const_s32, conv2d_53_t_in_0_fmt_zero_const_s8, conv2d_53_t_out_0_fmt_zero_const_s8, conv2d_53_t_in_0_fmt_scale_const_f32, conv2d_53_t_out_0_fmt_scale_const_f32, conv2d_53_t_weight_0_fmt_scale_const_f32, conv2d_53_l_out_ch_format_const_layer_format_type, conv2d_53_t_out_0_ptr_s8, 1, 1904, conv2d_53_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(53, 1, {(stai_ptr) conv2d_53_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_53 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_54 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_54(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_54 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_55 */
  {
      const ai_i8* conv2d_55_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[1] + 896);
    const ai_i8* conv2d_55_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 155568);
    const ai_i32* conv2d_55_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 174384);
    ai_i8* conv2d_55_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[1] + 1792);
    ai_i16* conv2d_55_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(55, 1, {(stai_ptr) conv2d_55_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_55_t_in_0_ptr_const_s8, conv2d_55_t_in_0_shape_w_const_u16, conv2d_55_t_in_0_shape_h_const_u16, conv2d_55_l_stride_1_const_u16, conv2d_55_l_stride_0_const_u16, conv2d_55_t_in_0_shape_ch_const_u16, conv2d_55_t_weight_0_ptr_const_s8, conv2d_55_t_out_0_shape_ch_const_u16, conv2d_55_t_weight_1_ptr_const_s32, conv2d_55_t_in_0_fmt_zero_const_s8, conv2d_55_t_out_0_fmt_zero_const_s8, conv2d_55_t_in_0_fmt_scale_const_f32, conv2d_55_t_out_0_fmt_scale_const_f32, conv2d_55_t_weight_0_fmt_scale_const_f32, conv2d_55_l_out_ch_format_const_layer_format_type, conv2d_55_t_out_0_ptr_s8, 1, 3584, conv2d_55_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(55, 1, {(stai_ptr) conv2d_55_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_55 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_56_pad_before */
  {
      const ai_ptr conv2d_56_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[1] + 1792);
    ai_ptr conv2d_56_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[1] + 7168);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(56, 1, {(stai_ptr) conv2d_56_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(conv2d_56_pad_before_t_in_0_ptr_const_ptr, conv2d_56_pad_before_t_out_0_ptr_ptr, (ai_handle)(conv2d_56_pad_before_v_pad_constant_value_const_s8), conv2d_56_pad_before_t_in_0_fmt_bitsize_const_s16, conv2d_56_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(1344), (ai_i32)(2016), (ai_i32)(2016), (ai_i32)(336), (ai_i32)(336));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(56, 1, {(stai_ptr) conv2d_56_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END conv2d_56_pad_before */
  /* LITE_KERNEL_SECTION BEGIN conv2d_56 */
  {
      const ai_i8* conv2d_56_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[1] + 7168);
    const ai_i8* conv2d_56_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 175728);
    const ai_i32* conv2d_56_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 178752);
    ai_i8* conv2d_56_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[1] + 1792);
    ai_i16* conv2d_56_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(56, 1, {(stai_ptr) conv2d_56_t_in_0_ptr_const_s8});
    
  forward_lite_dw_3x3_sssa8_ch(conv2d_56_t_in_0_ptr_const_s8, conv2d_56_t_in_0_shape_w_const_u16, conv2d_56_t_in_0_shape_h_const_u16, conv2d_56_t_in_0_shape_ch_const_u16, conv2d_56_t_weight_0_ptr_const_s8, conv2d_56_l_stride_1_const_u16, conv2d_56_l_stride_0_const_u16, conv2d_56_t_weight_1_ptr_const_s32, conv2d_56_t_in_0_fmt_zero_const_s8, conv2d_56_t_out_0_fmt_zero_const_s8, conv2d_56_t_in_0_fmt_scale_const_f32, conv2d_56_t_out_0_fmt_scale_const_f32, conv2d_56_t_weight_0_fmt_scale_const_f32, conv2d_56_t_out_0_ptr_s8, conv2d_56_t_out_0_shape_w_const_u16, conv2d_56_t_out_0_shape_h_const_u16, 0, 12433, conv2d_56_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(56, 1, {(stai_ptr) conv2d_56_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_56 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_57 */
  {
      const ai_i8* conv2d_57_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[1] + 1792);
    const ai_i8* conv2d_57_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 180096);
    const ai_i32* conv2d_57_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 198912);
    ai_i8* conv2d_57_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[1] + 0);
    ai_i16* conv2d_57_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(57, 1, {(stai_ptr) conv2d_57_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_57_t_in_0_ptr_const_s8, conv2d_57_t_in_0_shape_w_const_u16, conv2d_57_t_in_0_shape_h_const_u16, conv2d_57_l_stride_1_const_u16, conv2d_57_l_stride_0_const_u16, conv2d_57_t_in_0_shape_ch_const_u16, conv2d_57_t_weight_0_ptr_const_s8, conv2d_57_t_out_0_shape_ch_const_u16, conv2d_57_t_weight_1_ptr_const_s32, conv2d_57_t_in_0_fmt_zero_const_s8, conv2d_57_t_out_0_fmt_zero_const_s8, conv2d_57_t_in_0_fmt_scale_const_f32, conv2d_57_t_out_0_fmt_scale_const_f32, conv2d_57_t_weight_0_fmt_scale_const_f32, conv2d_57_l_out_ch_format_const_layer_format_type, conv2d_57_t_out_0_ptr_s8, 1, 1904, conv2d_57_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(57, 1, {(stai_ptr) conv2d_57_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_57 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_58 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_58(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_58 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_59 */
  {
      const ai_i8* conv2d_59_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[1] + 1792);
    const ai_i8* conv2d_59_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 199136);
    const ai_i32* conv2d_59_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 217952);
    ai_i8* conv2d_59_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[1] + 2688);
    ai_i16* conv2d_59_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(59, 1, {(stai_ptr) conv2d_59_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_59_t_in_0_ptr_const_s8, conv2d_59_t_in_0_shape_w_const_u16, conv2d_59_t_in_0_shape_h_const_u16, conv2d_59_l_stride_1_const_u16, conv2d_59_l_stride_0_const_u16, conv2d_59_t_in_0_shape_ch_const_u16, conv2d_59_t_weight_0_ptr_const_s8, conv2d_59_t_out_0_shape_ch_const_u16, conv2d_59_t_weight_1_ptr_const_s32, conv2d_59_t_in_0_fmt_zero_const_s8, conv2d_59_t_out_0_fmt_zero_const_s8, conv2d_59_t_in_0_fmt_scale_const_f32, conv2d_59_t_out_0_fmt_scale_const_f32, conv2d_59_t_weight_0_fmt_scale_const_f32, conv2d_59_l_out_ch_format_const_layer_format_type, conv2d_59_t_out_0_ptr_s8, 1, 3584, conv2d_59_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(59, 1, {(stai_ptr) conv2d_59_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_59 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_60_pad_before */
  {
      const ai_ptr conv2d_60_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[1] + 2688);
    ai_ptr conv2d_60_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[1] + 8064);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(60, 1, {(stai_ptr) conv2d_60_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(conv2d_60_pad_before_t_in_0_ptr_const_ptr, conv2d_60_pad_before_t_out_0_ptr_ptr, (ai_handle)(conv2d_60_pad_before_v_pad_constant_value_const_s8), conv2d_60_pad_before_t_in_0_fmt_bitsize_const_s16, conv2d_60_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(1344), (ai_i32)(2016), (ai_i32)(2016), (ai_i32)(336), (ai_i32)(336));
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(60, 1, {(stai_ptr) conv2d_60_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END conv2d_60_pad_before */
  /* LITE_KERNEL_SECTION BEGIN conv2d_60 */
  {
      const ai_i8* conv2d_60_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[1] + 8064);
    const ai_i8* conv2d_60_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 219296);
    const ai_i32* conv2d_60_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 222320);
    ai_i8* conv2d_60_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[1] + 0);
    ai_i16* conv2d_60_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(60, 1, {(stai_ptr) conv2d_60_t_in_0_ptr_const_s8});
    
  forward_lite_dw_3x3_sssa8_ch(conv2d_60_t_in_0_ptr_const_s8, conv2d_60_t_in_0_shape_w_const_u16, conv2d_60_t_in_0_shape_h_const_u16, conv2d_60_t_in_0_shape_ch_const_u16, conv2d_60_t_weight_0_ptr_const_s8, conv2d_60_l_stride_1_const_u16, conv2d_60_l_stride_0_const_u16, conv2d_60_t_weight_1_ptr_const_s32, conv2d_60_t_in_0_fmt_zero_const_s8, conv2d_60_t_out_0_fmt_zero_const_s8, conv2d_60_t_in_0_fmt_scale_const_f32, conv2d_60_t_out_0_fmt_scale_const_f32, conv2d_60_t_weight_0_fmt_scale_const_f32, conv2d_60_t_out_0_ptr_s8, conv2d_60_t_out_0_shape_w_const_u16, conv2d_60_t_out_0_shape_h_const_u16, 0, 12433, conv2d_60_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(60, 1, {(stai_ptr) conv2d_60_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_60 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_61 */
  {
      const ai_i8* conv2d_61_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[1] + 0);
    const ai_i8* conv2d_61_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 223664);
    const ai_i32* conv2d_61_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 261296);
    ai_i8* conv2d_61_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[1] + 5376);
    ai_i16* conv2d_61_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(61, 1, {(stai_ptr) conv2d_61_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_61_t_in_0_ptr_const_s8, conv2d_61_t_in_0_shape_w_const_u16, conv2d_61_t_in_0_shape_h_const_u16, conv2d_61_l_stride_1_const_u16, conv2d_61_l_stride_0_const_u16, conv2d_61_t_in_0_shape_ch_const_u16, conv2d_61_t_weight_0_ptr_const_s8, conv2d_61_t_out_0_shape_ch_const_u16, conv2d_61_t_weight_1_ptr_const_s32, conv2d_61_t_in_0_fmt_zero_const_s8, conv2d_61_t_out_0_fmt_zero_const_s8, conv2d_61_t_in_0_fmt_scale_const_f32, conv2d_61_t_out_0_fmt_scale_const_f32, conv2d_61_t_weight_0_fmt_scale_const_f32, conv2d_61_l_out_ch_format_const_layer_format_type, conv2d_61_t_out_0_ptr_s8, 1, 2464, conv2d_61_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(61, 1, {(stai_ptr) conv2d_61_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_61 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_62 */
  {
      const ai_i8* conv2d_62_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[1] + 5376);
    const ai_i8* conv2d_62_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 261744);
    const ai_i32* conv2d_62_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 405104);
    ai_i8* conv2d_62_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[1] + 7168);
    ai_i16* conv2d_62_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(62, 1, {(stai_ptr) conv2d_62_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_62_t_in_0_ptr_const_s8, conv2d_62_t_in_0_shape_w_const_u16, conv2d_62_t_in_0_shape_h_const_u16, conv2d_62_l_stride_1_const_u16, conv2d_62_l_stride_0_const_u16, conv2d_62_t_in_0_shape_ch_const_u16, conv2d_62_t_weight_0_ptr_const_s8, conv2d_62_t_out_0_shape_ch_const_u16, conv2d_62_t_weight_1_ptr_const_s32, conv2d_62_t_in_0_fmt_zero_const_s8, conv2d_62_t_out_0_fmt_zero_const_s8, conv2d_62_t_in_0_fmt_scale_const_f32, conv2d_62_t_out_0_fmt_scale_const_f32, conv2d_62_t_weight_0_fmt_scale_const_f32, conv2d_62_l_out_ch_format_const_layer_format_type, conv2d_62_t_out_0_ptr_s8, 1, 13248, conv2d_62_t_scratch_0_ptr_s16);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(62, 1, {(stai_ptr) conv2d_62_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_62 */
  /* LITE_KERNEL_SECTION BEGIN pool_63 */
  {
    
  forward_lite_ap_integer_INT8_pool_63(net_ctx);
  }
  /* LITE_KERNEL_SECTION END pool_63 */
  /* LITE_KERNEL_SECTION BEGIN gemm_64 */
  {
    
  forward_lite_dense_integer_SSSA_ch_gemm_64(net_ctx);
  }
  /* LITE_KERNEL_SECTION END gemm_64 */
  /* LITE_KERNEL_SECTION BEGIN nl_65 */
  {
      ai_i8* nl_65_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[1] + 0);
    const ai_i8* nl_65_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[1] + 1280);
    ai_i32* nl_65_t_scratch_0_ptr_s32 = (ai_i32*)(net_ctx->_activations[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(65, 1, {(stai_ptr) nl_65_t_in_0_ptr_const_s8});
    
  forward_lite_nl_softmax_is8os8(nl_65_t_out_0_ptr_s8, nl_65_t_in_0_ptr_const_s8, nl_65_t_in_0_shape_ch_prod_const_u32, 1, 3, 1505727616, 24, -124, nl_65_t_scratch_0_ptr_s32);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(65, 1, {(stai_ptr) nl_65_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END nl_65 */
  /* LITE_KERNEL_SECTION BEGIN conversion_66 */
  {
      const ai_i8* conversion_66_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[1] + 0);
    ai_float* conversion_66_t_out_0_ptr_f32 = (ai_float*)(net_ctx->_outputs[0] + 0);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(66, 1, {(stai_ptr) conversion_66_t_in_0_ptr_const_s8});
    
  forward_lite_node_convert_integer_is8of32(conversion_66_t_in_0_ptr_const_s8, conversion_66_t_out_0_ptr_f32, conversion_66_t_out_0_shape_h_w_ch_d_prod_const_u32, conversion_66_t_in_0_fmt_scale_const_f32, conversion_66_t_in_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(66, 1, {(stai_ptr) conversion_66_t_out_0_ptr_f32});
  }
  /* LITE_KERNEL_SECTION END conversion_66 */
  return net_ctx->_return_code;
}

/*****************************************************************************/
/*  Getters APIs Section  */
STAI_API_ENTRY
stai_size stai_network_get_context_size()
{
  return (stai_size)STAI_NETWORK_CONTEXT_SIZE;
}

#if defined(HAVE_NETWORK_INFO)
STAI_API_ENTRY
stai_return_code stai_network_get_info(
  stai_network* network,
  stai_network_info* info)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, info==NULL, STAI_ERROR_NETWORK_INVALID_INFO, net_ctx->_return_code)

  // Copy of network info struct
  *info = g_network_info;

  return STAI_SUCCESS;
}
#endif


STAI_API_ENTRY
stai_return_code stai_network_get_activations(
  stai_network* network, stai_ptr* activations, stai_size* n_activations)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  _STAI_SET_ERROR(net_ctx, !n_activations, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_activations = STAI_NETWORK_ACTIVATIONS_NUM;
for (stai_size idx=0; activations && (idx<STAI_NETWORK_ACTIVATIONS_NUM); idx++) {
    // get address of the activations buffers
    activations[idx] = net_ctx->_activations[idx];
  }return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_weights(
  stai_network* network, stai_ptr* weights, stai_size* n_weights)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_weights, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_weights = STAI_NETWORK_WEIGHTS_NUM;
for (stai_size idx=0; weights && (idx<STAI_NETWORK_WEIGHTS_NUM); idx++) {
    // get address of the weights buffers
    weights[idx] = net_ctx->_weights[idx];
  }return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_inputs(
  stai_network* network, stai_ptr* inputs, stai_size* n_inputs)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_inputs, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_inputs = STAI_NETWORK_IN_NUM;
  for (stai_size idx=0; inputs && (idx<STAI_NETWORK_IN_NUM); idx++) {
    inputs[idx] = net_ctx->_inputs[idx];
  }
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_outputs(
  stai_network* network, stai_ptr* outputs, stai_size* n_outputs)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_outputs, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_outputs = STAI_NETWORK_OUT_NUM;
  for (stai_size idx=0; outputs && (idx<STAI_NETWORK_OUT_NUM); idx++) {
    outputs[idx] = net_ctx->_outputs[idx];
  }
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_error(
  stai_network* network)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  /* return 1st generated error or STAI_SUCCESS if no errors so far */
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_states(
  stai_network* network, stai_ptr* states, stai_size* n_states)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_states, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  /* get the number of internals states (supporting multi-heap also for internal states) */
  *n_states = STAI_NETWORK_STATES_NUM;

  STAI_UNUSED(states)
return net_ctx->_return_code;
}


/*****************************************************************************/
/*  Setters APIs Section  */

STAI_API_ENTRY
stai_return_code stai_network_set_activations(
  stai_network* network,
  const stai_ptr* activations,
  const stai_size n_activations)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
const uintptr_t _activations_alignment[] = STAI_NETWORK_ACTIVATIONS_ALIGNMENTS;
  STAI_PRINT("  [stai_network_set_activations] network(%p) activations[%d]: %p\n\n", net_ctx, n_activations, activations)
  _STAI_SET_ERROR(net_ctx, !activations,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_activations!=STAI_NETWORK_ACTIVATIONS_NUM,
                  STAI_ERROR_NETWORK_INVALID_ACTIVATIONS_NUM, net_ctx->_return_code)

  for (stai_size idx=0; activations && idx<STAI_NETWORK_ACTIVATIONS_NUM; idx++) {
    STAI_PRINT("  activation[%d]: %p\n", idx, activations[idx])
    _STAI_SET_ERROR(net_ctx, activations[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_ACTIVATIONS_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)activations[idx]) & (_activations_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_activations[idx] = activations[idx];
  }
  net_ctx->_inputs[0] = activations[1] + 24896;

  net_ctx->_outputs[0] = activations[1] + 4;
_stai_network_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_set_weights(
  stai_network* network,
  const stai_ptr* weights,
  const stai_size n_weights)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
const uintptr_t _weights_alignment[] = STAI_NETWORK_WEIGHTS_ALIGNMENTS;
  _STAI_SET_ERROR(net_ctx, !weights,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_weights!=STAI_NETWORK_WEIGHTS_NUM,
                  STAI_ERROR_NETWORK_INVALID_WEIGHTS_NUM, net_ctx->_return_code)
  for (stai_size idx=0; weights && idx<STAI_NETWORK_WEIGHTS_NUM; idx++) {
    STAI_PRINT("  weight[%d]: %p\n", idx, weights[idx])
    _STAI_SET_ERROR(net_ctx, weights[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_WEIGHTS_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)weights[idx]) & (_weights_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_weights[idx] = weights[idx];
  }_stai_network_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_set_inputs(
  stai_network* network,
  const stai_ptr* inputs,
  const stai_size n_inputs)
{
  const uintptr_t _inputs_alignment[] = STAI_NETWORK_IN_ALIGNMENTS;
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !inputs,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_inputs!=STAI_NETWORK_IN_NUM,
                  STAI_ERROR_NETWORK_INVALID_IN_NUM, net_ctx->_return_code)

  for (stai_size idx=0; inputs && idx<STAI_NETWORK_IN_NUM; idx++) {
    STAI_PRINT("  input[%d]: %p\n", idx, inputs[idx])
    _STAI_SET_ERROR(net_ctx, inputs[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_IN_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)inputs[idx]) & (_inputs_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_inputs[idx] = inputs[idx];
  }

  _stai_network_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_set_outputs(
  stai_network* network,
  const stai_ptr* outputs,
  const stai_size n_outputs)
{
  const uintptr_t _outputs_alignment[] = STAI_NETWORK_OUT_ALIGNMENTS;
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !outputs,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_outputs!=STAI_NETWORK_OUT_NUM,
                  STAI_ERROR_NETWORK_INVALID_OUT_NUM, net_ctx->_return_code)

  for (stai_size idx=0; outputs && idx<n_outputs; idx++) {
    STAI_PRINT("  output[%d]: %p\n", idx, outputs[idx])
    _STAI_SET_ERROR(net_ctx, outputs[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_OUT_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)outputs[idx]) & (_outputs_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_outputs[idx] = outputs[idx];
  }

  _stai_network_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_set_states(
  stai_network* network,
  const stai_ptr* states,
  const stai_size n_states)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  STAI_UNUSED(states)
  STAI_UNUSED(n_states)
_stai_network_check(net_ctx);
  return net_ctx->_return_code;
}

STAI_API_ENTRY
stai_return_code stai_network_set_callback(
  stai_network* network, const stai_event_cb cb, void* cb_cookie)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  STAI_PRINT("  set_callback %p cb %p cookie %p\n", net_ctx, cb, cb_cookie)
  // _STAI_SET_ERROR(net_ctx, cb==NULL, STAI_ERROR_NETWORK_INVALID_CALLBACK, net_ctx->_return_code)
  net_ctx->_callback = cb;
  net_ctx->_callback_cookie = cb_cookie;
  return net_ctx->_return_code;
}

#undef _STAI_SET_ERROR
#undef _STAI_CONTEXT_ALIGNMENT
#undef _STAI_CONTEXT_ACQUIRE
#undef _STAI_NETWORK_EVENT_NODE_START_CB
#undef _STAI_NETWORK_EVENT_NODE_STOP_CB
#undef _STAI_NETWORK_MODEL_SIGNATURE
#undef _STAI_NETWORK_DATETIME
#undef _STAI_NETWORK_COMPILE_DATETIME

