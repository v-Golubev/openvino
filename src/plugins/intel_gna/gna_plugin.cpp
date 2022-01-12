// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define NOMINMAX

#include <cstdlib>
#include <vector>
#include <cstring>
#include <list>
#include <algorithm>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <utility>
#include <limits>

#include <ie_common.h>
#include <legacy/graph_tools.hpp>
#include <legacy/net_pass.h>
#include <debug.h>
#include <gna/gna_config.hpp>
#include "gna_plugin_config.hpp"
#include "gna_plugin.hpp"
#include "optimizer/gna_pass_manager.hpp"
#include "layers/gna_layer_type.hpp"
#include "preprocessing.hpp"
#include "frontend/weights_converter.hpp"
#include "frontend/model_quantizer.hpp"
#include "gna_fused_iterator.hpp"
#include "backend/am_intel_dnn.hpp"
#include "memory/gna_allocator.hpp"
#include "memory/gna_memory_state.hpp"
#include "gna_model_serial.hpp"
#include "runtime/gna_float_runtime.hpp"
#include <layers/gna_fake_quantize_layer.hpp>
#include "gna_graph_patterns.hpp"
#include "gna_tensor_tools.hpp"
#include "gna_itt.hpp"

#include <ngraph/pass/manager.hpp>
#include <legacy/convert_function_to_cnn_network.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_opset1_to_legacy.hpp>

#include <transformations/common_optimizations/common_optimizations.hpp>
#include <transformations/control_flow/unroll_tensor_iterator.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/opset_conversions/convert_opset3_to_opset2.hpp>
#include <transformations/opset_conversions/convert_opset2_to_opset1.hpp>
#include "transformations/common_optimizations/concat_reduce_fusion.hpp"
#include <transformations/common_optimizations/fq_mul_fusion.hpp>
#include <transformations/common_optimizations/fq_reshape_fusion.hpp>
#include <transformations/common_optimizations/pull_transpose_through_fq.hpp>
#include <transformations/common_optimizations/relu_fake_quantize_fusion.hpp>
#include <transformations/common_optimizations/add_fake_quantize_fusion.hpp>
#include <transformations/common_optimizations/transpose_sinking.hpp>
#include "transformations/common_optimizations/convert_compression_only_to_legacy.hpp"
#include "transformations/disable_decompression_convert_constant_folding.hpp"
#include <transformations/utils/utils.hpp>

#include "transformations/remove_extra_reshapes.hpp"
#include "transformations/insert_transpose_after_convolution_or_pooling.hpp"
#include "transformations/reorder_activation_and_pooling.hpp"
#include "transformations/swap_input_matmul_gna.hpp"
#include "transformations/convert_matmul_to_pointwise_convolution.hpp"
#include "transformations/split_convolution_with_large_buffer_size.hpp"
#include "transformations/handle_transposes_around_matmul.hpp"
#include "transformations/decompose_2d_convolution.hpp"
#include "transformations/convert_padded_to_valid_convolution.hpp"
#include "transformations/insert_reshape_around_matmul.hpp"
#include "transformations/convert_dwsc_to_scaleshifts.hpp"
#include "transformations/op_conversions/lstm_cell_decomposition.hpp"
#include "transformations/remove_single_input_concat.hpp"
#include "transformations/remove_converts.hpp"
#include "transformations/broadcast_const.hpp"
#include "transformations/op_conversions/convert_mvn1_to_mvn6.hpp"
#include "transformations/decompose_mvn.hpp"
#include "transformations/substitute_softsign.hpp"
#include "transformations/convert_precision.hpp"

#include <ngraph/opsets/opset7.hpp>

#include <gna2-model-api.h>

inline uint32_t ToByteSize(const Gna2DataType type) {
    switch (type) {
    case Gna2DataTypeInt8:
    case Gna2DataTypeUint8:
        return 1;
    case Gna2DataTypeInt16:
    case Gna2DataTypeUint16:
        return 2;
    case Gna2DataTypeInt32:
    case Gna2DataTypeUint32:
        return 4;
    case Gna2DataTypeInt64:
    case Gna2DataTypeUint64:
        return 8;
    default:
        return 0;
    }
}

constexpr uint32_t GNAPluginNS::GNAPlugin::FAKE_REQUEST_CONFIG_ID;

using namespace InferenceEngine;
using namespace std;
using namespace GNAPluginNS;
using namespace InferenceEngine::details;

namespace InferenceEngine {
    template<>
    InferenceEngine::TBlob<gna_compound_bias_t, std::enable_if<true, void> >::~TBlob() { free(); }
}

template <typename T, typename U>
void GNAPlugin::copyInputData(T *dst,
                const U *src,
                uint32_t num_frames,
                uint32_t num_group,
                uint32_t num_vector_elements,
                uint32_t num_vector_stride,
                intel_dnn_orientation_t orientation,
                float scaleFactor) {
    if (!dst || !src) {
        return;
    }
    if (orientation == kDnnInterleavedOrientation) {
        for (uint32_t i = 0; i < num_frames; i++) {
            for (uint32_t j = 0; j < num_vector_elements; j++) {
                if (!std::is_same<T, U>::value) {
                    if (!gnaFlags->input_low_precision) {
                        dst[j * num_group + i] = GNAPluginNS::ConvertFloatToInt16(src[i * num_vector_elements + j] * scaleFactor);
                    } else {
                        dst[j * num_group + i] = GNAPluginNS::ConvertFloatToInt8(src[i * num_vector_elements + j] * scaleFactor);
                    }
                } else {
                    dst[j * num_group + i] = src[i * num_vector_elements + j];
                }
            }
            // pad to meet weight matrix row length requirement
            for (uint32_t j = num_vector_elements; j < num_vector_stride; j++) {
                dst[j * num_group + i] = 0;
            }
        }
        // pad partial group
        for (uint32_t i = num_frames; i < num_group; i++) {
            for (uint32_t j = 0; j < num_vector_stride; j++) {
                dst[j * num_group + i] = 0;
            }
        }
    } else {
        if (!std::is_same<T, U>::value) {
            for (uint32_t i = 0; i < num_frames; i++) {
                T *ptr_dst_vec = reinterpret_cast<T *>(dst) + i * num_vector_stride;
                const U *ptr_src_vec = reinterpret_cast<const U *>(src) + i * num_vector_elements;
                std::memset(ptr_dst_vec, 0, num_vector_stride * sizeof(T));
                if (!gnaFlags->input_low_precision) {
                    for (uint32_t j = 0; j < num_vector_elements; j++) {
                        ptr_dst_vec[j] = GNAPluginNS::ConvertFloatToInt16(ptr_src_vec[j] * scaleFactor);
                    }
                } else {
                    for (uint32_t j = 0; j < num_vector_elements; j++) {
                        ptr_dst_vec[j] = GNAPluginNS::ConvertFloatToInt8(ptr_src_vec[j] * scaleFactor);
                    }
                }
            }

        } else {
            for (uint32_t i = 0; i < num_frames; i++) {
                void *ptr_dst_vec = reinterpret_cast<uint8_t *>(dst) + i * num_vector_stride * sizeof(T);
                const void *ptr_src_vec = reinterpret_cast<const uint8_t *>(src) + i * num_vector_elements * sizeof(U);
                std::memset(ptr_dst_vec, 0, num_vector_stride * sizeof(T));
                ie_memcpy(ptr_dst_vec, num_vector_elements * sizeof(T),
                    ptr_src_vec, num_vector_elements * sizeof(T));
            }
        }

        for (uint32_t i = num_frames; i < num_group; i++) {
            void *ptr_dst_vec = reinterpret_cast<uint8_t *>(dst) + i * num_vector_stride * sizeof(T);
            std::memset(ptr_dst_vec, 0, num_vector_stride * sizeof(T));
        }
    }
}

void GNAPlugin::ExportScores(void *ptr_dst,
                  const void *ptr_src,
                  intel_dnn_orientation_t orientation,
                  uint32_t num_frames,
                  uint32_t num_group,
                  uint32_t num_vector_elements,
                  uint32_t num_active_elements,
                  uint32_t num_vector_stride,
                  uint32_t num_bytes_per_element_input,
                  uint32_t num_bytes_per_element) {
    // source scores are possibly padded to multiple of 8 and possibly interleaved
    // rotate if necessary and only copy actual scores (not padding)
    if (orientation == kDnnInterleavedOrientation) {
        if (num_bytes_per_element == 2) {
            int16_t *dst = reinterpret_cast<int16_t *>(ptr_dst);
            const int16_t *src = reinterpret_cast<const int16_t *>(ptr_src);
            for (uint32_t i = 0; i < num_frames; i++) {
                for (uint32_t j = 0; j < num_active_elements; j++) {
                    dst[i * num_vector_elements + j] = src[j * num_group + i];
                }
                for (uint32_t j = num_active_elements; j < num_vector_elements; j++) {
                    dst[i * num_vector_elements + j] = 0;
                }
            }
        } else if (num_bytes_per_element == 4) {  // should work for both int and float
            int32_t *dst = reinterpret_cast<int32_t *>(ptr_dst);
            const int8_t *src = reinterpret_cast<const int8_t*>(ptr_src);
            for (uint32_t i = 0; i < num_frames; i++) {
                for (uint32_t j = 0; j < num_active_elements; j++) {
                    auto input_ptr = src + (j * num_group + i) * num_bytes_per_element_input;
                    auto dst_ptr = dst + (i * num_vector_elements + j);

                    switch (num_bytes_per_element_input) {
                        case 1: {
                            *dst_ptr = static_cast<int32_t>(*reinterpret_cast<const int8_t*>(input_ptr));
                            break;
                        }
                        case 2 : {
                            *dst_ptr  = static_cast<int32_t>(*reinterpret_cast<const int16_t*>(input_ptr));
                            break;
                        }
                        case 4 : {
                            *dst_ptr = *reinterpret_cast<const int32_t *>(input_ptr);
                            break;
                        }
                        default:
                            THROW_GNA_EXCEPTION << "Unsupported output layer precision: " << num_bytes_per_element_input << "bytes";
                    }
                }
                for (uint32_t j = num_active_elements; j < num_vector_elements; j++) {
                    dst[i * num_vector_elements + j] = 0;
                }
            }
        } else {
            THROW_GNA_EXCEPTION << "Unsupported target precision for infer : " << num_bytes_per_element << "bytes";
        }
    } else {
        if (num_bytes_per_element == 2) {
            for (uint32_t i = 0; i < num_frames; i++) {
                auto ptr_dst_vec = reinterpret_cast<uint8_t *>(ptr_dst) + i * num_vector_elements * sizeof(int16_t);
                auto ptr_src_vec = reinterpret_cast<const uint8_t *>(ptr_src) + i * num_vector_stride * sizeof(int16_t);
                memset(ptr_dst_vec, 0, num_vector_elements * sizeof(int16_t));
                ie_memcpy(ptr_dst_vec, num_active_elements * sizeof(int16_t),
                    ptr_src_vec, num_active_elements * sizeof(int16_t));
            }
        } else if (num_bytes_per_element == 4) {  // should work for both int and float
            if (num_bytes_per_element_input == 2) {
                for (uint32_t i = 0; i < num_frames; i++) {
                    auto ptr_dst_vec = reinterpret_cast<int32_t*>(ptr_dst) + i * num_vector_elements;
                    auto ptr_src_vec = reinterpret_cast<const int16_t*>(ptr_src) + i * num_vector_stride;
                    for (uint32_t j = 0; j < num_vector_elements; j++) {
                        ptr_dst_vec[j] = ptr_src_vec[j];
                    }
                }
            } else {
                for (uint32_t i = 0; i < num_frames; i++) {
                    void* ptr_dst_vec = reinterpret_cast<uint8_t*>(ptr_dst) + i * num_vector_elements * sizeof(float);
                    const void* ptr_src_vec = reinterpret_cast<const uint8_t*>(ptr_src) + i * num_vector_stride * sizeof(float);
                    memset(ptr_dst_vec, 0, num_vector_elements * sizeof(float));
                    ie_memcpy(ptr_dst_vec, num_active_elements * sizeof(float),
                        ptr_src_vec, num_active_elements * sizeof(float));
                }
            }
        } else {
            THROW_GNA_EXCEPTION << "Unsupported target precision for infer : " << num_bytes_per_element << "bytes";
        }
    }
}

void GNAPlugin::ImportFrames(void *ptr_dst,
                            const void *ptr_src,
                            Precision input_precision,
                            float scaleFactor,
                            intel_dnn_orientation_t orientation,
                            uint32_t num_frames,
                            uint32_t num_group,
                            uint32_t num_vector_elements,
                            uint32_t num_vector_stride) {
    switch (input_precision) {
    case Precision::U8:
    case Precision::I8:
    {
        auto src = reinterpret_cast<const uint8_t *>(ptr_src);
        if (!gnaFlags->input_low_precision) {
            auto dst = reinterpret_cast<int16_t*>(ptr_dst);
            copyInputData(dst, src, num_frames, num_group, num_vector_elements, num_vector_stride, orientation, scaleFactor);
        } else {
            auto dst = reinterpret_cast<int8_t*>(ptr_dst);
            copyInputData(dst, src, num_frames, num_group, num_vector_elements, num_vector_stride, orientation, scaleFactor);
        }
        break;
    }
    case Precision::I16:
    {
        auto src = reinterpret_cast<const int16_t *>(ptr_src);
        if (!gnaFlags->input_low_precision) {
            auto dst = reinterpret_cast<int16_t*>(ptr_dst);
            copyInputData(dst, src, num_frames, num_group, num_vector_elements, num_vector_stride, orientation, scaleFactor);
        } else {
            auto dst = reinterpret_cast<int8_t*>(ptr_dst);
            copyInputData(dst, src, num_frames, num_group, num_vector_elements, num_vector_stride, orientation, scaleFactor);
        }
        break;
    }
    case Precision::FP32:
    case Precision::I32:
    {
        auto src = reinterpret_cast<const float *>(ptr_src);
        if (!gnadevice) {
            auto dst = reinterpret_cast<float *>(ptr_dst);
            copyInputData(dst, src, num_frames, num_group, num_vector_elements, num_vector_stride, orientation, scaleFactor);
        } else {
            if (!gnaFlags->input_low_precision) {
                auto dst = reinterpret_cast<int16_t*>(ptr_dst);
                copyInputData(dst, src, num_frames, num_group, num_vector_elements, num_vector_stride, orientation, scaleFactor);
            } else {
                auto dst = reinterpret_cast<int8_t*>(ptr_dst);
                copyInputData(dst, src, num_frames, num_group, num_vector_elements, num_vector_stride, orientation, scaleFactor);
            }
        }
        break;
    }
    default:
        break;
    }
}

GNAPlugin::GNAPlugin() {
    Init();
    UpdateFieldsFromConfig();
}

GNAPlugin::GNAPlugin(const std::map<std::string, std::string>& configMap) {
    Init();
    SetConfig(configMap);
}

void GNAPlugin::Init() {
    OV_ITT_SCOPED_TASK(itt::domains::GNAPlugin, "Init");
    dnn = std::make_shared<backend::AMIntelDNN>(backend::AMIntelDNN());
    gnaFlags = std::make_shared<GNAPluginNS::GNAFlags>(GNAPluginNS::GNAFlags());
    inputs_ptr_ = std::make_shared<GNAPluginNS::GnaInputs>(GNAPluginNS::GnaInputs());
    outputs_ = GNAPluginNS::GnaOutputs();

    graphCompiler.setDNNPtr(dnn);
    graphCompiler.setGNAFlagsPtr(gnaFlags);
    graphCompiler.setInputsPtr(inputs_ptr_);
}

void GNAPlugin::InitGNADevice() {
    OV_ITT_SCOPED_TASK(itt::domains::GNA_LT, "InitGNADevice");
    gnadevice = std::make_shared<GNADeviceHelper>(config.gnaExecTarget,
                config.gnaCompileTarget,
                config.swExactMode,
                gnaFlags->gna_lib_async_threads_num,
                gnaFlags->gna_openmp_multithreading,
                gnaFlags->performance_counting,
                !config.dumpXNNPath.empty(),
                GetDeviceVersionFromString(config.dumpXNNGeneration));
    size_t page_size_bytes = 4096;
    gnamem = std::make_shared<gna_memory_type>(memory::make_polymorph<memory::GNAAllocator>(gnadevice), page_size_bytes);
    graphCompiler.setGNAMemoryPtr(gnamem);
}

void GNAPlugin::UpdateInputScaleFromNetwork(InferenceEngine::CNNNetwork& network) {
    OV_ITT_SCOPED_TASK(itt::domains::GNA_LT, "UpdateInputScaleFromNetwork");
    // fp32 emulation mode dont need any modifications to configuration
    if (config.gnaFlags.sw_fp32) return;

    // search for FQ layers
    // only supports cases of int16 or int8
    InputsDataMap inputs = network.getInputsInfo();
    size_t inputIdx = 0;
    for (auto&& input : inputs) {
        auto data = input.second->getInputData();
        for (auto && nextToInputLayer : getInputTo(data)) {
            if (!LayerInfo(nextToInputLayer.second).isFakeQuantize()) {
                continue;
            }

            // replacing scale factor from this fq layer
            GNAFakeQuantizeLayer fqLayer(nextToInputLayer.second);
            auto inputRange = fqLayer.getInputRange();
            auto outputRange = fqLayer.getOutputRange();
            if (inputRange.second.size() != 1 || inputRange.second.size() != 1 ||
                outputRange.second.size() != 1 || outputRange.second.size() != 1) {
                THROW_GNA_LAYER_EXCEPTION(nextToInputLayer.second)
                    << "unsupported, per-channel quantization for input layer : " << input.second->name();
            }

            // GNA input is always quantized to int16, so number of levels can't be greater than max uint16
            // todo: should be solved in POT (issue 63330)
            size_t levels = std::min(fqLayer.getLevels(), static_cast<size_t>(std::numeric_limits<uint16_t>::max() + 1));
            auto scaleInput = frontend::CalculateScaleFactorFromStats(levels, inputRange.first[0], inputRange.second[0]);

            IE_ASSERT(config.inputScaleFactors.size() > inputIdx);

            if (!config.inputScaleFactors.empty()) {
                gnawarn() << "WARNING: Scale factor calculated during model quantization (" << scaleInput
                    << ") will be used instead of user input (" << (*inputs_ptr_)[input.first].scale_factor << ").\n";
                if ((*inputs_ptr_)[input.first].scale_factor < scaleInput) {
                    gnawarn() << "WARNING: Scale factor calculated based on input values (" << (*inputs_ptr_)[input.first].scale_factor
                        << ") is smaller than scale factor used to quantize model (" << scaleInput << "). "
                        << "Input values will be clamped.\n";
                }
            }

            config.inputScaleFactors[inputIdx] = scaleInput;
            (*inputs_ptr_)[input.first].scale_factor = scaleInput;
        }

        inputIdx++;
    }
}

void GNAPlugin::UpdateInputsAndOutputsInfoFromNetwork(InferenceEngine::CNNNetwork & network) {
    OV_ITT_SCOPED_TASK(itt::domains::GNA_LT, "UpdateInputsAndOutputsInfoFromNetwork");

    // update inputs
    {
        InputsDataMap network_inputs = network.getInputsInfo();

        size_t id = 0;
        for (const auto input : network_inputs) {
            (*inputs_ptr_)[input.first].Update(input.second);

            // update scale factor from config
            if (id < config.inputScaleFactors.size()) {
                (*inputs_ptr_)[input.first].scale_factor = config.inputScaleFactors[id];
            }

            id++;
        }
    }
    // update outputs
    {
        OutputsDataMap outputs = network.getOutputsInfo();
        for (const auto output : outputs) {
            outputs_[output.first].Update(output.second);
        }
    }
}

void GNAPlugin::UpdateInputs(const std::vector<std::shared_ptr<const ov::Node>>& params) {
    OV_ITT_SCOPED_TASK(itt::domains::GNA_LT, "UpdateInputs");
    for (const auto& param : params) {
        const std::string ie_name = param->get_friendly_name();
        (*inputs_ptr_)[ie_name].name = param->get_friendly_name();
        (*inputs_ptr_)[ie_name].tensor_names = param->get_output_tensor(0).get_names();
    }
}

void GNAPlugin::UpdateOutputs(const std::vector<std::shared_ptr<const ov::Node>>& results) {
    OV_ITT_SCOPED_TASK(itt::domains::GNA_LT, "UpdateOutputs");
    for (const auto& result : results) {
        const std::string ie_name = ngraph::op::util::create_ie_output_name(result->input_value(0));
        outputs_[ie_name].name = ie_name;
        outputs_[ie_name].tensor_names = result->get_output_tensor(0).get_names();
    }
}

void GNAPlugin::UpdateInputsAndOutputsInfoFromModel(const std::shared_ptr<ov::Model> &model) {
    OV_ITT_SCOPED_TASK(itt::domains::GNA_LT, "UpdateInputsAndOutputsInfoFromFModel");

    // update inputs
    {
        std::vector<std::shared_ptr<const ov::Node>> node_vector;
        for (auto& param : model->get_parameters()) {
            node_vector.emplace_back(param);
        }
        UpdateInputs(node_vector);
    }

    // update outputs
    {
        std::vector<std::shared_ptr<const ov::Node>> node_vector;
        for (auto& result : model->get_results()) {
            node_vector.emplace_back(result);
        }
        UpdateOutputs(node_vector);
    }
}

bool GNAPlugin::TryToInitOutput(const std::string &portName, InferenceEngine::CNNLayerPtr layer) {
    auto initOutput = [this, portName, layer]
            (intel_dnn_orientation_t orientation, size_t numBytesPerElem, size_t numElem, void* outputPtr) {
        auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(layer);

        outputs_.at(portName).ptrs.resize(gnaFlags->gna_lib_async_threads_num);
        outputs_.at(portName).orientation = orientation;
        outputs_.at(portName).num_bytes_per_element = numBytesPerElem;
        outputs_.at(portName).scale_factor = quantized != nullptr ? quantized->_dst_quant.GetScale() : GNAPluginNS::kScaleFactorDefault;
        outputs_.at(portName).num_elements = numElem;

        // binding ptr for first infer request - then others will be setup during relocation
        gnamem->bind_ptr(layer, &outputs_.at(portName).ptrs.front(), outputPtr);
    };

    // probing gna_primitives
    auto irLayerAvatar = std::find_if(
        graphCompiler.dnnComponents.components.begin(),
        graphCompiler.dnnComponents.components.end(),
        [&layer](const backend::DnnComponents::storage_type::value_type & value) {
            return value.name == layer->name;
    });
    if (irLayerAvatar != graphCompiler.dnnComponents.components.end()) {
        initOutput(irLayerAvatar->dnnComponent.orientation_out, irLayerAvatar->dnnComponent.num_bytes_per_output,
                   irLayerAvatar->dnnComponent.num_rows_out, &irLayerAvatar->dnnComponent.ptr_outputs);
        return true;
    }

    // probing concatInfo
    if (LayerInfo(layer).isConcat()) {
        auto concatConnection  = graphCompiler.concat_connection.find(layer->name);
        if (concatConnection != graphCompiler.concat_connection.end()) {
            auto precision = layer->outData.front()->getPrecision().size();
            initOutput(kDnnInterleavedOrientation, precision, concatConnection->second.reserved_size / precision,
                       &concatConnection->second.gna_ptr);
            return true;
        }
    }

    // probing a constant info, for constant trivial networks support
    if (LayerInfo(layer).isConst()) {
        auto const_blob = layer->blobs["custom"];
        auto constConnection  = graphCompiler.const_connections.find(layer->name);
        if (constConnection != graphCompiler.const_connections.end()) {
            initOutput(kDnnInterleavedOrientation, layer->outData.front()->getPrecision().size(),
                       const_blob->size(), &constConnection->second);
            return true;
        }
    }

    return false;
}

void GNAPlugin::FillInputsAndOutputsTranspositionInfo(const InferenceEngine::CNNNetwork& net) {
    OV_ITT_SCOPED_TASK(itt::domains::GNA_LT, "FillInputsAndOutputsTranspositionInfo");
    auto printTranspositionInfo = [](const std::vector<TranspositionInfo> &transpositionInfo) {
        for (const auto &transpositionInfoPart : transpositionInfo) {
            gnalog() << "transpose=" << transpositionInfoPart.transpose << " rows_num=" << transpositionInfoPart.num_transpose_rows
                     << " columns_num=" << transpositionInfoPart.num_transpose_columns << "\n";
        }
    };

    auto inputLayers = CNNNetGetAllInputLayers(net);
    for (const auto& inputLayer : inputLayers) {
        // Collect information for inputs transposition
        if (!LayerInfo(inputLayer).isInput()) continue;
        auto transpositionInfo = FindTranspositionInfoFromNextLayers(inputLayer);
        if (transpositionInfo.empty()) continue;

        transpose_inputs_info.insert({inputLayer->name, transpositionInfo});
        gnalog() << "Input " << inputLayer->name << " transposition info: \n";
        printTranspositionInfo(transpositionInfo);
    }

    auto outputsMap = net.getOutputsInfo();
    for (const auto& outPort : outputsMap) {
        auto outLayer = getCreatorLayer(outPort.second).lock();
        // Collect information for outputs transposition
        if (!LayerInfo(outLayer).isOutput()) continue;
        auto transpositionInfo = FindTranspositionInfoFromPrevLayers(outLayer);
        if (transpositionInfo.empty()) continue;

        // Swap transposition info rows and columns since we need to transpose output back from NHWC to NCHW
        for (auto && transpositionInfoPart : transpositionInfo) {
            if (transpositionInfoPart.transpose) {
                std::swap(transpositionInfoPart.num_transpose_rows, transpositionInfoPart.num_transpose_columns);
            }
        }
        transpose_outputs_info.insert({outLayer->name, transpositionInfo});
        gnalog() << "Output " << outLayer->name << " transposition info: \n";
        printTranspositionInfo(transpositionInfo);
    }
}

#ifdef PLOT
void GNAPlugin::AddDebugProperties(const InferenceEngine::CNNLayerPtr layer,
    InferenceEngine::ordered_properties& printed_properties,
    InferenceEngine::ordered_properties& node_properties) {
    // printing quantized params
    auto quantized = InferenceEngine::getInjectedData<QuantizedLayerParams>(layer);
    if (!quantized) {
        return;
    }
    if (LayerInfo(layer).isWeightable() || LayerInfo(layer).isEltwise()) {
        printed_properties.emplace_back(
            "weights scale factor", std::to_string(quantized->_weights_quant.GetScale()));
        if (quantized->_weights_quant.IsStatsSet()) {
            for (auto& min : quantized->_weights_quant.GetMinValues()) {
                printed_properties.emplace_back(
                    "weights min val", std::to_string(min));
            }
            for (auto& max : quantized->_weights_quant.GetMaxValues()) {
                printed_properties.emplace_back(
                    "weights max val", std::to_string(max));
            }
        }

        if (quantized->_bias_quant.IsStatsSet()) {
            for (auto& min : quantized->_bias_quant.GetMinValues()) {
                printed_properties.emplace_back(
                    "bias min val", std::to_string(min));
            }
            for (auto& max : quantized->_bias_quant.GetMaxValues()) {
                printed_properties.emplace_back(
                    "bias max val", std::to_string(max));
            }
        }
    }
    printed_properties.emplace_back(
        "src scale factor", std::to_string(quantized->_src_quant.GetScale()));
    if (quantized->_src_quant.IsStatsSet()) {
        for (auto& min : quantized->_src_quant.GetMinValues()) {
            printed_properties.emplace_back(
                "src min val", std::to_string(min));
        }
        for (auto& max : quantized->_src_quant.GetMaxValues()) {
            printed_properties.emplace_back(
                "src max val", std::to_string(max));
        }
    }

    printed_properties.emplace_back(
        "dst scale factor", std::to_string(quantized->_dst_quant.GetScale()));
    if (quantized->_dst_quant.IsStatsSet()) {
        for (auto& min : quantized->_dst_quant.GetMinValues()) {
            printed_properties.emplace_back(
                "dst min val", std::to_string(min));
        }
        for (auto& max : quantized->_dst_quant.GetMaxValues()) {
            printed_properties.emplace_back(
                "dst max val", std::to_string(max));
        }
    }
}
#endif

void GNAPlugin::LoadNetwork(CNNNetwork & _network) {
    OV_ITT_SCOPED_TASK(itt::domains::GNAPlugin, "LoadNetwork");
    std::shared_ptr<InferenceEngine::details::CNNNetworkImpl> convertedNetwork;

    if (!gnaFlags->sw_fp32) {
        InitGNADevice();
    }

    std::string effectiveGnaCompileTarget = config.gnaCompileTarget;
    if (gnadevice) {
        effectiveGnaCompileTarget = gnadevice->getEffectiveGnaCompileTarget();
    }

    bool isNgraphPassesUsed = false;
    bool fake_quantized = false;

    if (_network.getFunction()) {
        CNNNetwork clonedNetwork = InferenceEngine::cloneNetwork(_network);
        const auto& graph = clonedNetwork.getFunction();
        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        fake_quantized = ngraph::op::util::has_op_with_type<ngraph::opset7::FakeQuantize>(graph);
        // In OV API 2.0(IRv10) default convertion to fp32 (inputs, outputs and weights) is disabled
        // and we need to run the ConvertPrecision transformation to support old networks.
        manager.register_pass<ngraph::pass::ConvertPrecision>(precisions_array{{ngraph::element::f16, ngraph::element::f32}});
        manager.register_pass<ngraph::pass::ConvertMVN1ToMVN6>();
        manager.register_pass<DecomposeMVN>();
        manager.register_pass<ngraph::pass::CommonOptimizations>();
        manager.register_pass<RemoveInputConvert>();
        manager.register_pass<RemoveOutputConvert>();
        manager.register_pass<ngraph::pass::LSTMCellDecomposition>();
        manager.register_pass<ConvertDWSCToScaleShifts>();
        manager.register_pass<ConvertPaddedToValidConv>();
        manager.register_pass<Decompose2DConvTransposedWithBiasAF>(effectiveGnaCompileTarget, config.gnaPrecision);
        manager.register_pass<Decompose2DConvTransposedWithBias>(effectiveGnaCompileTarget, config.gnaPrecision);
        manager.register_pass<Decompose2DConv>(effectiveGnaCompileTarget, config.gnaPrecision);
        // TODO enable this transformation for networks with convolutions
        if (!ngraph::op::util::has_op_with_type<ngraph::opset7::Convolution>(graph)) {
            manager.register_pass<ConvertMatmulWithFqToPointWiseConvolution>();
            manager.register_pass<ConvertMatmulWithBiasToPointWiseConvolution>();
            manager.register_pass<ConvertMatmulToPointWiseConvolution>();
        }
        manager.register_pass<SplitConvolutionWithFq>();
        manager.register_pass<SplitConvolutionWithBias>();
        manager.register_pass<SplitConvolution>();
        manager.register_pass<InsertReshapeAroundMatmulWithTranspose>();
        manager.register_pass<InsertReshapeAroundMatmulWithFq>();
        manager.register_pass<InsertReshapeAroundMatmulWithAdd>();
        manager.register_pass<InsertReshapeAroundMatmul>();
        manager.register_pass<SwapInputMatMulWithTrailingTranspose>();
        manager.register_pass<SwapInputMatMulWithAct>();
        manager.register_pass<SwapInputMatMulWithFq>();
        manager.register_pass<SwapInputMatMulWithBias>();
        manager.register_pass<SwapInputMatMul>();
        manager.register_pass<HandleTransposesAroundMatMul>();
        manager.register_pass<InsertTransposeAfterConvOrPool>();
        manager.register_pass<ReorderActivationAndPooling>();
        manager.register_pass<RemoveSingleInputConcat>();
        manager.register_pass<SubstituteSoftsign>();
        manager.register_pass<ngraph::pass::ConvertOpSet3ToOpSet2>();
        manager.register_pass<ngraph::pass::ConvertOpSet2ToOpSet1>();
        manager.register_pass<ngraph::pass::ConvertOpSet1ToLegacy>();
        manager.register_pass<RemoveExtraReshapes>();
        /*
          Put BroadcastAddMultiplyConst here after ConvertOpSet..() transformations since there are conficts with them.
          ngraph::pass::ConvertOpSet1ToLegacy -> ngraph::pass::BiasFusions ->
                                                    ngraph::pass::ConvAddFusion, ngraph::pass::ConvMultiplyFusion
          That transormations fuse bias into convolution and recognizes const node as [1, C, 1, 1].
          TODO: move that transformation just beyond RemoveSingleInputConcat pass after removing ConvertOpSet1ToLegacy
              transormations
        */
        manager.register_pass<BroadcastAddMultiplyConst>();
        // UnrollTI should be the last transformation in the transformation pipeline
        manager.register_pass<ngraph::pass::UnrollTensorIterator>();
        const auto& pass_config = manager.get_pass_config();

        // Allowing FP16 Converts to be folded and FP16 constants to upgrade to FP32 data type
        pass_config->disable<ov::pass::ConvertCompressedOnlyToLegacy>();
        pass_config->disable<ov::pass::DisableDecompressionConvertConstantFolding>();

        pass_config->disable<ngraph::pass::FakeQuantizeMulFusion>();
        pass_config->disable<ngraph::pass::FakeQuantizeReshapeFusion>();
        pass_config->disable<ngraph::pass::PullTransposeThroughFQUp>();
        pass_config->disable<ngraph::pass::ReluFakeQuantizeFusion>();
        // Consider to enable after per-channel quantization on FakeQuantize layer is supported in GNAPlugin, see issue 52034
        pass_config->disable<ngraph::pass::AddFakeQuantizeFusion>();
        // TransposeReduction can be enabled when Transpose-Conv-Transpose patterns will be handled in ngraph transformations
        pass_config->disable<ngraph::pass::TransposeReduction>();
        // Operations Max and Min aren't supported
        pass_config->disable<ngraph::pass::ConcatReduceFusion>();
        manager.run_passes(graph);
        convertedNetwork = InferenceEngine::details::convertFunctionToICNNNetwork(graph, clonedNetwork);
        isNgraphPassesUsed = true;
    }
    IE_SUPPRESS_DEPRECATED_START
    InferenceEngine::CNNNetwork network = convertedNetwork ? InferenceEngine::CNNNetwork{convertedNetwork} : _network;
    IE_SUPPRESS_DEPRECATED_END

    NetPass::ConvertPrecision(network, Precision::I64, Precision::I32);
    NetPass::ConvertPrecision(network, Precision::U64, Precision::I32);
    NetPass::ConvertPrecision(network, Precision::U32, Precision::I32);


    //  Check the network
    std::string error;
    if (!GNAPluginNS::GNALimitations::AreLayersSupported(network, error, gnaFlags->log_level == PluginConfigParams::LOG_WARNING)) {
        THROW_GNA_EXCEPTION << error.c_str();
    }

    // Set input and output information from ngraph function
    if (_network.getFunction()) {
        UpdateInputsAndOutputsInfoFromModel(_network.getFunction());
    }

    // Set input and output information from orginal network
    UpdateInputsAndOutputsInfoFromNetwork(network);

    if (fake_quantized) {
        UpdateInputScaleFromNetwork(network);
    }

    if (MustBeConvertedFromNCHWToNHWC(details::CNNNetSortTopologically(network))) {
        FillInputsAndOutputsTranspositionInfo(network);
    }

    // network optimisation phases
    int passIdx = 0;
    auto run_passes = [&] (const CNNNetwork& network, bool runBeforeCopy, bool lowPrecision) {
        auto passes = make_shared<PassManager>(PassManagerSettings{runBeforeCopy, lowPrecision}, network);
        passes->registerPass<RemoveConstPass>();
        if (!isNgraphPassesUsed) {
            passes->registerPass<UnrollTIPass>();
            passes->registerPass<RemoveConstPass>();
            passes->registerPass<UnrollLSTMCellPass>();
            passes->registerPass<RemoveSingleInputConcatPass>();
            passes->registerPass<BroadcastConstPass>();
            passes->registerPass<SubstituteScaleShiftBroadCastPass>();
        }

        if (fake_quantized)
            passes->registerPass<SubstituteSoftSignPass>();

        // fake quantisation aware passes
        passes->registerPass<FuseFQIntoWeightsPass>();
        passes->registerPass<MoveFakeQuantizeLayerIntoQuantParamsPass>();

        passes->registerPass<TransposeWeightsFromNCHWToNHWCPass>();

        passes->registerPass<SubstitutePReluPass>();

        passes->registerPass<ReorderMaxPoolPass>();
        passes->registerPass<EltwiseSplitOverChannelsPass>();
        passes->registerPass<InsertSplitAligningFilterPass>();

        passes->registerPass<InsertCopyLayerPass>();

        passes->registerPass<FlattenTrivialConcatPass>();
        passes->registerPass<InsertConcatAligningFilterPass>();
        passes->registerPass<ReorderConcatInputsPass>();
        passes->registerPass<RemovePermutationsNHWCToNCHWPass>();
        passes->registerPass<InsertIdentityLayerPass>();
        passes->registerPass<BreakFusingOfOutputLayersPass>();
        passes->registerPass<InsertDiagonalLayerPass>();
        passes->registerPass<HandleMultipleActivationsForTheLayerPass>();
        passes->registerPass<ForbidActivationFusingPass>();
        passes->registerPass<FuseMultipleIdentitiesPass>();
        passIdx = passes->run(passIdx);
    };

    InferenceEngine::CNNNetwork newNet;

    if (gnaFlags->sw_fp32) {
        auto visitor = [&](InferenceEngine::CNNLayerPtr lp) {
            transformLayer(lp, WeightsConverter());
            return lp;
        };
        newNet = InferenceEngine::CNNNetCopy(network, visitor);
        // to run all passes need to have two calls to pass manager
        run_passes(newNet, true, gnaFlags->input_low_precision);
        run_passes(newNet, false, gnaFlags->input_low_precision);
    } else if (fake_quantized) {
        ModelQuantizer<FakeQuant> modelQuantizer;
        newNet = modelQuantizer.quantize(network, run_passes, *inputs_ptr_);
    } else {
        switch (config.gnaPrecision) {
            case Precision::I16:
                ModelQuantizer<QuantI16> q16;
                newNet = q16.quantize(network, run_passes, *inputs_ptr_);
                break;
            case Precision::I8:
                if (gnaFlags->input_low_precision == false) {
                    ModelQuantizer<QuantI8> q8;
                    newNet = q8.quantize(network, run_passes, *inputs_ptr_);
                } else {
                    ModelQuantizer<QuantI8_I8> q8_8;
                    newNet = q8_8.quantize(network, run_passes, *inputs_ptr_);
                }
                break;
            default:
                THROW_GNA_EXCEPTION << "unsupported GNA precision for quantisation: " << config.gnaPrecision;
        }
    }

    auto inputLayers = CNNNetGetAllInputLayers(newNet);

#ifdef PLOT
    std::ofstream file("gna_passes.dot");
    saveGraphToDot(newNet, file, [this](const CNNLayerPtr layer,
        ordered_properties& printed_properties,
        ordered_properties& node_properties) {
            AddDebugProperties(layer, printed_properties, node_properties);
        });
#endif

    auto sortedNet = CNNNetSortTopologicallyEx(newNet, make_fuzed_order);

    if (sortedNet.empty()) {
        THROW_GNA_EXCEPTION << "Sorted network is empty";
    }

    std::vector<CNNLayerPtr> sortedNoMem;
    std::unordered_map<std::string, std::vector<InferenceEngine::CNNLayerPtr>> memoryPairs;
    // find all memory layers pairs and mark which one used as outputs
    for (auto &layer : sortedNet) {
        auto generic = dynamic_cast<GenericLayer *>(layer.get());
        if (generic == nullptr) {
            sortedNoMem.push_back(layer);
            continue;
        }
        LayerInfo layerInfo(layer);
        if (layerInfo.isMemory()) {
            // collect all memory pairs
            auto id = generic->GetParamAsString("id");
            memoryPairs[id].resize(generic->GetParamAsInt("size"));
            memoryPairs[id][generic->GetParamAsInt("index")] = layer;
            continue;
        } else if (layerInfo.isConcat()) {
            graphCompiler.fillConcatConnections(layer);
        } else if (layerInfo.isSplit() || layerInfo.isSlice()) {
            graphCompiler.fillSplitConnections(layer);
        }
        sortedNoMem.push_back(layer);
    }

    // fill in extra storage with memory layers
    graphCompiler.fillMemoryConnections(memoryPairs);

    if (!graphCompiler.memory_connection.empty() && gnaFlags->gna_lib_async_threads_num != 1) {
        // TODO: check if updating the number of threads is needed for sw_fp32
        gnaFlags->gna_lib_async_threads_num = 1;
        if (!gnaFlags->sw_fp32)
            InitGNADevice();
    }

    if (gnaFlags->sw_fp32) {
        gnamem.reset(new gna_memory_type(memory::make_polymorph<std::allocator<uint8_t>>()));
        graphCompiler.setGNAMemoryPtr(gnamem);
    }

    // keep inputs information and create input primitives
    inputs_data_map_ = newNet.getInputsInfo();
    if (inputs_data_map_.empty()) {
        gnawarn() << "No inputs for the topology\n";
    }

    // keep output dims
    outputs_data_map_ = newNet.getOutputsInfo();
    if (outputs_data_map_.empty()) {
        THROW_GNA_EXCEPTION << "No outputs for the topology";
    }

    for (auto && input : inputs_data_map_) {
        inputs_ptr_->at(input.first).ptrs.resize(gnaFlags->gna_lib_async_threads_num);
    }

    // Creating Layer primitives
    uint16_t id = 0;
    for (auto & layer : sortedNoMem) {
        IE_SUPPRESS_DEPRECATED_START
        layer->userValue.v_int = id++;
        IE_SUPPRESS_DEPRECATED_END
        graphCompiler.CreateLayerPrimitive(layer);
    }

    for (auto& inputLayer : inputLayers) {
        auto layerInfo = LayerInfo(inputLayer);
        if (layerInfo.isInput() && 0 == inputs_ptr_->at(inputLayer->name).get_allocated_size()) {
            graphCompiler.connectOutput(inputLayer, &inputs_ptr_->at(inputLayer->name).ptrs.front(), 0);
        }
    }

    if (graphCompiler.dnnComponents.components.empty()) {
        gnawarn() << "No GNA primitives created based on topology. This might indicate trivial topology\n";
        trivialTopology = true;
    }

    /// setting-up output layers information
    int portId = 0;
    for (auto && outPort : outputs_data_map_) {
        // gets output layer pointer in original topology not in cloned
        auto outLayer = getCreatorLayer(outPort.second).lock();

        // Memory layers are not dnnComponents hence we need to make switch with identity layer
        if (outLayer->type == "Memory") {
            // traverse memory connection to find corresponding output_memory
            for (auto && memConnection : graphCompiler.memory_connection) {
                if (memConnection.second.getInput()->name == outLayer->name) {
                    // if connection is found, replace memory input layer with memory output layer
                    outLayer = memConnection.second.getOutput();
                    break;
                }
            }
        }

        // searching for outData represented in GNA blob
        // using ufs - upper first search
        gnalog() << "[UFS] searching for : "<< outPort.first << " representation in GNA\n";
        bool stopSearching = false;

        CNNNetDFS(outLayer, [this, &outPort, &stopSearching](CNNLayerPtr layer) {
            gnalog() << "[UFS] from : "<< outPort.first <<" reached: " << layer->name << "\n";
            stopSearching = TryToInitOutput(outPort.first, layer);
        }, true, [&stopSearching](InferenceEngine::CNNLayer* from) {
            return make_upstream_order(!stopSearching ? from : nullptr);
        });
        if (!stopSearching) {
            THROW_GNA_EXCEPTION << "unsupported topology: cannot locate " << outPort.first
                                << " after compiling GNA graph";
        }
        portId++;
    }

    // TODO: how active list will work in multioutput case
    // make room for active list
    gnamem->reserve_ptr(nullptr, nullptr, ALIGN64(outputs_.Get().begin()->get_required_size()), 64);

    void *pParallelExecutionData  = nullptr;

    // reserving more bytes for intermediate data in parallel case - TODO: this works incorrectly in compact mode at lest
    rwSegmentSize = gnamem->getRWBytes();
    if (gnaFlags->gna_lib_async_threads_num > 1) {
        gnamem->reserve_ptr(nullptr, &pParallelExecutionData, gnamem->getRWBytes() * (gnaFlags->gna_lib_async_threads_num - 1), 64);
    }

    gnamem->commit(gnaFlags->compact_mode);

    dnn->Init(gnamem->getBasePtr(),
             gnamem->getTotalBytes(),
             gnaFlags->sw_fp32 ? kDnnFloat : kDnnInt,
             1);

    // TODO: this copy is unneeded; in fact, we can directly create gna structs from list
    auto execOrder = graphCompiler.dnnComponents.getExecutionOrder();
    dnn->component.insert(dnn->component.begin(), execOrder.begin(), execOrder.end());

    // in fp32 mode last PWL cannot be computed without that
    if (!graphCompiler.dnnComponents.components.empty()) {
        dnn->InitActiveList(NULL);
    }

    gnaModels.push_back(std::make_tuple(make_shared<CPPWrapper<Gna2Model>>()));

    if (!gnaFlags->sw_fp32 && !graphCompiler.dnnComponents.components.empty()) {
        // number of layer gets calculated inside that InitGNAStruct function
        dnn->InitGNAStruct(&std::get<0>(gnaModels.front())->obj, effectiveGnaCompileTarget);
    }

    // creating same gna RW segment for parallel infer requests
    for (int i = 1; i != gnaFlags->gna_lib_async_threads_num; i++) {
        gnaModels.push_back(std::make_tuple(make_shared<CPPWrapper<Gna2Model>>()));
        // this can be improved by just copy all structures, but we are too lazy
        dnn->InitGNAStruct(&std::get<0>(gnaModels.back())->obj, effectiveGnaCompileTarget);
        // relocate rw pointers to new offset
        auto basePtr = reinterpret_cast<uint8_t*>(pParallelExecutionData) + rwSegmentSize * (i - 1);

        auto relocate = [basePtr, this](void *& ptr_out, void * ptr_in) {
            if (ptr_in == nullptr) {
                ptr_out = nullptr;
            } else {
                auto offset = reinterpret_cast<uint8_t *>(ptr_in) - reinterpret_cast<uint8_t *>(gnamem->getBasePtr());
                ptr_out = basePtr + offset;
            }
        };

        for (auto &input : inputs_ptr_->Get()) {
            relocate(input.ptrs[i], input.ptrs[0]);
        }

        // relocating all output pointers
        for (auto &output : outputs_.Get()) {
            relocate(output.ptrs[i], output.ptrs[0]);
        }

        for (int j = 0; j != std::get<0>(gnaModels.front())->obj.NumberOfOperations; j++) {
            auto & gnaOperation = std::get<0>(gnaModels[i])->obj.Operations[j];
            relocate(const_cast<Gna2Tensor*>(gnaOperation.Operands[0])->Data, gnaOperation.Operands[0]->Data);
            relocate(const_cast<Gna2Tensor*>(gnaOperation.Operands[1])->Data, gnaOperation.Operands[1]->Data);
        }
    }

    // calculating input orientation without memory layers, since their orientation not changed during infer right now
    std::unordered_map<string, std::vector<string>> skippedLayers;

    bool withConv = false;
    for (auto &layer : sortedNet) {
        auto layerInfo = LayerInfo(layer);
        if (layerInfo.isConvolution()) {
            withConv = true;
            break;
        }
    }
    if (withConv) {
        for (auto &inputLayer : sortedNet) {
            if (!LayerInfo(inputLayer).isInput()) {
                continue;
            }
            auto doesntHaveGnaMapping = [this] (CNNLayerPtr l) {
                auto dnnLayer = graphCompiler.dnnComponents.findComponent(l);
                return dnnLayer == nullptr;
            };

            auto nextLayers = CNNNetGetAllNextLayersSkipCertain(inputLayer, -1, doesntHaveGnaMapping);

            std::vector<intel_dnn_orientation_t> orientations;
            for (auto &nextLayer : nextLayers) {
                auto dnnLayer = graphCompiler.dnnComponents.findComponent(nextLayer);
                // non functional layer - skipped by gna
                if (nullptr == dnnLayer) {
                    THROW_GNA_LAYER_EXCEPTION(inputLayer) << " gna mapped layer search connection failed";
                }
                // Orientation of an input doesn't make sense for components transposing the data and
                // components with identity dimensions, so skip them
                if (dnnLayer->operation != kDnnInterleaveOp && dnnLayer->operation != kDnnDeinterleaveOp &&
                    dnnLayer->num_rows_in > 1 && dnnLayer->num_columns_in > 1) {
                    orientations.push_back(dnnLayer->orientation_in);
                }
            }

            if (orientations.empty()) {
                // in this case orientation doesn't make a sense
                inputs_ptr_->at(inputLayer->name).orientation = kDnnNonInterleavedOrientation;
            } else if (std::adjacent_find(orientations.begin(), orientations.end(),
                           std::not_equal_to<intel_dnn_orientation_t>()) == orientations.end()) {
                // all orientations are equal
                inputs_ptr_->at(inputLayer->name).orientation = orientations.front();
            } else {
                // unsupported case: orientations are different and they are important for these components
                THROW_GNA_EXCEPTION << "orientation for input layer: " << inputLayer->name << " cannot be calculated";
            }
        }
    } else {
        for (auto &inputLayer : inputLayers) {
            if (LayerInfo(inputLayer).isInput()) {
                inputs_ptr_->at(inputLayer->name).orientation = kDnnInterleavedOrientation;
            }
        }
    }

    if (dnn->do_rotate_input && transpose_inputs_info.empty()) {
        for (auto &inputLayer : inputLayers) {
            transpose_inputs_info.insert({inputLayer->name,
                {TranspositionInfo{dnn->do_rotate_input, dnn->num_rotate_rows, dnn->num_rotate_columns}}});
        }
    }

    DumpXNNToFile();

#ifdef PLOT
    dnn->WriteGraphWizModel("gna-blob.dot");
#endif
    createRequestConfigsForGnaModels();
}

void GNAPlugin::createRequestConfigsForGnaModels() {
    if (!gnadevice || trivialTopology) {
        gnaRequestConfigToRequestIdMap.push_back(std::make_tuple(FAKE_REQUEST_CONFIG_ID, -1, InferenceEngine::BlobMap()));
        return;
    }
    for (auto& model : gnaModels) {
        auto& gnaNnet = std::get<0>(model).get()->obj;
        const auto modelId = gnadevice->createModel(gnaNnet);
        const auto requestConfigId = gnadevice->createRequestConfig(modelId);
        gnaRequestConfigToRequestIdMap.push_back(std::make_tuple(requestConfigId, -1, InferenceEngine::BlobMap()));
    }
}

int GNAPlugin::GetDeviceVersionFromString(const std::string deviceString) {
    if (deviceString.empty())
        return static_cast<int>(Gna2DeviceVersionEmbedded1_0);
    THROW_GNA_EXCEPTION << "Wrong GNA generation for embedded model dump: " << deviceString;
}

void GNAPlugin::DumpXNNToFile() const {
    // TODO: output  precision as well as pointer might be incorrect, LSTM for sure
    // gna looks automatically set layer 0 as output and adjust it's pointer / precision/ size respectively
    if (config.dumpXNNPath.empty()) {
        return;
    }

    if (!gnadevice) {
        THROW_GNA_EXCEPTION << "Cannot generate XNNDump for float network";
    }
    std::ofstream dumpStream(config.dumpXNNPath, std::ios::out | std::ios::binary);

    auto const modelId = gnadevice->createModel(std::get<0>(gnaModels.front())->obj);
    auto dump = gnadevice->dumpXnn(modelId);
    dump.header.RwRegionSize = gnamem->getRWBytes();
    dump.header.InputScalingFactor = inputs_ptr_->Get().begin()->scale_factor;
    dump.header.OutputScalingFactor = outputs_.Get().begin()->scale_factor;
    dumpStream.write(reinterpret_cast<char*>(&dump.header), sizeof(Gna2ModelSueCreekHeader));
    dumpStream.write(reinterpret_cast<char*>(dump.model.get()), dump.header.ModelSize);
    gnadevice->releaseModel(modelId);
}

uint32_t GNAPlugin::QueueInference(const InferenceEngine::BlobMap &inputs, InferenceEngine::BlobMap &result) {
    auto& nnets = gnaRequestConfigToRequestIdMap;
    auto freeNnet = std::find_if(std::begin(nnets), std::end(nnets), [](decltype(nnets.front()) & item) {
        return std::get<1>(item) == -1;
    });

    if (freeNnet == nnets.end()) {
        if (!graphCompiler.memory_connection.empty()) {
            Wait(0);
            freeNnet = nnets.begin();
        } else {
            IE_THROW(RequestBusy)
                               << "GNA executable network has max of "
                               << static_cast<uint32_t >(gnaFlags->gna_lib_async_threads_num)
                               << " parallel infer requests, please sync one of already running";
        }
    }

    auto idx = static_cast<uint32_t>(std::distance(std::begin(nnets), freeNnet));

    int inputNum = 0;
    for (auto &input : inputs) {
        auto inputLayout = input.second->getTensorDesc().getLayout();
        if (inputLayout != Layout::C && inputLayout != Layout::NC && inputLayout != Layout::CN &&
            inputLayout != Layout::CHW && inputLayout != Layout::NCHW) {
            THROW_GNA_EXCEPTION << "Expected input blob to have Layout::C, Layout::NC, Layout::CN, Layout::NCHW or Layout::CHW. But was: "
                                << input.second->getTensorDesc().getLayout();
        }

        if (inputLayout == Layout::NCHW || inputLayout == Layout::CHW) {
            // specific case that can be squeezed to 2d
            inputLayout = Layout::NC;
        }

        auto is1D = input.second->getTensorDesc().getLayout() == Layout::C;
        auto is3D = input.second->getTensorDesc().getLayout() == Layout::CHW;

        if (inputs_ptr_->at(input.first).ptrs.empty()) {
            // should not happen in user code however might happen if there any non executable network based integration of GNAPlugin instance
            THROW_GNA_EXCEPTION << "network not loaded : input pointer for " << input.first << " not set";
        }

        if (inputs_ptr_->at(input.first).ptrs[idx] == nullptr) {
            // should not happen in user code however might happen if there any non executable network based integration of GNAPlugin instance
            THROW_GNA_EXCEPTION << "network not loaded : input pointer for (" << input.first << " at inferRequest #"
                                << idx << " not set";
        }
        const auto inputOrientation = inputs_ptr_->at(input.first).orientation;
        if (inputOrientation == kDnnUnknownOrientation) {
            // should not happen in user code however might happen if there any non executable network based integration of GNAPlugin instance
            THROW_GNA_EXCEPTION << "network not loaded : input orientation for " << input.first << " not set";
        }

        for (auto& output : outputs_.Get()) {
            if (output.orientation == kDnnUnknownOrientation) {
                // should not happen in user code however might happen if there any non executable network based integration of GNAPlugin instance
                THROW_GNA_EXCEPTION << "network not loaded : output orientation not set";
            }
        }

        auto dims = input.second->getTensorDesc().getDims();
        auto  importedElements = is1D ? dims[0] : details::product(++std::begin(dims), std::end(dims));
        auto  importedFrames = (is3D || is1D) ? 1 : dims[0];
        auto  targetGroups = is1D ? 1 : dims[0]; // TODO: no proper support for groups yet

        auto  importedElementSizeBytes = gnaFlags->sw_fp32 ? 4 : (gnaFlags->input_low_precision ? 1 : 2);
        auto  importedBytes = importedElements * importedFrames * importedElementSizeBytes;

        if (inputs_ptr_->at(input.first).get_required_size() < importedBytes) {
            THROW_GNA_EXCEPTION << "Cannot import input frames for :" << input.first
                                  << ", allocated size: " << inputs_ptr_->at(input.first).get_required_size()
                                  << ", but input blob size: " << importedBytes;
        }

        ImportFrames(inputs_ptr_->at(input.first).ptrs[idx],
                     input.second->cbuffer().as<float *>(),
                     input.second->getTensorDesc().getPrecision(),
                     gnaFlags->sw_fp32 ? GNAPluginNS::kScaleFactorDefault : inputs_ptr_->at(input.first).scale_factor,
                     inputOrientation,
                     importedFrames,
                     targetGroups,
                     importedElements,
                     importedElements);

        auto transpose_info = transpose_inputs_info.find(input.first);
        if (transpose_info != std::end(transpose_inputs_info)) {
            size_t batchSize = (dims.size() > 1) ? dims[0] : 1;
            size_t elementsPerBatch = (dims.size() > 1) ? InferenceEngine::details::product(dims) / dims[0] : dims[0];
            size_t transposed_data_size = 0;
            for (const auto &part_transposition_info : transpose_info->second) {
                transposed_data_size += part_transposition_info.num_transpose_rows * part_transposition_info.num_transpose_columns;
            }
            if (elementsPerBatch != transposed_data_size) {
                THROW_GNA_EXCEPTION << "Transposed data size (" << transposed_data_size
                                    << ") do not match input buffer length of " << elementsPerBatch;
            }
            auto input_ptr = reinterpret_cast<uint8_t *>(inputs_ptr_->at(input.first).ptrs[idx]);
            ConvertTensorFromNCHWToNHWC(gnadevice ? 2 : 4, batchSize, elementsPerBatch, input_ptr, true, transpose_info->second);
        }
        ++inputNum;
    }
    // If there is no gnadevice infer using reference FP32 transforamtions
    if (!gnadevice || trivialTopology) {
        auto runtime = runtime::FP(dnn);
        runtime.infer();
        if (freeNnet != nnets.end()) {
            std::get<1>(*freeNnet) = 1;
        }
    } else {
        const auto reqConfigId = std::get<0>(*freeNnet);
        if (ptr_active_indices != nullptr && num_active_indices > 0 && activeLayerIndex != 0xffffffff)
            gnadevice->setUpActiveList(reqConfigId, activeLayerIndex, ptr_active_indices, num_active_indices);
        std::get<1>(*freeNnet) = gnadevice->propagate(reqConfigId, config.pluginGna2AccMode);
    }

#ifdef PLOT
    dnn->BeginNewWrite(dnn_dump_write_index);
    if (dnn->num_components() != 0) {
        dnn->WriteDnnText("Net_.txt", kDnnFloat);
    }
    dnn_dump_write_index++;
#endif
    if (freeNnet != nnets.end()) {
        // TODO: GNA2: Substitute properly when using GNA 2.0 Library setting and CPU
        std::get<2>(*freeNnet) = result;
    }
    return idx;
}

bool GNAPlugin::Wait(uint32_t request_idx) {
    return GNA_REQUEST_COMPLETED == WaitFor(request_idx, MAX_TIMEOUT);
}

GnaWaitStatus GNAPlugin::WaitFor(uint32_t request_idx, int64_t millisTimeout) {
    auto& nnets = gnaRequestConfigToRequestIdMap;
    // TODO: GNA2: check whether necessary
    if (nnets.size() <= request_idx) return GNA_REQUEST_COMPLETED;
    // already synced TODO: might be copy required ???
    if (std::get<1>(nnets[request_idx]) == -1) return GNA_REQUEST_COMPLETED;

    if (gnadevice && !trivialTopology) {
        const auto waitStatus = gnadevice->wait(std::get<1>(nnets[request_idx]), millisTimeout);
        if (waitStatus == GNA_REQUEST_ABORTED) {
            std::get<1>(nnets[request_idx]) = -1;
            return GNA_REQUEST_ABORTED;
        }
        if (waitStatus == GNA_REQUEST_PENDING) {
            return GNA_REQUEST_PENDING;
        }
    }

    std::get<1>(nnets[request_idx]) = -1;
    auto &request = std::get<2>(nnets[request_idx]);
#ifdef PLOT
    if (dnn->num_components() != 0) {
        dnn->WriteInputAndOutputText();
    }
    dnn->WriteInputAndOutputTextGNA(std::get<0>(gnaModels[request_idx])->obj);
#endif
    int output_idx = 0;
    for (auto && outputBlobIt : request) {
        auto & outputBlob = outputBlobIt.second;
        auto & outputDesc = outputs_.at(outputBlobIt.first);
        if (outputBlob->getTensorDesc().getLayout() != Layout::C && outputBlob->getTensorDesc().getLayout() != Layout::NC &&
            outputBlob->getTensorDesc().getLayout() != Layout::CN && outputBlob->getTensorDesc().getLayout() != Layout::NCHW &&
            outputBlob->getTensorDesc().getLayout() != Layout::CHW && outputBlob->getTensorDesc().getLayout() != Layout::SCALAR) {
            THROW_GNA_EXCEPTION << "Expected output blob to have Layout::C, Layout::NC, Layout::CN, Layout::NCHW or Layout::CHW. But was "
                << outputBlob->getTensorDesc().getLayout();
        }

        auto dims = outputBlob->getTensorDesc().getDims();
        auto is1D = outputBlob->getTensorDesc().getLayout() == Layout::C;
        auto isScalar = outputBlob->getTensorDesc().getLayout() == Layout::SCALAR;
        auto is3D = outputBlob->getTensorDesc().getLayout() == Layout::CHW;
        auto batchSize = (is1D || isScalar || is3D) ? 1 : dims[0];
        auto elementsPerBatch = isScalar ? 1 : (is1D ? dims.front() : details::product(++std::begin(dims), std::end(dims)));

        auto transpose_output_info = transpose_outputs_info.find(outputBlobIt.first);
        if (transpose_output_info != std::end(transpose_outputs_info) && FoundPartToTranspose(transpose_output_info->second)) {
            size_t transposed_data_size = 0;
            for (const auto &part_transposition_info : transpose_output_info->second) {
                transposed_data_size += part_transposition_info.num_transpose_rows * part_transposition_info.num_transpose_columns;
            }
            if (elementsPerBatch != transposed_data_size) {
                THROW_GNA_EXCEPTION << "Transposed data size (" << transposed_data_size
                                    << ") do not match output buffer length of " << elementsPerBatch;
            }
            ConvertTensorFromNCHWToNHWC(outputDesc.num_bytes_per_element,
                                        batchSize,
                                        elementsPerBatch,
                                        reinterpret_cast<uint8_t*>(outputDesc.ptrs[request_idx]),
                                        true,
                                        transpose_output_info->second);
        }

        ExportScores(outputBlob->buffer(),
                        outputDesc.ptrs[request_idx],
                        outputDesc.orientation,
                        batchSize,
                        batchSize,
                        elementsPerBatch,
                        elementsPerBatch,
                        elementsPerBatch,
                        outputDesc.num_bytes_per_element,
                        sizeof(float));

        if (gnadevice) {
#ifdef PLOT
            FILE* f = nullptr;
            static int num_infers = 0;
            {
                f = fopen("ex_scores.txt", "w");
            }
            num_infers++;
            if (f) {
                if (isScalar) {
                    fprintf(f, "%d ", outputBlob->cbuffer().as<int32_t*>()[0]);
                } else {
                    for (int i = 0; i < batchSize; i++) {
                        for (int j = 0; j < dims[dims.size() - 1]; j++) {
                            fprintf(f, "%d ", outputBlob->cbuffer().as<int32_t*>()[dims[dims.size() - 1] * i + j]);
                        }
                        fprintf(f, "\n");
                    }
                }
                fprintf(f, "\n\n");
            }
#endif
            switch (outputBlob->getTensorDesc().getPrecision()) {
            case InferenceEngine::Precision::FP32 :
                UnscaleAndCast(outputBlob->buffer().as<float*>(),
                              outputBlob->buffer().as<int32_t*>(),
                              elementsPerBatch, batchSize, outputDesc.scale_factor);
                break;

            case InferenceEngine::Precision::I32 :
                UnscaleAndCast(outputBlob->buffer().as<int32_t*>(),
                              outputBlob->buffer().as<int32_t*>(),
                              elementsPerBatch, batchSize, outputDesc.scale_factor);
                break;

            default:
                THROW_GNA_EXCEPTION << "Unsupported target precision: " << outputBlob->getTensorDesc().getPrecision() << std::endl;
                break;
            }

#ifdef PLOT
            if (f) {
                if (isScalar) {
                    fprintf(f, "%.2f ", outputBlob->cbuffer().as<float*>()[0]);
                } else {
                    auto dims = outputBlob->getTensorDesc().getDims();
                    for (int i = 0; i < batchSize; i++) {
                        for (int j = 0; j < dims[dims.size() - 1]; j++) {
                            fprintf(f, "%.2f ", outputBlob->cbuffer().as<float*>()[dims[dims.size() - 1] * i + j]);
                        }
                        fprintf(f, "\n");
                    }
                }
                fclose(f);
            }
#endif
        }

        output_idx++;
    }
    return GNA_REQUEST_COMPLETED;
}

void GNAPlugin::Reset() {
    graphCompiler.Reset();
}

bool GNAPlugin::Infer(const InferenceEngine::Blob &input, InferenceEngine::Blob &output) {
    BlobMap bmInput;
    BlobMap bmOutput;
    if (inputs_data_map_.size() != 1) {
        THROW_GNA_EXCEPTION << "cannot infer using Infer(Blob&, Blob&)"<< "model accepts " << inputs_data_map_.size() << " inputs";
    }

    IE_ASSERT(!inputs_data_map_.empty());
    bmInput[inputs_data_map_.begin()->first] = std::shared_ptr<Blob>(const_cast<Blob*>(&input), [](Blob*){});
    IE_ASSERT(!outputs_data_map_.empty());
    bmOutput[outputs_data_map_.begin()->first] = std::shared_ptr<Blob>(&output, [](Blob*){});
    return Infer(bmInput, bmOutput);
}

bool GNAPlugin::Infer(const InferenceEngine::BlobMap &input, InferenceEngine::BlobMap &result) {
    return  Wait(QueueInference(input, result));
}

static InferenceEngine::Layout GetLayoutForDims(const InferenceEngine::SizeVector &dims) {
    switch (dims.size()) {
    case 0: return SCALAR;
    case 1: return C;
    case 2: return NC;
    case 3: return CHW;
    case 4: return NCHW;
    default:
        THROW_GNA_EXCEPTION << "Unsupported dimensions size in GNA: " << dims.size();
    }
}

Blob::Ptr GNAPlugin::GetOutputBlob(const std::string& name, InferenceEngine::Precision precision) {
    // need to have intermediate blob for interleave conversion
    InferenceEngine::Blob::Ptr outputBlob;
    auto outputDataIt = outputs_data_map_.find(name);
    if (outputDataIt == std::end(outputs_data_map_)) {
        THROW_GNA_EXCEPTION << "Output " << name << " isn't found";
    }
    auto outputDims = outputDataIt->second->getTensorDesc().getDims();
    outputBlob = make_blob_with_precision(TensorDesc(precision, outputDims, GetLayoutForDims(outputDims)));
    outputBlob->allocate();
    return outputBlob;
}

Blob::Ptr GNAPlugin::GetInputBlob(const std::string& name, InferenceEngine::Precision precision) {
    InferenceEngine::Blob::Ptr inputBlob;
    // need to have intermediate blob for interleave conversion
    // TODO: NCHW format support is experimental = c++ MO did insert reshape, while TF mo - not
    auto inputDataIt = inputs_data_map_.find(name);
    if (inputDataIt == std::end(inputs_data_map_)) {
        THROW_GNA_EXCEPTION << "Input " << name << " isn't found";
    }
    auto inputDims = inputDataIt->second->getTensorDesc().getDims();
    inputBlob = make_blob_with_precision(TensorDesc(precision, inputDims, GetLayoutForDims(inputDims)));
    inputBlob->allocate();
    return inputBlob;
}

std::vector<InferenceEngine::IVariableStateInternal::Ptr>  GNAPlugin::QueryState() {
    if (memoryStates.size() != graphCompiler.memory_connection.size()) {
        memoryStates.clear();
        for (auto& connection : graphCompiler.memory_connection) {
            auto state = std::make_shared<memory::GNAVariableState>(connection.first, std::make_shared <GNAMemoryLayer>(connection.second));
            memoryStates.emplace_back(state);
        }
    }
    return memoryStates;
}

std::string GNAPlugin::GetName() const noexcept {
    return _pluginName;
}

void GNAPlugin::SetName(const std::string & pluginName) noexcept {
    _pluginName = pluginName;
}

InferenceEngine::IExecutableNetworkInternal::Ptr GNAPlugin::ImportNetwork(std::istream& networkModel) {
    auto header = GNAModelSerial::ReadHeader(networkModel);

    InitGNADevice();

    graphCompiler.setGNAMemoryPtr(gnamem);
    void *basePtr = nullptr;
    gnamem->reserve_ptr(nullptr, &basePtr, header.gnaMemSize);
    gnamem->commit();
    gnaModels.push_back(std::make_tuple(make_shared<CPPWrapper<Gna2Model>>(header.layersCount)));
    GNAModelSerial::MemoryType  mt;
    auto serial = GNAModelSerial(&std::get<0>(gnaModels.back())->obj, mt);

    serial.setHeader(header);
    serial.Import(basePtr,
            header.gnaMemSize,
            networkModel,
            *(inputs_ptr_),
            outputs_,
            transpose_inputs_info,
            transpose_outputs_info);

    SetNetworkInputs();
    SetNetworkOutputs();

    // If scale factors are defined in configuration we still need to use them instead of imported values,
    // for example to change the scale factors for the old models.
    if (!config.inputScaleFactors.empty()) {
        IE_ASSERT(config.inputScaleFactors.size() <= inputs_ptr_->size());
        // TODO: config should  use the map of inputs as well
        for (size_t id = 0; id < config.inputScaleFactors.size(); ++id) {
            if (id < inputs_ptr_->size() && config.inputScaleFactors[id] != GNAPluginNS::kScaleFactorDefault) {
                gnalog() << "[Import Network] Using input scale factor defined in configuration for input " << id << std::endl;
                inputs_ptr_->Get().at(id).scale_factor = config.inputScaleFactors[id];
            }
        }
    }

    auto getOrientation = [](Gna2Operation & gnaOperation) {
        return gnaOperation.Type == Gna2OperationTypeConvolution ?
            kDnnNonInterleavedOrientation : kDnnInterleavedOrientation;
    };
    (void)getOrientation;

    if (header.doRotateInput) {
        for (auto && input : inputs_data_map_) {
            transpose_inputs_info.insert({input.first, {{header.doRotateInput, header.nRotateRows, header.nRotateColumns}}});
        }
    }
    if (header.doRotateOutput) {
        for (auto && output : outputs_data_map_) {
            transpose_outputs_info.insert({output.first, {{header.doRotateOutput, header.nRotateOutputRows, header.nRotateOutputColumns}}});
        }
    }

    for (auto && memory : mt) {
        GNAMemoryLayer memoryLayer(nullptr, nullptr, gnaFlags->sw_fp32 ? 4 : 2);
        std::string name;
        std::tie(memoryLayer.gna_ptr, memoryLayer.reserved_size, name, memoryLayer.scale_factor) = memory;
        graphCompiler.memory_connection.emplace_back(make_pair(name, memoryLayer));
    }

    DumpXNNToFile();

#ifdef PLOT
    dnn->WriteGraphWizModel("gna-blob-imported.dot");
#endif
    trivialTopology = (std::get<0>(gnaModels.back())->obj.NumberOfOperations == 0);
    createRequestConfigsForGnaModels();
    return {};
}

void GNAPlugin::Export(const std::string &fileName) {
    std::fstream outStream(fileName, ios_base::out | ios_base::binary);
    Export(outStream);
}

void GNAPlugin::Export(std::ostream &outStream) {
    if (inputs_ptr_->empty() || outputs_.empty()) {
        THROW_GNA_EXCEPTION << " network not loaded";
    }

    // TODO: nnet group parameter looks only used in application - so can we move this line into load network.
    IE_ASSERT(!inputs_data_map_.empty());
    auto inputDims = inputs_data_map_.begin()->second->getTensorDesc().getDims();

    Gna2Model* modelToSerial = &std::get<0>(gnaModels.front())->obj;
    auto serial = GNAModelSerial(modelToSerial,
                                 *(inputs_ptr_),
                                 outputs_)
                    .SetInputRotation(transpose_inputs_info)
                    .SetOutputRotation(transpose_outputs_info);

    for (auto && memoryConnection : graphCompiler.memory_connection) {
        auto state = std::make_shared<memory::GNAVariableState>(memoryConnection.first, std::make_shared <GNAMemoryLayer>(memoryConnection.second));
        gnalog() << "Scale factor Memory layer " << state->GetScaleFactor() << std::endl;
        serial.AddState(memoryConnection.second.gna_ptr, memoryConnection.second.reserved_size, memoryConnection.first, state->GetScaleFactor());
    }

    serial.Export(gnamem->getBasePtr(), gnamem->getTotalBytes(), outStream);
}

std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> GNAPlugin::GetPerformanceCounts() {
    if (gnaFlags->performance_counting) {
        std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> perfMap;
        gnadevice->getGnaPerfCounters(perfMap);
        return perfMap;
    } else {
        return {};
    }
}

void GNAPlugin::AddExtension(const InferenceEngine::IExtensionPtr& extension) {}

void GNAPlugin::SetConfig(const std::map<std::string, std::string> &config_map) {
    config.UpdateFromMap(config_map);
    UpdateFieldsFromConfig();
}

void GNAPlugin::UpdateFieldsFromConfig() {
    *gnaFlags = config.gnaFlags;
}

void GNAPlugin::SetNetworkInputs() {
    inputs_data_map_.clear();
    for (auto & input : inputs_ptr_->Get()) {
        inputs_data_map_[input.name] = input.ToIEInputInfo();
    }
}

void GNAPlugin::SetNetworkOutputs() {
    outputs_data_map_.clear();
    for (auto & output : outputs_.Get()) {
        outputs_data_map_[output.name] = output.to_ie_data();
    }
}

std::vector<std::shared_ptr<const ov::Node>> GNAPlugin::GetInputs() {
    std::vector<std::shared_ptr<const ov::Node>> params;
    params.reserve(inputs_ptr_->size());
    for (auto&& input : inputs_ptr_->Get()) {
        auto param = std::make_shared<ov::op::v0::Parameter>(
            convertPrecision(input.model_precision),
            ov::PartialShape(input.dims));
        param->set_friendly_name(input.name);
        param->get_output_tensor(0).add_names(input.tensor_names);
        params.emplace_back(move(param));
    }
    return params;
}

std::vector<std::shared_ptr<const ov::Node>> GNAPlugin::GetOutputs() {
    std::vector<std::shared_ptr<const ov::Node>> results;
    results.reserve(outputs_.size());
    for (auto&& output : outputs_.Get()) {
        auto param = std::make_shared<ov::op::v0::Parameter>(
            convertPrecision(output.model_precision),
            ov::PartialShape(output.dims));
        param->set_friendly_name(output.name);
        auto result = std::make_shared<ov::op::v0::Result>(param);
        result->get_output_tensor(0).add_names(output.tensor_names);
        results.emplace_back(std::move(result));
    }
    return results;
}

InferenceEngine::QueryNetworkResult GNAPlugin::QueryNetwork(const InferenceEngine::CNNNetwork& network,
                                                            const std::map<std::string, std::string>& config) const {
    InferenceEngine::QueryNetworkResult res;

    if (network.getFunction()) {
        IE_THROW(NotImplemented) << " ngraph::Function is not supported natively";
    }

    std::unordered_set<CNNLayer *> allLayers;
    InferenceEngine::InputsDataMap inputs = network.getInputsInfo();
    std::vector<CNNLayerPtr> sortedLayers = CNNNetSortTopologically(network);

    if (inputs.empty()) {
        THROW_GNA_EXCEPTION << "Network is empty (GNA)\n";
    }

    auto const & secondLayers = getInputTo(inputs.begin()->second->getInputData());
    if (secondLayers.empty()) {
        THROW_GNA_EXCEPTION << "Network consists of input layer only (GNA)\n";
    }

    InferenceEngine::details::UnorderedDFS(allLayers,
                                           secondLayers.begin()->second,
                                           [&](CNNLayerPtr const& layer) {
                                                if (LayerTypeFromStr(layer->type) != LayerType::NO_TYPE) {
                                                    res.supportedLayersMap.insert({ layer->name, GetName() });
                                                }
                                            }, false);

    return res;
}
