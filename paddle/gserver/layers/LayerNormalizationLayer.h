/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "Layer.h"

namespace paddle {
/**
 *
 * The config file api is layer_norm_layer.
 */

class LayerNormalizationLayer : public Layer {
public:
  explicit LayerNormalizationLayer(const LayerConfig& config) : Layer(config) {}
  ~LayerNormalizationLayer() {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;
  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback = nullptr) override;

protected:
  static const real EPS;
  std::unique_ptr<Weight> weight_;
  std::unique_ptr<Weight> biases_;
};

}  // namespace paddle
