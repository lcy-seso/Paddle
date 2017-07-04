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

#include "LayerNormalizationLayer.h"
#include "paddle/utils/Stat.h"

namespace paddle {

REGISTER_LAYER(layer_norm, LayerNormalizationLayer);

const real LayerNormalizationLayer::EPS = 1E-5;

bool LayerNormalizationLayer::init(const LayerMap& layerMap,
                                   const ParameterMap& parameterMap) {
  if (!Layer::init(layerMap, parameterMap)) return false;

  CHECK_EQ(inputLayers_.size(), 1U);
  CHECK_EQ(inputLayers_.size(), parameters_.size());
  CHECK_EQ(inputLayers_.size(), size_t(config_.inputs_size()));

  weight_.reset(new Weight(1, getSize(), parameters_[0]));

  if (biasParameter_.get() != NULL) {
    biases_ = std::unique_ptr<Weight>(new Weight(1, getSize(), biasParameter_));
  }
  return true;
}

void LayerNormalizationLayer::forward(PassType passType) {
  Layer::forward(passType);

  /* activation */ {
    REGISTER_TIMER_INFO("FwAtvTimer", getName().c_str());
    forwardActivation();
  }
}

void LayerNormalizationLayer::backward(const UpdateCallback& callback) {
  /* Do derivation */ {
    REGISTER_TIMER_INFO("BpAvtTimer", getName().c_str());
    backwardActivation();
  }

  // compute derivatives.
  if (biases_ && biases_->getWGrad()) {
    REGISTER_TIMER_INFO("BpBiasTimer", getName().c_str());

    /* Increasing the number of gradient */
    biases_->getParameterPtr()->incUpdate(callback);
  }
  if (weight_->getWGrad()) {
  }

  // compute input gradients.
  {
    REGISTER_TIMER_INFO("WeightUpdate", getName().c_str());
    weight_->getParameterPtr()->incUpdate(callback);
  }
}

}  // namespace paddle
