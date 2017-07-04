#!/usr/bin/env python
#coding=utf-8
from paddle.trainer_config_helpers import *

data = data_layer(name='input', size=300)
hidden = fc_layer(input=data, size=100, bias_attr=False)

layer_norm = layer_norm_layer(input=hidden, act=TanhActivation())

outputs(layer_norm)
