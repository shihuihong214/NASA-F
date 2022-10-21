import os
import math
import copy
import torch.nn as nn
import numpy as np
# from thop import profile
import torch
from distutils.version import LooseVersion
from thop.vision.basic_hooks import *
from thop.rnn_hooks import *

OPs_list = []
OPs_list.append({"idx": 0 , "type": "Conv", "kernel_size": 3, "stride": 1, "padding": 1, "input_H": 32, "input_W": 32, "input_C": 16, "output_E": 32, "output_F": 32, "output_M": 16})
OPs_list.append({"idx": 1 , "type": "Conv", "kernel_size": 3, "stride": 1, "padding": 1, "input_H": 16, "input_W": 16, "input_C": 32, "output_E": 16, "output_F": 16, "output_M": 32})
OPs_list.append({"idx": 2 , "type": "Conv", "kernel_size": 3, "stride": 1, "padding": 1, "input_H": 8 , "input_W": 8 , "input_C": 64, "output_E": 8 , "output_F": 8 , "output_M": 64})
OPs_list.append({"idx": 3 , "type": "Conv", "kernel_size": 3, "stride": 1, "padding": 1, "input_H": 16, "input_W": 16, "input_C": 16, "output_E": 16, "output_F": 16, "output_M": 16})
OPs_list.append({"idx": 4 , "type": "Conv", "kernel_size": 3, "stride": 1, "padding": 1, "input_H": 8 , "input_W": 8 , "input_C": 32, "output_E": 8 , "output_F": 8 , "output_M": 32})
OPs_list.append({"idx": 5 , "type": "Conv", "kernel_size": 3, "stride": 1, "padding": 1, "input_H": 4 , "input_W": 4 , "input_C": 64, "output_E": 4 , "output_F": 4 , "output_M": 64})
# Cell: 1*1 Conv
OPs_list.append({"idx": 6 , "type": "Conv", "kernel_size": 1, "stride": 1, "padding": 0, "input_H": 32, "input_W": 32, "input_C": 16, "output_E": 32, "output_F": 32, "output_M": 16})
OPs_list.append({"idx": 7 , "type": "Conv", "kernel_size": 1, "stride": 1, "padding": 0, "input_H": 16, "input_W": 16, "input_C": 32, "output_E": 16, "output_F": 16, "output_M": 32})
OPs_list.append({"idx": 8 , "type": "Conv", "kernel_size": 1, "stride": 1, "padding": 0, "input_H": 8 , "input_W": 8 , "input_C": 64, "output_E": 8 , "output_F": 8 , "output_M": 64})
OPs_list.append({"idx": 9 , "type": "Conv", "kernel_size": 1, "stride": 1, "padding": 0, "input_H": 16, "input_W": 16, "input_C": 16, "output_E": 16, "output_F": 16, "output_M": 16})
OPs_list.append({"idx": 10, "type": "Conv", "kernel_size": 1, "stride": 1, "padding": 0, "input_H": 8 , "input_W": 8 , "input_C": 32, "output_E": 8 , "output_F": 8 , "output_M": 32})
OPs_list.append({"idx": 11, "type": "Conv", "kernel_size": 1, "stride": 1, "padding": 0, "input_H": 4 , "input_W": 4 , "input_C": 64, "output_E": 4 , "output_F": 4 , "output_M": 64})
# Cell: 3*3 AvgP
OPs_list.append({"idx": 12, "type": "AvgP", "kernel_size": 3, "stride": 1, "padding": 1, "input_H": 32, "input_W": 32, "input_C": 16, "output_E": 32, "output_F": 32, "output_M": 16})
OPs_list.append({"idx": 13, "type": "AvgP", "kernel_size": 3, "stride": 1, "padding": 1, "input_H": 16, "input_W": 16, "input_C": 32, "output_E": 16, "output_F": 16, "output_M": 32})
OPs_list.append({"idx": 14, "type": "AvgP", "kernel_size": 3, "stride": 1, "padding": 1, "input_H": 8 , "input_W": 8 , "input_C": 64, "output_E": 8 , "output_F": 8 , "output_M": 64})
OPs_list.append({"idx": 15, "type": "AvgP", "kernel_size": 3, "stride": 1, "padding": 1, "input_H": 16, "input_W": 16, "input_C": 16, "output_E": 16, "output_F": 16, "output_M": 16})
OPs_list.append({"idx": 16, "type": "AvgP", "kernel_size": 3, "stride": 1, "padding": 1, "input_H": 8 , "input_W": 8 , "input_C": 32, "output_E": 8 , "output_F": 8 , "output_M": 32})
OPs_list.append({"idx": 17, "type": "AvgP", "kernel_size": 3, "stride": 1, "padding": 1, "input_H": 4 , "input_W": 4 , "input_C": 64, "output_E": 4 , "output_F": 4 , "output_M": 64})
# Res1 - CIFAR
OPs_list.append({"idx": 18, "type": "Conv", "kernel_size": 3, "stride": 2, "padding": 1, "input_H": 32, "input_W": 32, "input_C": 16, "output_E": 16, "output_F": 16, "output_M": 32})
OPs_list.append({"idx": 19, "type": "Conv", "kernel_size": 3, "stride": 1, "padding": 1, "input_H": 16, "input_W": 16, "input_C": 32, "output_E": 16, "output_F": 16, "output_M": 32})
OPs_list.append({"idx": 20, "type": "AvgP", "kernel_size": 2, "stride": 2, "padding": 0, "input_H": 32, "input_W": 32, "input_C": 16, "output_E": 16, "output_F": 16, "output_M": 16})
OPs_list.append({"idx": 21, "type": "Conv", "kernel_size": 1, "stride": 1, "padding": 0, "input_H": 16, "input_W": 16, "input_C": 16, "output_E": 16, "output_F": 16, "output_M": 32})
# Res2 - CIFAR
OPs_list.append({"idx": 22, "type": "Conv", "kernel_size": 3, "stride": 2, "padding": 1, "input_H": 16, "input_W": 16, "input_C": 32, "output_E": 8 , "output_F": 8 , "output_M": 64})
OPs_list.append({"idx": 23, "type": "Conv", "kernel_size": 3, "stride": 1, "padding": 1, "input_H": 8 , "input_W": 8 , "input_C": 64, "output_E": 8 , "output_F": 8 , "output_M": 64})
OPs_list.append({"idx": 24, "type": "AvgP", "kernel_size": 2, "stride": 2, "padding": 0, "input_H": 16, "input_W": 16, "input_C": 32, "output_E": 8 , "output_F": 8 , "output_M": 32})
OPs_list.append({"idx": 25, "type": "Conv", "kernel_size": 1, "stride": 1, "padding": 0, "input_H": 8 , "input_W": 8 , "input_C": 32, "output_E": 8 , "output_F": 8 , "output_M": 64})
# Res1 - ImageNet16-120
OPs_list.append({"idx": 26, "type": "Conv", "kernel_size": 3, "stride": 2, "padding": 1, "input_H": 16, "input_W": 16, "input_C": 16, "output_E": 8 , "output_F": 8 , "output_M": 32})
OPs_list.append({"idx": 27, "type": "Conv", "kernel_size": 3, "stride": 1, "padding": 1, "input_H": 8 , "input_W": 8 , "input_C": 32, "output_E": 8 , "output_F": 8 , "output_M": 32})
OPs_list.append({"idx": 28, "type": "AvgP", "kernel_size": 2, "stride": 2, "padding": 0, "input_H": 16, "input_W": 16, "input_C": 16, "output_E": 8 , "output_F": 8 , "output_M": 16})
OPs_list.append({"idx": 29, "type": "Conv", "kernel_size": 1, "stride": 1, "padding": 0, "input_H": 8 , "input_W": 8 , "input_C": 16, "output_E": 8 , "output_F": 8 , "output_M": 32})
# Res2 - ImageNet16-120
OPs_list.append({"idx": 30, "type": "Conv", "kernel_size": 3, "stride": 2, "padding": 1, "input_H": 8 , "input_W": 8 , "input_C": 32, "output_E": 4 , "output_F": 4 , "output_M": 64})
OPs_list.append({"idx": 31, "type": "Conv", "kernel_size": 3, "stride": 1, "padding": 1, "input_H": 4 , "input_W": 4 , "input_C": 64, "output_E": 4 , "output_F": 4 , "output_M": 64})
OPs_list.append({"idx": 32, "type": "AvgP", "kernel_size": 2, "stride": 2, "padding": 0, "input_H": 8 , "input_W": 8 , "input_C": 32, "output_E": 4 , "output_F": 4 , "output_M": 32})
OPs_list.append({"idx": 33, "type": "Conv", "kernel_size": 1, "stride": 1, "padding": 0, "input_H": 4 , "input_W": 4 , "input_C": 32, "output_E": 4 , "output_F": 4 , "output_M": 64})
# FC Head
OPs_list.append({"idx": 34, "type": "FC"  , "kernel_size": 1, "stride": 1, "padding": 0, "input_H": 1 , "input_W": 1 , "input_C": 64, "output_E": 1 , "output_F": 1 , "output_M": 10})
OPs_list.append({"idx": 35, "type": "FC"  , "kernel_size": 1, "stride": 1, "padding": 0, "input_H": 1 , "input_W": 1 , "input_C": 64, "output_E": 1 , "output_F": 1 , "output_M":100})
OPs_list.append({"idx": 36, "type": "FC"  , "kernel_size": 1, "stride": 1, "padding": 0, "input_H": 1 , "input_W": 1 , "input_C": 64, "output_E": 1 , "output_F": 1 , "output_M":120})



register_hooks = {
    nn.ZeroPad2d: zero_ops,  # padding does not involve any multiplication.

    nn.Conv1d: count_convNd,
    nn.Conv2d: count_convNd,
    nn.Conv3d: count_convNd,
    nn.ConvTranspose1d: count_convNd,
    nn.ConvTranspose2d: count_convNd,
    nn.ConvTranspose3d: count_convNd,

    nn.BatchNorm1d: count_bn,
    nn.BatchNorm2d: count_bn,
    nn.BatchNorm3d: count_bn,
    nn.SyncBatchNorm: count_bn,

    nn.ReLU: zero_ops,
    nn.ReLU6: zero_ops,
    nn.LeakyReLU: count_relu,

    nn.MaxPool1d: zero_ops,
    nn.MaxPool2d: zero_ops,
    nn.MaxPool3d: zero_ops,
    nn.AdaptiveMaxPool1d: zero_ops,
    nn.AdaptiveMaxPool2d: zero_ops,
    nn.AdaptiveMaxPool3d: zero_ops,

    nn.AvgPool1d: count_avgpool,
    nn.AvgPool2d: count_avgpool,
    nn.AvgPool3d: count_avgpool,
    nn.AdaptiveAvgPool1d: count_adap_avgpool,
    nn.AdaptiveAvgPool2d: count_adap_avgpool,
    nn.AdaptiveAvgPool3d: count_adap_avgpool,

    nn.Linear: count_linear,
    nn.Dropout: zero_ops,

    nn.Upsample: count_upsample,
    nn.UpsamplingBilinear2d: count_upsample,
    nn.UpsamplingNearest2d: count_upsample,

    nn.RNNCell: count_rnn_cell,
    nn.GRUCell: count_gru_cell,
    nn.LSTMCell: count_lstm_cell,
    nn.RNN: count_rnn,
    nn.GRU: count_gru,
    nn.LSTM: count_lstm,
}

def profile(model: nn.Module, inputs, custom_ops=None, verbose=True):
    handler_collection = {}
    types_collection = set()
    if custom_ops is None:
        custom_ops = {}

    def add_hooks(m: nn.Module):
        m.register_buffer('total_ops', torch.zeros(1, dtype=torch.float64))
        m.register_buffer('total_params', torch.zeros(1, dtype=torch.float64))

        # for p in m.parameters():
        #     m.total_params += torch.DoubleTensor([p.numel()])

        m_type = type(m)

        fn = None
        if m_type in custom_ops:  # if defined both op maps, use custom_ops to overwrite.
            fn = custom_ops[m_type]
            # if m_type not in types_collection and verbose:
                # print("[INFO] Customize rule %s() %s." % (fn.__qualname__, m_type))
        elif m_type in register_hooks:
            fn = register_hooks[m_type]
            # if m_type not in types_collection and verbose:
                # print("[INFO] Register %s() for %s." % (fn.__qualname__, m_type))
        else:
            if m_type not in types_collection and verbose:
                prRed("[WARN] Cannot find rule for %s. Treat it as zero Macs and zero Params." % m_type)

        if fn is not None:
            handler_collection[m] = (m.register_forward_hook(fn), m.register_forward_hook(count_parameters))
        types_collection.add(m_type)

    prev_training_status = model.training

    model.eval()
    model.apply(add_hooks)

    with torch.no_grad():
        model(*inputs)

    def dfs_count(module: nn.Module, prefix="\t") -> (int, int):
        total_ops, total_params = 0, 0
        for m in module.children():
            # if not hasattr(m, "total_ops") and not hasattr(m, "total_params"):  # and len(list(m.children())) > 0:
            #     m_ops, m_params = dfs_count(m, prefix=prefix + "\t")
            # else:
            #     m_ops, m_params = m.total_ops, m.total_params
            if m in handler_collection and not isinstance(m, (nn.Sequential, nn.ModuleList)):
                m_ops, m_params = m.total_ops.item(), m.total_params.item()
            else:
                m_ops, m_params = dfs_count(m, prefix=prefix + "\t")
            total_ops += m_ops
            total_params += m_params
        #  print(prefix, module._get_name(), (total_ops.item(), total_params.item()))
        return total_ops, total_params

    total_ops, total_params = dfs_count(model)

    # reset model to original status
    model.train(prev_training_status)
    for m, (op_handler, params_handler) in handler_collection.items():
        op_handler.remove()
        params_handler.remove()
        m._buffers.pop("total_ops")
        m._buffers.pop("total_params")

    return total_ops, total_params

def valid_tiling(tiling_factor, num_pe, memory, FC_flag):
    valid = True
    M3 = tiling_factor["M3"]
    M2 = tiling_factor["M2"]
    M1 = tiling_factor["M1"]
    M0 = tiling_factor["M0"]
    M = tiling_factor["M"]
    if not (M == M3*M2*M1*M0):
        valid = False
    if not (M0 <= 16):
        valid = False
    C3 = tiling_factor["C3"]
    C2 = tiling_factor["C2"]
    C1 = tiling_factor["C1"]
    C0 = tiling_factor["C0"]
    C = tiling_factor["C"]
    if not (C == C3*C2*C1*C0):
        valid = False
    if not (C3 == 1):
        valid = False
    E1 = tiling_factor["E1"]
    E3 = tiling_factor["E3"]
    E = tiling_factor["E"]
    if not (E == E1*E3):
        valid = False
    if not FC_flag:
        if not (E1 != 1):
            valid = False
    R = tiling_factor["R"]
    S = tiling_factor["S"]
    if not (M1*C1*E1*R < num_pe):
        valid = False
    if not (C0*S < 12):
        valid = False
    if not (M0*C0*S < 192):
        valid = False
    stride = tiling_factor["stride"]
    F = tiling_factor["F"]
    if not (C1*C0*((E1-1)*stride+R)*((F-1)*stride+S) + M2*M1*M0*E1*F < memory):
        valid = False
    return valid

def get_latency(tiling_factor, unit_latency):
    M3 = tiling_factor["M3"]
    M2 = tiling_factor["M2"]
    M1 = tiling_factor["M1"]
    M0 = tiling_factor["M0"]
    M = tiling_factor["M"]
    C3 = tiling_factor["C3"]
    C2 = tiling_factor["C2"]
    C1 = tiling_factor["C1"]
    C0 = tiling_factor["C0"]
    C = tiling_factor["C"]
    E1 = tiling_factor["E1"]
    E3 = tiling_factor["E3"]
    E = tiling_factor["E"]
    R = tiling_factor["R"]
    S = tiling_factor["S"]
    stride = tiling_factor["stride"]
    F = tiling_factor["F"]

    latency = M3*M2*M0*C3*C2*C0*E3*F*S*unit_latency
    return latency

def get_memory(tiling_factor):
    M3 = tiling_factor["M3"]
    M2 = tiling_factor["M2"]
    M1 = tiling_factor["M1"]
    M0 = tiling_factor["M0"]
    M = tiling_factor["M"]
    C3 = tiling_factor["C3"]
    C2 = tiling_factor["C2"]
    C1 = tiling_factor["C1"]
    C0 = tiling_factor["C0"]
    C = tiling_factor["C"]
    E1 = tiling_factor["E1"]
    E3 = tiling_factor["E3"]
    E = tiling_factor["E"]
    R = tiling_factor["R"]
    S = tiling_factor["S"]
    stride = tiling_factor["stride"]
    F = tiling_factor["F"]

    memory = C1*C0*((E1-1)*stride+R)*((F-1)*stride+S) + M2*M1*M0*E1*F
    return memory

def get_energy(tiling_factor, unit_energy, bit):
    M3 = tiling_factor["M3"]
    M2 = tiling_factor["M2"]
    M1 = tiling_factor["M1"]
    M0 = tiling_factor["M0"]
    M = tiling_factor["M"]
    C3 = tiling_factor["C3"]
    C2 = tiling_factor["C2"]
    C1 = tiling_factor["C1"]
    C0 = tiling_factor["C0"]
    C = tiling_factor["C"]
    E1 = tiling_factor["E1"]
    E3 = tiling_factor["E3"]
    E = tiling_factor["E"]
    R = tiling_factor["R"]
    S = tiling_factor["S"]
    stride = tiling_factor["stride"]
    F = tiling_factor["F"]

    H = tiling_factor["H"]
    W = tiling_factor["W"]
    num_ifmap = H*W*C # input feature map size
    num_weight = R*S*C*M # weight size
    num_ofmap = E*F*M # output size

    computation = E*F*M*R*S*C
    DRAM_ifmap = M3 * num_ifmap
    DRAM_weight = E3 * num_weight
    DRAM_ofmap = ( max((2*C3-1), 1) ) * num_ofmap
    GB_ifmap = M3 * M2 * num_ifmap
    GB_ofmap = ( max(2 * C3 * (C2-1), 1) ) * num_ofmap
    NoC_ifmap = M3 * M2 * M1 * R * E / H * num_ifmap
    NoC_weight = E3 * E1 * num_weight
    NoC_ofmap = ( max(C3 * C2 * (C1 * R - 1), 1) ) * num_ofmap
    RF_ifmap = M3 * M2 * M1 * R * E / H * M0 * S * F / W * num_ifmap
    RF_weight = E3 * E1 * F * num_weight
    RF_ofmap = ( max(C3 * C2 * C1 * R * (C0 * S - 1 ) * 2, 1) ) * num_ofmap
    energy = computation * unit_energy["unit_comp"] \
             + (DRAM_ifmap + DRAM_weight*bit/8 + DRAM_ofmap) * unit_energy["unit_DRAM"] \
             + (DRAM_ifmap + DRAM_ofmap) * unit_energy["unit_DRAM_GB"] \
             + (GB_ifmap + GB_ofmap) * unit_energy["unit_GB"] \
             + (NoC_ifmap + NoC_weight*bit/8 + NoC_ofmap) * unit_energy["unit_NoC"] \
             + (RF_ifmap + RF_weight*bit/8 + RF_ofmap) * unit_energy["unit_RF"]
    return energy

# Refer: https://www.geeksforgeeks.org/print-all-prime-factors-of-a-given-number/
def primeFactors(n): 
    prime_list = []
    # Print the number of two's that divide n 
    while n % 2 == 0: 
        prime_list.append(2)
        n = n / 2
          
    # n must be odd at this point 
    # so a skip of 2 ( i = i + 2) can be used 
    for i in range(3,int(math.sqrt(n))+1,2): 
          
        # while i divides n , print i ad divide n 
        while n % i== 0: 
            prime_list.append(int(i))
            n = n / i 
              
    # Condition if n is a prime 
    # number greater than 2 
    if n > 2: 
        prime_list.append(int(n))
    return prime_list

def possible_mul(x,l):
    if len(l) == 1:
        raw_list = [x*l[0], x*1]
        clean_list = list(dict.fromkeys(raw_list))
        return clean_list
    else:
        raw_list = possible_mul(x*l[0], l[1:]) + possible_mul(x*1, l[1:])
        clean_list = list(dict.fromkeys(raw_list))
        return clean_list

def tile(num, tile_size):
    if tile_size == 1:
        return [[num]]
    else:
        if num == 1:
            prime_list = [1]
        else:
            prime_list = primeFactors(num)

        tile_list = []
        selected_list = possible_mul(1, prime_list)
        for selected in selected_list:
            # select 1 for current the first position
            for options in tile(int(num/selected), tile_size-1):
                to_append = [selected,] + options
                if to_append not in tile_list:
                    tile_list.append(to_append)
        return tile_list

# gives the energy (mJ), latency (ms)
def get_OPs_HW_metric(layer_dict, type, num_pe, memory, v=False):
    # constant defination
    if layer_dict["type"] == "FC":
        FC_flag = True
    else:
        FC_flag = False
    unit_energy = {}
    if type=='conv':
        unit_energy["unit_comp"] = 0.2/(1e9) # mJ/MAC
        bit = 8
    elif type=='shift':
        unit_energy["unit_comp"] = 0.024(1e9) # mJ/MAC
        bit = 6
    unit_energy["unit_DRAM"] = 200/(1e9)*8/16 # mJ/16 bits
    unit_energy["unit_DRAM_GB"] = 0.0/(1e9) # mJ/16 bits
    unit_energy["unit_GB"] = 6/(1e9) *3.7*8/16# mJ/16 bits
    unit_energy["unit_NoC"] = 2.0/(1e9)*3.7*8/16 # mJ/16 bits
    unit_energy["unit_RF"] = 1/(1e9) *3.7*8/16# mJ/16 bits
    unit_latency = 1.0/(250e6)*(1e3) # ms

    # Add basic information to tiling_factor
    base_tiling_factor = {}
    base_tiling_factor["H"] = layer_dict["input_H"]
    base_tiling_factor["W"] = layer_dict["input_W"]
    base_tiling_factor["C"] = layer_dict["input_C"]
    base_tiling_factor["R"] = layer_dict["kernel_size"]
    base_tiling_factor["S"] = layer_dict["kernel_size"]
    base_tiling_factor["M"] = layer_dict["output_M"]
    base_tiling_factor["E"] = layer_dict["output_E"]
    base_tiling_factor["F"] = layer_dict["output_F"]
    base_tiling_factor["stride"] = layer_dict["stride"]
    # tile M to M0 * M1 * M2 * M3
    M_tile_list = tile(base_tiling_factor["M"], 4)
    # filter out M0 > 16 options
    for tile_option in M_tile_list:
        if tile_option[0] > 16:
            M_tile_list.remove(tile_option)
    # tile C to C0 * C1 * C2 * C3
    C_tile_list = tile(base_tiling_factor["C"], 4)
    # filter out C3 != 1 options
    for tile_option in C_tile_list:
        if tile_option[3] != 1:
            C_tile_list.remove(tile_option)
    # tile E to E1 * E3
    E_tile_list = tile(base_tiling_factor["E"], 2)
    # filter out E1 == 1 options
    if not FC_flag:
        for tile_option in E_tile_list:
            if tile_option[0] == 1:
                E_tile_list.remove(tile_option)

    energy_list = []
    latency_list = []
    edp_list = []
    memory_list = []
    tiling_factor_list = []

    for M_tile in M_tile_list:
        for C_tile in C_tile_list:
            for E_tile in E_tile_list:
                tiling_factor = copy.deepcopy(base_tiling_factor)
                tiling_factor["M0"] = M_tile[0]
                tiling_factor["M1"] = M_tile[1]
                tiling_factor["M2"] = M_tile[2]
                tiling_factor["M3"] = M_tile[3]
                tiling_factor["C0"] = C_tile[0]
                tiling_factor["C1"] = C_tile[1]
                tiling_factor["C2"] = C_tile[2]
                tiling_factor["C3"] = C_tile[3]
                tiling_factor["E1"] = E_tile[0]
                tiling_factor["E3"] = E_tile[1]
                if valid_tiling(tiling_factor, num_pe, memory, FC_flag):
                    energy = get_energy(tiling_factor, unit_energy, bit)
                    latency = get_latency(tiling_factor, unit_latency)
                    memory = get_memory(tiling_factor)

                    energy_list.append(energy)
                    latency_list.append(latency)
                    edp_list.append(energy*latency)
                    memory_list.append(memory)
                    tiling_factor_list.append(tiling_factor)

    # tiling factor search M, C, E
    # max_energy = max(energy_list)
    min_energy = min(energy_list)

    # max_latency = max(latency_list)
    min_latency = min(latency_list)

    # max_edp = max(edp_list)
    min_edp = min(edp_list)

    # min_normal_metric = (energy_list[0]-min_energy)/max_energy + (latency_list[0]-min_latency)/max_latency

    # total_optimal_tiling_factor_idx = [0]
    # latency_optimal_tiling_factor_idx = []
    # energy_optimal_tiling_factor_idx = []
    edp_optimal_tiling_factor_idx = []

    for i in range(len(tiling_factor_list)):
        tiling_factor = tiling_factor_list[i]
        # energy = energy_list[i]
        # latency = latency_list[i]

        # normal_metric = (energy_list[i]-min_energy)/max_energy + (latency_list[i]-min_latency)/max_latency
        edp = edp_list[i]

        # update total optimal
        # if edp < min_edp:
        #     min_edp = edp
        #     edp_optimal_tiling_factor_idx = [i]
        if edp == min_edp:
            edp_optimal_tiling_factor_idx.append(i)
        # if latency == min_latency:
        #     latency_optimal_tiling_factor_idx.append(i)
        # if energy == min_energy:
        #     energy_optimal_tiling_factor_idx.append(i)
    
    return energy_list[edp_optimal_tiling_factor_idx[0]], latency_list[edp_optimal_tiling_factor_idx[0]], edp_list[edp_optimal_tiling_factor_idx[0]], min_energy, min_latency, min_edp, memory_list[edp_optimal_tiling_factor_idx[0]]

# for item in OPs_list:
#     idx = item["idx"]
#     energy, latency, min_energy, min_latency = get_OPs_HW_metric(OPs_list[idx],v=False)
#     OPs_list[idx]["energy"] = energy
#     OPs_list[idx]["latency"]= latency
#     print("============================>{}st OPs, energy: {} (min: {}) mJ, latency: {} (min: {}) ms".format(idx, energy, min_energy, latency, min_latency))

def PE_allocation(OPs_list, total_pe, ratio):
    conv_flops = 0
    shift_flops = 0
    conv_list = []
    shift_list = []
    for item in OPs_list:
        padding = int(np.ceil((1 * (item["kernel_size"] - 1) + 1 - item["stride"]) / 2.))
        layer = nn.Conv2d(item["input_C"], item["output_M"], kernel_size=item["kernel_size"], stride=item["stride"], padding=padding, bias=False)
        flops, params = profile(layer, inputs=(torch.randn(1, item["input_C"], item["input_H"], item["input_W"]),))
        if item["type"] == 'conv':
            conv_flops += flops
            conv_list.append(item)
        elif item["type"] == 'shift':
            shift_flops += flops
            shift_list.append(item)
    if shift_flops!= 0:
        conv_pe = int(np.floor(total_pe/(1+shift_flops/(conv_flops*ratio))))
    else:
        conv_pe = total_pe
    shift_pe = int((total_pe-conv_pe)*ratio)
    print('The (Conv_PE, Shift_PE) is ({}, {})'.format(conv_pe, shift_pe))

    return conv_pe, shift_pe, conv_list, shift_list      


def predictor():
    # conv_pe, shift_pe, conv_list, shift_list = PE_allocation(OPs_list, total_pe=168, ratio=1.88)
    conv_pe=168
    shift_pe=0 
    conv_list=OPs_list
    shift_list=[]
    conv_iter = iter(conv_list)
    shift_iter = iter(shift_list)
    shift_energy = 0
    shift_latency = 0
    shift_edp = 0
    shift_min_energy = 0
    shift_min_latency = 0
    shift_min_edp = 0
    conv_energy = 0
    conv_latency = 0
    conv_edp = 0
    conv_min_energy = 0
    conv_min_latency = 0
    conv_min_edp = 0
    shift = next(shift_iter, 'over')
    conv = next(conv_iter, 'over')
    # for i in range(len(conv_list)+len(shift_list)):
    for i in range(50000):
        
        if i==0 or shift_latency==conv_latency:
            # ################ Shift Layer ###############
            if shift != 'over':
                energy, latency, edp, min_energy, min_latency, min_edp, memory = get_OPs_HW_metric(shift, type='shift', num_pe=shift_pe, memory=65536*shift_pe/(conv_pe+shift_pe))
                shift_energy += energy
                shift_latency += latency
                shift_edp += edp
                shift_min_energy += min_energy
                shift_min_latency += min_latency
                shift_min_edp += min_edp
                shift_memory = memory
                shift = next(shift_iter, 'over')
            else:
                print('!!!!!!!!!!! Shift layers are processed completely !!!!!!!!!!!') 
                shift_memory = 0
                Shift_latency = shift_latency
                shift_latency = 5000
                if conv == 'over':
                    print('!!!!!!!!!!! All processed completely !!!!!!!!!!!') 
                    break

            # ################ Conv Layer ###############
            if conv != 'over':
                energy, latency, edp, min_energy, min_latency, min_edp, memory = get_OPs_HW_metric(conv, type='conv', num_pe=conv_pe, memory=65536*conv_pe/(conv_pe+shift_pe))
                conv_energy += energy
                conv_latency += latency
                conv_edp += edp
                conv_min_energy += min_energy
                conv_min_latency += min_latency
                conv_min_edp += min_edp
                conv_memory = memory
                conv = next(conv_iter, 'over')
            else:
                print('!!!!!!!!!!! Conv layers are processed completely !!!!!!!!!!!') 
                conv_memory = 0
                Conv_latency = conv_latency
                conv_latency = 5000
                if shift == 'over':
                    print('!!!!!!!!!!! All processed completely !!!!!!!!!!!') 
                    break

        # ################ Shift Layer first ###############
        elif shift_latency < conv_latency:
            # shift = next(shift_list)
            if shift != 'over':
                energy, latency, edp, min_energy, min_latency, min_edp, memory = get_OPs_HW_metric(shift, type='shift', num_pe=shift_pe, memory=65536-conv_memory)
                shift_energy += energy
                shift_latency += latency
                shift_edp += edp
                shift_min_energy += min_energy
                shift_min_latency += min_latency
                shift_min_edp += min_edp
                shift_memory = memory
                shift = next(shift_iter, 'over')
            else:
                print('!!!!!!!!!!! Shift layers are processed completely !!!!!!!!!!!') 
                shift_memory = 0
                Shift_latency = shift_latency
                shift_latency = 5000
                if conv == 'over':
                    print('!!!!!!!!!!! All processed completely !!!!!!!!!!!') 
                    break

        # ################ Conv Layer first ###############
        elif conv_latency < shift_latency:
            # shift = next(shift_list)
            if conv != 'over':
                energy, latency, edp, min_energy, min_latency, min_edp, memory = get_OPs_HW_metric(conv, type='conv', num_pe=conv_pe, memory=65536-shift_memory)
                conv_energy += energy
                conv_latency += latency
                conv_edp += edp
                conv_min_energy += min_energy
                conv_min_latency += min_latency
                conv_min_edp += min_edp
                conv_memory = memory
                conv = next(conv_iter, 'over')
            else:
                print('!!!!!!!!!!! Conv layers are processed completely !!!!!!!!!!!') 
                conv_memory = 0
                Conv_latency = conv_latency
                conv_latency = 5000
                if shift == 'over':
                    print('!!!!!!!!!!! All processed completely !!!!!!!!!!!') 
                    break
    
    print("============================>Shift layers:, energy: {} (min: {}) mJ, latency: {} (min: {}) ms".format(shift_energy, shift_min_energy, Shift_latency, shift_min_latency))
    print("============================>Conv layers:, energy: {} (min: {}) mJ, latency: {} (min: {}) ms".format(conv_energy, conv_min_energy, Conv_latency, conv_min_latency))
    edp = (conv_energy+shift_energy)*max((Conv_latency,Shift_latency))
    min_edp = (conv_min_energy+shift_min_energy)*max(conv_min_latency, shift_min_latency)
    print("============================>Total:, energy: {} (min: {}) mJ, latency: {} (min: {}) ms, EDP: {} (min: {}) mJ*ms".format(conv_energy+shift_energy, conv_min_energy+shift_min_energy, max(Conv_latency,Shift_latency), max(conv_min_latency, shift_min_latency), edp, min_edp))
    return edp


if __name__ == '__main__':
    predictor()

    



        
    
