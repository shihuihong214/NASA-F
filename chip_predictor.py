import os
import math
import copy
from thop import profile
import torch.nn as nn
# ########################### Compute FLOPs ###############################
import numpy as np
import torch
import random
from models.modules.nn_utils import make_divisible

import argparse
from models.modules.static_layers import ShortcutLayer, MBInvertedConvLayer

# parser = argparse.ArgumentParser(description='ShiftAdd Simulator')
# parser.add_argument('--bit_conv', type=int, default=8, help='the bit number of conv')
# parser.add_argument('--bit_shift', type=int, default=6, help='the bit number of shift')
# parser.add_argument('--bit_Adder', type=int, default=6, help='the bit number of Adder')
# parser.add_argument('--Conv_PE', type=int, default=None, help='the number of PE allocated for Conv')
# parser.add_argument('--Shift_PE', type=int, default=None, help='the number of PE allocated for Shift')
# parser.add_argument('--Adder_PE', type=int, default=None, help='the number of PE allocated for Adder')
# parser.add_argument('--total_PE', type=int, default=168, help='the number of total PE')
# parser.add_argument('--hybrid',  type=str, default='cifar10_conv', help='the type of hybrid models')
# parser.add_argument('--dataflow_Conv',  type=str, default='rs', help='the dataflow for Conv')
# parser.add_argument('--dataflow_Shift', type=str, default='rs', help='the dataflow for Shift')
# parser.add_argument('--dataflow_Adder', type=str, default='rs', help='the dataflow for Adder')
# parser.add_argument('--energy_constraint', type=float, default=None, help='the energy_constraint')
# args = parser.parse_args()

bit_conv=8
bit_shift=6
bit_Adder=6
Ratio_Adder = 1.88
Ratio_Shift = 1.88
# TODO:
# energy_constraint = args.energy_constraint


def valid_tiling(tiling_factor, type, num_PE, ratio, memory, bit, dataflow, FC_flag):
    valid = True
    # ####################### RS ##############################
    if dataflow == 'rs':
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
            if not ((E1!= 1)):
                valid = False
        R = tiling_factor["R"]
        S = tiling_factor["S"]
        if not (M1*C1*E1*R < num_PE):
            valid = False
        # ########## feature map ############
        if not (C0*S < 48):
            valid = False
        stride = tiling_factor["stride"]
        F = tiling_factor["F"]
        # ############# weight #############
        if not (M0*C0*S < 112):
            valid = False
        # ############# GB ##############
        if not ((C1*C0*((E1-1)*stride+R)*((F-1)*stride+S) + M2*M1*M0*E1*F) < (131070-memory)):
            valid = False
    
    # ####################### IS ##############################
    elif dataflow == 'is': 
        C3 = tiling_factor["C3"]
        C2 = tiling_factor["C2"]
        C1 = tiling_factor["C1"]
        C0 = tiling_factor["C0"]
        C = tiling_factor["C"]
        if not (C == C3*C2*C1*C0):
            valid = False
        if not (C3 == 1):
            valid = False
        # M3 = tiling_factor["M3"]
        # M2 = tiling_factor["M2"]
        M1 = tiling_factor["M1"]
        M0 = tiling_factor["M0"]
        M = tiling_factor["M"]
        if not (M == M1*M0):
            valid = False
        if not (M0 <= 16):
            valid = False
        E0 = tiling_factor["E0"]
        E1 = tiling_factor["E1"]
        E3 = tiling_factor["E3"]
        E = tiling_factor["E"]
        if not (E == E0*E1*E3):
            valid = False
        if not FC_flag:
            if not ((E0!= 1)):
                valid = False
        F0 = tiling_factor["F0"]
        F1 = tiling_factor["F1"]
        F3 = tiling_factor["F3"]
        F =  tiling_factor["F"]
        if not (F == F0*F1*F3):
            valid = False
        if not FC_flag:
            if not ((F0!= 1)):
                valid = False
        R = tiling_factor["R"]
        S = tiling_factor["S"]
        stride = tiling_factor["stride"]
        
        if not (C1*E1*F1*M1 < num_PE):
            valid = False
        # ########## feature map ############
        stride = tiling_factor["stride"]
        if F0 > E0:
            if not (C0*S*((E0-1)*stride+R) < 48):
                valid = False
        else:
            if not (C0*((F0-1)*stride+R)*S < 48):
                valid = False
        # ############# weight #############
        if not (C0*M0*R*S < 112):
            valid = False
        # ############# GB ##############
        if not ((C1*C0*((E1*E0-1)*stride+R)*((F1*F0-1)*stride+S) + M1*M0*E1*E0*F1*F0) < (131070-memory)):
            valid = False
    
    # ####################### WS ##############################
    elif dataflow == 'ws':
        C3 = tiling_factor["C3"]
        C2 = tiling_factor["C2"]
        C1 = tiling_factor["C1"]
        C0 = tiling_factor["C0"]
        C = tiling_factor["C"]
        if not (C == C3*C2*C1*C0):
            valid = False
        if not (C3 == 1):
            valid = False
        M3 = tiling_factor["M3"]
        M2 = tiling_factor["M2"]
        M1 = tiling_factor["M1"]
        M0 = tiling_factor["M0"]
        M = tiling_factor["M"]
        if not (M == M3*M2*M1*M0):
            valid = False
        if not (M0 <= 16):
            valid = False
        E0 = tiling_factor["E0"]
        E1 = tiling_factor["E1"]
        # E3 = tiling_factor["E3"]
        E = tiling_factor["E"]
        if not (E == E0*E1):
            valid = False
        if not FC_flag:
            if not ((E0!= 1)):
                valid = False
        F0 = tiling_factor["F0"]
        F1 = tiling_factor["F1"]
        # F3 = tiling_factor["F3"]
        F =  tiling_factor["F"]
        if not (F == F0*F1):
            valid = False
        if not FC_flag:
            if not ((F0!= 1)):
                valid = False
        R = tiling_factor["R"]
        S = tiling_factor["S"]
        stride = tiling_factor["stride"]
        
        if not (C1*R*S*M1*E1*F1 < num_PE):
            valid = False
        # ########## feature map ############
        if not (C0 < 48):
            valid = False
        # ############# weight #############
        if not (C0*M0 < 112):
            valid = False
        # ############# GB ##############
        if not ((C1*C0*((E-1)*stride+R)*((F-1)*stride+S) + M2*M1*M0*E*F) < (131070-memory)):
            valid = False

    # # ####################### OS ##############################
    elif dataflow == 'os':
        M3 = tiling_factor["M3"]
        M2 = tiling_factor["M2"]
        M1 = tiling_factor["M1"]
        M0 = tiling_factor["M0"]
        M = tiling_factor["M"]
        if not (M == M3*M2*M1*M0):
            valid = False
        if not (M0 <= 16):
            valid = False
    #     C3 = tiling_factor["C3"]
    #     C2 = tiling_factor["C2"]
        C1 = tiling_factor["C1"]
        C0 = tiling_factor["C0"]
        C = tiling_factor["C"]
        if not (C == C1*C0):
            valid = False
        # if not (C3 == 1):
        #     valid = False
        E0 = tiling_factor["E0"]
        E1 = tiling_factor["E1"]
        E3 = tiling_factor["E3"]
        E = tiling_factor["E"]
        if not (E == E0*E1*E3):
            valid = False
        if not FC_flag:
            if not ((E0!= 1)):
                valid = False
        F0 = tiling_factor["F0"]
        F1 = tiling_factor["F1"]
        F3 = tiling_factor["F3"]
        F =  tiling_factor["F"]
        if not (F == F0*F1*F3):
            valid = False
        if not FC_flag:
            if not ((F0!= 1)):
                valid = False
        R = tiling_factor["R"]
        S = tiling_factor["S"]
        if not (M1*F1*E1*C1 < num_PE):
            valid = False
        # ########## feature map ############
        stride = tiling_factor["stride"]
        if F0 > E0:
            if not (C0*S*((E0-1)*stride+R) < 48):
                valid = False
        else:
            if not (C0*((F0-1)*stride+R)*S < 48):
                valid = False
        # if not (C0*((F0-1)*stride+R)*((E0-1)*stride+R) < 48):
        #     valid = False
        # ############# weight #############
        if not (M0*R*S*C0 < 112):
            valid = False
    #     # ############# GB ##############
        if not ((C0*C1*((E1*E0-1)*stride+R)*((F1*F0-1)*stride+S) + M2*M1*M0*E1*E0*F1*F0) < (131070-memory)):
            valid = False

    return valid

def get_latency(tiling_factor, unit_latency, dataflow):
    # ####################### RS ##############################
    if dataflow == 'rs': 
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

    # ####################### IS ##############################
    elif dataflow == 'is':
        C3 = tiling_factor["C3"]
        C2 = tiling_factor["C2"]
        C1 = tiling_factor["C1"]
        C0 = tiling_factor["C0"]
        C = tiling_factor["C"]
        # M3 = tiling_factor["M3"]
        # M2 = tiling_factor["M2"]
        M1 = tiling_factor["M1"]
        M0 = tiling_factor["M0"]
        M = tiling_factor["M"]
        E0 = tiling_factor["E0"]
        E1 = tiling_factor["E1"]
        E3 = tiling_factor["E3"]
        E =  tiling_factor["E"]
        F0 = tiling_factor["F0"]
        F1 = tiling_factor["F1"]
        F3 = tiling_factor["F3"]
        F =  tiling_factor["F"]
        E = tiling_factor["E"]
        R = tiling_factor["R"]
        S = tiling_factor["S"]
        stride = tiling_factor["stride"]
        F = tiling_factor["F"]
        # latency = M3*M2*M0*C3*C2*C0*E3*E0*F3*F0*R*S*unit_latency
        latency = M0*C3*C2*C0*E3*E0*F3*F0*R*S*unit_latency
    
    # # ####################### WS ##############################
    elif dataflow == 'ws':
        C3 = tiling_factor["C3"]
        C2 = tiling_factor["C2"]
        C1 = tiling_factor["C1"]
        C0 = tiling_factor["C0"]
        C = tiling_factor["C"]
        M3 = tiling_factor["M3"]
        M2 = tiling_factor["M2"]
        M1 = tiling_factor["M1"]
        M0 = tiling_factor["M0"]
        M = tiling_factor["M"]
        E0 = tiling_factor["E0"]
        E1 = tiling_factor["E1"]
        # E3 = tiling_factor["E3"]
        E =  tiling_factor["E"]
        F0 = tiling_factor["F0"]
        F1 = tiling_factor["F1"]
        # F3 = tiling_factor["F3"]
        F =  tiling_factor["F"]
        E = tiling_factor["E"]
        R = tiling_factor["R"]
        S = tiling_factor["S"]
        stride = tiling_factor["stride"]
        F = tiling_factor["F"]
        latency = M3*M2*M0*C3*C2*C0*E0*F0*unit_latency
    
    elif dataflow == 'os':
        M3 = tiling_factor["M3"]
        M2 = tiling_factor["M2"]
        M1 = tiling_factor["M1"]
        M0 = tiling_factor["M0"]
        M = tiling_factor["M"]
        # C3 = tiling_factor["C3"]
        # C2 = tiling_factor["C2"]
        C0 = tiling_factor["C0"]
        C = tiling_factor["C"]
        E0 = tiling_factor["E0"]
        E1 = tiling_factor["E1"]
        E3 = tiling_factor["E3"]
        E = tiling_factor["E"]
        F0 = tiling_factor["F0"]
        F1 = tiling_factor["F1"]
        F3 = tiling_factor["F3"]
        F =  tiling_factor["F"]
        R = tiling_factor["R"]
        S = tiling_factor["S"]
        stride = tiling_factor["stride"]
        F = tiling_factor["F"]
        latency = M3*M2*M0*C0*E3*E0*F3*F0*R*S*unit_latency
    
    return latency

def get_energy(tiling_factor, unit_energy, type, bit=32, dataflow='rs'):
    # ####################### RS ##############################
    if dataflow == 'rs':
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


    # ####################### IS ##############################
    elif  dataflow == 'is':
        
        C3 = tiling_factor["C3"]
        C2 = tiling_factor["C2"]
        C1 = tiling_factor["C1"]
        C0 = tiling_factor["C0"]
        C = tiling_factor["C"]
        # M3 = tiling_factor["M3"]
        # M2 = tiling_factor["M2"]
        M1 = tiling_factor["M1"]
        M0 = tiling_factor["M0"]
        M = tiling_factor["M"]
        E0 = tiling_factor["E0"]
        E1 = tiling_factor["E1"]
        E3 = tiling_factor["E3"]
        E =  tiling_factor["E"]
        F0 = tiling_factor["F0"]
        F1 = tiling_factor["F1"]
        F3 = tiling_factor["F3"]
        F =  tiling_factor["F"]
        R = tiling_factor["R"]
        S = tiling_factor["S"]
        stride = tiling_factor["stride"]
        F = tiling_factor["F"]
        E = tiling_factor["E"]
        R = tiling_factor["R"]
        S = tiling_factor["S"]
        F = tiling_factor["F"]
        H = tiling_factor["H"]
        W = tiling_factor["W"]
        
        num_ifmap = H*W*C # input feature map size
        num_weight = R*S*C*M # weight size
        num_ofmap = E*F*M # output size
        computation = E*F*M*R*S*C

        DRAM_ifmap = num_ifmap
        DRAM_weight = E3 * F3 * num_weight
        DRAM_ofmap = ( max((2*C3-1), 1) ) * num_ofmap
        GB_ifmap = num_ifmap
        GB_ofmap = ( max(2 * C3 * (C2-1), 1) ) * num_ofmap
        
        NoC_ifmap = M1 * E3*((E0*E1-1)*stride+R)*F3*((F0*F1-1)*stride+S)*C
        NoC_weight = E3 * F3 * E1 * F1 * num_weight
        NoC_ofmap = ( max(C3 * C2 * (C1 - 1), 1) ) * num_ofmap
        RF_ifmap = M1 * M0 * num_ifmap 
        RF_weight = E3 * F3 * E1* F1 * E0 * F0 * num_weight
        RF_ofmap = ( max(C3 * C2 * C1 * (C0*R*S - 1 ) * 2, 1) ) * num_ofmap

    # ####################### WS ##############################
    elif dataflow == 'ws':

        C3 = tiling_factor["C3"]
        C2 = tiling_factor["C2"]
        C1 = tiling_factor["C1"]
        C0 = tiling_factor["C0"]
        C = tiling_factor["C"]
        M3 = tiling_factor["M3"]
        M2 = tiling_factor["M2"]
        M1 = tiling_factor["M1"]
        M0 = tiling_factor["M0"]
        M = tiling_factor["M"]
        E0 = tiling_factor["E0"]
        E1 = tiling_factor["E1"]
        # E3 = tiling_factor["E3"]
        E =  tiling_factor["E"]
        F0 = tiling_factor["F0"]
        F1 = tiling_factor["F1"]
        # F3 = tiling_factor["F3"]
        F =  tiling_factor["F"]
        R = tiling_factor["R"]
        S = tiling_factor["S"]
        stride = tiling_factor["stride"]
        F = tiling_factor["F"]
        E = tiling_factor["E"]
        R = tiling_factor["R"]
        S = tiling_factor["S"]
        F = tiling_factor["F"]
        H = tiling_factor["H"]
        W = tiling_factor["W"]
        
        num_ifmap = H*W*C # input feature map size
        num_weight = R*S*C*M # weight size
        num_ofmap = E*F*M # output size
        computation = E*F*M*R*S*C

        # DRAM_ifmap = M3*num_ifmap
        # DRAM_weight = num_weight
        # DRAM_ofmap = ( max((2*C3-1), 1) ) * num_ofmap
        # GB_ifmap = M3 * M2 * num_ifmap
        # GB_ofmap = ( max(2 * C3 * (C2-1), 1) ) * num_ofmap
        
        # NoC_ifmap = M3 * M2 * M1 * num_ifmap * R * E / H * S * F / W 
        # NoC_weight = E1 * F1 * num_weight
        # NoC_ofmap = ( max(C3 * C2 * (C1*R*S - 1), 1) ) * num_ofmap
        # RF_ifmap = M3 * M2 * M1 * M0 * num_ifmap * R * E / H * S * F / W 
        # RF_weight = E3 * F3 * E1* F1 * E0 * F0 * num_weight
        # RF_ofmap = ( max(C3 * C2 * C1 *R*S*(C0 - 1 ) * 2, 1) ) * num_ofmap

        DRAM_ifmap = M3*num_ifmap
        DRAM_weight = num_weight
        DRAM_ofmap = ( max((2*C3-1), 1) ) * num_ofmap
        GB_ifmap = M3 * M2 * num_ifmap
        GB_ofmap = ( max(2 * C3 * (C2-1), 1) ) * num_ofmap
        
        NoC_ifmap = M3 * M2 * M1 * num_ifmap * R * E / H * S * F / W 
        NoC_weight = E1 * F1 * num_weight
        NoC_ofmap = ( max(C3 * C2 * (C1*R*S - 1), 1) ) * num_ofmap
        RF_ifmap = M3 * M2 * M1 * M0 * num_ifmap * R * E / H * S * F / W 
        RF_weight = E1* F1 * E0 * F0 * num_weight
        RF_ofmap = ( max(C3 * C2 * C1 *R*S*(C0 - 1 ) * 2, 1) ) * num_ofmap

    # ####################### OS ##############################
    elif dataflow == 'os':
        # C3 = tiling_factor["C3"]
        # C2 = tiling_factor["C2"]
        C1 = tiling_factor["C1"]
        C0 = tiling_factor["C0"]
        C = tiling_factor["C"]
        M3 = tiling_factor["M3"]
        M2 = tiling_factor["M2"]
        M1 = tiling_factor["M1"]
        M0 = tiling_factor["M0"]
        M = tiling_factor["M"]
        E0 = tiling_factor["E0"]
        E1 = tiling_factor["E1"]
        E3 = tiling_factor["E3"]
        E =  tiling_factor["E"]
        F0 = tiling_factor["F0"]
        F1 = tiling_factor["F1"]
        F3 = tiling_factor["F3"]
        F =  tiling_factor["F"]
        R = tiling_factor["R"]
        S = tiling_factor["S"]
        stride = tiling_factor["stride"]
        F = tiling_factor["F"]
        E = tiling_factor["E"]
        R = tiling_factor["R"]
        S = tiling_factor["S"]
        F = tiling_factor["F"]
        H = tiling_factor["H"]
        W = tiling_factor["W"]
        
        num_ifmap = H*W*C # input feature map size
        num_weight = R*S*C*M # weight size
        num_ofmap = E*F*M # output size
        computation = E*F*M*R*S*C

        DRAM_ifmap = M3 * num_ifmap
        DRAM_weight = E3 * F3 * num_weight
        DRAM_ofmap =  num_ofmap
        GB_ifmap = M3 * M2 * num_ifmap
        GB_ofmap = num_ofmap
        
        NoC_ifmap = M3 * M2 * M1 * E3*((E0*E1-1)*stride+R)*F3*((F0*F1-1)*stride+S)*C
        NoC_weight = E3 * F3 * E1 * F1 * num_weight
        NoC_ofmap = ( max((C1 - 1), 1) ) * num_ofmap
        RF_ifmap = M3 * M2 * M1 * M0 * num_ifmap * R * E / H * S * F / W 
        RF_weight = E3 * F3 * E1* F1 * E0 * F0 * num_weight
        RF_ofmap = ( max(C1 * (C0*R*S - 1 ) * 2, 1) ) * num_ofmap

        # DRAM_ifmap = M3 * num_ifmap
        # DRAM_weight = E3 * F3 * num_weight
        # DRAM_ofmap =  num_ofmap
        # GB_ifmap = M3 * M2 * num_ifmap
        # GB_ofmap = num_ofmap
        
        # NoC_ifmap = M3 * M2 * M1 * num_ifmap * R * E / H * S * F / W 
        # NoC_weight = E3 * F3 * E1 * F1 * num_weight
        # NoC_ofmap = ( max((C1*R*S - 1), 1) ) * num_ofmap
        # RF_ifmap = M3 * M2 * M1 * M0 * num_ifmap * R * E / H * S * F / W 
        # RF_weight = E3 * F3 * E1* F1 * E0 * F0 * num_weight
        # RF_ofmap = ( max(C3 * C2 * C1 *R*S* (C0 - 1 ) * 2, 1) ) * num_ofmap

    energy = computation * unit_energy["unit_Conv"] \
            + (DRAM_ifmap + DRAM_weight + DRAM_ofmap) * unit_energy["unit_DRAM"] \
            + (DRAM_ifmap + DRAM_ofmap) * unit_energy["unit_DRAM_GB"] \
            + (GB_ifmap + GB_ofmap) * unit_energy["unit_GB"] \
            + (NoC_ifmap + NoC_weight + NoC_ofmap) * unit_energy["unit_NoC"] \
            + (RF_ifmap + RF_weight + RF_ofmap) * unit_energy["unit_RF"]
    return energy, computation * unit_energy["unit_Conv"], (DRAM_ifmap + DRAM_weight + DRAM_ofmap) * unit_energy["unit_DRAM"], (GB_ifmap + GB_ofmap) * unit_energy["unit_GB"], (NoC_ifmap + NoC_weight + NoC_ofmap) * unit_energy["unit_NoC"], (RF_ifmap + RF_weight + RF_ofmap) * unit_energy["unit_RF"]

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


def mapping(FC_flag, H, W, C, R, S, M, E, F, stride, type, PE, unit_energy, unit_latency, bit):
    base_tiling_factor = {}
    base_tiling_factor["H"] = H
    base_tiling_factor["W"] = W
    base_tiling_factor["C"] = C
    base_tiling_factor["R"] = R
    base_tiling_factor["S"] = S
    base_tiling_factor["M"] = M
    base_tiling_factor["E"] = E
    base_tiling_factor["F"] = F
    base_tiling_factor["stride"] = stride
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
    compute_energy_list = []
    DRAM_energy_list = []
    GB_energy_list = []
    NoC_energy_list = []
    RF_energy_list = []
    latency_list = []
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
                if valid_tiling(tiling_factor, type, PE, ratio, FC_flag):
                    energy, compute, DRAM, GB, NoC, RF = get_energy(tiling_factor, unit_energy, type, bit)
                    latency = get_latency(tiling_factor, unit_latency)

                    energy_list.append(energy)
                    compute_energy_list.append(compute)
                    DRAM_energy_list.append(DRAM)
                    GB_energy_list.append(GB)
                    NoC_energy_list.append(NoC)
                    RF_energy_list.append(RF)
                    latency_list.append(latency)
                    tiling_factor_list.append(tiling_factor)

    # tiling factor search M, C, E
    max_energy = max(energy_list)
    min_energy = min(energy_list)

    max_latency = max(latency_list)
    min_latency = min(latency_list)

    min_normal_metric = (energy_list[0]-min_energy)/max_energy + (latency_list[0]-min_latency)/max_latency

    total_optimal_tiling_factor_idx = [0]
    latency_optimal_tiling_factor_idx = []
    energy_optimal_tiling_factor_idx = []

    for i in range(len(tiling_factor_list)):
        tiling_factor = tiling_factor_list[i]
        energy = energy_list[i]
        latency = latency_list[i]

        normal_metric = (energy_list[i]-min_energy)/max_energy + (latency_list[i]-min_latency)/max_latency

        # update total optimal
        if normal_metric < min_normal_metric:
            min_normal_metric = normal_metric
            total_optimal_tiling_factor_idx = [i]
        elif normal_metric == min_normal_metric:
            total_optimal_tiling_factor_idx.append(i)
        if latency == min_latency:
            latency_optimal_tiling_factor_idx.append(i)
        if energy == min_energy:
            energy_optimal_tiling_factor_idx.append(i)
    return energy_list[total_optimal_tiling_factor_idx[0]], latency_list[total_optimal_tiling_factor_idx[0]], compute_energy_list[total_optimal_tiling_factor_idx[0]], DRAM_energy_list[total_optimal_tiling_factor_idx[0]], GB_energy_list[total_optimal_tiling_factor_idx[0]], NoC_energy_list[total_optimal_tiling_factor_idx[0]], RF_energy_list[total_optimal_tiling_factor_idx[0]]

def get_memory(tiling, bit, dataflow):
    if dataflow == 'rs':
        C1 = tiling["C1"]
        C0 = tiling["C0"]
        E1 = tiling["E1"]
        stride = tiling["stride"]
        R = tiling["R"]
        F = tiling["F"]
        S = tiling["S"]
        M2 = tiling["M2"]
        M1 = tiling["M1"]
        M0 = tiling["M0"]

        return (C1*C0*((E1-1)*stride+R)*((F-1)*stride+S) + M2*M1*M0*E1*F) 
    
    # ####################### IS ##############################
    elif dataflow == 'is':
        C3 = tiling["C3"]
        C2 = tiling["C2"]
        C1 = tiling["C1"]
        C0 = tiling["C0"]
        C =  tiling["C"]
        # M3 = tiling["M3"]
        # M2 = tiling["M2"]
        M1 = tiling["M1"]
        M0 = tiling["M0"]
        M =  tiling["M"]
        E0 = tiling["E0"]
        E1 = tiling["E1"]
        E3 = tiling["E3"]
        E =  tiling["E"]
        F0 = tiling["F0"]
        F1 = tiling["F1"]
        F3 = tiling["F3"]
        F =  tiling["F"]
        R =  tiling["R"]
        S =  tiling["S"]
        stride = tiling["stride"]
        return (C1*C0*((E1*E0-1)*stride+R)*((F1*F0-1)*stride+S) + M1*M0*E1*E0*F1*F0)

    # ####################### WS ##############################
    elif dataflow == 'ws':
        C3 = tiling["C3"]
        C2 = tiling["C2"]
        C1 = tiling["C1"]
        C0 = tiling["C0"]
        C =  tiling["C"]
        M3 = tiling["M3"]
        M2 = tiling["M2"]
        M1 = tiling["M1"]
        M0 = tiling["M0"]
        M =  tiling["M"]
        E0 = tiling["E0"]
        E1 = tiling["E1"]
        # E3 = tiling_factor["E3"]
        E =  tiling["E"]
        F0 = tiling["F0"]
        F1 = tiling["F1"]
        # F3 = tiling_factor["F3"]
        F =  tiling["F"]
        R =  tiling["R"]
        S =  tiling["S"]
        stride = tiling["stride"]
        return (C1*C0*((E-1)*stride+R)*((F-1)*stride+S) + M2*M1*M0*E*F)

    elif dataflow == 'os':
        M3 = tiling["M3"]
        M2 = tiling["M2"]
        M1 = tiling["M1"]
        M0 = tiling["M0"]
        M =  tiling["M"]
        C1 = tiling["C1"]
        C0 = tiling["C0"]
        C =  tiling["C"]
        E0 = tiling["E0"]
        E1 = tiling["E1"]
        E3 = tiling["E3"]
        E =  tiling["E"]
        F0 = tiling["F0"]
        F1 = tiling["F1"]
        F3 = tiling["F3"]
        F =  tiling["F"]
        R =  tiling["R"]
        S =  tiling["S"]
        stride = tiling["stride"]
        return (C0*C1*((E1*E0-1)*stride+R)*((F1*F0-1)*stride+S) + M2*M1*M0*E1*E0*F1*F0)

# gives the energy (mJ), latency (ms)
def get_OPs_HW_metric(layer_dict, PE, bit, ratio, memory, dataflow, v=False):
    # constant defination
    
    if layer_dict["type"] == "FC":
        FC_flag = True
    else:
        FC_flag = False
    unit_energy = {}
    # if (layer_dict["type"] == "AvgP") or ((layer_dict["type"] == "MaxP")):
    #     unit_energy["unit_comp"] = 0.0/(1e9) # mJ/MAC
    # else:
    #     unit_energy["unit_comp"] = 1.0/(1e9) # mJ/MAC
    if bit == 32:
        conv_energy = 3.7
        add_energy = 0.9
        shift_energy = 0.13
    elif bit == 16:
        conv_energy = 1
        add_energy = 0.03
        shift_energy = 0.103
    elif bit == 8:
        conv_energy = 0.2
        add_energy = 0.03
        shift_energy = 0.024
    elif bit == 6:
        conv_energy = 0.2
        add_energy = 0.022
        shift_energy = 0.018

    unit_energy["unit_Shift"] = shift_energy/(1e9) # mJ/MAC
    unit_energy["unit_Add"] = add_energy/(1e9) # mJ/MAC
    unit_energy["unit_Conv"] = conv_energy/(1e9)# mJ/MAC

    unit_energy["unit_DRAM"] = 200/(1e9) * 1 * bit/16 # mJ/16 bits
    unit_energy["unit_DRAM_GB"] = 0.0/(1e9) * bit/16  # mJ/16 bits
    unit_energy["unit_GB"] = 6.0/(1e9) * 1 * bit/16  # mJ/16 bits
    unit_energy["unit_NoC"] = 2.0/(1e9) * 1 * bit/16  # mJ/16 bits
    # unit_energy["unit_NoC_psum"] = 1.0/(1e9) * 1   # mJ/16 bits
    unit_energy["unit_RF"] = 1.0/(1e9) * 1 * bit/16  # mJ/16 bits    

    unit_latency = 1.0/(250e6)*(1e3)  # ms

    # #################### different dataflows ####################
    base_tiling_factor = {}
    base_tiling_factor["H"] = layer_dict["H"]
    base_tiling_factor["W"] = layer_dict["W"]
    base_tiling_factor["C"] = layer_dict["C"]
    base_tiling_factor["R"] = layer_dict["R"]
    base_tiling_factor["S"] = layer_dict["S"]
    base_tiling_factor["M"] = layer_dict["M"]
    base_tiling_factor["E"] = layer_dict["E"]
    base_tiling_factor["F"] = layer_dict["F"]
    base_tiling_factor["stride"] = layer_dict["stride"]

    energy_list = []
    compute_energy_list = []
    DRAM_energy_list = []
    GB_energy_list = []
    NoC_energy_list = []
    RF_energy_list = []
    latency_list = []
    tiling_factor_list = []
    
    # #################### RS ################################
    if dataflow == 'rs':
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
        # E_tile_list = tile(base_tiling_factor["E"], 2)
        # filter out E1 == 1 options
        if not FC_flag:
            for tile_option in E_tile_list:
                if tile_option[0] == 1:
                    E_tile_list.remove(tile_option)

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
                    if valid_tiling(tiling_factor, type, PE, ratio, memory, bit, dataflow, FC_flag):
                        
                        latency = get_latency(tiling_factor, unit_latency, dataflow)
                        # if (latency < latency_constraint):
                        energy, compute, DRAM, GB, NoC, RF = get_energy(tiling_factor, unit_energy, type, bit, dataflow)
                        energy_list.append(energy)
                        compute_energy_list.append(compute)
                        DRAM_energy_list.append(DRAM)
                        GB_energy_list.append(GB)
                        NoC_energy_list.append(NoC)
                        RF_energy_list.append(RF)
                        latency_list.append(latency)
                        tiling_factor_list.append(tiling_factor)
    
    # ################################### IS/WS/OS ############################
    elif dataflow == 'is':
        # tile M to M0 * M1 * M2 * M3
        M_tile_list = tile(base_tiling_factor["M"], 2)
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
        
        # tile H to H1 * H3
        E_tile_list = tile(base_tiling_factor["E"], 3)
        # filter out H0 == 1 options
        if not FC_flag:
            for tile_option in E_tile_list:
                if tile_option[0] == 1:
                    E_tile_list.remove(tile_option)

        # tile F to F0 * F1
        F_tile_list = tile(base_tiling_factor["F"], 3)
        if not FC_flag:
            for tile_option in F_tile_list:
                if tile_option[0] == 1:
                    F_tile_list.remove(tile_option)
        
        for M_tile in M_tile_list:
            for C_tile in C_tile_list:
                for E_tile in E_tile_list:
                    for F_tile in F_tile_list:
                        tiling_factor = copy.deepcopy(base_tiling_factor)
                        tiling_factor["C0"] = C_tile[0]
                        tiling_factor["C1"] = C_tile[1]
                        tiling_factor["C2"] = C_tile[2]
                        tiling_factor["C3"] = C_tile[3]
                        tiling_factor["M0"] = M_tile[0]
                        tiling_factor["M1"] = M_tile[1]
                        # tiling_factor["M2"] = M_tile[2]
                        # tiling_factor["M3"] = M_tile[3]
                        tiling_factor["E0"] = E_tile[0]
                        tiling_factor["E1"] = E_tile[1]
                        tiling_factor["E3"] = E_tile[2]
                        tiling_factor["F0"] = F_tile[0]
                        tiling_factor["F1"] = F_tile[1]
                        tiling_factor["F3"] = F_tile[2]
                        
                        if valid_tiling(tiling_factor, type, PE, ratio, memory, bit, dataflow, FC_flag):
                            latency = get_latency(tiling_factor, unit_latency, dataflow)
                            # if (latency < latency_constraint):
                            energy, compute, DRAM, GB, NoC, RF = get_energy(tiling_factor, unit_energy, type, bit, dataflow)
                            energy_list.append(energy)
                            compute_energy_list.append(compute)
                            DRAM_energy_list.append(DRAM)
                            GB_energy_list.append(GB)
                            NoC_energy_list.append(NoC)
                            RF_energy_list.append(RF)
                            latency_list.append(latency)
                            tiling_factor_list.append(tiling_factor)

    # #################### WS ################################
    elif dataflow == 'ws':
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
        
        # tile H to H1 * H3
        E_tile_list = tile(base_tiling_factor["E"], 2)
        # filter out H0 == 1 options
        if not FC_flag:
            for tile_option in E_tile_list:
                if tile_option[0] == 1:
                    E_tile_list.remove(tile_option)

        # tile F to F0 * F1
        F_tile_list = tile(base_tiling_factor["F"], 2)
        if not FC_flag:
            for tile_option in F_tile_list:
                if tile_option[0] == 1:
                    F_tile_list.remove(tile_option)
        
        for M_tile in M_tile_list:
            for C_tile in C_tile_list:
                for E_tile in E_tile_list:
                    for F_tile in F_tile_list:
                        tiling_factor = copy.deepcopy(base_tiling_factor)
                        tiling_factor["C0"] = C_tile[0]
                        tiling_factor["C1"] = C_tile[1]
                        tiling_factor["C2"] = C_tile[2]
                        tiling_factor["C3"] = C_tile[3]
                        tiling_factor["M0"] = M_tile[0]
                        tiling_factor["M1"] = M_tile[1]
                        tiling_factor["M2"] = M_tile[2]
                        tiling_factor["M3"] = M_tile[3]
                        tiling_factor["E0"] = E_tile[0]
                        tiling_factor["E1"] = E_tile[1]
                        # tiling_factor["E3"] = E_tile[3]
                        tiling_factor["F0"] = F_tile[0]
                        tiling_factor["F1"] = F_tile[1]
                        # tiling_factor["F3"] = F_tile[3]
                        if valid_tiling(tiling_factor, type, PE, ratio, memory, bit, dataflow, FC_flag):
                            latency = get_latency(tiling_factor, unit_latency, dataflow)
                            # if (latency < latency_constraint):
                            energy, compute, DRAM, GB, NoC, RF = get_energy(tiling_factor, unit_energy, type, bit, dataflow)
                            energy_list.append(energy)
                            compute_energy_list.append(compute)
                            DRAM_energy_list.append(DRAM)
                            GB_energy_list.append(GB)
                            NoC_energy_list.append(NoC)
                            RF_energy_list.append(RF)
                            latency_list.append(latency)
                            tiling_factor_list.append(tiling_factor)
    
    # # #################### OS ################################
    elif dataflow == 'os':
        # tile M to M0 * M1 * M2 * M3
        M_tile_list = tile(base_tiling_factor["M"], 4)
        # filter out M0 > 16 options
        for tile_option in M_tile_list:
            if tile_option[0] > 16:
                M_tile_list.remove(tile_option)

        C_tile_list = tile(base_tiling_factor["C"], 2)
        # filter out C3 != 1 options
        # for tile_option in C_tile_list:
        #     if tile_option[1] != 1:
        #         C_tile_list.remove(tile_option)
       
        # tile E to E1 * E3
        E_tile_list = tile(base_tiling_factor["E"], 3)
        # filter out E1 == 1 options
        if not FC_flag:
            for tile_option in E_tile_list:
                if tile_option[0] == 1:
                    E_tile_list.remove(tile_option)

        # tile F to F0 * F1
        F_tile_list = tile(base_tiling_factor["F"], 3)
        if not FC_flag:
            for tile_option in F_tile_list:
                if tile_option[0] == 1:
                    F_tile_list.remove(tile_option)

        for M_tile in M_tile_list:
            for C_tile in C_tile_list:
                for F_tile in F_tile_list:
                    for E_tile in E_tile_list:
                        tiling_factor = copy.deepcopy(base_tiling_factor)
                        tiling_factor["M0"] = M_tile[0]
                        tiling_factor["M1"] = M_tile[1]
                        tiling_factor["M2"] = M_tile[2]
                        tiling_factor["M3"] = M_tile[3]
                        tiling_factor["C0"] = C_tile[0]
                        tiling_factor["C1"] = C_tile[1]
                        # tiling_factor["C2"] = C_tile[2]
                        # tiling_factor["C3"] = C_tile[3]
                        tiling_factor["F0"] = F_tile[0]
                        tiling_factor["F1"] = F_tile[1]
                        tiling_factor["F3"] = F_tile[2]
                        tiling_factor["E0"] = E_tile[0]
                        tiling_factor["E1"] = E_tile[1]
                        tiling_factor["E3"] = E_tile[2]
                        if valid_tiling(tiling_factor, type, PE, ratio, memory, bit, dataflow, FC_flag):
                            latency = get_latency(tiling_factor, unit_latency, dataflow)
                            # if (latency < latency_constraint):
                            energy, compute, DRAM, GB, NoC, RF = get_energy(tiling_factor, unit_energy, type, bit, dataflow)
                            energy_list.append(energy)
                            compute_energy_list.append(compute)
                            DRAM_energy_list.append(DRAM)
                            GB_energy_list.append(GB)
                            NoC_energy_list.append(NoC)
                            RF_energy_list.append(RF)
                            latency_list.append(latency)
                            tiling_factor_list.append(tiling_factor)
        

    # tiling factor search M, C, E
    # if energy_list:
    #     max_energy = max(energy_list)
    #     min_energy = min(energy_list)

    #     max_latency = max(latency_list)
    #     min_latency = min(latency_list)

    #     # TODO:
    #     min_normal_metric = (energy_list[0]-min_energy)/max_energy + 2*(latency_list[0]-min_latency)/max_latency

    #     total_optimal_tiling_factor_idx = [0]
    #     latency_optimal_tiling_factor_idx = []
    #     energy_optimal_tiling_factor_idx = []

    #     for i in range(len(tiling_factor_list)):
    #         tiling_factor = tiling_factor_list[i]
    #         energy = energy_list[i]
    #         latency = latency_list[i]
    #         # TODO:
    #         normal_metric = (energy_list[i]-min_energy)/max_energy + 2*(latency_list[i]-min_latency)/max_latency

    #         # update total optimal
    #         if normal_metric < min_normal_metric:
    #             min_normal_metric = normal_metric
    #             total_optimal_tiling_factor_idx = [i]
    #         elif normal_metric == min_normal_metric:
    #             total_optimal_tiling_factor_idx.append(i)
    #         if latency == min_latency:
    #             latency_optimal_tiling_factor_idx.append(i)
    #         if energy == min_energy:
    #             energy_optimal_tiling_factor_idx.append(i)
    #     return energy_list[total_optimal_tiling_factor_idx[0]], latency_list[total_optimal_tiling_factor_idx[0]], min_energy, min_latency, compute_energy_list[total_optimal_tiling_factor_idx[0]], DRAM_energy_list[total_optimal_tiling_factor_idx[0]], GB_energy_list[total_optimal_tiling_factor_idx[0]], NoC_energy_list[total_optimal_tiling_factor_idx[0]], RF_energy_list[total_optimal_tiling_factor_idx[0]], get_memory(tiling_factor_list[total_optimal_tiling_factor_idx[0]],bit,dataflow)

    if energy_list:
        # max_energy = max(energy_list)
        # min_energy = min(energy_list)

        # max_latency = max(latency_list)
        # min_latency = min(latency_list)

        # # TODO:
        # # min_normal_metric = (energy_list[0]-min_energy)/max_energy + 2*(latency_list[0]-min_latency)/max_latency

        # total_optimal_tiling_factor_idx = [0]
        # latency_optimal_tiling_factor_idx = []
        # energy_optimal_tiling_factor_idx = []

        # for i in range(len(tiling_factor_list)):
        #     tiling_factor = tiling_factor_list[i]
        #     energy = energy_list[i]
        #     latency = latency_list[i]
        #     # TODO:
        #     # normal_metric = (energy_list[i]-min_energy)/max_energy + 2*(latency_list[i]-min_latency)/max_latency

        #     # update total optimal
        #     if latency < min_latency:
        #         min_latency = latency
        #         latency_optimal_tiling_factor_idx = [i]
        #     elif latency == min_latency:
        #         latency_optimal_tiling_factor_idx.append(i)
        #     # if latency == min_latency:
        #     #     latency_optimal_tiling_factor_idx.append(i)
        #     # if energy == min_energy:
        #     #     energy_optimal_tiling_factor_idx.append(i)
        # return energy_list[latency_optimal_tiling_factor_idx[0]], latency_list[latency_optimal_tiling_factor_idx[0]], min_energy, min_latency, compute_energy_list[latency_optimal_tiling_factor_idx[0]], DRAM_energy_list[latency_optimal_tiling_factor_idx[0]], GB_energy_list[latency_optimal_tiling_factor_idx[0]], NoC_energy_list[latency_optimal_tiling_factor_idx[0]], RF_energy_list[latency_optimal_tiling_factor_idx[0]], get_memory(tiling_factor_list[latency_optimal_tiling_factor_idx[0]],bit,dataflow)
        
        # FIXME: change from latency priority to EDP first
        # max_energy = max(energy_list)
        min_energy = min(energy_list)

        # max_latency = max(latency_list)
        min_latency = min(latency_list)
        
        min_EDP = energy_list[0]*latency_list[0]

        total_optimal_tiling_factor_idx = [0]
        # latency_optimal_tiling_factor_idx = []
        # energy_optimal_tiling_factor_idx = []

        for i in range(len(tiling_factor_list)):
            tiling_factor = tiling_factor_list[i]
            energy = energy_list[i]
            latency = latency_list[i]
        
            normal_EDP = energy_list[i]*latency_list[i]

            # update total optimal
            if normal_EDP < min_EDP:
                min_EDP = normal_EDP
                total_optimal_tiling_factor_idx = [i]
            elif normal_EDP == min_EDP:
                total_optimal_tiling_factor_idx.append(i)
            # if latency == min_latency:
            #     latency_optimal_tiling_factor_idx.append(i)
            # if energy == min_energy:
            #     energy_optimal_tiling_factor_idx.append(i)
        return energy_list[total_optimal_tiling_factor_idx[0]], latency_list[total_optimal_tiling_factor_idx[0]], min_energy, min_latency, compute_energy_list[total_optimal_tiling_factor_idx[0]], DRAM_energy_list[total_optimal_tiling_factor_idx[0]], GB_energy_list[total_optimal_tiling_factor_idx[0]], NoC_energy_list[total_optimal_tiling_factor_idx[0]], RF_energy_list[total_optimal_tiling_factor_idx[0]], get_memory(tiling_factor_list[total_optimal_tiling_factor_idx[0]],bit,dataflow)
    else:
        return None, None,None,None,None,None,None,None,None,None
    

def predictor(OPs_list, stage): 
# ################################# PE Allocation ########################################
    flops_conv = 0
    flops_shift = 0
    flops_Adder = 0
    para_conv = 0
    para_shift = 0
    para_Adder = 0
    Conv_list = []
    Shift_list = []
    Adder_list = []
    for item in OPs_list:
        # if item["type"] == "Skip":
        #     layer = Skip(C_in=item["input_C"], C_out=item["output_M"], layer_id=item["idx"], stride=item["stride"])
        # else: 
        layer = MBInvertedConvLayer(in_channels=item["input_C"], out_channels=item["output_M"], expand_ratio=item["expansion"], kernel_size=item["kernel_size"], stride=item["stride"], use_se=item["use_se"], mid_channels=item["middle_channel"])
        flops, params = profile(layer, inputs=(torch.randn(1, item["input_C"], item["input_H"], item["input_W"]),), verbose=False)
        flop = flops
        # if item["input_C"] != item["output_M"]:
        #     layer = ShortcutLayer(in_channels=item["input_C"], out_channels=item["output_M"])
        #     flops, params = profile(layer, inputs=(torch.randn(1, item["output_E"], item["output_F"], item["input_W"]),))
        #     flop += flops

        if item["type"] == "shift":
            flops_shift += flop
            para_shift += params
            Shift_list.append(item)
        elif item["type"] == "add":
            flops_Adder += flop
            para_Adder += params
            Adder_list.append(item)
        else:
            flops_conv += flop
            para_conv += params
            Conv_list.append(item)

    FLOPs = round(((flops_conv + flops_shift/1.88 + flops_Adder/1.88)/1e7),3)
    buffer = round((para_shift*6 + para_Adder*6 + para_conv*8)/1024,1)
    # print(FLOPs)
    
    # TODO:
    if stage == 'coarse':
        return FLOPs
    # else:
    # if FLOPs > 
    # if FLOPs > targeted_max_FLOPs:
    #     return 50, 50, 50, FLOPs
    # x + flops_shift/flops_conv * x / 1.93 =  14*48

    def expand_operator(list):
        new_list = []
        if len(list):
            for item in enumerate(list):
                # print(item[1]["type"])
                for i in range(3):
                    if i == 0:
                        if item[1]["expansion"] == 1:
                            pass
                        else:
                            new_item = {}
                            new_item["type"] = item[1]["type"]
                            new_item["H"] = item[1]["input_H"]
                            new_item["W"] = item[1]["input_W"]
                            new_item["C"] = item[1]["input_C"]
                            new_item["R"] = 1
                            new_item["S"] = 1
                            new_item["M"] = item[1]["middle_channel"]
                            new_item["E"] = item[1]["input_H"]
                            new_item["F"] = item[1]["input_W"]
                            new_item["stride"] = 1
                            new_list.append(new_item)
                    elif i == 1:
                        for j in range (item[1]["middle_channel"]):
                            new_item = {}
                            new_item["type"] = item[1]["type"]
                            new_item["H"] = item[1]["input_H"]
                            new_item["W"] = item[1]["input_W"]
                            new_item["C"] = 1
                            new_item["R"] = item[1]["kernel_size"]
                            new_item["S"] = item[1]["kernel_size"]
                            new_item["M"] = 1
                            new_item["E"] = item[1]["output_E"]
                            new_item["F"] = item[1]["output_F"]
                            new_item["stride"] = item[1]["stride"]
                            new_list.append(new_item)
                        if item[1]["use_se"]:
                            num_mid = make_divisible(item[1]["middle_channel"] // 4, divisor=8)
                            # reduce layer
                            new_item = {}
                            new_item["type"] = item[1]["type"]
                            new_item["H"] = item[1]["output_E"]
                            new_item["W"] = item[1]["output_F"]
                            new_item["C"] = item[1]["middle_channel"]
                            new_item["R"] = 1
                            new_item["S"] = 1
                            new_item["M"] = num_mid
                            new_item["E"] = item[1]["output_E"]
                            new_item["F"] = item[1]["output_F"]
                            new_item["stride"] = 1
                            new_list.append(new_item)
                            # expand layer
                            new_item = {}
                            new_item["type"] = item[1]["type"]
                            new_item["H"] = item[1]["output_E"]
                            new_item["W"] = item[1]["output_F"]
                            new_item["C"] = num_mid
                            new_item["R"] = 1
                            new_item["S"] = 1
                            new_item["M"] = item[1]["middle_channel"]
                            new_item["E"] = item[1]["output_E"]
                            new_item["F"] = item[1]["output_F"]
                            new_item["stride"] = 1
                            new_list.append(new_item)
                    elif i == 2:
                        new_item = {}
                        new_item["type"] = item[1]["type"]
                        new_item["H"] = item[1]["output_E"]
                        new_item["W"] = item[1]["output_F"]
                        new_item["C"] = item[1]["middle_channel"]
                        new_item["R"] = 1
                        new_item["S"] = 1
                        new_item["M"] = item[1]["output_M"]
                        new_item["E"] = item[1]["output_E"]
                        new_item["F"] = item[1]["output_F"]
                        new_item["stride"] = 1
                        new_list.append(new_item)
                # if item[1]["output_M"] != item[1]["input_C"]:
                #     new_item = {}
                #     new_item["type"] = item[1]["type"]
                #     new_item["H"] = item[1]["output_E"]
                #     new_item["W"] = item[1]["output_F"]
                #     new_item["C"] = item[1]["input_C"]
                #     new_item["R"] = 1
                #     new_item["S"] = 1
                #     new_item["M"] = item[1]["output_M"]
                #     new_item["E"] = item[1]["output_E"]
                #     new_item["F"] = item[1]["output_F"]
                #     new_item["stride"] = 1
                #     new_list.append(new_item)

            return new_list
        else:
            return None

    new_Conv_list = expand_operator(Conv_list)
    new_Shift_list = expand_operator(Shift_list)
    new_Adder_list = expand_operator(Adder_list)
    # print(len(new_Conv_list)+len(new_Shift_list))

    # if bit_conv == 32 and bit_shift == 16:
    #     flops_conv *= 4
    # print('flops_conv',flops_conv)
    # print('flops_shift',flops_shift)
    # print('flops_Adder',flops_Adder)
    per_flops = 168/(flops_conv + flops_shift/Ratio_Shift + flops_Adder/Ratio_Adder)
    # print('args.total_PE',args.total_PE)
    # print('per_flops',per_flops)
    Conv_PE = math.floor(per_flops*flops_conv)
    Shift_PE = math.ceil(per_flops*flops_shift)
    Adder_PE = math.ceil(per_flops*flops_Adder)
    # Conv_PE = int((args.total_PE)/(1+flops_shift/(flops_conv*2.92)))
    # Shift_PE = int(flops_shift/flops_conv*Conv_PE)

    ratio = flops_conv/(flops_conv+flops_shift)
    # print(flops_conv,flops_shift)
    # if args.Conv_PE != None:
    #     Conv_PE = args.Conv_PE
    # if args.Shift_PE != None:
    #     Shift_PE = args.Shift_PE
    # if args.Adder_PE != None:
    #     Adder_PE = args.Adder_PE
    # print("-------------------------------> The PE allocation is: ",(Conv_PE, Shift_PE, Adder_PE))
    num_type = 0
    if Conv_PE!=0:
        num_type += 1
    if Shift_PE!=0:
        num_type += 1
    if Adder_PE != 0:
        num_type += 1
        
    # print('The type number of the measured hybrid model is ',num_type) 


    if new_Shift_list!= None:
        shift_dataflow = 4
    else:
        shift_dataflow = 1

    if new_Adder_list!= None:
        Adder_dataflow = 4
    else:
        Adder_dataflow = 1

    dataflow = ['rs','is','ws','os']
    total_EDP = []
    total_min_latency = []
    total_min_energy = []
    total_energy = []
    total_wo_DRAM_energy = []
    total_latency = []
    total_compute_energy = []
    total_DRAM_energy = []
    total_GB_energy = []
    total_NoC_energy = []
    total_RF_energy = []
    # FIXME: four dataflows
    # for c in range(1):
    #     for s in range(1):
    #         for a in range(1):
    c = 0
    s = 0 
    a = 0
    # print("-------------------------------> The dataflows for Conv/ Shift/ Adder are: ",(dataflow[c], dataflow[s], dataflow[a]))

    Conv_total_energy=0; Conv_total_latency=0; Conv_total_min_energy=0; Conv_total_min_latency=0
    Conv_total_compute_energy=0; Conv_total_DRAM_energy=0; Conv_total_GB_energy=0; Conv_total_NoC_energy=0; Conv_total_RF_energy=0

    Shift_total_energy=0; Shift_total_latency=0; Shift_total_min_energy=0; Shift_total_min_latency=0
    Shift_total_compute_energy=0; Shift_total_DRAM_energy=0; Shift_total_GB_energy=0; Shift_total_NoC_energy=0; Shift_total_RF_energy=0

    Adder_total_energy=0; Adder_total_latency=0; Adder_total_min_energy=0; Adder_total_min_latency=0
    Adder_total_compute_energy=0; Adder_total_DRAM_energy=0; Adder_total_GB_energy=0; Adder_total_NoC_energy=0; Adder_total_RF_energy=0

    Shift_real_latency = 0
    Conv_real_latency = 0
    Adder_real_latency = 0

    if new_Conv_list!= None:
        conv_iter = iter(new_Conv_list)
        conv = next(conv_iter,'over')
    else:
        conv = None

    if new_Shift_list!= None:
        shift_iter = iter(new_Shift_list)
        shift = next(shift_iter,'over')
    else:
        shift = None

    if new_Adder_list!= None:
        Adder_iter = iter(new_Adder_list)
        Adder = next(Adder_iter,'over')
    else:
        Adder = None
    
    for i in range(100000):
        # idx = conv["idx"]
        if i == 0 :
            if shift != None:
                shift_energy, shift_latency, shift_min_energy, shift_min_latency, shift_compute_energy, shift_DRAM_energy, shift_GB_energy, shift_NoC_energy, shift_RF_energy, shift_memory = get_OPs_HW_metric(shift, PE=Shift_PE, bit=bit_shift, ratio=ratio, memory=0, dataflow=dataflow[s], v=False)
                if shift_energy != None:
                    Shift_total_energy += shift_energy
                    Shift_total_latency += shift_latency
                    Shift_total_min_energy += shift_min_energy
                    Shift_total_min_latency += shift_min_latency
                    Shift_total_compute_energy += shift_compute_energy
                    Shift_total_DRAM_energy += shift_DRAM_energy
                    Shift_total_GB_energy += shift_GB_energy
                    Shift_total_NoC_energy += shift_NoC_energy
                    Shift_total_RF_energy += shift_RF_energy
                    shift = next(shift_iter,'over')
                    if shift == 'over':
                        # print('Shift Complete!!!!!!!!!!!!!!!!!') 
                        Shift_real_latency = Shift_total_latency
                else:
                    # print('!!!!!!!!!!!!!!!!!!!!!!! PASS !!!!!!!!!!!!!!!!!!!!!!!! ')
                    # print("------------------------------------> The passed dataflows for Conv/ Shift/ Adder are: ",(dataflow[c], dataflow[s], dataflow[a]))
                    Shift_total_energy = 5000
                    Shift_real_latency = 5000
                    break
            else:
                Shift_total_latency += 500000
                Shift_real_latency = 0
                shift_memory = 0

            if Adder != None:
                Adder_energy, Adder_latency, Adder_min_energy, Adder_min_latency, Adder_compute_energy, Adder_DRAM_energy, Adder_GB_energy, Adder_NoC_energy, Adder_RF_energy, Adder_memory = get_OPs_HW_metric(Adder, PE=Adder_PE, bit=bit_Adder, ratio=ratio, memory=shift_memory, dataflow=dataflow[a], v=False)
                if Adder_energy != None:
                    Adder_total_energy += Adder_energy
                    Adder_total_latency += Adder_latency
                    Adder_total_min_energy += Adder_min_energy
                    Adder_total_min_latency += Adder_min_latency
                    Adder_total_compute_energy += Adder_compute_energy
                    Adder_total_DRAM_energy += Adder_DRAM_energy
                    Adder_total_GB_energy += Adder_GB_energy
                    Adder_total_NoC_energy += Adder_NoC_energy
                    Adder_total_RF_energy += Adder_RF_energy
                    Adder = next(Adder_iter,'over')
                    if Adder == 'over':
                        # print('Adder Complete!!!!!!!!!!!!!!!!!') 
                        Adder_real_latency = Adder_total_latency
                else:
                    # print('!!!!!!!!!!!!!!!!!!!!!!! PASS !!!!!!!!!!!!!!!!!!!!!!!! ')
                    # print("------------------------------------> The passed dataflows for Conv/ Shift/ Adder are: ",(dataflow[c], dataflow[s], dataflow[a]))
                    Adder_total_energy = 5000
                    Adder_real_latency = 5000
                    break
            else:
                Adder_total_latency += 500000
                Adder_memory = 0
            
            if conv != None:
                conv_energy, conv_latency, conv_min_energy, conv_min_latency, conv_compute_energy, conv_DRAM_energy, conv_GB_energy, conv_NoC_energy, conv_RF_energy, conv_memory = get_OPs_HW_metric(conv, PE=Conv_PE, bit=bit_conv, ratio=ratio, memory=shift_memory+Adder_memory, dataflow=dataflow[c], v=False)
                if conv_energy != None:
                    Conv_total_energy += conv_energy
                    Conv_total_latency += conv_latency
                    Conv_total_min_energy += conv_min_energy
                    Conv_total_min_latency += conv_min_latency
                    Conv_total_compute_energy += conv_compute_energy
                    Conv_total_DRAM_energy += conv_DRAM_energy
                    Conv_total_GB_energy += conv_GB_energy
                    Conv_total_NoC_energy += conv_NoC_energy
                    Conv_total_RF_energy += conv_RF_energy
                    conv = next(conv_iter,'over')
                    if conv == 'over':
                        # print('Conv Complete!!!!!!!!!!!!!!!!!') 
                        Conv_real_latency = Conv_total_latency
                else:
                    # print('!!!!!!!!!!!!!!!!!!!!!!! PASS !!!!!!!!!!!!!!!!!!!!!!!! ')
                    # print("------------------------------------> The passed dataflows for Conv/ Shift/ Adder are: ",(dataflow[c], dataflow[s], dataflow[a]))
                    Conv_total_energy = 5000
                    Conv_real_latency = 5000
                    break
            else:
                Conv_total_latency += 500000
                conv_memory = 0

        
        # ######################## Normal Case #########################
        
        if (Conv_total_latency == min(Shift_total_latency, Conv_total_latency, Adder_total_latency)):
            if conv != 'over':
                # if shift == 'over':
                #     shift_memory == 0
                # if Adder == 'over':
                #     Adder_memory == 0
                if Conv_total_latency == Shift_total_latency:
                    shift_memory = 0
                if Conv_total_latency == Adder_total_latency:
                    Adder_memory = 0
                conv_energy, conv_latency, conv_min_energy, conv_min_latency, conv_compute_energy, conv_DRAM_energy, conv_GB_energy, conv_NoC_energy, conv_RF_energy, conv_memory = get_OPs_HW_metric(conv, PE=Conv_PE, bit=bit_conv, ratio=ratio, memory=shift_memory+Adder_memory, dataflow=dataflow[c], v=False)
                if conv_energy != None:
                    Conv_total_energy += conv_energy
                    Conv_total_latency += conv_latency
                    Conv_total_min_energy += conv_min_energy
                    Conv_total_min_latency += conv_min_latency
                    Conv_total_compute_energy += conv_compute_energy
                    Conv_total_DRAM_energy += conv_DRAM_energy
                    Conv_total_GB_energy += conv_GB_energy
                    Conv_total_NoC_energy += conv_NoC_energy
                    Conv_total_RF_energy += conv_RF_energy
                    conv = next(conv_iter,'over')
                    if conv == 'over':
                        # print('Conv complete !!!!!!!')
                        Conv_real_latency = Conv_total_latency
                else:
                    # print('!!!!!!!!!!!!!!!!!!!!!!! PASS !!!!!!!!!!!!!!!!!!!!!!!! ')
                    # print("------------------------------------> The passed dataflows for Conv/ Shift/ Adder are: ",(dataflow[c], dataflow[s], dataflow[a]))
                    Conv_total_energy = 5000
                    Conv_real_latency = 5000
                    break
            else:
                # print('Conv already completed !!!!!!!')
                conv_memory = 0
                Conv_total_latency = 500000
        
        elif (Shift_total_latency == min(Shift_total_latency, Conv_total_latency, Adder_total_latency)):
            if shift != 'over':
                if Shift_total_latency == Conv_total_latency:
                    conv_memory = 0
                if Shift_total_latency == Adder_total_latency:
                    Adder_memory = 0
                shift_energy, shift_latency, shift_min_energy, shift_min_latency, shift_compute_energy, shift_DRAM_energy, shift_GB_energy, shift_NoC_energy, shift_RF_energy, shift_memory = get_OPs_HW_metric(shift, PE=Shift_PE, bit=bit_shift, ratio=ratio, memory=conv_memory+Adder_memory, dataflow=dataflow[s], v=False)
                if shift_energy != None:
                    Shift_total_energy += shift_energy
                    Shift_total_latency += shift_latency
                    Shift_total_min_energy += shift_min_energy
                    Shift_total_min_latency += shift_min_latency
                    Shift_total_compute_energy += shift_compute_energy
                    Shift_total_DRAM_energy += shift_DRAM_energy
                    Shift_total_GB_energy += shift_GB_energy
                    Shift_total_NoC_energy += shift_NoC_energy
                    Shift_total_RF_energy += shift_RF_energy
                    shift = next(shift_iter,'over')
                    if shift == 'over':
                        # print('Shift complete !!!!!!!')
                        Shift_real_latency = Shift_total_latency
                else:
                    # print('!!!!!!!!!!!!!!!!!!!!!!! PASS !!!!!!!!!!!!!!!!!!!!!!!! ')
                    # print("------------------------------------> The passed dataflows for Conv/ Shift/ Adder are: ",(dataflow[c], dataflow[s], dataflow[a]))
                    Shift_total_energy = 5000
                    Shift_real_latency = 5000
                    break
            else:
                # print('Conv already completed !!!!!!!')
                shift_memory = 0
                Shift_total_latency = 500000

        elif (Adder_total_latency == min(Shift_total_latency, Conv_total_latency, Adder_total_latency)):
            if Adder != 'over':
                if Adder_total_latency == Conv_total_latency:
                    conv_memory = 0
                if Adder_total_latency == Shift_total_latency:
                    shift_memory = 0
                Adder_energy, Adder_latency, Adder_min_energy, Adder_min_latency, Adder_compute_energy, Adder_DRAM_energy, Adder_GB_energy, Adder_NoC_energy, Adder_RF_energy, Adder_memory = get_OPs_HW_metric(Adder, PE=Adder_PE, bit=bit_Adder, ratio=ratio, memory=conv_memory+Adder_memory, dataflow=dataflow[a], v=False)
                if Adder_energy != None:
                    Adder_total_energy += Adder_energy
                    Adder_total_latency += Adder_latency
                    Adder_total_min_energy += Adder_min_energy
                    Adder_total_min_latency += Adder_min_latency
                    Adder_total_compute_energy += Adder_compute_energy
                    Adder_total_DRAM_energy += Adder_DRAM_energy
                    Adder_total_GB_energy += Adder_GB_energy
                    Adder_total_NoC_energy += Adder_NoC_energy
                    Adder_total_RF_energy += Adder_RF_energy
                    Adder = next(Adder_iter,'over')
                    if Adder == 'over':
                        # print('Adder complete !!!!!!!')
                        Adder_real_latency = Adder_total_latency
                else:
                    # print('!!!!!!!!!!!!!!!!!!!!!!! PASS !!!!!!!!!!!!!!!!!!!!!!!! ')
                    # print("------------------------------------> The passed dataflows for Conv/ Shift/ Adder are: ",(dataflow[c], dataflow[s], dataflow[a]))
                    Adder_total_energy = 5000
                    Adder_real_latency = 5000
                    break
            else:
                # print('Conv already completed !!!!!!!')
                Adder_memory = 0
                Adder_total_latency = 500000 

        if Adder_total_latency==500000 and Conv_total_latency==500000 and Shift_total_latency==500000:
            # print('All complete!!!!!!!!')
            break

    # print("-------------------------------> Conv Total: energy: {} (min: {}, compute: {}, DRAM: {}, GB: {}, NoC: {}, RF: {}) mJ, latency: {} (min: {}) ms".format(Conv_total_energy, Conv_total_min_energy,Conv_total_compute_energy, Conv_total_DRAM_energy, Conv_total_GB_energy, Conv_total_NoC_energy, Conv_total_RF_energy, Conv_real_latency, Conv_total_min_latency))
    # print("-------------------------------> Shift Total: energy: {} (min: {}, compute: {}, DRAM: {}, GB: {}, NoC: {}, RF: {}) mJ, latency: {} (min: {}) ms".format(Shift_total_energy, Shift_total_min_energy,Shift_total_compute_energy, Shift_total_DRAM_energy, Shift_total_GB_energy, Shift_total_NoC_energy, Shift_total_RF_energy, Shift_real_latency, Shift_total_min_latency))
    # print("-------------------------------> Adder Total: energy: {} (min: {}, compute: {}, DRAM: {}, GB: {}, NoC: {}, RF: {}) mJ, latency: {} (min: {}) ms".format(Adder_total_energy, Adder_total_min_energy,Adder_total_compute_energy, Adder_total_DRAM_energy, Adder_total_GB_energy, Adder_total_NoC_energy, Adder_total_RF_energy, Adder_real_latency, Adder_total_min_latency))
    # print("-------------------------------> Total: EDP: {} mJ*ms, energy: {}, energy_woDRAM: {} (compute: {}, DRAM: {}, GB: {}, NoC: {}, RF: {}) mJ, latency: {} ms".format((Shift_total_energy+Conv_total_energy+Adder_total_energy)*max(Shift_real_latency,Conv_real_latency,Adder_real_latency), 
    # (Shift_total_energy+Conv_total_energy+Adder_total_energy),(Shift_total_energy+Conv_total_energy+Adder_total_energy)-(Shift_total_DRAM_energy+Conv_total_DRAM_energy+Adder_total_DRAM_energy), (Shift_total_compute_energy+Conv_total_compute_energy+Adder_total_compute_energy),(Shift_total_DRAM_energy+Conv_total_DRAM_energy+Adder_total_DRAM_energy),(Shift_total_GB_energy+Conv_total_GB_energy+Adder_total_GB_energy), (Shift_total_NoC_energy+Conv_total_NoC_energy+Adder_total_NoC_energy), (Shift_total_RF_energy+Conv_total_RF_energy+Adder_total_RF_energy), max(Shift_real_latency,Conv_real_latency,Adder_real_latency)))

    total_EDP.append((Shift_total_energy+Conv_total_energy+Adder_total_energy)*max(Shift_real_latency,Conv_real_latency,Adder_real_latency))
    total_energy.append((Shift_total_energy+Conv_total_energy+Adder_total_energy))
    total_energy.append((Shift_total_energy+Conv_total_energy+Adder_total_energy))
    total_wo_DRAM_energy.append(((Shift_total_energy+Conv_total_energy+Adder_total_energy)-(Shift_total_DRAM_energy+Conv_total_DRAM_energy+Adder_total_DRAM_energy)))
    total_latency.append((max(Shift_real_latency,Conv_real_latency,Adder_real_latency)))
    total_min_latency.append(max(Shift_total_min_latency,Conv_total_min_latency, Adder_total_min_latency))
    total_min_energy.append(Shift_total_min_energy+Conv_total_min_energy+Adder_total_min_energy)
    # total_min_energy.append()
    # total_min_latency.append()
    total_compute_energy.append((Shift_total_compute_energy+Conv_total_compute_energy+Adder_total_compute_energy))
    total_DRAM_energy.append((Shift_total_DRAM_energy+Conv_total_DRAM_energy+Adder_total_DRAM_energy))
    total_GB_energy.append((Shift_total_GB_energy+Conv_total_GB_energy+Adder_total_GB_energy))
    total_NoC_energy.append((Shift_total_NoC_energy+Conv_total_NoC_energy+Adder_total_NoC_energy))
    total_RF_energy.append((Shift_total_RF_energy+Conv_total_RF_energy+Adder_total_RF_energy))


    # # ################################### latency first #######################################
    # latency_optimal_tiling_factor_idx = []
    # # min_energy = total_energy[0]
    # min_latency = max(total_latency)
    # for i in range(len(total_latency)):
    #     # tiling_factor = tiling_factor_list[i]
    #     energy = total_energy[i]
    #     latency = total_latency[i]

    #     if not(energy > energy_constraint):
    #     # update total optimal
    #         if not(latency > min_latency):
    #             min_latency = latency
    #             latency_optimal_tiling_factor_idx = [i]
    
    # TODO: best dataflows
    # ################################### EDP first #######################################
    # EDP_optimal_tiling_factor_idx = []
    # # min_energy = total_energy[0]
    # min_EDP = max(total_EDP)
    # for i in range(len(total_EDP)):
    #     # tiling_factor = tiling_factor_list[i]
    #     # energy = total_energy[i]
    #     EDP = total_EDP[i]

    #     # if not(energy > energy_constraint):
    #     # update total optimal
    #     if not(EDP > min_EDP):
    #         min_EDP = EDP
    #         EDP_optimal_tiling_factor_idx = [i]
    # print(' ')
    # print(' ')
    # print("!!!!!!!!!!!!!!!!!!!!! Best EDP !!!!!!!!!!!!!!!!!!!!!")
    # print("------------------------------------> The best of all: EDP: {}, energy: {}, energy_woDRAM: {} (compute: {}, DRAM: {}, GB: {}, NoC: {}, RF: {}) mJ, latency: {} ms".format(
    # total_EDP[EDP_optimal_tiling_factor_idx[0]],
    # total_energy[EDP_optimal_tiling_factor_idx[0]],
    # total_wo_DRAM_energy[EDP_optimal_tiling_factor_idx[0]],
    # total_compute_energy[EDP_optimal_tiling_factor_idx[0]],
    # total_DRAM_energy[EDP_optimal_tiling_factor_idx[0]],
    # total_GB_energy[EDP_optimal_tiling_factor_idx[0]],
    # total_NoC_energy[EDP_optimal_tiling_factor_idx[0]],
    # total_RF_energy[EDP_optimal_tiling_factor_idx[0]],
    # total_EDP[EDP_optimal_tiling_factor_idx[0]],
    # ))
    # print("------------------------------------> The chosed dataflows for Conv/ Shift/ Adder are: ",(dataflow[EDP_optimal_tiling_factor_idx[0]//shift_dataflow//Adder_dataflow], 
    # dataflow[EDP_optimal_tiling_factor_idx[0]//Adder_dataflow % shift_dataflow], 
    # dataflow[EDP_optimal_tiling_factor_idx[0] % Adder_dataflow]
    # ))


    # # ################################### overall optimal #######################################
    # max_energy = max(total_energy)
    # min_energy = min(total_energy)

    # max_latency = max(total_latency)
    # min_latency = min(total_latency)

    # min_normal_metric = (total_energy[0]-min_energy)/max_energy + (total_latency[0]-min_latency)/max_latency

    # total_optimal_tiling_factor_idx = [0]

    # for i in range(len(total_energy)):

    #     normal_metric = (total_energy[i]-min_energy)/max_energy + (total_latency[i]-min_latency)/max_latency

    #     # update total optimal
    #     if normal_metric < min_normal_metric:
    #         min_normal_metric = normal_metric
    #         total_optimal_tiling_factor_idx = [i]
    #     elif normal_metric == min_normal_metric:
    #         total_optimal_tiling_factor_idx.append(i)

    # print(' ')
    # print("!!!!!!!!!!!!!!!!!!!!! Overall Optimal !!!!!!!!!!!!!!!!!!!!!")
    # print("------------------------------------> The best of all: energy: {}, energy_woDRAM: {} (compute: {}, DRAM: {}, GB: {}, NoC: {}, RF: {}) mJ, latency: {} ms".format(
    # total_energy[total_optimal_tiling_factor_idx[0]],
    # total_wo_DRAM_energy[total_optimal_tiling_factor_idx[0]],
    # total_compute_energy[total_optimal_tiling_factor_idx[0]],
    # total_DRAM_energy[total_optimal_tiling_factor_idx[0]],
    # total_GB_energy[total_optimal_tiling_factor_idx[0]],
    # total_NoC_energy[total_optimal_tiling_factor_idx[0]],
    # total_RF_energy[total_optimal_tiling_factor_idx[0]],
    # total_latency[total_optimal_tiling_factor_idx[0]],
    # ))
    # print("------------------------------------> The chosed dataflows for Conv/ Shift/ Adder are: ",(dataflow[total_optimal_tiling_factor_idx[0]//shift_dataflow//Adder_dataflow], 
    # dataflow[total_optimal_tiling_factor_idx[0]//Adder_dataflow % shift_dataflow], 
    # dataflow[total_optimal_tiling_factor_idx[0] % Adder_dataflow]
    # ))
    # return total_EDP[EDP_optimal_tiling_factor_idx[0]]
    
    # return round(total_EDP[0],3), round(total_min_energy[0],3), round(total_energy[0],3), round(total_wo_DRAM_energy[0],3), round(total_min_latency[0],3), round(total_latency[0],3), FLOPs, buffer
    return round(total_wo_DRAM_energy[0],3), round(total_latency[0],3), FLOPs