# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Implementation adapted from OFA: https://github.com/mit-han-lab/once-for-all

import copy
import random
import collections
import math

import torch
import torch.nn as nn
from torch.nn import init

from .modules.dynamic_layers import DynamicMBConvLayer, DynamicConvBnActLayer, DynamicLinearLayer, DynamicShortcutLayer
from .modules.static_layers import MobileInvertedResidualBlock
from .modules.nn_utils import make_divisible, int2list
from .modules.nn_base import MyNetwork
from .attentive_nas_static_model import AttentiveNasStaticModel
from .modules.nn_utils import make_divisible
# from chip_predictor import predictor
from predictor_reconfig import predictor


class shared_block(nn.Module):
    def __init__(self, C_in, C_out, kernel, expansion, use_se):
        super(shared_block, self).__init__()

        self.conv1 = nn.Parameter(torch.randn(C_in*expansion, C_in, 1, 1))
        self.conv2 = nn.Parameter(torch.randn(C_in*expansion, 1, kernel, kernel))
        self.conv3 = nn.Parameter(torch.randn(C_out, C_in*expansion, 1, 1))
        if use_se:
            num_mid = make_divisible(C_in*expansion // 4, divisor=8)
            self.se1 = nn.Parameter(torch.randn(num_mid, C_in*expansion, 1, 1))
            self.se2 = nn.Parameter(torch.randn(C_in*expansion, num_mid, 1, 1))

        # ############## initialization ######################
        init.kaiming_normal_(self.conv1, mode='fan_out')
        init.kaiming_normal_(self.conv2, mode='fan_out')
        init.kaiming_normal_(self.conv3, mode='fan_out')
        if use_se:
            init.kaiming_normal_(self.se1, mode='fan_out')
            init.kaiming_normal_(self.se2, mode='fan_out')
    
    def forward(self, x):
        # for key, item in self.shared_weight.items():
        #     x = x * item
        return x


class shared_shortcut(nn.Module):
    def __init__(self, C_in, C_out):
        super(shared_shortcut, self).__init__()

        self.conv = nn.Parameter(torch.randn(C_out, C_in, 1, 1))

        # ############## initialization ######################
        init.kaiming_normal_(self.conv, mode='fan_out')
    
    def forward(self, x):
        # for key, item in self.shared_weight.items():
        #     x = x * item
        return x


class AttentiveNasDynamicModel(MyNetwork):

    def __init__(self, supernet, n_classes=1000, bn_param=(0., 1e-5)):
        super(AttentiveNasDynamicModel, self).__init__()

        self.pretrain_conv = False
        self.pretrain_add = False
        self.mix = False
        self.mix_training = False
        self.prob = 0
        self.supernet = supernet
        self.n_classes = n_classes
        self.use_v3_head = getattr(self.supernet, 'use_v3_head', False)
        self.stage_names = ['first_conv', 'mb1', 'mb2', 'mb3', 'mb4', 'mb5', 'mb6', 'mb7', 'last_conv']

        self.width_list, self.depth_list, self.ks_list, self.expand_ratio_list, self.type = [], [], [], [], []
        for name in self.stage_names:
            block_cfg = getattr(self.supernet, name)
            self.width_list.append(block_cfg.c)
            if name.startswith('mb'):
                self.depth_list.append(block_cfg.d)
                self.ks_list.append(block_cfg.k)
                self.expand_ratio_list.append(block_cfg.t)
                self.type.append(block_cfg.type)
                # print('self.type',self.type)
        self.resolution_list = self.supernet.resolutions

        self.cfg_candidates = {
            'type': self.type ,
            'resolution': self.resolution_list ,
            'width': self.width_list,
            'depth': self.depth_list,
            'kernel_size': self.ks_list,
            'expand_ratio': self.expand_ratio_list
        }
        # print('self.cfg_candidates',self.cfg_candidates)

        #first conv layer, including conv, bn, act
        out_channel_list, act_func, stride = \
            self.supernet.first_conv.c, self.supernet.first_conv.act_func, self.supernet.first_conv.s
        self.first_conv = DynamicConvBnActLayer(
            in_channel_list=int2list(3), out_channel_list=out_channel_list, 
            kernel_size=3, stride=stride, act_func=act_func,
        )

        # inverted residual blocks
        self.block_group_info = []
        blocks = []
        _block_index = 0
        feature_dim = out_channel_list
        for stage_id, key in enumerate(self.stage_names[1:-1]):
            block_cfg = getattr(self.supernet, key)
            type = block_cfg.type
            # print('type',type)
            width = block_cfg.c
            n_block = max(block_cfg.d)
            act_func = block_cfg.act_func
            ks = block_cfg.k
            expand_ratio_list = block_cfg.t
            use_se = block_cfg.se

            self.block_group_info.append([_block_index + i for i in range(n_block)])
            _block_index += n_block

            output_channel = width
            for i in range(n_block):
                stride = block_cfg.s if i == 0 else 1
                if min(expand_ratio_list) >= 4:
                    expand_ratio_list = [_s for _s in expand_ratio_list if _s >= 4] if i == 0 else expand_ratio_list
                
                shared_block_para = shared_block(max(feature_dim), max(output_channel), max(ks), max(expand_ratio_list), use_se)
                shared_shortcut_para = shared_shortcut(max(feature_dim), max(output_channel))

                # for j, key in enumerate(type):
                    # print('key',key)
                mobile_inverted_conv = DynamicMBConvLayer(
                    in_channel_list=feature_dim, 
                    out_channel_list=output_channel, 
                    kernel_size_list=ks,
                    expand_ratio_list=expand_ratio_list, 
                    stride=stride, 
                    act_func=act_func, 
                    use_se=use_se,
                    channels_per_group=getattr(self.supernet, 'channels_per_group', 1),
                    shared_weight=shared_block_para,
                )
                shortcut = DynamicShortcutLayer(feature_dim, output_channel, reduction=stride, shared_weight=shared_shortcut_para)
                blocks.append(MobileInvertedResidualBlock(mobile_inverted_conv, shortcut))
                feature_dim = output_channel
        self.blocks = nn.ModuleList(blocks)

        last_channel, act_func = self.supernet.last_conv.c, self.supernet.last_conv.act_func
        if not self.use_v3_head:
            self.last_conv = DynamicConvBnActLayer(
                    in_channel_list=feature_dim, out_channel_list=last_channel,
                    kernel_size=1, act_func=act_func,
            )
        else:
            expand_feature_dim = [f_dim * 6 for f_dim in feature_dim]
            self.last_conv = nn.Sequential(collections.OrderedDict([
                ('final_expand_layer', DynamicConvBnActLayer(
                    feature_dim, expand_feature_dim, kernel_size=1, use_bn=True, act_func=act_func)
                ),
                ('pool', nn.AdaptiveAvgPool2d((1,1))),
                ('feature_mix_layer', DynamicConvBnActLayer(
                    in_channel_list=expand_feature_dim, out_channel_list=last_channel,
                    kernel_size=1, act_func=act_func, use_bn=False,)
                ),
            ]))

        #final conv layer
        self.classifier = DynamicLinearLayer(
            in_features_list=last_channel, out_features=n_classes, bias=True
        )

        # set bn param
        self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

        # runtime_depth
        self.runtime_depth = [len(block_idx) for block_idx in self.block_group_info]

        self.zero_residual_block_bn_weights()

        self.active_dropout_rate = 0
        self.active_drop_connect_rate = 0
        # TODO:
        self.active_resolution = 32


    def zero_residual_block_bn_weights(self):
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, MobileInvertedResidualBlock):
                    if isinstance(m.mobile_inverted_conv, DynamicMBConvLayer) and m.shortcut is not None:
                        m.mobile_inverted_conv.point_linear_conv.bn.bn.weight.zero_()
                        # TODO:
                        # m.mobile_inverted_conv.point_linear_shift.bn.bn.weight.zero_()
                        m.mobile_inverted_conv.point_linear_adder.bn.bn.weight.zero_()


    @staticmethod
    def name():
        return 'AttentiveNasModel'


    def forward(self, x, conv_ws=False):
        # resize input to target resolution first
        # print('ok!!!!!!!')
        self.kl_loss = 0
        if x.size(-1) != self.active_resolution:
            x = torch.nn.functional.interpolate(x, size=self.active_resolution, mode='bicubic')

        # first conv
        x = self.first_conv(x)
        # print('ok!!!!!!!')
        # blocks
        for stage_id, block_idx in enumerate(self.block_group_info):
            # print('index:',stage_id)
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            for idx in active_idx:
                self.blocks[idx].mobile_inverted_conv.conv_ws = conv_ws
                self.blocks[idx].shortcut.conv_ws = conv_ws
                x = self.blocks[idx](x)
                # if self.blocks[idx].mobile_inverted_conv.type == 'add' and self.training:
                # FIXME:
                # if self.training:
                #     self.kl_loss += self.blocks[idx].mobile_inverted_conv.kl_loss
                #     self.kl_loss += self.blocks[idx].shortcut.kl_loss

        x = self.last_conv(x)
        x = x.mean(3, keepdim=True).mean(2, keepdim=True)  # global average pooling
        x = torch.squeeze(x)

        if self.active_dropout_rate > 0 and self.training:
            x = torch.nn.functional.dropout(x, p = self.active_dropout_rate)

        x = self.classifier(x)
        return x


    @property
    def module_str(self):
        _str = self.first_conv.module_str + '\n'
        _str += self.blocks[0].module_str + '\n'

        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            for idx in active_idx:
                _str += self.blocks[idx].module_str + '\n'
        if not self.use_v3_head:
            _str += self.last_conv.module_str + '\n'
        else:
            _str += self.last_conv.final_expand_layer.module_str + '\n'
            _str += self.last_conv.feature_mix_layer.module_str + '\n'
        _str += self.classifier.module_str + '\n'
        return _str

    @property
    def config(self):
        return {
            'name': AttentiveNasDynamicModel.__name__,
            'bn': self.get_bn_param(),
            'first_conv': self.first_conv.config,
            'blocks': [
                block.config for block in self.blocks
            ],
            'last_conv': self.last_conv.config if not self.use_v3_head else None,
            'final_expand_layer': self.last_conv.final_expand_layer if self.use_v3_head else None,
            'feature_mix_layer': self.last_conv.feature_mix_layer if self.use_v3_head else None,
            'classifier': self.classifier.config,
            'resolution': self.active_resolution
        }


    @staticmethod
    def build_from_config(config):
        raise ValueError('do not support this function')
    

    def get_kl_loss(self):
        return self.kl_loss

    # FIXME:
    """ set, sample and get active sub-networks """
    def set_active_subnet(self, resolution=224, width=None, depth=None, kernel_size=None, expand_ratio=None, type='conv', **kwargs):
        assert len(depth) == len(kernel_size) == len(expand_ratio) == len(width) - 2
        #set resolution
        self.active_resolution = resolution

        # first conv
        self.first_conv.active_out_channel = width[0] 

        for stage_id, (c, k, e, d, t) in enumerate(zip(width[1:-1], kernel_size, expand_ratio, depth, type)):
        
            start_idx, end_idx = min(self.block_group_info[stage_id]), max(self.block_group_info[stage_id])
            i = 0
            for block_id in range(start_idx, start_idx+d):
                block = self.blocks[block_id]
                #block output channels
                # mix training 
                if self.mix_training and self.pretrain_add:
                    block.mobile_inverted_conv.shared_weight.conv1.requires_grad = False
                    block.mobile_inverted_conv.shared_weight.conv2.requires_grad = False
                    block.mobile_inverted_conv.shared_weight.conv3.requires_grad = False
                    if block.mobile_inverted_conv.use_se:
                        block.mobile_inverted_conv.shared_weight.se1.requires_grad = False
                        block.mobile_inverted_conv.shared_weight.se2.requires_grad = False
                else:
                    block.mobile_inverted_conv.shared_weight.conv1.requires_grad = True
                    block.mobile_inverted_conv.shared_weight.conv2.requires_grad = True
                    block.mobile_inverted_conv.shared_weight.conv3.requires_grad = True
                    if block.mobile_inverted_conv.use_se:
                        block.mobile_inverted_conv.shared_weight.se1.requires_grad = True
                        block.mobile_inverted_conv.shared_weight.se2.requires_grad = True
                
                block.mobile_inverted_conv.active_out_channel = c
                if block.shortcut is not None:
                    block.shortcut.active_out_channel = c
                    # mix training 
                    if self.mix_training and self.pretrain_add:
                        block.shortcut.shared_weight.conv.requires_grad = False
                    else:
                        block.shortcut.shared_weight.conv.requires_grad = True

                #dw kernel size
                block.mobile_inverted_conv.active_kernel_size = k

                #dw expansion ration
                block.mobile_inverted_conv.active_expand_ratio = e

                # TODO: type 
                block.mobile_inverted_conv.type = t[i]
                block.shortcut.type = t[i]
                # block.mobile_inverted_conv.type = t
                # block.shortcut.type = t
                i += 1

        #IRBlocks repated times
        for i, d in enumerate(depth):
            self.runtime_depth[i] = min(len(self.block_group_info[i]), d)

        #last conv
        if not self.use_v3_head:
            self.last_conv.active_out_channel = width[-1]
        else:
            # default expansion ratio: 6
            self.last_conv.final_expand_layer.active_out_channel = width[-2] * 6
            self.last_conv.feature_mix_layer.active_out_channel = width[-1]
    

    def get_active_subnet_settings(self):
        r = self.active_resolution
        width, depth, kernel_size, expand_ratio, type= [], [], [],  [], []

        #first conv
        width.append(self.first_conv.active_out_channel)
        for stage_id in range(len(self.block_group_info)):
            start_idx = min(self.block_group_info[stage_id])
            # print(self.block_group_info[stage_id])
            block = self.blocks[start_idx]  #first block
            width.append(block.mobile_inverted_conv.active_out_channel)
            kernel_size.append(block.mobile_inverted_conv.active_kernel_size)
            expand_ratio.append(block.mobile_inverted_conv.active_expand_ratio)
            # type.append((block.mobile_inverted_conv.type))
            depth.append(self.runtime_depth[stage_id])
            t = []
            for index in self.block_group_info[stage_id]:
                block = self.blocks[index]
                t.append((block.mobile_inverted_conv.type))
            # print(t)
            type.append(t)

        
        if not self.use_v3_head:
            width.append(self.last_conv.active_out_channel)
        else:
            width.append(self.last_conv.feature_mix_layer.active_out_channel)

        return {
            'resolution': r,
            'width': width,
            'kernel_size': kernel_size,
            'expand_ratio': expand_ratio,
            'depth': depth,
            'type': type,
        }

    def set_dropout_rate(self, dropout=0, drop_connect=0, drop_connect_only_last_two_stages=True):
        self.active_dropout_rate = dropout
        for idx, block in enumerate(self.blocks):
            if drop_connect_only_last_two_stages:
                if idx not in self.block_group_info[-1] + self.block_group_info[-2]:
                    continue
            this_drop_connect_rate = drop_connect * float(idx) / len(self.blocks)
            block.drop_connect_rate = this_drop_connect_rate


    def sample_min_subnet(self):
        return self._sample_active_subnet(min_net=True)


    def sample_max_subnet(self):
        # cfg = self._sample_active_subnet(max_net=True)
        # cfg['EDP'] = self.compute_active_subnet_flops()
        # print('FLOPs: ', cfg['EDP'])
        return self._sample_active_subnet(max_net=True)
    

    def sample_active_subnet(self, compute_flops=False):
        cfg = self._sample_active_subnet(
            False, False
        ) 
        if compute_flops:
            cfg['flops'] = self.compute_active_subnet_flops()
        return cfg


    def settings_for_mixture_training(self, pretrain_conv=False, pretrain_add=False, mix=True, mix_training=False, prob=0):
        self.pretrain_conv = pretrain_conv
        self.pretrain_add = pretrain_add
        self.mix = mix
        self.mix_training = mix_training
        self.prob = prob

    
    def creat_model(self, cfg):
        OPs_list = []
        # ######################## first conv ##########################
        # OPs_list.append({"type": "conv", "kernel_size": 3, "stride": self.supernet.first_conv.s, "input_H": cfg['resolution'], "input_W": cfg['resolution'], "input_C": 3, "output_E": cfg['resolution']//self.supernet.first_conv.s, "output_F": cfg['resolution']//self.supernet.first_conv.s, "output_M": cfg['width'][0]})

        in_channel = cfg['width'][0]
        input_size = cfg['resolution']//self.supernet.first_conv.s
        # ########################## block ##########################
        for stage_id, (key, c, k, e, d, t) in enumerate(zip(self.stage_names[1:-1], cfg['width'][1:-1], cfg['kernel_size'], cfg['expand_ratio'], cfg['depth'], cfg['type'])):
        
            block_cfg = getattr(self.supernet, key)
            
            for block_id in range(d):
                # first pointwise layer
                # OPs_list.append({"type": t, "kernel_size": 1, "stride": 1, "input_H": input_size, "input_W": input_size, "input_C": in_channel, "output_E": input_size, "output_F": input_size, "output_M": in_channel*e})
                
                # # middle depthwise layer
                stride = block_cfg.s if block_id == 0 else 1
                use_se = block_cfg.se
                middle_channel = make_divisible(round(in_channel * e), 8)
                # for i in range(in_channel*e):
                #     OPs_list.append({"type": t, "kernel_size": k, "stride": stride, "input_H": input_size, "input_W": input_size, "input_C": 1, "output_E": input_size//stride, "output_F": input_size//stride, "output_M": 1})
                #     input_size = input_size//stride
                
                # # last pointwise layer
                # OPs_list.append({"type": t, "kernel_size": 1, "stride": 1, "input_H": input_size, "input_W": input_size, "input_C": in_channel*e, "output_E": input_size, "output_F": input_size, "output_M": c})
                # in_channel = c

                OPs_list.append({"type": t[block_id], "kernel_size": k, "expansion": e, "stride": stride, "input_H": input_size, "input_W": input_size, "input_C": in_channel, "output_E": input_size//stride, "output_F": input_size//stride, "output_M": c, "use_se":use_se, "middle_channel":middle_channel})
                in_channel = c
                input_size = input_size//stride

        # ######################### last conv #########################
        # if not self.use_v3_head:
        #     OPs_list.append({"type": 'Conv', "kernel_size": 1, "stride": 1, "input_H": input_size, "input_W": input_size, "input_C": in_channel, "output_E": input_size, "output_F": input_size, "output_M": cfg['width'][-1]})
        # else:
        #     OPs_list.append({"type": 'Conv', "kernel_size": 1, "stride": 1, "input_H": input_size, "input_W": input_size, "input_C": in_channel, "output_E": input_size, "output_F": input_size, "output_M": cfg['width'][-2]*6})
        #     OPs_list.append({"type": 'Conv', "kernel_size": 1, "stride": 1, "input_H": 1, "input_W": 1, "input_C": cfg['width'][-2]*6, "output_E": 1, "output_F": 1, "output_M": cfg['width'][-1]})
        
        return OPs_list


    def compute_EDP(self, cfg, stage='coarse', Conv_PE=168):
        model = self.creat_model(cfg)
        # print(model)
        return predictor(model, stage, Conv_PE)
        # return self.compute_active_subnet_flops()


    # TODO:
    def sample_active_subnet_within_range(self, targeted_min_EDP, targeted_max_EDP):
        while True:
            cfg = self._sample_active_subnet() 
            print('cfg',cfg)
            model = self.creat_model(cfg)
            cfg['EDP'] = predictor(model)
            # cfg['EDP'] = self.compute_active_subnet_flops()
            # print('FLOPs: ', cfg['EDP'])
            if cfg['EDP'] >= targeted_min_EDP and cfg['EDP'] <= targeted_max_EDP:
                return cfg
            # if cfg['flops'] >= targeted_min_flops and cfg['flops'] <= targeted_max_flops:
                # return cfg


    # ##################### no adder constraint #########################
    def _sample_active_subnet(self, min_net=False, max_net=False):

        sample_cfg = lambda candidates, sample_min, sample_max: \
            min(candidates) if sample_min else (max(candidates) if sample_max else random.choice(candidates))

        def _choose_helper(g1, g2, prob=0.3):
            # assert type(g1) == type(g2)
            return g1 if random.random() < prob else g2

        cfg = {}
        # sample a resolution
        cfg['resolution'] = sample_cfg(self.cfg_candidates['resolution'], min_net, max_net)
        for k in ['width', 'depth', 'kernel_size', 'expand_ratio']:
            cfg[k] = []
            if k == 'depth':
                cfg['type'] = []
            i = 0
            for vv in self.cfg_candidates[k]:
                cfg[k].append(sample_cfg(int2list(vv), min_net, max_net))
                # support mixture training 
                if k == 'depth':
                    type = []
                    for j in range(cfg[k][i]):
                        if self.mix_training:
                            # cfg['type'].append('shift')
                            if self.pretrain_conv:
                            # if self.pretrain_conv or i==5 or i==6:
                                type.append('conv')
                            else:
                                type.append(_choose_helper(self.cfg_candidates['type'][0][1], self.cfg_candidates['type'][0][0], self.prob))   
                        
                        else:
                            type.append(_choose_helper(self.cfg_candidates['type'][0][1], self.cfg_candidates['type'][0][0], self.prob))
                            
                    i += 1
                    cfg['type'].append(type)
                    # print(type)
            
        self.set_active_subnet(
            cfg['resolution'], cfg['width'], cfg['depth'], cfg['kernel_size'], cfg['expand_ratio'], cfg['type']
        )
        # print(cfg)
        return cfg


    # # ##################### 1stage/propro training #########################
    # def _sample_active_subnet(self, min_net=False, max_net=False):

    #     sample_cfg = lambda candidates, sample_min, sample_max: \
    #         min(candidates) if sample_min else (max(candidates) if sample_max else random.choice(candidates))

    #     def _choose_helper(g1, g2, prob=0.3):
    #         # assert type(g1) == type(g2)
    #         return g1 if random.random() < prob else g2

    #     cfg = {}
    #     # sample a resolution
    #     cfg['resolution'] = sample_cfg(self.cfg_candidates['resolution'], min_net, max_net)
    #     for k in ['width', 'depth', 'kernel_size', 'expand_ratio']:
    #         cfg[k] = []
    #         if k == 'depth':
    #             cfg['type'] = []
    #         i = 0
    #         for vv in self.cfg_candidates[k]:
    #             cfg[k].append(sample_cfg(int2list(vv), min_net, max_net))
    #             # support mixture training 
    #             if k == 'depth':
    #                 type = []
    #                 for j in range(cfg[k][i]):
    #                     if self.mix_training:
    #                         # cfg['type'].append('shift')
    #                         if self.pretrain_conv or i==1 or (i==2 and j<2) or (i==5 and j>cfg[k][i]-5) or i==6:
    #                         # if self.pretrain_conv or i==5 or i==6:
    #                             type.append('conv')
    #                         else:
    #                             # cfg['type'].append('add')
    #                             # type.append(random.choice(self.cfg_candidates['type'][0]))
    #                             type.append(_choose_helper(self.cfg_candidates['type'][0][1], self.cfg_candidates['type'][0][0], self.prob))   
                        
    #                     else:
    #                         if i==1 or (i==2 and j<2) or (i==5 and j>cfg[k][i]-5) or i==6:
    #                         # if i==5 or i==6:
    #                             type.append('conv')
    #                         # type.append(random.choice(self.cfg_candidates['type'][0]))
    #                         else:
    #                             type.append(_choose_helper(self.cfg_candidates['type'][0][1], self.cfg_candidates['type'][0][0], self.prob))
                            
    #                 i += 1
    #                 cfg['type'].append(type)
    #                 # print(type)
            
    #     self.set_active_subnet(
    #         cfg['resolution'], cfg['width'], cfg['depth'], cfg['kernel_size'], cfg['expand_ratio'], cfg['type']
    #     )
    #     # print(cfg)
    #     return cfg


    def mutate_and_reset(self, cfg, prob=0.1, keep_resolution=False):
        cfg = copy.deepcopy(cfg)
        pick_another = lambda x, candidates: x if len(candidates) == 1 else random.choice([v for v in candidates if v != x])
        # sample a resolution
        r = random.random()
        if r < prob and not keep_resolution:
            cfg['resolution'] = pick_another(cfg['resolution'], self.cfg_candidates['resolution'])

        # sample channels, depth, kernel_size, expand_ratio
        for k in ['width', 'depth', 'kernel_size', 'expand_ratio', 'type']:
            # FIXME:
            if k is not 'type':
                for _i, _v in enumerate(cfg[k]):
                    r = random.random()
                    if r < prob:
                        cfg[k][_i] = pick_another(cfg[k][_i], int2list(self.cfg_candidates[k][_i]))
            else:
                # print(cfg['type'])
                j = 0
                length = []
                for type in cfg[k]:
                    # print(len(type))
                    # print(cfg['depth'][j])
                    if len(type) == cfg['depth'][j] or len(type) < cfg['depth'][j]:
                        for _i, _v in enumerate(type):
                            r = random.random()
                            if r < prob:
                                type[_i] = pick_another(type[_i], int2list(self.cfg_candidates[k][j]))
                        
                        if len(type) != cfg['depth'][j]:
                            for i in range(cfg['depth'][j]-len(type)):
                                type.append(pick_another('conv', int2list(self.cfg_candidates[k][j])))
                    else:
                        new_type = []
                        for i in range(cfg['depth'][j]):
                            r = random.random()
                            if r < prob:
                                new_type.append(pick_another(type[i], int2list(self.cfg_candidates[k][j])))
                            else:
                                new_type.append(type[i])
                        type = new_type
                    
                    length.append(len(type))
                    j += 1 
                # print(cfg['type'])
        # print(cfg['depth'])
        # print(length)
        # print(cfg['depth']==length)
        self.set_active_subnet(
            cfg['resolution'], cfg['width'], cfg['depth'], cfg['kernel_size'], cfg['expand_ratio'], cfg['type']
        )
        
        return cfg


    def crossover_and_reset(self, cfg1, cfg2, p=0.5):
        def _cross_helper(g1, g2, prob):
            assert type(g1) == type(g2)
            if isinstance(g1, int):
                return g1 if random.random() < prob else g2
            if isinstance(g1, str):
                return g1 if random.random() < prob else g2
            elif isinstance(g1, list):
                return [v1 if random.random() < prob else v2 for v1, v2 in zip(g1, g2)]
            else:
                raise NotImplementedError

        # print(cfg1['depth'])
        # print(cfg2['depth'])
        cfg = {}
        cfg['resolution'] = cfg1['resolution'] if random.random() < p else cfg2['resolution']
        for k in ['width', 'depth', 'kernel_size', 'expand_ratio', 'type']:
            # FIXME:
            if k is not 'type':
                cfg[k] = _cross_helper(cfg1[k], cfg2[k], p)
            else:
                # print(cfg['depth'])
                cfg[k] = []
                for i, (type1, type2) in enumerate(zip(cfg1[k], cfg2[k])):
                    type_new = []
                    for j in range(min(len(type1), len(type2))):
                        type_new.append(_cross_helper(type1[j], type2[j], p))
                    # print(cfg['depth'][i])
                    # print(len(type_new))
                    # print(len(type1))
                    # print(len(type2))
                    if len(type_new) != cfg['depth'][i]:
                        index = len(type_new)
                        if len(type1) == cfg['depth'][i] or len(type1) > cfg['depth'][i]:
                            for n in range(cfg['depth'][i]-index): 
                                type_new.append(type1[index+n])
                        else:
                            # print(cfg['depth'][i])
                            # print(len(type_new))
                            # print(len(type2))
                            for n in range(cfg['depth'][i]-index): 
                                # print(n)
                                type_new.append(type2[index+n])
                    
                    cfg[k].append(type_new)

        self.set_active_subnet(
            cfg['resolution'], cfg['width'], cfg['depth'], cfg['kernel_size'], cfg['expand_ratio'], cfg['type']
        )
        return cfg


    def get_active_subnet(self, preserve_weight=True):
        with torch.no_grad():
            first_conv = self.first_conv.get_active_subnet(3, preserve_weight)

            blocks = []
            input_channel = first_conv.out_channels
            # blocks
            for stage_id, block_idx in enumerate(self.block_group_info):
                depth = self.runtime_depth[stage_id]
                active_idx = block_idx[:depth]
                stage_blocks = []
                for idx in active_idx:
                    stage_blocks.append(MobileInvertedResidualBlock(
                        self.blocks[idx].mobile_inverted_conv.get_active_subnet(input_channel, preserve_weight),
                        self.blocks[idx].shortcut.get_active_subnet(input_channel, preserve_weight) if self.blocks[idx].shortcut is not None else None
                    ))
                    input_channel = stage_blocks[-1].mobile_inverted_conv.out_channels
                blocks += stage_blocks

            if not self.use_v3_head:
                last_conv = self.last_conv.get_active_subnet(input_channel, preserve_weight)
                in_features = last_conv.out_channels
            else:
                final_expand_layer = self.last_conv.final_expand_layer.get_active_subnet(input_channel, preserve_weight)
                feature_mix_layer = self.last_conv.feature_mix_layer.get_active_subnet(input_channel*6, preserve_weight)
                in_features = feature_mix_layer.out_channels
                last_conv = nn.Sequential(
                    final_expand_layer,
                    nn.AdaptiveAvgPool2d((1,1)),
                    feature_mix_layer
                )

            classifier = self.classifier.get_active_subnet(in_features, preserve_weight)

            _subnet = AttentiveNasStaticModel(
                first_conv, blocks, last_conv, classifier, self.active_resolution, use_v3_head=self.use_v3_head
            )
            _subnet.set_bn_param(**self.get_bn_param())
            return _subnet


    def get_active_net_config(self):
        raise NotImplementedError

    # TODO:
    def compute_active_subnet_flops(self):

        def count_conv(c_in, c_out, size_out, groups, k):
            kernel_ops = k**2
            output_elements = c_out * size_out**2
            ops = c_in * output_elements * kernel_ops / groups
            return ops

        def count_linear(c_in, c_out):
            return c_in * c_out

        total_ops = 0

        c_in = 3
        size_out = self.active_resolution // self.first_conv.stride
        c_out = self.first_conv.active_out_channel

        total_ops += count_conv(c_in, c_out, size_out, 1, 3)
        c_in = c_out

        # mb blocks
        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            for idx in active_idx:
                block = self.blocks[idx]
                c_middle = make_divisible(round(c_in * block.mobile_inverted_conv.active_expand_ratio), 8)
                # 1*1 conv
                # TODO:inverted_bottleneck_conv and inverted_bottleneck_shift
                if block.mobile_inverted_conv.inverted_bottleneck_conv is not None:
                    total_ops += count_conv(c_in, c_middle, size_out, 1, 1)
                # dw conv
                stride = 1 if idx > active_idx[0] else block.mobile_inverted_conv.stride
                if size_out % stride == 0:
                    size_out = size_out // stride
                else:
                    size_out = (size_out +1) // stride
                total_ops += count_conv(c_middle, c_middle, size_out, c_middle, block.mobile_inverted_conv.active_kernel_size)
                # 1*1 conv
                c_out = block.mobile_inverted_conv.active_out_channel
                total_ops += count_conv(c_middle, c_out, size_out, 1, 1)
                #se
                if block.mobile_inverted_conv.use_se:
                    num_mid = make_divisible(c_middle // block.mobile_inverted_conv.depth_conv.se.reduction, divisor=8)
                    total_ops += count_conv(c_middle, num_mid, 1, 1, 1) * 2
                if block.shortcut and c_in != c_out:
                    total_ops += count_conv(c_in, c_out, size_out, 1, 1)
                c_in = c_out

        if not self.use_v3_head:
            c_out = self.last_conv.active_out_channel
            total_ops += count_conv(c_in, c_out, size_out, 1, 1)
        else:
            c_expand = self.last_conv.final_expand_layer.active_out_channel
            c_out = self.last_conv.feature_mix_layer.active_out_channel
            total_ops += count_conv(c_in, c_expand, size_out, 1, 1)
            total_ops += count_conv(c_expand, c_out, 1, 1, 1)

        # n_classes
        total_ops += count_linear(c_out, self.n_classes)
        return total_ops / 1e6


    def load_weights_from_pretrained_models(self, checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            checkpoint = torch.load(f, map_location='cpu')
        assert isinstance(checkpoint, dict)
        pretrained_state_dicts = checkpoint['state_dict']
        for k, v in self.state_dict().items():
            name = 'module.' + k if not k.startswith('module') else k
            v.copy_(pretrained_state_dicts[name])

