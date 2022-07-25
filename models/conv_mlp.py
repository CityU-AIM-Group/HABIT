import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
# from torchvision.models.utils import load_state_dict_from_url
from torch.hub import load_state_dict_from_url
from torch import Tensor
from torch.jit.annotations import List

# import torch
from torch import einsum
from einops.layers.torch import Rearrange, Reduce

from .g_mlp import gMLP, gMLPVision, gMLPBlock, SpatialGatingUnit, Residual, PreNorm


__all__ = ['DenseNet', 'ssl_densenet121', 'densenet169', 'densenet201', 'densenet161']

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs):
        # type: (List[Tensor]) -> Tensor
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input):
        # type: (List[Tensor]) -> bool
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, input):
        # type: (List[Tensor]) -> Tensor
        def closure(*inputs):
            return self.bn_function(inputs)

        return cp.checkpoint(closure, *input)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input):
        # type: (List[Tensor]) -> (Tensor)
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input):
        # type: (Tensor) -> (Tensor)
        pass

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input):  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, memory_efficient=False, fusion=False):
        super(DenseNet, self).__init__()

        self.fusion = fusion
        self.block_config = block_config

        # First convolution
        # self.input_layer = nn.Sequential(
        #     nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
        #                         padding=3, bias=False),
        #     nn.BatchNorm2d(num_init_features),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        # )
        self.input_layer = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # self.input_layer = nn.ModuleList([
        #     nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
        #                         padding=3, bias=False),
        #     nn.BatchNorm2d(num_init_features),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        # ])

        # Each denseblock
        num_features = num_init_features

        # self.block = []
        # self.trans = []
        self.block = nn.ModuleList()
        self.transition = nn.ModuleList()

        for i, num_layers in enumerate(block_config):
            # self.block.append(_DenseBlock(
            #     num_layers=num_layers,
            #     num_input_features=num_features,
            #     bn_size=bn_size,
            #     growth_rate=growth_rate,
            #     drop_rate=drop_rate,
            #     memory_efficient=memory_efficient
            # ))
            
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            
            # self.dense_layer = nn.Sequential(OrderedDict([
            #     ('denseblock%d' % (i + 1), block),
            #     ]))
            self.block.append(block)
            # self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                # self.transition.append(trans)
                # self.transition_layer = nn.Sequential(OrderedDict([
                #     ('transition%d' % (i + 1), trans)
                #     ]))
                self.transition.append(trans)
                # self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.norm5 = nn.Sequential(OrderedDict([
                    ('norm5', nn.BatchNorm2d(num_features))
                    ]))
        # self.norm5 = nn.Sequential(nn.BatchNorm2d(num_features))
        # self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x, out_fea=False, out_fea_map=False):
        # features = self.features(x)
        side_features = []
        features = self.input_layer(x)
        for i, num_layers in enumerate(self.block_config):
            # import ipdb; ipdb.set_trace()
            features = self.block[i].cuda()(features)
            if i != len(self.block_config) - 1:
                features = self.transition[i].cuda()(features)
            if self.fusion == True:
                side_features.append(features)
        features = self.norm5(features)
        # import ipdb; ipdb.set_trace()
        if self.fusion == True:
            side_features[-1] = features

        out = F.relu(features, inplace=True)
        feat_map = out
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        feat = out
        out = self.classifier(out)

        if self.fusion:
            if out_fea == True and out_fea_map == False:
                return out, feat, side_features
            elif out_fea == True and out_fea_map==True:
                return out, feat, feat_map, side_features
            else:
                return out, side_features
        else:
            if out_fea == True and out_fea_map == False:
                return out, feat
            elif out_fea == True and out_fea_map==True:
                return out, feat, feat_map
            else:
                return out


def _load_state_dict(model, model_url, progress):
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
    pattern_transition = re.compile(
        r'^(.*transition\d+\.(?:norm|relu|conv))\.(?:weight|bias|running_mean|running_var)$')
    pattern_norm = re.compile(
        r'^(.*norm\d)+\.(?:weight|bias|running_mean|running_var)$')

    state_dict = load_state_dict_from_url(model_url, progress=progress)
    num = 0
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            new_key = new_key[14:19]+ '.' + str(int(new_key[19])-1) + new_key[20:] ###
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
        
        res_transition = pattern_transition.match(key)
        if res_transition:
            new_key = res_transition.group(0)
            new_key = new_key[9:19] + '.' + str(int(new_key[19])-1) + new_key[20:] ###
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
        
        res_norm = pattern_norm.match(key)
        if res_norm:
            num += 1
            new_key = res_norm.group(0)
            if new_key[9:14] == 'norm5':
                new_key = 'norm5.' + new_key[9:]
            elif new_key[9:14] == 'norm0':
                new_key = 'input_layer.' + new_key[9:]
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
        
        if key == 'features.conv0.weight':
            new_key = 'input_layer.conv0.weight'
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
        
    model.load_state_dict(state_dict)


def _densenet(arch, growth_rate, block_config, num_init_features, pretrained, progress,
              **kwargs):
    model = DenseNet(growth_rate, block_config, num_init_features, **kwargs)
    if pretrained:
        _load_state_dict(model, model_urls[arch], progress)
    return model


def ssl_densenet121(pretrained=False, progress=True, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _densenet('densenet121', 32, (6, 12, 24, 16), 64, pretrained, progress,
                     **kwargs)





# ===================== ResMLP =====================
def pair(val):
    return (val, val) if not isinstance(val, tuple) else val

class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, 1, dim))
        self.b = nn.Parameter(torch.zeros(1, 1, dim))
    def forward(self, x):
        return x * self.g + self.b

class PreAffinePostLayerScale(nn.Module): # https://arxiv.org/abs/2103.17239
    def __init__(self, dim, depth, fn):
        super().__init__()
        if depth <= 18:
            init_eps = 0.1
        elif depth > 18 and depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        scale = torch.zeros(1, 1, dim).fill_(init_eps)
        self.scale = nn.Parameter(scale)
        self.affine = Affine(dim)
        self.fn = fn
    def forward(self, x):
        return self.fn(self.affine(x)) * self.scale + x

class ssl_ResMLP(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, num_classes, expansion_factor=4, fusion=False):  # depth=4 if fusion=True
        super(ssl_ResMLP, self).__init__()
        image_height, image_width = pair(image_size)
        assert (image_height % patch_size) == 0 and (image_width % patch_size) == 0, 'image height and width must be divisible by patch size'
        num_patches = (image_height // patch_size) * (image_width // patch_size)
        wrapper = lambda i, fn: PreAffinePostLayerScale(dim, i + 1, fn)

        self.fusion = fusion
        self.depth = depth

        self.to_patch_embed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear((patch_size ** 2) * 3, dim),
        )

        self.block = nn.ModuleList()

        for i in range(depth):
            self.block.append(nn.Sequential(
                wrapper(i, nn.Conv1d(num_patches, num_patches, 1)),
                wrapper(i, nn.Sequential(
                    nn.Linear(dim, dim * expansion_factor),
                    nn.GELU(),
                    nn.Linear(dim * expansion_factor, dim)
                ))
            ))

        self.output_layer = nn.Sequential(
            Affine(dim),
            Reduce('b n c -> b c', 'mean'),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x, out_fea=False, out_fea_map=False):
        side_features = []
        
        fea = self.to_patch_embed(x)
        
        for i in range(self.depth):
            fea = self.block[i](fea)
            if self.fusion == True:
                side_features.append(fea)
        
        out = self.output_layer(fea)
        
        if self.fusion == False:
            return out
        else:
            return out, side_features


# ===================== gMLP =====================
class ssl_gMLPVision(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads = 1,
        ff_mult = 4,
        channels = 3,
        attn_dim = None,
        prob_survival = 1,
        fusion=False
    ):
        super().__init__()
        assert (dim % heads) == 0, 'dimension must be divisible by number of heads'

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert (image_height % patch_height) == 0 and (image_width % patch_width) == 0, 'image height and width must be divisible by patch size'
        num_patches = (image_height // patch_height) * (image_width // patch_width)

        dim_ff = dim * ff_mult

        self.fusion = fusion
        self.depth = depth

        self.to_patch_embed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1 = patch_height, p2 = patch_width),
            nn.Linear(channels * patch_height * patch_width, dim)
        )

        self.prob_survival = prob_survival

        self.layers = nn.ModuleList([Residual(PreNorm(dim, gMLPBlock(dim = dim, heads = heads, dim_ff = dim_ff, seq_len = num_patches, attn_dim = attn_dim))) for i in range(depth)])

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            Reduce('b n d -> b d', 'mean'),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        side_features = []
        x = self.to_patch_embed(x)
        # layers = self.layers if not self.training else dropout_layers(self.layers, self.prob_survival)
        # x = nn.Sequential(*layers)(x)
        # self.layers
        for i in range(self.depth):
            # x = self.layers[i](x) if not self.training else dropout_layers(self.layers, self.prob_survival)
            x = self.layers[i](x)
            if self.fusion == True:
                side_features.append(x)
        
        out = self.to_logits(x)
        
        if self.fusion == False:
            return out
        else:
            return out, side_features












