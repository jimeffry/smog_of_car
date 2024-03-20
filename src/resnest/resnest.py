##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020
##
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""ResNeSt models"""
import collections
import torch
from resnet import ResNet, Bottleneck

__all__ = ['resnest50', 'resnest101', 'resnest200', 'resnest269']

_url_format = 'https://s3.us-west-1.wasabisys.com/resnest/torch/{}-{}.pth'

_model_sha256 = {name: checksum for checksum, name in [
    ('528c19ca', 'resnest50'),
    ('22405ba7', 'resnest101'),
    ('75117900', 'resnest200'),
    ('0cc87c48', 'resnest269'),
    ]}

def short_hash(name):
    if name not in _model_sha256:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
    return _model_sha256[name][:8]

resnest_model_urls = {name: _url_format.format(name, short_hash(name)) for
    name in _model_sha256.keys()
}

def resnest50(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    # if pretrained:
    #     model.load_state_dict(torch.hub.load_state_dict_from_url(
    #         resnest_model_urls['resnest50'], progress=True, check_hash=True))

    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(resnest_model_urls['resnest50'], progress=True, check_hash=True)
        # model.load_state_dict(state_dict, strict=False)
        fsd = collections.OrderedDict()
        # 10 convlution *(weight, bias) = 20 parameters
        res_iter = state_dict.items()
        for i in range(len(res_iter)):
            temp_key = list(res_iter)[i][0]
            # print(temp_key)
            if 'fc' in temp_key:
                continue
            fsd[temp_key] = list(res_iter)[i][1]
        model.load_state_dict(fsd, strict=False)
    return model

def resnest101(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    # if pretrained:
    #     model.load_state_dict(torch.hub.load_state_dict_from_url(
    #         resnest_model_urls['resnest101'], progress=True, check_hash=True))
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(resnest_model_urls['resnest101'], progress=True, check_hash=True)
        # model.load_state_dict(state_dict, strict=False)
        fsd = collections.OrderedDict()
        # 10 convlution *(weight, bias) = 20 parameters
        res_iter = state_dict.items()
        for i in range(len(res_iter)):
            temp_key = list(res_iter)[i][0]
            # print(temp_key)
            if 'fc' in temp_key:
                continue
            fsd[temp_key] = list(res_iter)[i][1]
        model.load_state_dict(fsd, strict=False)
    return model

def resnest200(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [3, 24, 36, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            resnest_model_urls['resnest200'], progress=True, check_hash=True))
    return model

def resnest269(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [3, 30, 48, 8],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            resnest_model_urls['resnest269'], progress=True, check_hash=True))
    return model

if __name__=='__main__':
    net_50 = resnest101(pretrained=True,root='D:\models',num_classes=2)