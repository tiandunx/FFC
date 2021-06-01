from .resnet_arcface import iresnet50
from .resnet_std import resnet50
from .mobilefacenet_def import MobileFaceNet


def create_net(net_type, **kwargs):
    net_creator = {'ir50': iresnet50, 'r50': resnet50, 'mobile': MobileFaceNet}
    if net_type not in net_creator:
        raise Exception('Unknown architecture')
    return net_creator[net_type](**kwargs)
