from densenet import MyDensenet
from resnet import MyResnet
from vggnet import MyVGGNet
from torchvision import models

_MODELS = ['resnet-50', 'resnet-101', 'densenet-121', 'vgg-13', 'vgg-16', 'vgg-19']

def set_model (model_name, num_class, pretrained_model, neurons_reducer_block=0, comb_method=None, comb_config=None, pretrained=True,
         freeze_conv=False):

    if pretrained:
        pre_torch = True
    else:
        pre_torch = False

    if model_name not in _MODELS:
        raise Exception("The model {} is not available!".format(model_name))

    model = None
    if model_name == 'resnet-50':
        if(pretrained_model is not None):
            model = MyResnet(pretrained_model, num_class, neurons_reducer_block, freeze_conv,
                        comb_method=comb_method, comb_config=comb_config)
        else:
            model = MyResnet(models.resnet50(pretrained=pre_torch), num_class, neurons_reducer_block, freeze_conv,
                        comb_method=comb_method, comb_config=comb_config)

    elif model_name == 'densenet-121':
        if(pretrained_model is not None):
            model = MyDensenet(models.densenet121(pretrained=pre_torch), num_class, neurons_reducer_block, freeze_conv,
                        comb_method=comb_method, comb_config=comb_config)
        else:
            model = MyDensenet(pretrained_model, num_class, neurons_reducer_block, freeze_conv,
                        comb_method=comb_method, comb_config=comb_config)

    elif model_name == 'vgg-13':
        if(pretrained_model is not None):
            model = MyVGGNet(pretrained_model, num_class, neurons_reducer_block, freeze_conv,
                         comb_method=comb_method, comb_config=comb_config)
        else:
            model = MyVGGNet(models.vgg13_bn(pretrained=pre_torch), num_class, neurons_reducer_block, freeze_conv,
                         comb_method=comb_method, comb_config=comb_config)

    return model


