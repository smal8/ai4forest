from bisect import bisect_right

import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F
import random
from collections import OrderedDict

class GeneralUtility:
    """General helper functions"""
    @staticmethod
    def fill_dict_with_none(d):
        for key in d:
            if isinstance(d[key], dict):
                GeneralUtility.fill_dict_with_none(d[key])  # Recursive call for nested dictionaries
            else:
                d[key] = None
        return d
    
    @staticmethod
    def update_config_with_default(configDict, defaultDict):
        """Update config with default values recursively."""
        for key, default_value in defaultDict.items():
            if key not in configDict:
                configDict[key] = default_value
            elif isinstance(default_value, dict):
                configDict[key] = GeneralUtility.update_config_with_default(configDict.get(key, {}), default_value)
        return configDict
    
    @staticmethod
    def load_pretrained_unet_sixmonth(
        new_model, 
        old_state_dict, 
        geo_encoding_location='none', 
        n_geo_channels=4
    ):
        """
        Loads as many weights as possible from the old UNetSixMonth model into the new one,
        handling geo-encoding channel differences.
        
        Freezes all layers except for the ones that are initialized with Kaiming initialization for re-training.
        """
        new_state_dict = new_model.state_dict()
        adapted_state_dict = OrderedDict()
        
        # Track parameters to keep unfrozen
        unfrozen_params = []

        for name, param in new_state_dict.items():
            if geo_encoding_location == 'first' and name in ['conv1.conv.0.weight', 'conv1.conv.0.bias']:
                # only need to change the conv1 layer weights
                adapted = torch.zeros_like(param)
                # Initialize with Kaiming (He) initialization for better training
                if 'weight' in name:
                    torch.nn.init.kaiming_normal_(adapted)
                adapted_state_dict[name] = adapted
                unfrozen_params.append(name)
                
            elif geo_encoding_location == 'last' and name in ['last_conv.weight', 'last_conv.bias']:
                # only need to change the last_conv layer weights
                adapted = torch.zeros_like(param)
                # Initialize with Kaiming (He) initialization for better training
                if 'weight' in name:
                    torch.nn.init.kaiming_normal_(adapted)
                adapted_state_dict[name] = adapted
                unfrozen_params.append(name)
                
            elif geo_encoding_location == 'bottleneck' and (name in ['up1.up.up_scale.weight', 'up1.up.up_scale.bias'] 
                                                            or name in ['up1.conv.conv.0.weight', 'up1.conv.conv.0.bias']):
                    adapted = torch.zeros_like(param)
                    # Initialize with Kaiming (He) initialization for better training
                    if 'weight' in name:
                        torch.nn.init.kaiming_normal_(adapted)
                    adapted_state_dict[name] = adapted
                    unfrozen_params.append(name)
            
            else:
                if name in old_state_dict and old_state_dict[name].shape == param.shape:
                    adapted_state_dict[name] = old_state_dict[name]

                else:
                    adapted = torch.zeros_like(param)
                    if 'weight' in name:
                        torch.nn.init.kaiming_normal_(adapted)
                    adapted_state_dict[name] = adapted
                    unfrozen_params.append(name)
        
        
        # Load adapted state dict
        new_model.load_state_dict(adapted_state_dict, strict=False)
        
        if geo_encoding_location == 'none':
            return new_model
        
        # Freeze all parameters except the ones we initialized with Kaiming
        for name, param in new_model.named_parameters():
            if name in unfrozen_params:
                param.requires_grad = True
            else:
                param.requires_grad = False
                
        return new_model
    

class JointRandomRotationTransform:
    def __init__(self):
        self.angles = [90, 180, 270, 360]

    def __call__(self, image, label):
        angle = random.choice(self.angles)
        if angle == 360:
            return image, label
        return F.rotate(image, angle), F.rotate(label, angle)


class SequentialSchedulers(torch.optim.lr_scheduler.SequentialLR):
    """
    Repairs SequentialLR to properly use the last learning rate of the previous scheduler when reaching milestones
    """

    def __init__(self, **kwargs):
        self.optimizer = kwargs['schedulers'][0].optimizer
        super(SequentialSchedulers, self).__init__(**kwargs)

    def step(self):
        self.last_epoch += 1
        idx = bisect_right(self._milestones, self.last_epoch)
        self._schedulers[idx].step()
