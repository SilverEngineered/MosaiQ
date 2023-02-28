import sys
import json
import os
import torch
from Components.models.small_model import SmallNet
from Components.models.full_model_stripped import Unet
from Components.models.full_model import Unet as FUNet
from Components.sinusoidal import SinusoidalPositionEmbeddings
from Components.models.quantum_model import QNet
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def check_and_parse_args():
    if len(sys.argv) < 2:
        print("Please input name of config file... exiting")
        exit()
    config_file_name = sys.argv[1]
    config_file_path = os.path.join('Configurations', config_file_name)
    config = json.load(open(config_file_path, "r"))
    model = get_model(config)
    return config, model

def get_model(config):
    sinusoidal = SinusoidalPositionEmbeddings
    if config['model'] == 'Unet':
        model = Unet(
        dim=config['img_size'],
        channels=1        )
    elif config['model'] == 'Full':
        model = FUNet(config['img_size'], channels=1, dim_mults=(1,2))
    elif config['model'] == 'Quantum':
        model = QNet(config['n_qubits'], config['n_layers']) 
    else:
        raise NotImplementedError(f'{config["model"]} not implemented')
    if torch.cuda.is_available():
        return model.cuda()
    else:
        return model