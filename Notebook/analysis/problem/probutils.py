import os
import sys
sys.path.append('../../../')

import torch
import numpy as np
        
import torch.nn as nn
from cka import CKACalculator
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.DataList import dataset_dict

from src.simulator.utils import get_key_by_value


def vizualize_cka_model(model_dict, client_loader_list, device, criterion, unbiased=False):
    torch.cuda.set_device(device)
    fig, axs = plt.subplots(2, 5, figsize=(25, 12))
    
    layers = (nn.Conv3d, nn.Conv2d)  # dummy_val: nn.Conv2d
    model_types = list(model_dict.keys())
    model_types.remove(criterion)

    y_ticks = [f"{_type}" for _type in model_types]
        
    for client_idx in range(len(client_loader_list)):
        ax = axs[client_idx//5, client_idx%5]
        dataLoader = client_loader_list[client_idx]
        client_cka = []

        for _type in model_types:
            except_layer = []
            layer_names = []

            if criterion == 'Local':
                calculator = CKACalculator(model1=model_dict['Local'][client_idx].to(device),
                                        model2=model_dict[_type].to(device),
                                        dataloader=dataLoader,
                                        hook_layer_types=layers,
                                        num_epochs=1,
                                        epsilon=1e-10,)
                
            else:
                if _type == 'Local':
                    calculator = CKACalculator(model1=model_dict['Center'].to(device),
                                                model2=model_dict[_type][client_idx].to(device),
                                                dataloader=dataLoader,
                                                hook_layer_types=layers,
                                                num_epochs=1,
                                                epsilon=1e-4,)
                else:
                    calculator = CKACalculator(model1=model_dict['Center'].to(device),
                                               model2=model_dict[_type].to(device),
                                               dataloader=dataLoader,
                                               hook_layer_types=layers,
                                               num_epochs=1,
                                               epsilon=1e-4,)
                                    
            cka_output = calculator.calculate_cka_matrix().detach().cpu().numpy()
            cka_output = np.diag(cka_output)
            # cka_output = np.nan_to_num(cka_output, nan=0, posinf=1, neginf=1)

            for i, name in enumerate(calculator.module_names_X):
                if 'down'in name:
                    except_layer.append(i)
                else:
                    layer_names.append(name)

            cka_output = np.delete(cka_output, except_layer)
            
            client_cka.append(cka_output)
            calculator.reset()
            torch.cuda.empty_cache()
        
        vmin = 0
        vmax = 1.0
        cmap = 'inferno'

        image = sns.heatmap(client_cka, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, cbar=False)

        ax.set_xticks(range(len(layer_names)))
        ax.set_xticklabels(layer_names, rotation=90)
        
        y_tick_positions = np.arange(len(y_ticks)) + 0.5
        ax.set_yticks(y_tick_positions)
        ax.set_yticklabels(y_ticks)

        ax.set_title(f"Client {client_idx} || {get_key_by_value(dataset_dict, client_idx)} (n={len(dataLoader.dataset)})",
                     fontsize=12)

    fig.subplots_adjust(right=0.85)

    cbar_ax = fig.add_axes([1.02, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(image.get_children()[0], cax=cbar_ax, fraction=0.98)
    cbar.set_label("CKA Score")
    
    plt.tight_layout()
    plt.show()