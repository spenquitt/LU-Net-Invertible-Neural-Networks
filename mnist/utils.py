import torch
from functools import partial
from PIL import Image

def reverse_state_dict_lunet(state_dict):
    """create checkpoint of LUNet inverse from LUNet forward checkpoint"""
    if isinstance(state_dict, str):
        state_dict = torch.load(state_dict)
    state_dict_inverse_lunet = {}
    i = 0
    for key, value in reversed(state_dict.items()):
        if "bias" in key:
            new_key = "lu_layers.{}.bias".format(i)
            state_dict_inverse_lunet[new_key] = value
        else:
            new_key = "lu_layers.{}.weight".format(i)
            state_dict_inverse_lunet[new_key] = value
            i += 1
    return state_dict_inverse_lunet

def mutlivariate_normal_params(train_loader, model, device="cpu"):
    model.eval()
    mean = torch.zeros(model.layer_size).to(device)
    covariance = torch.zeros((model.layer_size, model.layer_size)).to(device)
    for inputs, _ in train_loader:
        with torch.no_grad():
            latents = model(inputs.to(device))
            mean += torch.mean(latents, dim=0)
            covariance += torch.cov(torch.t(latents))
    mean = mean / len(train_loader)
    covariance = covariance / len(train_loader)
    stats = {}
    stats["latent_mean"] = mean
    stats["latent_covariance"] = covariance
    return stats

def mutlivariate_normal_params_gaussian(train_loader, model, batch_size, device="cpu"):
    model.eval()
    mean = torch.zeros(model.layer_size).to(device)
    covariance = torch.zeros((model.layer_size, model.layer_size)).to(device)
    for k in range(int(len(train_loader) / batch_size)):
        inputs = train_loader[k * batch_size : k * batch_size + batch_size]
        with torch.no_grad():
            latents = model(inputs.to(device))
            mean += torch.mean(latents, dim=0)
            covariance += torch.cov(torch.t(latents))
    mean = mean / len(train_loader)
    covariance = covariance / len(train_loader)
    stats = {}
    stats["latent_mean"] = mean
    stats["latent_covariance"] = covariance
    return stats

"""
* helper functions to store activations and parameters in intermediate layers of the model
* use forward hooks for this, which are functions executed automatically during forward pass
* in PyTorch hooks are registered for nn.Module and are triggered by forward pass of object
"""

def save_activations(activations_dict, blu, bla, blub, output):
    activations_dict[blu] = output.detach()

def register_activation_hooks(model, layer_name):
    """register forward hooks in specified layers"""
    activations_dict = {}
    for name, module in model.named_modules():
        if layer_name + "." in name:
            module.register_forward_hook(partial(save_activations, activations_dict, name))
    return activations_dict



def save_image_from_numpy(np_array, save_path):
    Image.fromarray(np_array).save(save_path)