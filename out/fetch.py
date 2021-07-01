import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-n', '--name', type=str,
        help='name of training session to copy')

parser.add_argument('-e', '--epoch', type=int, default=20,
        help='epoch')

args = parser.parse_args()

out_path = '/home/vol08/scarf1018/STFC-unet_main/out'

training_path = os.path.join(out_path, args.name)


models_path = os.path.join(args.name, 'models')
denoised_path = os.path.join(args.name, 'denoised')
plots_path = os.path.join(args.name, 'plots')

path_exists = os.path.exists(args.name)

if not path_exists:
    os.mkdir(args.name)
    os.mkdir(models_path)
    os.mkdir(denoised_path)
    os.mkdir(plots_path)

scarf_path = os.path.join(out_path, args.name)


scarf_models = os.path.join(scarf_path, 'models')
scarf_denoised = os.path.join(scarf_path, 'denoised')
scarf_plots = os.path.join(scarf_path, 'plots')

scarf_inputs = os.path.join(scarf_path, 'inputs.yaml')
scarf_model = os.path.join(scarf_models, f'epoch_{args.epoch}.pt')
scarf_denoise = os.path.join(scarf_denoised, f'e_{args.epoch}_0552.map')
scarf_plot = os.path.join(scarf_plots, f'e_{args.epoch}.png')

local_inputs = os.path.join(args.name, 'inputs.yaml')
local_model = os.path.join(args.name, 'models', 
        f'epoch_{args.epoch}.pt')
local_denoise = os.path.join(args.name, 'denoised',
        f'e_{args.epoch}_0552.map')
local_plot = os.path.join(args.name, 'plots', 
        f'e_{args.epoch}.png')

exists_local_inputs = os.path.exists(local_inputs)
exists_local_model = os.path.exists(local_model)
exists_local_denoise = os.path.exists(local_denoise)
exists_local_plot = os.path.exists(local_plot)

if not exists_local_inputs:
    os.system(f"scp uyk:{scarf_inputs} {local_inputs}")
if not exists_local_model:
    os.system(f"scp uyk:{scarf_model} {local_model}")
if not exists_local_denoise:
    os.system(f"scp uyk:{scarf_denoise} {local_denoise}")
if not exists_local_plot:
    os.system(f"scp uyk:{scarf_plot} {local_plot}")
