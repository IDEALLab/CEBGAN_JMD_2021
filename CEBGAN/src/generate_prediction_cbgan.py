import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from models.cgans import AirfoilAoAGenerator
from train_final_cbgan import read_configs
from utils.shape_plot import plot_samples, plot_grid, plot_comparision
from utils.dataloader import NoiseGenerator

def load_generator(gen_cfg, save_dir, checkpoint, device='cpu'):
    ckp = torch.load(os.path.join(save_dir, checkpoint), map_location='cuda:0')
    generator = AirfoilAoAGenerator(**gen_cfg).to(device)
    generator.load_state_dict(ckp['generator'])
    generator.eval()
    return generator

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_dir = '../saves/final/'
    _, gen_cfg, _, cz, _ = read_configs('cbgan')

    epoch = 5000

    inp_paras = np.load('../data/inp_paras_995.npy')
    mean, std = inp_paras.mean(0), inp_paras.std(0)
    inp_paras = np.load('../data/inp_paras_347.npy') # reload inp_paras from Jun's test set.
    tr_inp_paras = (inp_paras - mean) / std


    generator = load_generator(gen_cfg, save_dir, 'cbgan{}.tar'.format(epoch-1), device=device)
    params = torch.tensor(tr_inp_paras, dtype=torch.float, device=device)
    samples = []; aoas = []

    noise = torch.zeros([len(params), cz[0]], device=device, dtype=torch.float)
    pred = generator(noise, params)[0]
    samples.append(pred[0].cpu().detach().numpy().transpose([0, 2, 1]))
    aoas.append(pred[1].cpu().detach().numpy())
    
    np.save('../data/pred_cbgan/single/new/aoas_pred.npy', aoas[0])
    np.save('../data/pred_cbgan/single/new/airfoils_pred.npy', samples[0])
    np.save('../data/pred_cbgan/single/new/inp_params_pred.npy', params.cpu().detach().numpy())

    # num_trials = 10
    # for _ in range(num_trials):
    #     noise = torch.randn([len(params), cz[0]], device=device, dtype=torch.float)
    #     pred = generator(noise, params)[0]
    #     samples.append(pred[0].cpu().detach().numpy().transpose([0, 2, 1]))
    #     aoas.append(pred[1].cpu().detach().numpy())

    # np.save('aoas_pred.npy', np.stack(aoas, axis=1))
    # np.save('airfoils_pred.npy', np.stack(samples, axis=1))
    # np.save('inp_params_pred.npy', params.cpu().detach().numpy())
