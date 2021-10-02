import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from models.cgans import AirfoilAoAGenerator
from train_final_cebgan import read_configs
from utils.shape_plot import plot_samples, plot_grid, plot_comparision
from utils.dataloader import NoiseGenerator

def load_generator(gen_cfg, save_dir, checkpoint, device='cpu'):
    ckp = torch.load(os.path.join(save_dir, checkpoint))
    generator = AirfoilAoAGenerator(**gen_cfg).to(device)
    generator.load_state_dict(ckp['generator'])
    generator.eval()
    return generator

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_dir = '../saves/complete/runs/Feb05_01-07-14'
    _, gen_cfg, _, cz = read_configs('conditional')

    epoch = 5000

    airfoils_opt = np.load('../data/airfoils_opt.npy')
    inp_paras = np.load('../data/inp_paras.npy')
    mean, std = inp_paras.mean(0), inp_paras.std(0)
    tr_inp_paras = (inp_paras - mean) / std

    generator = load_generator(gen_cfg, save_dir, 'conditional{}.tar'.format(epoch-1), device=device)
    params = torch.tensor(tr_inp_paras[-100:], dtype=torch.float, device=device)
    samples = []; aoas = []
    for _ in range(10):
        noise = torch.tensor(
            np.hstack([
                np.random.rand(len(params), cz[0]),
                np.random.randn(len(params), cz[1])
                ]), device=device, dtype=torch.float)
        pred = generator(noise, params)[0]
        samples.append(pred[0].cpu().detach().numpy().transpose([0, 2, 1]))
        aoas.append(pred[1].cpu().detach().numpy())

    plot_comparision(None, airfoils_opt[-100:], samples, scale=1.0, scatter=False, symm_axis=None, fname=os.path.join(save_dir, 'compare'))