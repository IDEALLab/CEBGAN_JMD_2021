import torch
import numpy as np
import os, json

from datetime import datetime
from sklearn.model_selection import train_test_split, KFold
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from models.cgans import AirfoilAoACEGAN, AirfoilAoADiscriminator1D, AirfoilAoAGenerator
from utils.dataloader import AirfoilDataset, NoiseGenerator
from utils.shape_plot import plot_samples, plot_comparision
from torchvision.transforms import Normalize
from utils.metrics import ci_cons, ci_mll, ci_rsmth, ci_rdiv, ci_mmd

cost1 = lambda x1, x2: torch.cdist(x1[0].flatten(1), x2[0].flatten(1), p=1) \
    + torch.cdist(x1[1], x2[1], p=1) \
    + torch.cdist(x1[2], x2[2], p=1)

def read_configs(name, base_dir="./"):
    with open(os.path.join(base_dir, 'configs', name+'.json')) as f:
        configs = json.load(f)
        dis_cfg = configs['dis']
        gen_cfg = configs['gen']
        egan_cfg = configs['egan']
        cz = configs['cz']
        noise_type = configs['noise_type']
    return dis_cfg, gen_cfg, egan_cfg, cz, noise_type

def assemble_new_gan(dis_cfg, gen_cfg, egan_cfg, device='cpu'):
    discriminator = AirfoilAoADiscriminator1D(**dis_cfg).to(device)
    generator = AirfoilAoAGenerator(**gen_cfg).to(device)
    egan = AirfoilAoACEGAN(generator, discriminator, 
        cost_func=cost1, 
        **egan_cfg)
    return egan

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch = 128
    epochs = 15000
    save_intvl = 5000

    dis_cfg, gen_cfg, egan_cfg, cz, noise_type = read_configs('cebgan')
    save_dir = '../saves/final'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'runs'), exist_ok=True)

    airfoils_opt = np.load('../data/airfoils_opt_995.npy').astype(np.float32)
    inp_paras = np.load('../data/inp_paras_995.npy').astype(np.float32)
    aoas_opt = np.load('../data/aoas_opt_995.npy').astype(np.float32).reshape(-1, 1)
    mean_std = (inp_paras.mean(0), inp_paras.std(0))

    save_iter_list = list(np.linspace(1, epochs/save_intvl, dtype=int) * save_intvl - 1)
    
    time = datetime.now().strftime('%b%d_%H-%M-%S')

    # build entropic gan on the device specified
    egan = assemble_new_gan(dis_cfg, gen_cfg, egan_cfg, device=device)

    # build dataloader and noise generator on the device specified
    dataset = AirfoilDataset(inp_paras, airfoils_opt, aoas_opt, inp_mean_std=mean_std, device=device)
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=True)
    noise_gen = NoiseGenerator(batch, sizes=cz, noise_type=noise_type, device=device) # all Gaussian noise

    # build tensorboard summary writer
    tb_dir = os.path.join(save_dir, 'runs', time)
    os.makedirs(os.path.join(tb_dir, 'images'), exist_ok=True)
    writer = SummaryWriter(tb_dir)
    
    def epoch_plot(epoch, batch, fake, *args, **kwargs):
        if (epoch + 1) % 1000 == 0:
            num = min(36, len(fake[0]))
            airfoils, aoas_opt, _ = batch
            airfoils = airfoils.cpu().detach().numpy().transpose([0, 2, 1])[:num]
            aoas_opt = aoas_opt.cpu().detach().numpy().squeeze()[:num]
            pred_airfoils = fake[0].cpu().detach().numpy().transpose([0, 2, 1])[:num]
            pred_aoas = fake[1].cpu().detach().numpy().squeeze()[:num]

            plot_comparision(
                None, airfoils, [pred_airfoils], aoas_opt, pred_aoas, scale=1.0, scatter=False, symm_axis=None, 
                fname=os.path.join(tb_dir, 'images', 'epoch {}'.format(epoch+1))
                )

    egan.train(
        epochs=epochs,
        num_iter_D=1, 
        num_iter_G=1,
        dataloader=dataloader, 
        noise_gen=noise_gen, 
        tb_writer=writer,
        report_interval=1,
        save_dir=save_dir,
        save_iter_list=save_iter_list,
        plotting=epoch_plot
        )