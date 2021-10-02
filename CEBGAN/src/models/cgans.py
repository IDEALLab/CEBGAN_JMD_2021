"""
Components customized for airfoil prediction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from .gans import BezierSEGAN, BezierGAN, _eps
from .sinkhorn import sinkhorn_divergence, regularized_ot, sink
from .cmpnts import MLP, Conv1DNetwork, BezierGenerator, CPWGenerator
from . import layers
from .utils import first_element

class AirfoilAoADiscriminator1D(Conv1DNetwork):
    def __init__(
        self, in_channels: int, in_features: int, cond_dim: int,
        n_critics: int, aoa_dim: int = 1,
        conv_channels: list = [64, 64*2, 64*4, 64*8, 64*16, 64*32],
        crt_layers: list = [1024,],
        pred_layers: list = [512,]
        ):
        super().__init__(
            in_channels, in_features, conv_channels,
            )
        self.cond_dim = cond_dim
        self.m_features += cond_dim
        self.m_features += aoa_dim
        self.n_critics = n_critics
        self.critics = nn.Sequential(
            MLP(self.m_features, crt_layers[-1], crt_layers[:-1]),
            nn.Linear(crt_layers[-1], n_critics)
        )
    
    def forward(self, input, condition, aoa):
        x = torch.hstack([self.conv(input), condition, aoa])
        critics = self.critics(x)
        return critics

class AirfoilAoAGenerator(nn.Module):
    def __init__(
        self, in_features: int, n_control_points: int, n_data_points: int, 
        m_features: int = 256,
        feature_gen_layers: list = [1024,],
        dense_layers: list = [1024,],
        deconv_channels: list = [96*8, 96*4, 96*2, 96],
        mlp_layers: list = [128,]
        ):
        super().__init__()
        self.in_features = in_features
        self.n_control_points = n_control_points
        self.n_data_points = n_data_points

        self.airfoil_generator = BezierGenerator(in_features, n_control_points, n_data_points, 
            m_features, feature_gen_layers, dense_layers, deconv_channels)
        self.aoa_generator = MLP(in_features, 1, layer_width=mlp_layers)
    
    def forward(self, noise, inp_paras):
        dp, cp, w, pv, intvls = self.airfoil_generator(torch.hstack([noise, inp_paras]))
        aoa = self.aoa_generator(torch.hstack([noise, inp_paras]))
        return (dp, aoa), cp, w, pv, intvls
    
    def extra_repr(self) -> str:
        return 'in_features={}, n_control_points={}, n_data_points={}'.format(
            self.in_features, self.n_control_points, self.n_data_points
        )

class AirfoilAoACEGAN(BezierSEGAN):
    def _update_D(self, num_iter_D, batch, noise_gen, **kwargs): pass
    
    def loss_G(self, batch, noise_gen, **kwargs):
        _, _, inp_paras = batch
        noise = noise_gen()[:len(inp_paras)]; latent_code = noise[:, :noise_gen.sizes[0]]
        fake, cp, w, pv, intvls = self.generator(noise, inp_paras)
        sinkhorn_loss = self.sinkhorn_divergence(batch, (*fake, inp_paras))
        reg_loss = self.regularizer(cp, w, pv, intvls)
        return sinkhorn_loss + 1*reg_loss
    
    def generate(self, noise, condition, output_condition=True, additional_info=False):
        if additional_info:
            return self.generator(noise, condition)
        else:
            airfoils, aoas = first_element(self.generator(noise, condition))
            if output_condition: 
                return airfoils, aoas, condition
            else:
                return airfoils, aoas
    
    def surrogate_ll(self, test_loader, noise_gen):
        noise, p_x = noise_gen()
        num_samples = len(test_loader.dataset)
        ll = torch.zeros(num_samples, device=noise_gen.device)
        for i, test in tqdm(enumerate(test_loader), total=len(test_loader)):
            fake = self.generate(noise, test[-1].expand(len(noise), -1))
            prob = self._estimate_prob(test, fake, p_x) # [n_test, batch]
            ep_dist = (self.cost(test, fake) * prob).sum(dim=1) # [n_test]
            entropy = (-prob * torch.log(prob + _eps)).sum(dim=1) # [n_test]
            # ep_lpx = (torch.log(p_x).T * prob).sum(dim=1) # [n_test]
            ep_lpx = (torch.log(p_x).T / len(p_x)).sum(dim=1) # [n_test]
            ll[i] = (-ep_dist / self.lamb + entropy + ep_lpx).detach() # [n_test] log likelihood surrogate 2.7
        return ll
    
    def _estimate_prob(self, test, fake, p_x): 
        v_star, _, _ = self._cal_v(test, fake) # [n_test, batch] term inside exp() of 4.3. 
        exp_v = torch.exp((v_star - v_star.max(dim=1, keepdim=True).values) / self.lamb) # [n_test, batch] avoid numerical instability.
        prob = 1 / len(p_x) * exp_v # [n_test, batch] with uniform distribution for empirical
        return prob / prob.sum(dim=1, keepdim=True) # [n_test, batch] normalize over x to obtain 4.3 P(x|y)

    def _epoch_report(self, epoch, epochs, batch, noise_gen, report_interval, tb_writer, **kwargs):
        if epoch % report_interval == 0:
            _, aoas_opt, inp_paras = batch
            noise = noise_gen()[:len(inp_paras)]; latent_code = noise[:, :noise_gen.sizes[0]]
            fake, cp, w, pv, intvls = self.generator(noise, inp_paras)
            sinkhorn_loss = self.sinkhorn_divergence(batch, (*fake, inp_paras))
            # info_loss = self.info_loss(fake, inp_paras, latent_code)
            reg_loss = self.regularizer(cp, w, pv, intvls)
            mse = F.mse_loss(fake[1], aoas_opt)
            if tb_writer:
                tb_writer.add_scalar('Sinkhorn Divergence', sinkhorn_loss, epoch)
                # tb_writer.add_scalar('Info Loss', info_loss, epoch)
                tb_writer.add_scalar('Regularization Loss', reg_loss, epoch)
                tb_writer.add_scalar('MSE of AoA', mse, epoch)

            try: 
                kwargs['plotting'](epoch, batch, fake)
                kwargs['metrics'](epoch, self.generator, tb_writer)
            except:
                pass

class CBGAN(BezierGAN):

    def generate(self, noise, inp_paras):
        return self.generator(noise, inp_paras)

    def train(
        self, dataloader, noise_gen, epochs, num_iter_D=5, num_iter_G=1, report_interval=5,
        save_dir=None, save_iter_list=[100,], tb_writer=None, **kwargs
        ):
        for epoch in range(epochs):
            self._epoch_hook(epoch, epochs, noise_gen, tb_writer, **kwargs)
            for i, (dp, aoa, inp_paras) in enumerate(dataloader):
                #self._batch_hook(i, batch, noise_gen, tb_writer, **kwargs)
                self._update_D(num_iter_D, dp, aoa, inp_paras, noise_gen, **kwargs)
                #if not self._train_gen_criterion(batch, noise_gen, epoch): continue
                self._update_G(num_iter_G, dp, aoa, inp_paras, noise_gen, **kwargs)
                #self._batch_report(i, dp, aoa, noise_gen, tb_writer, **kwargs)
            self._epoch_report(epoch, epochs, dp, aoa, inp_paras, noise_gen, report_interval, tb_writer, **kwargs)

            if save_dir:
                if save_iter_list and epoch in save_iter_list:
                    self.save(save_dir, epoch=epoch, noise=noise_gen)

    def _update_D(self, num_iter_D, real_dp, real_aoa, inp_paras, noise_gen, **kwargs):
        for _ in range(num_iter_D):
            self.optimizer_D.zero_grad()
            self.loss_D(real_dp, real_aoa, inp_paras, noise_gen, **kwargs).backward()
            self.optimizer_D.step()
    
    def _update_G(self, num_iter_G, real_dp, real_aoa, inp_paras, noise_gen, **kwargs):
        for _ in range(num_iter_G):
            self.optimizer_G.zero_grad()
            self.loss_G(real_dp, real_aoa, inp_paras, noise_gen, **kwargs).backward()
            self.optimizer_G.step()

    def loss_G(self, real_dp, real_aoa, inp_paras, noise_gen, **kwargs):
        noise = noise_gen(); latent_code = noise[:, :noise_gen.sizes[0]]
        (fake_dp, fake_aoa), cp, w, pv, intvls = self.generator(noise, inp_paras)
        js_loss = self.js_loss_G(fake_dp, fake_aoa, inp_paras)
        reg_loss = self.regularizer(cp, w, pv, intvls)
        return js_loss + 10 * reg_loss

    def loss_D(self, real_dp, real_aoa, inp_paras, noise_gen, **kwargs):
        noise = noise_gen(); latent_code = noise[:, :noise_gen.sizes[0]]
        (fake_dp, fake_aoa), cp, w, pv, intvls = self.generator(noise, inp_paras)
        js_loss = self.js_loss_D(real_dp, real_aoa, fake_dp, fake_aoa, inp_paras)
        return js_loss

    def js_loss_D(self, real_dp, real_aoa, fake_dp, fake_aoa, inp_paras):
        return F.binary_cross_entropy_with_logits(
            first_element(self.discriminator(real_dp, inp_paras, real_aoa)),
            torch.ones(len(real_dp), 1, device=real_dp.device)
            ) + F.binary_cross_entropy_with_logits(
            first_element(self.discriminator(fake_dp, inp_paras, fake_aoa)),
            torch.zeros(len(fake_dp), 1, device=fake_dp.device)
            )

    def js_loss_G(self, fake_dp, fake_aoa, inp_paras):
        return F.binary_cross_entropy_with_logits(
            first_element(self.discriminator(fake_dp, inp_paras, fake_aoa)),
            torch.ones(len(fake_dp), 1, device=fake_dp.device)
            )

    def _epoch_report(self, epoch, epochs, real_dp, real_aoa, inp_paras,
                      noise_gen, report_interval, tb_writer, **kwargs):
        if epoch % report_interval == 0:
            noise = noise_gen(); latent_code = noise[:, :noise_gen.sizes[0]]
            (fake_dp, fake_aoa), cp, w, pv, intvls = self.generator(noise, inp_paras)
            js_loss = self.js_loss_D(real_dp, real_aoa, fake_dp, fake_aoa, inp_paras)
            reg_loss = self.regularizer(cp, w, pv, intvls)
            if tb_writer:
                tb_writer.add_scalar('JS Loss', js_loss, epoch)
                tb_writer.add_scalar('Regularization Loss', reg_loss, epoch)
            else:
                print('[Epoch {}/{}] JS loss: {:d}, Regularization loss: {:d}'.format(
                    epoch, epochs,  js_loss, reg_loss))
            try:
                kwargs['plotting'](epoch, (real_dp, real_aoa, inp_paras), (fake_dp, fake_aoa))
            except:
                pass
