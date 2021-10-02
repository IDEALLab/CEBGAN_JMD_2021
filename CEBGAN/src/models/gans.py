import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import strong_convex_func, first_element
from .sinkhorn import sinkhorn_divergence, regularized_ot, sink

_eps = 1e-7

class GAN:
    def __init__(
        self, generator: nn.Module, discriminator: nn.Module, 
        opt_g_lr: float=1e-4, opt_g_betas: tuple=(0.5, 0.99), opt_g_eps: float=1e-8,
        opt_d_lr: float=1e-4, opt_d_betas: tuple=(0.5, 0.99), opt_d_eps: float=1e-8,
        name: str=None, checkpoint: str=None, train_mode: bool=True
        ):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.name = name
        self.optimizer_G = torch.optim.Adam(
            self.generator.parameters(), lr=opt_g_lr, betas=opt_g_betas, eps=opt_g_eps)
        self.optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(), lr=opt_d_lr, betas=opt_d_betas, eps=opt_g_eps)
        if checkpoint:
            self.load(checkpoint, train_mode)
    
    def loss_G(self, batch, noise_gen, **kwargs):
        fake = self.generate(noise_gen())
        return self.js_loss_G(batch, fake)
    
    def loss_D(self, batch, noise_gen, **kwargs):
        fake = self.generate(noise_gen())
        return self.js_loss_D(batch, fake)
    
    def js_loss_D(self, real, fake):
        return F.binary_cross_entropy_with_logits(
            first_element(self.discriminator(real)), 
            torch.ones(len(real), 1, device=real.device)
            ) + F.binary_cross_entropy_with_logits(
            first_element(self.discriminator(fake)), 
            torch.zeros(len(fake), 1, device=fake.device)
            )
    
    def js_loss_G(self, real, fake):
        return F.binary_cross_entropy_with_logits(
            first_element(self.discriminator(fake)), 
            torch.ones(len(fake), 1, device=fake.device)
            )

    def _batch_hook(self, i, batch, noise_gen, tb_writer, **kwargs): pass
    
    def _batch_report(self, i, batch, noise_gen, tb_writer, **kwargs): pass

    def _epoch_hook(self, epoch, epochs, noise_gen, tb_writer, **kwargs): pass

    def _epoch_report(self, epoch, epochs, batch, noise_gen, report_interval, tb_writer, **kwargs):
        if epoch % report_interval == 0:
            if tb_writer:
                tb_writer.add_scalar('D Loss', loss_D(batch, noise_gen), epoch)
                tb_writer.add_scalar('G Loss', loss_G(batch, noise_gen), epoch)
            else:
                print('[Epoch {}/{}] D loss: {:d}, G loss: {:d}'.format(
                    epoch, epochs, loss_D(batch, noise_gen), loss_G(batch, noise_gen)))
    
    def _train_gen_criterion(self, batch, noise_gen, epoch): return True

    def save(self, save_dir, **kwargs):
        torch.save({
            'discriminator': self.discriminator.state_dict(),
            'generator': self.generator.state_dict(),
            'optimizer_D': self.optimizer_D.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
            'records': kwargs
            }, os.path.join(save_dir, self.name+str(kwargs['epoch'])+'.tar'))

    def load(self, checkpoint, train_mode):
        ckp = torch.load(checkpoint)
        self.discriminator.load_state_dict(ckp['discriminator'])
        self.generator.load_state_dict(ckp['generator'])
        if train_mode:  
            self.discriminator.train(); self.generator.train()
        else:
            self.discriminator.eval(); self.generator.eval()
        self.optimizer_D.load_state_dict(ckp['optimizer_D'])
        self.optimizer_G.load_state_dict(ckp['optimizer_G'])
    
    def generate(self, input, additional_info=False):
        if additional_info:
            return self.generator(input)
        else: 
            return first_element(self.generator(input))
    
    def _update_D(self, num_iter_D, batch, noise_gen, **kwargs):
        for _ in range(num_iter_D):
            self.optimizer_D.zero_grad()
            self.loss_D(batch, noise_gen, **kwargs).backward()
            self.optimizer_D.step()
    
    def _update_G(self, num_iter_G, batch, noise_gen, **kwargs):
        for _ in range(num_iter_G):
            self.optimizer_G.zero_grad()
            self.loss_G(batch, noise_gen, **kwargs).backward()
            self.optimizer_G.step()

    def train(
        self, dataloader, noise_gen, epochs, num_iter_D=5, num_iter_G=1, report_interval=5,
        save_dir=None, save_iter_list=[100,], tb_writer=None, **kwargs
        ):
        for epoch in range(epochs):
            self._epoch_hook(epoch, epochs, noise_gen, tb_writer, **kwargs)
            for i, batch in enumerate(dataloader):
                self._batch_hook(i, batch, noise_gen, tb_writer, **kwargs)
                self._update_D(num_iter_D, batch, noise_gen, **kwargs)
                if not self._train_gen_criterion(batch, noise_gen, epoch): continue
                self._update_G(num_iter_G, batch, noise_gen, **kwargs)
                self._batch_report(i, batch, noise_gen, tb_writer, **kwargs)
            self._epoch_report(epoch, epochs, batch, noise_gen, report_interval, tb_writer, **kwargs)

            if save_dir:
                if save_iter_list and epoch in save_iter_list:
                    self.save(save_dir, epoch=epoch, noise=noise_gen)

class InfoGAN(GAN):
    def loss_G(self, batch, noise_gen, **kwargs):
        noise = noise_gen(); latent_code = noise[:, :noise_gen.sizes[0]]
        fake = self.generate(noise)
        js_loss = self.js_loss_G(batch, fake)
        info_loss = self.info_loss(fake, latent_code)
        return js_loss + info_loss

    def loss_D(self, batch, noise_gen, **kwargs):
        noise = noise_gen(); latent_code = noise[:, :noise_gen.sizes[0]]
        fake = self.generate(noise)
        js_loss = self.js_loss_D(batch, fake)
        info_loss = self.info_loss(fake, latent_code)
        return js_loss + info_loss

    def info_loss(self, fake, latent_code):
        q = self.discriminator(fake)[1]
        q_mean = q[:, :, 0]
        q_logstd = q[:, :, 1]
        epsilon = (latent_code - q_mean) / (torch.exp(q_logstd) + _eps)
        return torch.mean(q_logstd + 0.5 * epsilon ** 2)

class BezierGAN(InfoGAN):
    def loss_G(self, batch, noise_gen, **kwargs):
        noise = noise_gen(); latent_code = noise[:, :noise_gen.sizes[0]]
        fake, cp, w, pv, intvls = self.generator(noise)
        js_loss = self.js_loss_G(batch, fake)
        info_loss = self.info_loss(fake, latent_code)
        reg_loss = self.regularizer(cp, w, pv, intvls)
        return js_loss + info_loss + 10 * reg_loss
    
    def regularizer(self, cp, w, pv, intvls):
        w_loss = torch.mean(w[:, :, 1:-1])
        cp_loss = torch.norm(cp[:, :, 1:] - cp[:, :, :-1], dim=1).mean()
        end_loss = torch.pairwise_distance(cp[:, :, 0], cp[:, :, -1]).mean()
        reg_loss = w_loss + cp_loss + end_loss
        return reg_loss
    
    def _epoch_report(self, epoch, epochs, batch, noise_gen, report_interval, tb_writer, **kwargs):
        if epoch % report_interval == 0:
            noise = noise_gen(); latent_code = noise[:, :noise_gen.sizes[0]]
            fake, cp, w, pv, intvls = self.generator(noise)
            js_loss = self.js_loss_D(batch, fake)
            info_loss = self.info_loss(fake, latent_code)
            reg_loss = self.regularizer(cp, w, pv, intvls)
            if tb_writer:
                tb_writer.add_scalar('JS Loss', js_loss, epoch)
                tb_writer.add_scalar('Info Loss', info_loss, epoch)
                tb_writer.add_scalar('Regularization Loss', reg_loss, epoch)
            else:
                print('[Epoch {}/{}] JS loss: {:d}, Info loss: {:d}, Regularization loss: {:d}'.format(
                    epoch, epochs,  js_loss, info_loss, reg_loss))
            try: 
                kwargs['plotting'](epoch, fake)
            except:
                pass

class EGAN(GAN):
    def __init__(self, generator: nn.Module, discriminator: nn.Module, lamb: float, 
        cost_func=lambda x1, x2: torch.cdist(x1.flatten(1), x2.flatten(1), p=1),
        opt_g_lr: float=1e-4, opt_g_betas: tuple=(0.5, 0.99), opt_g_eps: float=1e-8,
        opt_d_lr: float=1e-4, opt_d_betas: tuple=(0.5, 0.99), opt_d_eps: float=1e-8,
        name: str=None, checkpoint: str=None, train_mode: bool=True
        ):
        super().__init__(
            generator, discriminator, 
            opt_g_lr=opt_g_lr, opt_g_betas=opt_g_betas, opt_g_eps=opt_g_eps,
            opt_d_lr=opt_d_lr, opt_d_betas=opt_d_betas, opt_d_eps=opt_d_eps,
            name=name, checkpoint=checkpoint, train_mode=train_mode
            )
        self.lamb = lamb
        self.cost = cost_func

    def loss_D(self, batch, noise_gen, **kwargs):
        return -self.loss_G(batch, noise_gen)
    
    def loss_G(self, batch, noise_gen, **kwargs):
        fake = self.generate(noise_gen())
        return self.dual_loss(batch, fake)

    def dual_loss(self, real, fake):
        v, d_r, d_f = self._cal_v(real, fake) 
        smooth = strong_convex_func(v, lamb=self.lamb).mean()
        return d_r.mean() - d_f.mean() - smooth

    def surrogate_ll(self, test_samples, noise_gen): #### to be modified!!!!!!!!!!!!!
        noise, p_x = noise_gen(); fake = self.generate(noise)
        ll = torch.zeros(len(test_samples), device=test_samples.device)
        for i, test in enumerate(test_samples.unsqueeze(1)): #### to be modified!!!!!!!!!!!!!
            prob = self._estimate_prob(test, fake, p_x) # [n_test, batch]
            ep_dist = (self.cost(test, fake) * prob).sum(dim=1) # [n_test]
            entropy = (-prob * torch.log(prob + _eps)).sum(dim=1) # [n_test]
            ep_lpx = (torch.log(p_x).T * prob).sum(dim=1) # [n_test]
            ll[i] = (-ep_dist / self.lamb + entropy + ep_lpx).detach() # [n_test] log likelihood surrogate 2.7
        return ll
    
    def _cal_v(self, real, fake):
        d_r = first_element(self.discriminator(real))[:, 0] # [r_batch, 1]
        d_f = first_element(self.discriminator(fake))[:, 1] # [f_batch, 1]
        v = torch.squeeze(d_r.unsqueeze(1) - d_f.unsqueeze(0), dim=-1) \
            - self.cost(real, fake) # [r_batch, f_batch] v(y,^y) grid.
        return v, d_r, d_f

    def _estimate_prob(self, test, fake, p_x):
        v_star, _, _ = self._cal_v(test, fake) # [n_test, batch] term inside exp() of 4.3. 
        exp_v = torch.exp((v_star - v_star.max(dim=1, keepdim=True).values) / self.lamb) # [n_test, batch] avoid numerical instability.
        prob = p_x.T * exp_v # [n_test, batch] unnormalized 4.3
        return prob / prob.sum(dim=1, keepdim=True) # [n_test, batch] normalize over x to obtain 4.3 P(x|y)
    
    def _train_gen_criterion(self, batch, noise_gen, epoch):
        _, d_r, d_f = self._cal_v(batch, self.generate(noise_gen()))
        return d_r.mean() - d_f.mean() > 0

class SinkhornEGAN(EGAN):
    def loss_D(self, batch, noise_gen, **kwargs):
        return -self.cost(batch, self.generate(noise_gen())).mean() \
            if isinstance(self.cost, nn.Module) \
            else torch.tensor(0, device=batch.device)
    
    def loss_G(self, batch, noise_gen, **kwargs):
        fake = self.generate(noise_gen())
        return self.sinkhorn_divergence(batch, fake) 
    
    def dual_loss(self, real, fake):
        len_a = len(first_element(real)); len_b = len(first_element(fake))
        a = torch.ones(len_a, 1, device=first_element(real).device) / len_a
        b = torch.ones(len_b, 1, device=first_element(fake).device) / len_b
        return regularized_ot(
            a, real, b, fake, 
            eps=self.lamb, assume_convergence=True, cost_func=self.cost, nits=1000
            )

    def sinkhorn_divergence(self, real, fake):
        len_a = len(first_element(real)); len_b = len(first_element(fake))
        a = torch.ones(len_a, 1, device=first_element(real).device) / len_a
        b = torch.ones(len_b, 1, device=first_element(fake).device) / len_b
        return sinkhorn_divergence(
            a, real, b, fake, 
            eps=self.lamb, assume_convergence=True, cost_func=self.cost, nits=1000
            )
    
    def _cal_v(self, real, fake):
        len_a = len(first_element(real)); len_b = len(first_element(fake))
        a = torch.ones(len_a, 1, device=first_element(real).device) / len_a
        b = torch.ones(len_b, 1, device=first_element(fake).device) / len_b
        p_f, p_r = sink(
            a, real, b, fake, 
            eps=self.lamb, assume_convergence=True, cost_func=self.cost, nits=1000
            ) # [f_batch], [r_batch]
        v = p_r.unsqueeze(1) + p_f.unsqueeze(0) \
            - self.cost(real, fake) # [r_batch, f_batch]
        return v, p_r, p_f
    
    def _train_gen_criterion(self, batch, noise_gen, epoch): return True

class BezierEGAN(EGAN, BezierGAN):
    def loss_D(self, batch, noise_gen, **kwargs):
        noise = noise_gen(); latent_code = noise[:, :noise_gen.sizes[0]]
        fake = self.generate(noise)
        dual_loss = self.dual_loss(batch, fake)
        info_loss = self.info_loss(fake, latent_code)
        return -dual_loss + info_loss

    def loss_G(self, batch, noise_gen, **kwargs):
        noise = noise_gen(); latent_code = noise[:, :noise_gen.sizes[0]]
        fake, cp, w, pv, intvls = self.generator(noise)
        dual_loss = self.dual_loss(batch, fake)
        info_loss = self.info_loss(fake, latent_code)
        reg_loss = self.regularizer(cp, w, pv, intvls)
        return dual_loss + info_loss + 10 * reg_loss
    
    def _epoch_report(self, epoch, epochs, batch, noise_gen, report_interval, tb_writer, **kwargs):
        if epoch % report_interval == 0:
            noise = noise_gen(); latent_code = noise[:, :noise_gen.sizes[0]]
            fake, cp, w, pv, intvls = self.generator(noise)
            v, d_r, d_f = self._cal_v(batch, fake)
            smooth = strong_convex_func(v, lamb=self.lamb).mean()
            info_loss = self.info_loss(fake, latent_code)
            reg_loss = self.regularizer(cp, w, pv, intvls)
            if tb_writer:
                # tb_writer.add_scalar('Dual Loss', dual_loss, epoch)
                tb_writer.add_scalars('Dual Loss', {
                                        'dual': d_r.mean() - d_f.mean() - smooth,
                                        'emd': d_r.mean() - d_f.mean(),
                                        'smooth': smooth
                                        }, epoch)
                tb_writer.add_scalar('Info Loss', info_loss, epoch)
                tb_writer.add_scalar('Regularization Loss', reg_loss, epoch)
            else:
                print('[Epoch {}/{}] Dual loss: {:d}, Info loss: {:d}, Regularization loss: {:d}'.format(
                    epoch, epochs,  d_r.mean() - d_f.mean() - smooth, info_loss, reg_loss))
            try: 
                kwargs['plotting'](epoch, fake)
            except:
                pass

class BezierSEGAN(SinkhornEGAN, BezierGAN):
    def loss_D(self, batch, noise_gen, **kwargs):
        noise = noise_gen(); latent_code = noise[:, :noise_gen.sizes[0]]
        fake = self.generate(noise)
        info_loss = self.info_loss(fake, latent_code)
        cost_loss = -self.cost(batch, fake).mean() \
            if isinstance(self.cost, nn.Module) \
            else torch.tensor(0, device=first_element(batch).device)
        return cost_loss + info_loss

    def loss_G(self, batch, noise_gen, **kwargs):
        noise = noise_gen(); latent_code = noise[:, :noise_gen.sizes[0]]
        fake, cp, w, pv, intvls = self.generator(noise)
        sinkhorn_loss = self.sinkhorn_divergence(batch, fake)
        info_loss = self.info_loss(fake, latent_code)
        reg_loss = self.regularizer(cp, w, pv, intvls)
        return sinkhorn_loss + info_loss + 10 * reg_loss
    
    def _epoch_report(self, epoch, epochs, batch, noise_gen, report_interval, tb_writer, **kwargs):
        if epoch % report_interval == 0:
            noise = noise_gen(); latent_code = noise[:, :noise_gen.sizes[0]]
            fake, cp, w, pv, intvls = self.generator(noise)
            sinkhorn_loss = self.sinkhorn_divergence(batch, fake)
            info_loss = self.info_loss(fake, latent_code)
            reg_loss = self.regularizer(cp, w, pv, intvls)
            if tb_writer:
                # tb_writer.add_scalar('Dual Loss', dual_loss, epoch)
                tb_writer.add_scalar('Sinkhorn Divergence', sinkhorn_loss, epoch)
                tb_writer.add_scalar('Info Loss', info_loss, epoch)
                tb_writer.add_scalar('Regularization Loss', reg_loss, epoch)

            try: 
                kwargs['plotting'](epoch, fake)
            except:
                pass