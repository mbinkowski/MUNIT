"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from networks import AdaINGen, MsImageDis, VAEGen
import networks
from nets import ConvNet5, InvConvNet5
from utils import weights_init, get_model_list, vgg_preprocess, load_vgg16, get_scheduler, cuda
from torch.autograd import Variable
import torch
import torch.nn as nn
import os
import loss


class MUNIT_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(MUNIT_Trainer, self).__init__()
        # Initiate the networks
        self.gen_a = AdaINGen(hyperparameters['input_dim_a'], hyperparameters['gen'])  # auto-encoder for domain a
        self.gen_b = AdaINGen(hyperparameters['input_dim_b'], hyperparameters['gen'])  # auto-encoder for domain b
        self.dis_a = MsImageDis(hyperparameters['input_dim_a'], hyperparameters['dis'])  # discriminator for domain a
        self.dis_b = MsImageDis(hyperparameters['input_dim_b'], hyperparameters['dis'])  # discriminator for domain b
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.style_dim = hyperparameters['gen']['style_dim']

        # fix the noise used in sampling
        display_size = int(hyperparameters['display_size'])
        self.s_a = torch.randn(display_size, self.style_dim, 1, 1).cuda()
        self.s_b = torch.randn(display_size, self.style_dim, 1, 1).cuda()

        # Setup the optimizers
        opt_kwargs = {'lr': hyperparameters['lr'], 'betas': (hyperparameters['beta1'], hyperparameters['beta2']),
                        'weight_decay':hyperparameters['weight_decay']}
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad], **opt_kwargs)
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad], **opt_kwargs)
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))
        self.dis_a.apply(weights_init('gaussian'))
        self.dis_b.apply(weights_init('gaussian'))

        # Load VGG model if needed
        if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
            self.vgg = load_vgg16(hyperparameters['vgg_model_path'] + '/models')
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

        self.set_batch_weights(hyperparameters, opt_kwargs)

    def set_batch_weights(self, hyperparameters, opt_kwargs):
        if 'bw' not in hyperparameters:
            self.bw = self.bw_gen = False
            return
        config = hyperparameters['bw'].copy()
        config['size'] = hyperparameters['new_size']
        self.bw = True
        if config['AE']:
            i_dim = hyperparameters['input_dim_b']
        else:
            i_dim = hyperparameters['input_dim_a'] 
        self.weight = ConvNet5(i_dim=i_dim, o_dim=1, **config)
        bw_params = list(self.weight.parameters())

        if config['type'] == 'PQSEP2':
            self.weightB = ConvNet5(i_dim=hyperparameters['input_dim_b'], o_dim=1, **config)
            bw_params += list(self.weightB.parameters())

        self.bw_type = config['type']
        self.bw_full = config['full']
        self.bw_start = config['start']
        self.bw_recon = config['recon']
        self.bw_gen = config['gen']
        self.bw_opt = torch.optim.Adam(bw_params, **opt_kwargs)
        self.bw_scheduler = get_scheduler(self.bw_opt, hyperparameters)

        if config['AE']:
            kwargs = config.copy()
            kwargs.update({'bn': config['bn_AE']})
            self.BWenc = ConvNet5(i_dim=hyperparameters['input_dim_a'],
                                 o_dim=config['z_dim'], **kwargs)
            self.BWdec = InvConvNet5(o_dim=hyperparameters['input_dim_a'],
                                    i_dim=config['z_dim'], **kwargs)
            self.ae_opt = torch.optim.Adam(list(self.BWenc.parameters()) + list(self.BWdec.parameters()), **opt_kwargs)
            self.ae_scheduler = get_scheduler(self.ae_opt, hyperparameters)
        

    def _mean_a(self, x):
        if self.bw:
            return (x.view(x.size(0), -1).mean(dim=1) * self.w_a).sum()
        return x.mean()

    def _mean_b(self, x):
        if self.bw:
            return (x.view(x.size(0), -1).mean(dim=1) * self.w_b).sum()
        return x.mean()

    def recon_criterion(self, input, target, _mean=torch.mean):
        return _mean(torch.abs(input - target))

    def forward(self, x_a, x_b):
        self.eval()
        s_a = Variable(self.s_a)
        s_b = Variable(self.s_b)
        c_a, s_a_fake = self.gen_a.encode(x_a)
        c_b, s_b_fake = self.gen_b.encode(x_b)
        x_ba = self.gen_a.decode(c_b, s_a)
        x_ab = self.gen_b.decode(c_a, s_b)
        self.train()
        return x_ab, x_ba

    def gen_update(self, x_a, x_b, hyperparameters):
        self.gen_opt.zero_grad()
        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        # encode
        c_a, s_a_prime = self.gen_a.encode(x_a)
        c_b, s_b_prime = self.gen_b.encode(x_b)
        # decode (within domain)
        x_a_recon = self.gen_a.decode(c_a, s_a_prime)
        x_b_recon = self.gen_b.decode(c_b, s_b_prime)
        c_b, s_b_prime = self.gen_b.encode(x_b)
        # decode (within domain)
        x_a_recon = self.gen_a.decode(c_a, s_a_prime)
        x_b_recon = self.gen_b.decode(c_b, s_b_prime)

    def get_batch_weights(self, x, gen_y, y, step, train=True, gen_x=None):
        w_0 = cuda(torch.ones(x.size(0)) / x.size(0))
        n0 = self.bw_full - self.bw_start
        t = .25 + max(0, (step - self.bw_start) / n0)
        s = n0 / (n0 + 10 * (step - self.bw_start))
        _sm = lambda w: torch.nn.functional.softmax((w.view(-1) - w.mean()).clamp(-t, t), dim=0)

        if step < self.bw_start:
            w_x, w_y = w_0, w_0
        elif self.bw_type == 'P':
            w_x, w_y = _sm(self.weight(x)), w_0
        elif self.bw_type == 'PQ':
            w_x = .5 * (w_0 + _sm(self.weight(gen_y)))
            w_y = .5 * (w_0 + _sm(-self.weight(y)))
        elif self.bw_type == 'PQSEP2':
            assert gen_x is not None
            w_x = .25 * (2*w_0 + _sm(self.weight(x)) + _sm(-self.weightB(gen_y)))
            w_y = .25 * (2*w_0 + _sm(-self.weight(gen_x)) + _sm(self.weightB(y)))
        elif self.bw_type == 'A0P':
            r_x, r_y = self.weight(gen_y), self.weight(y).mean(dim=0, keepdim=True)
            a = torch.matmul(r_x, r_y.t()) / np.sqrt(r_x.size(1))
            w_x, w_y = _sm(a), w_0
        elif self.bw_type == 'A0PQ':
            r_x, r_y = self.weight(gen_y), self.weight(y)
            a = torch.matmul(r_x, r_y.t()) / np.sqrt(r_x.size(1))
            w_x = .5 * (_sm(a.mean(dim=1)) +  w_0)
            w_y = .5 * (_sm(a.t().mean(dim=1)) + w_0)
        elif self.bw_type == 'AP':
            _sm = lambda w: torch.nn.functional.softmax(w.clamp(-t, t), dim=0)
            r_x, r_y = self.weight(gen_y), self.weight(y)
            a = torch.matmul(r_x, r_y.t()) / np.sqrt(r_x.size(1))
            w_x, w_y = _sm(a).mean(dim=1), w_0
        elif self.bw_type == 'APQ':
            _sm = lambda w: torch.nn.functional.softmax(w.clamp(-t, t), dim=0)
            r_x, r_y = self.weight(gen_y), self.weight(y)
            a = torch.matmul(r_x, r_y.t()) / np.sqrt(r_x.size(1))
            w_x = .5 * (_sm(a).mean(dim=1) +  w_0)
            w_y = .5 * (_sm(a.t()).mean(dim=1) + w_0)
        else:
            raise Exception('Wrong batch weight type: ' + self.bw_type)

#        self._log_batch_weights(w_x, self.ground_truth_x, train=train, prefix='x ', step=step)
#        self._log_batch_weights(w_y, self.ground_truth_y, train=train, prefix='y ', sign=-1,
#                                step=step, yy_gen=torch.cat((y, gen_y), dim=0))
        return w_x, w_y

    def train_batch_weights(self, g_loss, y=None, gen_y=None, w_x=None, w_y=None):
        # Batch weight DIM
        log = {}
        if hasattr(self, 'BWenc'):
            yy = torch.cat((y, gen_y), dim=0)
            # yy = y
            cc = self.BWenc(yy)
            yy_dec, zz = self.BWdec(cc), cc.view(2 * y.size(0), -1)
            ae_loss = (yy_dec - yy).abs().mean()
            log.update({'AE BW ': ae_loss})
            
            self.ae_opt.zero_grad()
            ae_loss.backward(retain_graph=True)
            self.ae_opt.step()

            z, gen_z = zz[:y.size(0)], zz[y.size(0):]
            #z, gen_z = zz, self.WencZ(self.WencC(gen_y))
            g_loss = (gen_z * w_x.view(w_x.size(0), -1) - z * w_y.view(w_y.size(0), -1)).mean(dim=0)

        w_loss = (g_loss ** 2).sum()
        self.bw_opt.zero_grad()
        log.update({'w loss': w_loss})
        w_loss.backward(retain_graph=True)
        self.bw_opt.step()


    def forward(self, x_a, x_b):
        self.eval()
        s_a = Variable(self.s_a)
        s_b = Variable(self.s_b)
        c_a, s_a_fake = self.gen_a.encode(x_a)
        c_b, s_b_fake = self.gen_b.encode(x_b)
        x_ba = self.gen_a.decode(c_b, s_a)
        x_ab = self.gen_b.decode(c_a, s_b)
        self.train()
        return x_ab, x_ba

    def gen_update(self, x_a, x_b, hyperparameters, iteration):
        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        # encode
        c_a, s_a_prime = self.gen_a.encode(x_a)
        c_b, s_b_prime = self.gen_b.encode(x_b)
        # decode (within domain)
        x_a_recon = self.gen_a.decode(c_a, s_a_prime)
        x_b_recon = self.gen_b.decode(c_b, s_b_prime)
        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b, s_a)
        x_ab = self.gen_b.decode(c_a, s_b)
        # encode again
        c_b_recon, s_a_recon = self.gen_a.encode(x_ba)
        c_a_recon, s_b_recon = self.gen_b.encode(x_ab)
        # decode again (if needed)
        x_aba = self.gen_a.decode(c_a_recon, s_a_prime) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bab = self.gen_b.decode(c_b_recon, s_b_prime) if hyperparameters['recon_x_cyc_w'] > 0 else None

        _mean_a, _mean_b, _mean_recon_a, _mean_recon_b = torch.mean, torch.mean, torch.mean, torch.mean
        if self.bw:
            self.w_a, self.w_b = self.get_batch_weights(x_a, x_ab, x_b, iteration, train=True, gen_x=x_ba)
            if self.bw_gen:
                _mean_a, _mean_b = self._mean_a, self._mean_b
                if self.bw_recon:
                    _mean_recon_a, _mean_recon_b = self._mean_a, self._mean_b
        # reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a, _mean_recon_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b, _mean_recon_b)
        self.loss_gen_recon_s_a = self.recon_criterion(s_a_recon, s_a)
        self.loss_gen_recon_s_b = self.recon_criterion(s_b_recon, s_b)
        self.loss_gen_recon_c_a = self.recon_criterion(c_a_recon, c_a)
        self.loss_gen_recon_c_b = self.recon_criterion(c_b_recon, c_b)
        self.loss_gen_cycrecon_x_a = self.recon_criterion(x_aba, x_a, _mean_recon_a) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        self.loss_gen_cycrecon_x_b = self.recon_criterion(x_bab, x_b, _mean_recon_b) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba, _mean_fake=_mean_a)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab, _mean_fake=_mean_b)
        # domain-invariant perceptual loss
        self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, x_ba, x_b) if hyperparameters['vgg_w'] > 0 else 0
        self.loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, x_ab, x_a) if hyperparameters['vgg_w'] > 0 else 0
        # total loss
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_b + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_a + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_b + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_b + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_a + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_b + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_a + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_b

        self.gen_opt.zero_grad()
        self.loss_gen_total.backward()
        self.gen_opt.step()

    def compute_vgg_loss(self, vgg, img, target, _mean=torch.mean):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return _mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def sample(self, x_a, x_b):
        self.eval()
        s_a1 = Variable(self.s_a)
        s_b1 = Variable(self.s_b)
        s_a2 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b2 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        x_a_recon, x_b_recon, x_ba1, x_ba2, x_ab1, x_ab2 = [], [], [], [], [], []
        for i in range(x_a.size(0)):
            c_a, s_a_fake = self.gen_a.encode(x_a[i].unsqueeze(0))
            c_b, s_b_fake = self.gen_b.encode(x_b[i].unsqueeze(0))
            x_a_recon.append(self.gen_a.decode(c_a, s_a_fake))
            x_b_recon.append(self.gen_b.decode(c_b, s_b_fake))
            x_ba1.append(self.gen_a.decode(c_b, s_a1[i].unsqueeze(0)))
            x_ba2.append(self.gen_a.decode(c_b, s_a2[i].unsqueeze(0)))
            x_ab1.append(self.gen_b.decode(c_a, s_b1[i].unsqueeze(0)))
            x_ab2.append(self.gen_b.decode(c_a, s_b2[i].unsqueeze(0)))
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba1, x_ba2 = torch.cat(x_ba1), torch.cat(x_ba2)
        x_ab1, x_ab2 = torch.cat(x_ab1), torch.cat(x_ab2)
        self.train()
        return x_a, x_a_recon, x_ab1, x_ab2, x_b, x_b_recon, x_ba1, x_ba2

    def sample_many(self, x_a, x_b, n=8):
        self.eval()
        s_a = [torch.cat([torch.randn(1, self.style_dim, 1, 1).cuda()] * x_a.size(0), dim=0)
                for _ in range(n)]
        s_b = [torch.cat([torch.randn(1, self.style_dim, 1, 1).cuda()] * x_b.size(0), dim=0)
                for _ in range(n)]
        x_ba, x_ab = [], []
        for i in range(n):
            c_a, s_a_fake = self.gen_a.encode(x_a)
            c_b, s_b_fake = self.gen_b.encode(x_b)
            x_ba.append(self.gen_a.decode(c_b, s_a[i]))
            x_ab.append(self.gen_b.decode(c_a, s_b[i]))
        self.train()
        return [x_a] + x_ab + [x_b] + x_ba

    def dis_update(self, x_a, x_b, hyperparameters, iteration):
        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        # encode
        c_a, _ = self.gen_a.encode(x_a)
        c_b, _ = self.gen_b.encode(x_b)
        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b, s_a)
        x_ab = self.gen_b.decode(c_a, s_b)

        if self.bw:
            self.w_a, self.w_b = self.get_batch_weights(x_a, x_ab, x_b, iteration, train=True, gen_x=x_ba)
            _mean_a, _mean_b = self._mean_a, self._mean_b
        else:
            _mean_a, _mean_b = torch.mean, torch.mean
        # D loss
        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a, _mean_a, _mean_b)
        self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b, _mean_b, _mean_a)
        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_a + hyperparameters['gan_w'] * self.loss_dis_b

        self.dis_opt.zero_grad()
        self.loss_dis_total.backward(retain_graph=self.bw)
        self.dis_opt.step()

        if self.bw:
            self.train_batch_weights(self.loss_dis_total, y=x_b, gen_y=x_ab, w_x=self.w_a, w_y=self.w_b)

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen_a.load_state_dict(state_dict['a'])
        self.gen_b.load_state_dict(state_dict['b'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        if hasattr(self, 'dis_ab'):
            self.dis_ab.load_state_dict(state_dict['ab'])
        else:
            self.dis_a.load_state_dict(state_dict['a'])
            self.dis_b.load_state_dict(state_dict['b'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'a': self.gen_a.state_dict(), 'b': self.gen_b.state_dict()}, gen_name)
        if hasattr(self, 'dis_ab'):
            torch.save({'ab': self.dis_ab.state_dict()}, dis_name)
        else:
            torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)


class MUNITDD_Trainer(MUNIT_Trainer):
    def __init__(self, hyperparameters):
        super(MUNIT_Trainer, self).__init__()
        # Initiate the networks
        if 'net' in hyperparameters['dis']:
            Dis = getattr(networks, hyperparameters['dis']['net'])
        else:
            Dis = MsImageDis
        self.gen_a = AdaINGen(hyperparameters['input_dim_a'], hyperparameters['gen'])  # auto-encoder for domain a
        self.gen_b = AdaINGen(hyperparameters['input_dim_b'], hyperparameters['gen'])  # auto-encoder for domain b
        self.dis_ab = Dis(hyperparameters['input_dim_a'] + hyperparameters['input_dim_b'],
                          hyperparameters['dis'])  # joint discriminator for domains a & b
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.style_dim = hyperparameters['gen']['style_dim']

        # fix the noise used in sampling
        display_size = int(hyperparameters['display_size'])
        self.s_a = torch.randn(display_size, self.style_dim, 1, 1).cuda()
        self.s_b = torch.randn(display_size, self.style_dim, 1, 1).cuda()

        # Setup the optimizers
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
        opt_kwargs = {'lr': hyperparameters['lr'], 'weight_decay': hyperparameters['weight_decay'],
                      'betas': (hyperparameters['beta1'], hyperparameters['beta2'])}
        self.dis_opt = torch.optim.Adam([p for p in list(self.dis_ab.parameters()) if p.requires_grad], **opt_kwargs)
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad], **opt_kwargs)
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))
        self.dis_ab.apply(weights_init('gaussian'))

        # Load VGG model if needed
        if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
            self.vgg = load_vgg16(hyperparameters['vgg_model_path'] + '/models')
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

        self.set_batch_weights(hyperparameters, opt_kwargs)

    def gen_update(self, x_a, x_b, hyperparameters, iteration):
        self.gen_opt.zero_grad()
        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        # encode
        c_a, s_a_prime = self.gen_a.encode(x_a)
        c_b, s_b_prime = self.gen_b.encode(x_b)
        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b, s_a) # fake 'a' generated from b
        x_ab = self.gen_b.decode(c_a, s_b)

        _mean_a, _mean_b = torch.mean, torch.mean
        if self.bw:
            self.w_a, self.w_b = self.get_batch_weights(x_a, x_ab, x_b, iteration, train=True, gen_x=x_ba)
            if self.bw_gen:
                _mean_a, _mean_b = self._mean_a, self._mean_b
        # GAN loss
        loss_f = getattr(loss, self.dis_ab.gan_type.upper())
        scale_type = hyperparameters['dis']['scale_type'] if 'scale_type' in hyperparameters['dis'] else None
        scale = hyperparameters['dis']['gp'] if 'gp' in hyperparameters['dis'] else None
        self.loss_gen_adv_ab, log = loss_f(self.dis_ab, torch.cat((x_a, x_ab), dim=1),
                                           pos=torch.cat((x_ba, x_b), dim=1),
                                           phase='generator', weights=_mean_a,
                                           pos_weights=_mean_b, scale=scale,
                                           scale_type=scale_type )
#        self.loss_gen_adv_ab = self.dis_ab.calc_gen_loss(input_fake=torch.cat((x_a, x_ba.detach()), dim=1),
#                                                         input_real=torch.cat((x_ab.detach(), x_b), dim=1))
        # domain-invariant perceptual loss
        self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, x_ba, x_b) if hyperparameters['vgg_w'] > 0 else 0
        self.loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, x_ab, x_a) if hyperparameters['vgg_w'] > 0 else 0
        # total loss
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_ab + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_a + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_b

        self.loss_gen_total.backward()
        self.gen_opt.step()

        self.logger.print_log(log)

    def dis_update(self, x_a, x_b, hyperparameters, iteration):
        self.dis_opt.zero_grad()
        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        # encode
        c_a, s_a_prime = self.gen_a.encode(x_a)
        c_b, s_b_prime = self.gen_b.encode(x_b)
        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b, s_a) # fake 'a' generated from b
        x_ab = self.gen_b.decode(c_a, s_b)

        if self.bw:
            self.w_a, self.w_b = self.get_batch_weights(x_a, x_ab, x_b, iteration, train=True, gen_x=x_ba)
            _mean_a, _mean_b = self._mean_a, self._mean_b
        else:
            _mean_a, _mean_b = torch.mean, torch.mean
        # GAN loss
        loss_f = getattr(loss, self.dis_ab.gan_type.upper())
        combine_f = lambda x, gp: x + hyperparameters['dis']['gp'] * gp
        scale_type = hyperparameters['dis']['scale_type'] if 'scale_type' in hyperparameters['dis'] else None
        scale = hyperparameters['dis']['gp'] if 'gp' in hyperparameters['dis'] else None
        loss_dis_ab, log = loss_f(self.dis_ab, torch.cat((x_a, x_ab), dim=1),
                                  pos=torch.cat((x_ba, x_b), dim=1),
                                  phase='discriminator', weights=_mean_a,
                                  pos_weights=_mean_b, combine_f=combine_f,
                                  scale=scale, scale_type=scale_type)
        
        self.loss_dis_total = hyperparameters['gan_w'] * loss_dis_ab

        self.loss_dis_total.backward(retain_graph=self.bw)
        self.dis_opt.step()

        if self.bw:
            self.train_batch_weights(self.loss_dis_total, y=x_b, gen_y=x_ab, w_x=self.w_a, w_y=self.w_b)

        self.logger.print_log(log)

UNIT_Trainer = MUNIT_Trainer



#        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
#        x_ba = torch.cat(x_ba)
#        x_ab = torch.cat(x_ab)
#        self.train()
#        return x_a, x_a_recon, x_ab, x_b, x_b_recon, x_ba
#
#    def dis_update(self, x_a, x_b, hyperparameters):
#        self.dis_opt.zero_grad()
#        # encode
#        h_a, n_a = self.gen_a.encode(x_a)
#        h_b, n_b = self.gen_b.encode(x_b)
#        # decode (cross domain)
#        x_ba = self.gen_a.decode(h_b + n_b)
#        x_ab = self.gen_b.decode(h_a + n_a)
#        # D loss
#        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
#        self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)
#        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_a + hyperparameters['gan_w'] * self.loss_dis_b
#        self.loss_dis_total.backward()
#        self.dis_opt.step()
#
#    def update_learning_rate(self):
#        if self.dis_scheduler is not None:
#            self.dis_scheduler.step()
#        if self.gen_scheduler is not None:
#            self.gen_scheduler.step()
#
#    def resume(self, checkpoint_dir, hyperparameters):
#        # Load generators
#        last_model_name = get_model_list(checkpoint_dir, "gen")
#        state_dict = torch.load(last_model_name)
#        self.gen_a.load_state_dict(state_dict['a'])
#        self.gen_b.load_state_dict(state_dict['b'])
#        iterations = int(last_model_name[-11:-3])
#        # Load discriminators
#        last_model_name = get_model_list(checkpoint_dir, "dis")
#        state_dict = torch.load(last_model_name)
#        self.dis_a.load_state_dict(state_dict['a'])
#        self.dis_b.load_state_dict(state_dict['b'])
#        # Load optimizers
#        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
#        self.dis_opt.load_state_dict(state_dict['dis'])
#        self.gen_opt.load_state_dict(state_dict['gen'])
#        # Reinitilize schedulers
#        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
#        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
#        print('Resume from iteration %d' % iterations)
#        return iterations
#
#    def save(self, snapshot_dir, iterations):
#        # Save generators, discriminators, and optimizers
#        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
#        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
#        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
#        torch.save({'a': self.gen_a.state_dict(), 'b': self.gen_b.state_dict()}, gen_name)
#        torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict()}, dis_name)
#        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)
