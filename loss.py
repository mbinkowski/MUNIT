import torch
from torch import nn
import torchvision.utils as vutils
import glob
import numpy as np
import os, sys
import time
import kernel

_eps = 1e-6
_combine = lambda x: x

def pnorm(x, dim=None, p=2):
    if dim is None:
        if p == 2:
            return (x**2).sum()
        return (x.abs()**p).sum()
    if dim == -1:
        x = x.view(x.size()[0], -1)
        dim = 1
    if p == 2:
        return (x**2).sum(dim=dim)
    return (x.abs()**p).sum(dim=dim)


def hms(start, diff=False):
    if diff:
        t = start
    else:
        t = time.time() - start
    h = int(t / 3600)
    m = int((t - 3600*h)/60)
    s = int(t - 3600*h - 60*m)
    return '%02d:%02d:%02d' % (h, m, s)

def add_prefix(log, prefix):
    return dict([(prefix + k, v) for k, v in log.items()])

def accuracy(gt, prob):
    _, pred = torch.max(prob, 1)
    return (gt == pred).float().mean()


def cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x


def stack(x, mode='full'):
    if type(x) == list:
        if mode == 'full':
            return torch.cat([s.view(s.size(0), -1) for s in x], dim=1)
        elif mode == 'avgs':
            return torch.stack([s.view(s.size(0), -1).mean(dim=1) for s in x], dim=1)
        elif mode == 'avg':
            return torch.stack([s.view(s.size(0), -1).mean(dim=1) for s in x], dim=1).mean(dim=1)
    return x


def grad_penalty(fx, x, lip1=False):
    dfx = torch.autograd.grad(outputs=fx, inputs=x,
                              grad_outputs=cuda(torch.ones(fx.size())),
                              create_graph=True, retain_graph=True,
                              only_inputs=True)[0]
    if lip1:
        norm = torch.sqrt(pnorm(dfx, dim=-1) + _eps)
        return (norm - 1)**2
    return pnorm(dfx, dim=-1)


def CE(x, c):
    target = cuda(c * torch.ones(x.size()))
    return torch.nn.functional.binary_cross_entropy_with_logits(x, target, reduction='none')

def JSGP(disc, neg, pos=None, combine_f=_combine,
         phase='generator', prefix='', _pos_mean=torch.mean, _neg_mean=torch.mean, **kwargs):
    d_neg = stack(disc(neg), 'avgs')
    if pos is not None:
        d_pos = stack(disc(pos), 'avgs')
    if phase == 'generator':
        loss = _neg_mean(CE(d_neg, 1.))
        if pos is not None:
            loss += _pos_mean(CE(d_pos, 0))
        return loss, {prefix + 'g loss': loss}
    assert pos is not None, 'Positive sample must be given in discriminator phase'
    loss = _pos_mean(CE(d_pos, 1.)) + _neg_mean(CE(d_neg, 0.))
    gp_pos = grad_penalty(CE(d_pos, 1.), pos)
    gp_neg = grad_penalty(CE(d_neg, 0.), neg)
    gp = _pos_mean(gp_pos) + _neg_mean(gp_neg)
    log = {'disc pos': _pos_mean(d_pos), 'disc neg': _neg_mean(d_neg), 'gp': gp, 'd loss': loss}
    return combine_f(loss, gp), add_prefix(log, prefix)
    

def LSGAN(disc, neg, pos=None, phase='generator', prefix='',
          _neg_mean=torch.mean, _pos_mean=torch.mean, **kwargs):
    d_neg = stack(disc(neg), 'full')
    if pos is not None:
        d_pos = stack(disc(pos), 'full')
    if phase == 'generator':
        loss = _neg_mean((d_neg - 1.)**2)
        if pos is not None:
            loss += _pos_mean((d_pos - 0)**2)
        return loss, {prefix + 'g loss': loss}
    assert pos is not None, 'Positive sample must be given in discriminator phase'
    loss = _pos_mean((d_pos - 1.)**2) + _neg_mean((d_neg - 0.)**2)
    log = {'disc pos': _pos_mean(d_pos), 'disc neg': _neg_mean(d_neg), 'd loss': loss}
    return loss, add_prefix(log, prefix)
    
    

def WGANGP(disc, neg, pos=None, combine_f=_combine,
           phase='generator', prefix='', gp_type='Roth', _neg_mean=torch.mean,
           _pos_mean=torch.mean, **kwargs):
    d_neg = stack(disc(neg), 'avg')
    disc_neg = _neg_mean(d_neg)
    log = {'disc neg': disc_neg}
    if pos is not None:
        d_pos = stack(disc(pos), 'avg')
        disc_pos = _pos_mean(d_pos)
        log.update({'disc pos': disc_pos})
    if phase == 'generator':
        loss = -disc_neg
        if pos is not None:
            loss += disc_pos
        log.update({'g loss': loss})
        return loss, add_prefix(log, prefix)

    assert pos is not None, 'Positive sample must be given in discriminator phase'
    loss = disc_neg - disc_pos
    gp_pos = grad_penalty(d_pos, pos, lip1=True)
    gp_neg = grad_penalty(d_neg, neg, lip1=True)
    gp = _pos_mean(gp_pos) + _neg_mean(gp_neg)
    log.update({'gp': gp, 'd loss': loss})
    return combine_f(loss, gp), add_prefix(log, prefix)


#def MMDGANGP(disc, neg, pos=None, combine_f=_combine, unroll_act=_unroll, 
#                 phase='generator', prefix='', kernel_f='mix_rq_1dot', 
#                 gp_type='mid', gp_split=None, **kwargs):
#    assert pos is not None, 'Positive sample must be given for MMD loss'
#    K = getattr(kernel, kernel_f)
#    d_neg, act_neg = unroll_act(disc(neg))
#    d_pos, act_pos = unroll_act(disc(pos))
#    log = {'disc pos': d_pos.mean(), 'disc neg': d_neg.mean()}
#    mmd2 = kernel.mmd2(K(d_pos, Y=d_neg, K_XY_only=False))
#
#    if phase == 'generator':
#        log.update({'loss': mmd2})
#        return mmd2, add_prefix(log, prefix)
#
#    def witness(xx):
#        d_xx = unroll_act(disc(xx))[0]
#        return (K(d_pos, Y=d_xx, K_XY_only=True) - K(d_neg, Y=d_xx, K_XY_only=True)).mean(dim=1)
#
#    if gp_type == 'mid':
#        gp = w2gan.grad_penalty_f(witness, pos, neg, lip1=True).mean()
#    elif (gp_type == 'ends') or (gp_split is None):
#        gp_pos = w2gan.grad_penalty_f(witness, pos, lip1=True).mean()
#        gp_neg = w2gan.grad_penalty_f(witness, neg, lip1=True).mean()
#        gp = gp_pos + gp_neg
#    elif (gp_type == 'ends_split') and (type(gp_split) == int):
#        witness_p = lambda tt: witness(torch.cat((pos[:, :gp_split], tt), dim=1))
#        witness_n = lambda tt: witness(torch.cat((tt, pos[:, gp_split:]), dim=1))
#        gp_pos = w2gan.grad_penalty_f(witness_p, pos[:, gp_split:], lip1=True).mean()
#        gp_neg = w2gan.grad_penalty_f(witness_n, neg[:, :gp_split], lip1=True).mean()
#        gp = gp_pos + gp_neg
#    elif (gp_type == 'ends_split') and (type(gp_split) == tuple):
#        witness_ = lambda tt: witness(torch.cat((pos[:, :gp_split[0]], tt), dim=1))
#        gp_pos = w2gan.grad_penalty_f(witness_, pos[:, gp_split[0]:], lip1=True).mean()
#        gp_neg = w2gan.grad_penalty_f(witness_, neg[:, gp_split[0]:], lip1=True).mean()
#        gp = gp_pos + gp_neg
#    log.update({'gp': gp, 'act': act_pos + act_neg, 'loss': -mmd2})
#    return combine_f(-mmd2, gp, act_pos + act_neg), add_prefix(log, prefix)
#
#def MMDGANGPends(disc, neg, **kwargs):
#    return MMDGANGP(disc, neg, gp_type='ends', **kwargs)
#
#def MMDGANGPends_split(disc, neg, **kwargs):
#    return MMDGANGP(disc, neg, gp_type='ends_split', **kwargs)
