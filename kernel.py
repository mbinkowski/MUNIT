import torch
import utils

def HSIC(K, L):
    m = K.size(0)
    H = utils.cuda(torch.eye(m) - 1./(m**2))

    return 1/((m-1)**2) * torch.trace(K.mm(H).mm(L).mm(H))


def _level2(X, Y=None):
    XX = torch.mm(X, X.t())
    Xsqnorms = torch.diag(XX).repeat(1, 1)
    if Y is not None:
        XY = torch.mm(X, Y.t())
        YY = torch.mm(Y, Y.t())
        Ysqnorms = torch.diag(YY).repeat(1, 1)
        return XX, XY, YY, Xsqnorms, Ysqnorms
    return XX, XX, XX, Xsqnorms, Xsqnorms

def polynomial(X, Y, degree=3, gamma=None, coef0=1, K_XY_only=True):
    if gamma is None:
        gamma = 1. / X.size(1)
    p = lambda x, y: (torch.mm(x, y.t()) * gamma + coef0) ** degree

    if K_XY_only:
        return p(X, Y)
    return p(X, X), p(X, Y), p(Y, Y)


def mix_rbf(X, Y=None, sigmas=[2., 5., 10., 20., 40. ,80.], K_XY_only=True):
    XX, XY, YY, Xsqnorms, Ysqnorms = _level2(X, Y)
    K_XX, K_XY, K_YY = 0, 0, 0

    XYsqnorm = -2 * XY + Xsqnorms.t() + Ysqnorms
    for sigma in sigmas:
        gamma = 1/(2 * sigma**2)
        K_XY += torch.exp(-gamma * XYsqnorm)
    if K_XY_only:
        return K_XY
    if Y is None:
        return K_XY, K_XY, K_XY, False

    XXsqnorm = -2 * XX + Xsqnorms.t() + Xsqnorms
    YYsqnorm = -2 * YY + Ysqnorms.t() + Ysqnorms
    for sigma in sigmas:
        gamma = 1/(2 * sigma**2)
        K_XX += torch.exp(-gamma * XXsqnorm)
        K_YY += torch.exp(-gamma * YYsqnorm)

    return K_XX, K_XY, K_YY, False


def mix_rq(X, Y=None, alphas=[.2, .5, 1., 2., 5.], K_XY_only=True, add_dot=0.):
    XX, XY, YY, Xsqnorms, Ysqnorms = _level2(X, Y)
    K_XX, K_XY, K_YY = 0, 0, 0

    XYsqnorm = (-2 * XY + Xsqnorms.t() + Ysqnorms).clamp(min=0)
    for alpha in alphas:
        logXY = torch.log(1. + XYsqnorm/(2 * alpha))
        K_XY += torch.exp(-alpha * logXY)
    if add_dot > 0:
        K_XY += add_dot * XY
    if K_XY_only:
        return K_XY
    if Y is None:
        return K_XY, K_XY, K_XY, False

    XXsqnorm = (-2 * XX + Xsqnorms.t() + Xsqnorms).clamp(min=0)
    YYsqnorm = (-2 * YY + Ysqnorms.t() + Ysqnorms).clamp(min=0)
    for alpha in alphas:
        logXX = torch.log(1. + XXsqnorm/(2 * alpha))
        logYY = torch.log(1. + YYsqnorm/(2 * alpha))
        K_XX += torch.exp(-alpha * logXX)
        K_YY += torch.exp(-alpha * logYY)
    if add_dot > 0:
        K_XX += add_dot * XX
        K_YY += add_dot * YY

    return K_XX, K_XY, K_YY, False


def mix_rq_1dot(X, **kwargs):
    return mix_rq(X, add_dot=1, **kwargs)


def mix_rq_01dot(X, **kwargs):
    return mix_rq(X, add_dot=.1, **kwargs)


def mix_rq_001dot(X, **kwargs):
    return mix_rq(X, add_dot=.01, **kwargs)


def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m, n = K_XX.size(0), K_YY.size(0)

    if biased:
        mmd2 = K_XX.sum() / (m*m) + K_YY.sum() / (n*n) + K_XY.sum() / (m*n)
    else:
        if const_diagonal:
            trace_X, trace_Y = K_XX.trace(), K_YY.trace()
        else:
            trace_X, trace_Y = m * const_diagonal, n * const_diagonal

        mmd2 = ((K_XX.sum() - trace_X) / (m * (m-1))
              + (K_YY.sum() - trace_Y) / (n * (n-1))
              - 2 * K_XY.sum() / (m * n))
    return mmd2


def mmd2(K, biased=False):
    K_XX, K_XY, K_YY, const_diagonal = K
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal, biased)

