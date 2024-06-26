import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def get_bridges(args, inputs, labels, num_classes, device, other_inputs=None, other_labels=None, sigma=None):
    '''
    sample brownian bridges for inputs and labels
    '''
    if args.sigma_y == -1:
        args.sigma_y = args.sigma_x
        
    real_batch_size = inputs.shape[0]
    d = int(torch.prod(torch.tensor(inputs.shape[1:])))

    t = torch.linspace(0, args.T, args.n_t).unsqueeze(0).unsqueeze(-1)
    t = t.repeat(real_batch_size*args.n_b, 1, d).float().to(device)
    t_label = torch.linspace(0, args.T, args.n_t).unsqueeze(0).unsqueeze(-1)
    t_label = t_label.repeat(real_batch_size*args.n_b, 1, 1).float().to(device)
    
    if other_inputs is None:
        # Random endpoints
        in_shuffle = torch.randperm(real_batch_size).to(device)
        inflat_a = inputs.flatten(1)
        inflat_b = inflat_a[in_shuffle]
        labels_oh_a = F.one_hot(labels, num_classes=num_classes)
        labels_oh_b = labels_oh_a[in_shuffle]
    elif args.euclidean:
        # Euclidean distance based endpoints
        inflat_a = inputs.flatten(1)
        in_shuffle = torch.cdist(inflat_a,inflat_b,1).sort(-1)[1][:,1]
        inflat_b = inflat_a[in_shuffle]
        labels_oh_a = F.one_hot(labels, num_classes=num_classes)
        labels_oh_b = labels_oh_a[in_shuffle]
    else:
        # Predetermined endpoints
        inflat_a = inputs.flatten(1)
        inflat_b = other_inputs.flatten(1)
        labels_oh_a = labels
        labels_oh_b = other_labels
    
    inflat_a = inflat_a.repeat(args.n_b,1)
    inflat_b = inflat_b.repeat(args.n_b,1)
    labels_oh_a = labels_oh_a.repeat(args.n_b,1)
    labels_oh_b = labels_oh_b.repeat(args.n_b,1)
    
    sigma_x = None
    sigma_y = None
    if args.temper and sigma is not None:
        #import pdb
        #pdb.set_trace()
        sigma_a = sigma.unsqueeze(1).unsqueeze(2)
        sigma_b = sigma[in_shuffle].unsqueeze(1).unsqueeze(2)
        ratio = 1 if args.sigma_y == -1 else args.sigma_y/args.sigma_x
        sigma_x = t * sigma_a + (1-t)*sigma_b
        sigma_y = (t_label * sigma_a + (1-t_label)*sigma_b)*(ratio)
        
        
    mix_input = bbridges(t, inflat_a, inflat_b, 
                         var=sigma_x if args.temper else args.sigma_x, simplex=False).reshape(-1, *inputs.shape[1:])
    mix_label = bbridges(t_label, labels_oh_a, labels_oh_b, 
                         var=sigma_y if args.temper else args.sigma_y, simplex=True).reshape(-1, *labels_oh_a.shape[1:])
    return mix_input, mix_label
    

def bbridges(t, a, b, var=1, pp=True, simplex=False):
    '''
    Samples a Brownian Bridge from a to b.
    '''

    dt = t[:,1] - t[:,0]
    t = (t - t[:,0].unsqueeze(1)) / (t[:,-1] - t[:,0]).unsqueeze(1)

    if pp:
        pp = -(torch.rand_like(t) + 1e-4).log()
    else:
        pp = 1

    dW = torch.randn_like(t) * dt.sqrt().unsqueeze(1) * var * pp
    #import pdb; pdb.set_trace()
    W = dW.cumsum(1)
    W[:,0] = 0
    W = W + a.unsqueeze(1)
    
    BB = W - t * (W[:,-1] - b).unsqueeze(1)
    if simplex:
        bridge_abs = BB.abs()
        BB = bridge_abs / bridge_abs.sum(-1, keepdims=True)
    return BB

def brownian_bridge(t, N=1, noise=None, sigma=1):
    '''
    Samples a Brownian Bridge from 0 to 0.
    '''

    dt = t[:,1] - t[:,0]
    t = (t - t[:,0].unsqueeze(1)) / (t[:,-1] - t[:,0]).unsqueeze(1)

    if noise is None:
        noise = torch.randn((N, t.shape[0], t.shape[1]))

    dW = noise * dt.sqrt().unsqueeze(1) * sigma
    W = dW.cumsum(-1)
    W[:,:,0] = 0

    BB = W - t * (W[:,:,-1]).unsqueeze(-1)
    
    return BB, t

def brownian_bridge_nd(t, N = 100, noise=None):
    '''
    Samples a Brownian sheet from 0 to 0.
    '''

    dt = t[:,:,1] - t[:,:,0]
    t = (t - t[:,:,0].unsqueeze(-1)) / (t[:,:,-1] - t[:,:,0]).unsqueeze(-1)

    if noise is None:
        noise = torch.randn((N, t.shape[0], t.shape[1], t.shape[2]))

    dW = noise * dt.sqrt().unsqueeze(-1)
    W = dW.cumsum(-1)
    W[:,:,:,0] = 0

    BB = W - t.unsqueeze(0) * (W[:,:,:,-1]).unsqueeze(-1)
    
    return BB, t

def brownian_bridge_nd(t, noise=None):
    '''
    Samples a Brownian Bridge from 0 to 0.
    '''

    dta = (t[:,:,1] - t[:,:,0])
    dt = (t[:,:,1] - t[:,:,0]).clone()
    t = (t - t[:,:,0].unsqueeze(-1)) / (t[:,:,-1] - t[:,:,0]).unsqueeze(-1)

    if noise is None:
        noise = torch.randn_like(t)

    dW = noise * dt.sqrt().unsqueeze(-1)
    W = dW
    W[:,:,0] = 0
    W = W.cumsum(-1)

    BB = W - t * (W[:,:,-1]).unsqueeze(-1)
    
    return BB, t

def excursion(t, neg=True, N=2, noise=None, sigma=1):
    '''
    Simulates excursions from a brownian bridge.
    '''
    if len(t.shape) > 2:
        bb, t  = brownian_bridge_nd(t, noise=noise)
    else:
        bb, t  = brownian_bridge(t, N=N, noise=noise, sigma=sigma)
    m, idx = bb.min(-1)
    if len(t.shape) > 2:
        t_rep = t
    else:
        t_rep = t.unsqueeze(0).repeat(N, 1, 1)
    ini = torch.arange(bb.shape[0])
    inj = torch.arange(bb.shape[1])
    ij  = torch.meshgrid(ini, inj, indexing='ij')

    nt = ( t_rep[ij[0], ij[1], idx].unsqueeze(-1) + t_rep ) % 1
    j  = torch.floor(nt * t.shape[-1]).long()

    j[j<0] = 0 

    BE = (bb.gather(-1,j) - m.unsqueeze(-1))

    if neg:
        if len(t.shape) > 2:
            bernoulli = torch.randint(2, (t.shape[0],t.shape[1])) * 2 - 1
            BE = BE * bernoulli.unsqueeze(-1)
        else:
            bernoulli = torch.randint(2, (t.shape[0],1)) * 2 - 1
            BE = BE * bernoulli

    return BE, bb

def get_log_mixture(N):

    mix_param = nn.Parameter(torch.ones(N,))
    loc_param = nn.Parameter(torch.rand(N,))
    scale_param = nn.Parameter(torch.rand(N,))

    mix  = D.categorical.Categorical(mix_param)
    comp = D.log_normal.LogNormal(loc_param, scale_param)
    lmm  = D.mixture_same_family.MixtureSameFamily(mix, comp)

from scipy import stats
class GammaMixture():

    def _ppf(self, q, *args, **kwds):
        print(q)
        mix = D.categorical.Categorical(torch.ones(2,))
        comp = D.gamma.Gamma(torch.tensor([2.0, 7.5]), torch.tensor([2.0, 1.0]))
        m = D.mixture_same_family.MixtureSameFamily(mix, comp)
        return m.icdf(torch.tensor(q))


def compute_grad(model, x, y_true, y_b, lam):
    """ Compute grad_x L(f(X), Y)"""
    outputs = model(x)
    grad_xF = torch.autograd.grad(outputs[0,y_true[0]], x)
    # currently only works for CE
    return lam/outputs[0,y_true[0]] * (grad_xF[0]) + (1-lam)/outputs[0,y_b[0]] * (grad_xF[0])

def compute_sample_grads(model, x, outputs, y_a, mixup = False, y_b=None, lam=None):
    """ process each sample with per sample gradient """
    if mixup == False:
        y_b = [None]*x.shape[0]
        lam = [None]*x.shape[0]
    outputs_non_zero_grad_a = outputs[list(range(outputs.shape[0])), y_a].unsqueeze(1)
    grad_xF_a = torch.autograd.grad(outputs_non_zero_grad_a, x, 
                                  grad_outputs=torch.ones_like(outputs_non_zero_grad_a), 
                                  create_graph=True)
    outputs_non_zero_grad_b = outputs[list(range(outputs.shape[0])), y_b].unsqueeze(1)
    grad_xF_b = torch.autograd.grad(outputs_non_zero_grad_b, x, 
                                  grad_outputs=torch.ones_like(outputs_non_zero_grad_b), 
                                  create_graph=True)
    lam = lam.unsqueeze(1)
    return lam/outputs_non_zero_grad_a * (grad_xF_a[0]) + (1-lam)/outputs_non_zero_grad_b * (grad_xF_b[0])

def importance_weights(x, sample_grads, t, batch_size, n_t):
    """Calcualte importance weights by integrating the sample_grads according to girsanov"""
    if n_t > 1:
        x = x.reshape(batch_size, n_t, *x.shape[1:])
        sample_grads = sample_grads.reshape(batch_size, n_t, *sample_grads.shape[1:])
        dX = torch.diff(x, dim=1)
        dt = torch.diff(t, dim=1)
        weights = (dX.flatten(2)*sample_grads.flatten(2)[:,0:-1]).sum(-1) - \
            1/2 * torch.sum(sample_grads.flatten(2)[:,0:-1]*dt, dim=-1)
    else:
        weights = sample_grads.flatten(2).sum(-1) - \
            1/2 * torch.sum(sample_grads.flatten(2), dim=-1)
    return weights