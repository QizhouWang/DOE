import torch
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
EPS = 1E-20


def diff_in_weights(model, proxy):
    diff_dict = OrderedDict()
    model_state_dict = model.state_dict()
    proxy_state_dict = proxy.state_dict()
    for (old_k, old_w), (new_k, new_w) in zip(model_state_dict.items(), proxy_state_dict.items()):
        if len(old_w.size()) <= 1:
            continue
        if 'weight' in old_k:
            diff_w = new_w - old_w
            diff_dict[old_k] = old_w.norm() / (diff_w.norm() + EPS) * diff_w
    return diff_dict


def add_into_weights(model, diff, coeff=1.0):
    names_in_diff = diff.keys()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in names_in_diff:
                param.add_(coeff * diff[name])


def average_diff(cur_diff, new_diff, beta):
    for key in cur_diff.keys():
        cur_diff[key] = beta * cur_diff[key] + (1 - beta) * new_diff[key]
    return cur_diff

class AdvWeightPerturb(object):
    def __init__(self, model, proxy, proxy_optim, gamma):
        super(AdvWeightPerturb, self).__init__()
        self.model = model
        self.proxy = proxy
        self.proxy_optim = proxy_optim
        self.gamma = gamma

    '''def calc_awp(self, inputs_adv, cluster_head, steps = 1, gamma = None):
        def target_distribution(batch: torch.Tensor) -> torch.Tensor:
            weight = (batch ** 2) / torch.sum(batch, 0)
            return (weight.t() / torch.sum(weight, 1)).t()
        
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()
        
        if gamma is not None: 
            self.gamma = gamma
        for _ in range(steps):
            emb = self.proxy.intermediate_forward(inputs_adv, 0)
            # emb = F.avg_pool2d(emb, 8).view(-1,self.proxy.nChannels)
            # x = cluster_head(emb.view(emb.size(0), -1))
            x = cluster_head(emb)
            loss = nn.KLDivLoss(reduction='none')(x.log(), target_distribution(x).detach()).sum(-1).mean()
            self.proxy_optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.proxy.parameters(), 1)
            self.proxy_optim.step()

        # the adversary weight perturb
        diff = diff_in_weights(self.model, self.proxy)
        return diff'''

    def calc_awp_simple(self, inputs_adv, cluster_head, steps = 1, gamma = None):
        def target_distribution(batch: torch.Tensor) -> torch.Tensor:
            weight = (batch ** 2) / torch.sum(batch, 0)
            return (weight.t() / torch.sum(weight, 1)).t()
        
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()
        
        if gamma is not None: 
            self.gamma = gamma
        for _ in range(steps):
            emb = self.proxy.intermediate_forward_simple(inputs_adv, 0)
            # emb = F.avg_pool2d(emb, 8).view(-1,self.proxy.nChannels)
            # x = cluster_head(emb.view(emb.size(0), -1))
            x = cluster_head(emb)
            loss = nn.KLDivLoss(reduction='none')(x.log(), target_distribution(x).detach()).sum(-1).mean()
            self.proxy_optim.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.proxy.parameters(), 1)
            self.proxy_optim.step()

        # the adversary weight perturb
        diff = diff_in_weights(self.model, self.proxy)
        return diff

    def calc_awp(self, inputs_adv, steps = 1, gamma = None):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()
        
        if gamma is not None: 
            self.gamma = gamma
        for _ in range(steps):
            x = self.proxy(inputs_adv)
            l_oe = -(x.mean(1) - torch.logsumexp(x, dim=1)).mean() # / x.size(1)
            # l_oe = -(x.mean(1) - x.exp().sum(1)).mean()
            # l_oe = x.exp().mean(1).mean()
            # torch.logsumexp(x, dim=1)).mean() # / x.size(1)
            loss = l_oe
            self.proxy_optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.proxy.parameters(), 1)
            self.proxy_optim.step()

        # the adversary weight perturb
        diff = diff_in_weights(self.model, self.proxy)
        return diff

    def calc_awp_reg(self, inputs_adv, steps = 1, gamma = None):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()
        
        if gamma is not None: 
            self.gamma = gamma
        for _ in range(steps):
            scale = torch.Tensor([1]).cuda().requires_grad_()
            x = self.proxy(inputs_adv) * scale
            l_oe = -(x.mean(1) - torch.logsumexp(x, dim=1)).mean() / x.size(1)
            grads = torch.autograd.grad(l_oe, [scale], create_graph = True)[0]
            r_mr = torch.sum(grads ** 2)
            loss = l_oe - r_mr
            self.proxy_optim.zero_grad()
            loss.backward()
            self.proxy_optim.step()

        # the adversary weight perturb
        diff = diff_in_weights(self.model, self.proxy)
        return diff

    def calc_awp_smooth(self, inputs, targets, loss_fn, steps = 1):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()
        
        for _ in range(steps):
            outputs = self.proxy(inputs)
            loss = loss_fn(outputs, targets)
            self.proxy_optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.proxy.parameters(), 1)
            self.proxy_optim.step()

        # the adversary weight perturb
        diff = diff_in_weights(self.model, self.proxy)
        return diff

    def perturb(self, diff):
        add_into_weights(self.model, diff, coeff= - 1.0 * self.gamma)

    def restore(self, diff):
        add_into_weights(self.model, diff, coeff= 1.0 * self.gamma)


