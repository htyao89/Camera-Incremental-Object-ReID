import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch

device=None
optim = None
lossfunc=None


def modeldict_weighted_average(ws, weights=[]):
    w_avg = {}
    for layer in ws[0].keys():
        w_avg[layer] = torch.zeros_like(ws[0][layer])
    if weights == []: weights = [1.0/len(ws) for i in range(len(ws))]
    for wid in range(len(ws)):
        for layer in w_avg.keys():
            w_avg[layer] = w_avg[layer] + ws[wid][layer] * weights[wid]
    return w_avg

def modeldict_zeroslike(w):
    res = {}
    for layer in w.keys():
        res[layer] = w[layer] - w[layer]
    return res

def modeldict_scale(w, c):
    res = {}
    for layer in w.keys():
        res[layer] = w[layer] * c
    return res

def modeldict_sub(w1, w2):
    res = {}
    for layer in w1.keys():
        res[layer] = w1[layer] - w2[layer]
    return res

def modeldict_norm(w, p=2):
    return torch.norm(modeldict_to_tensor1D(w), p)

def modeldict_to_tensor1D(w):
    res = torch.Tensor().cuda()
    for layer in w.keys():
        res = torch.cat((res, w[layer].view(-1)))
    return res

def modeldict_add(w1, w2):
    res = {}
    for layer in w1.keys():
        res[layer] = w1[layer] + w2[layer]
    return res

def modeldict_dot(w1, w2):
    res = 0
    for layer in w1.keys():
        s = 1
        for l in w1[layer].shape:
            s *= l
        res += (w1[layer].view(1, s).float().mm(w2[layer].view(1, s).float().T))
    return res.item()

def modeldict_print(w):
    for layer in w.keys():
        print("{}:{}".format(layer, w[layer]))

if __name__ == '__main__':
    w= {'a': torch.Tensor([[1, 4], [3, 4]]),  'c':torch.Tensor([1])}
    res=modeldict_norm(w)
    print(res**2)
