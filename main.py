from node import Node
import torch

h = 0.1 # sanity check: following the direction of grad to see if Loss increases

def cust():
    w1 = Node(1.2); w1._label = 'w1'
    w2 = Node(1.0); w2._label = 'w2'
    x1 = Node(-2.0); x1._label = 'x1'
    x2 = Node(2.1); x2._label = 'x2'

    e = w1*x1; e._label = 'e'
    f = w2*x2; f._label = 'f'

    g = e+f; g._label = 'g'
    Loss = g.sigmoid(); Loss._label = 'Loss'
    Loss.backward()
    print(w1,w2,x1,x2,e,f,g,Loss)

def trch():
    w1 = torch.Tensor([1.2]); w1.requires_grad = True
    w2 = torch.Tensor([1.0]); w2.requires_grad = True
    x1 = torch.Tensor([-2.0]); x1.requires_grad = True
    x2 = torch.Tensor([2.1]); x2.requires_grad = True

    e = w1*x1; e.retain_grad()
    f = w2*x2; f.retain_grad()
    g = e+f; g.retain_grad()
    Loss = torch.sigmoid(g); Loss.retain_grad()

    Loss.backward()

    print(w1.grad.item())
    print(w2.grad.item())
    print(x1.grad.item())
    print(x2.grad.item())
    print(e.grad.item())
    print(f.grad.item())
    print(g.grad.item())
    print(Loss.grad.item())

cust()
trch()
