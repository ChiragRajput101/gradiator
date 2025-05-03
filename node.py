class Node:
    def __init__(self, value, grad=1.0, _prev=(), _op='', _label=''):
        self.value = value
        self.grad = grad
        self._prev = set(_prev)
        self._op = _op
        self._label = _label

    def backward(self):
        def calcGrad(root, _grad=1.0):
            root.grad *= _grad
            print(f"{root._label}, grad:{root.grad}")
            for v in root._prev:
                calcGrad(v, root.grad)

        calcGrad(self)
        
    def __add__(self, node):
        self.grad = 1.0
        return Node(self.value + node.value, 1.0, (self, node), '+')
        
    def __mul__(self, node):
        self.grad = node.value
        node.grad = self.value
        return Node(self.value * node.value, 1.0, (self, node), '*')
        
    def __repr__(self):
        return f"Node(value={self.value}, grad={self.grad})"


h = 0 # sanity check: following the direction of grad to see if Loss increases
 
w1 = Node(3.0-h); w1._label = 'w1'
w2 = Node(4.0); w2._label = 'w2'
x1 = Node(-2.0); x1._label = 'x1'
x2 = Node(12.0); x2._label = 'x2'

e = w1*x1; e._label = 'e'
f = w2*x2; f._label = 'f'

Loss = e+f; Loss._label = 'Loss'
# print(Loss)
Loss.backward()

