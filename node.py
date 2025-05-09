import math

class Node:
    def __init__(self, value, prev=(), _op='', _label=''):
        self.value = float(value)
        self.prev = prev
        self.grad = 0.0
        self._op = _op
        self._label = _label
        self._backward = lambda: None

    def backward(self):
        self.grad = 1.0
        
        self._backward()
        def dfs(root):
            for v in root.prev:
                v._backward()
                dfs(v) 
        dfs(self)  

    def help(self):
        def dfs(root):
            print(root)
            for v in root.prev:
                dfs(v) 
        dfs(self) 

    # Activation functions
    def relu(self):
        out = Node(max(0,self.value), (self, ), 'relu')

        def _backward():
            if self.value > 0:
                self.grad += 1.0 * out.grad
            else:
                self.grad = 0.0
        
        out._backward = _backward
        return out

    def sigmoid(self):
        x = self.value
        ret = 1 / (1 + math.exp(-x))
        out = Node(ret, (self, ), 'sigmoid')

        def _backward():
            self.grad += ret * (1 - ret) * out.grad
        
        out._backward = _backward
        return out

    def tanh(self):
        x = self.value
        ret = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Node(ret, (self, ), 'tanh')

        def _backward():
            self.grad += (1 - ret**2) * out.grad
        out._backward = _backward
        return out
        
    def __add__(self, node):
        out = Node(self.value + node.value, (self, node), '+')

        # out is captured by reference
        def _backward():
            self.grad += out.grad
            node.grad += out.grad

        # creates a linkage between the local input and output, triggers the gradient calculation from 'out' to input nodes
        out._backward = _backward 
        return out

    def __sub__(self, node):
        out = Node(self.value - node.value, (self, node), '-')

        # out is captured by reference
        def _backward():
            self.grad += 1.0 * out.grad
            node.grad += (-1.0) * out.grad

        # creates a linkage between the local input and output, triggers the gradient calculation from 'out' to input nodes
        out._backward = _backward 
        return out
        
    def __mul__(self, node):
        out = Node(self.value * node.value, (self, node), '*')

        def _backward():
            self.grad += node.value * out.grad
            node.grad += self.value * out.grad
                
        # creates a linkage between the local input and output, triggers the gradient calculation from 'out' to input nodes
        out._backward = _backward # fn
        return out
    
    def __pow__(self, p):
        out = Node(self.value ** p.value, (self, ), 'pow')

        def _backward():
            self.grad += p.value * (self.value ** (p.value - 1)) * out.grad
        
        out._backward = _backward
        return out
        
    def __repr__(self):
        return f"Node({self._op}: {self.value}, grad = {self.grad})"