class Node:
    def __init__(self, value, _prev=(), _op='', _label=''):
        self.value = value
        self._prev = _prev
        self.grad = 0.0
        self._op = _op
        self._label = _label
        self._backward = lambda: None

    def backward(self):
        dag = []

        def dfs(root):
            dag.append(root)
            for v in root._prev:
                dfs(v)

        dfs(self)
        self.grad = 1.0
        for v in dag:
            v._backward()
        
        def print_grad():
            for v in dag:
                print(f"{v._label} {v.grad}")
        
        # print_grad()

    # Activation functions
    def sigmoid(self):
        pass
        
    def __add__(self, node):
        out = Node(self.value + node.value, (self, node), '+')

        def _backward():
            self.grad += out.grad
            node.grad += out.grad

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
        
    def __repr__(self):
        return f"Node(value={self.value}, grad={self.grad})"

