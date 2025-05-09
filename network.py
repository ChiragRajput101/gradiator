from node import Node
from functional import Functional as F

class Layer:
   def __init__(self, inp, out=1):
       self.inp = inp
       self.out = out
       self.F = F()
   def get_in_dims(self):
       return self.inp
   def get_out_dims(self):
       return self.out
   def get_weights(self):
       w = self.F.init_weights((self.out, self.inp))
       return w

# network class - abstracts the way to create a NN
class Network:
   def __init__(self, x, y, lr=0.001, iters=50):
       self.x = x
       self.y = y
       self.model_layers = []
       self.iters = iters
       self.lr = lr
       self.model = []
       self.F = F()

       self.F.tensorize(self.x)
       self.F.tensorize(self.y)

   def get_model(self, layers=[]):
       if self.model_layers:  # only initialize once
           return
       for layer in layers:
           self.model_layers.append(layer.get_weights())

   def gradient_descent(self):
        lrN = Node(self.lr)
        # update all weights in self.model_layers (all are weights)
        new_weights = []
        for weight_tensors in self.model_layers:
            new_weight_tensor = []
            for i in range(len(weight_tensors)):
                v=[]
                for j in range(len(weight_tensors[i])):
                    weight_tensors[i][j] = weight_tensors[i][j] - lrN * Node(weight_tensors[i][j].grad)
                    v.append(Node(weight_tensors[i][j].value))
                new_weight_tensor.append(v)
                
            new_weights.append(new_weight_tensor)

        self.model_layers = new_weights
  
   def train(self):
        x_in = self.x
        out = None
        # print(self.model_layers)
        for model_layer in self.model_layers:
            z = self.F.dot_prod(x_in, model_layer)
            a = [[zz.relu() for zz in z]]
            self.model.append(z)
            self.model.append(a)
            x_in = a
            out = a


        loss = ((out[0][0] - self.y[0][0]) ** Node(2))
        loss._op='loss'
        loss.backward()
        self.gradient_descent()
        return loss
      
   def run(self, layers=[]):
        self.get_model(layers)
        for i in range(self.iters):
            loss = self.train()
            if i%2 == 0:
                print(f"iteration {i}:, {loss}")
        return self.model
