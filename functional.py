import random
from node import Node

class Functional:
   def __init__(self):
       pass

   # initializes Node-based vector of dimensions = dims, random values in [0,1]
   def init_weights(self, dims=()):
       assert len(dims) > 1, "can't initialize tensor with less than 2 dimensions"
       weight_tensor = []
       r = dims[0]; c = dims[1]
       for i in range(r):
           row_vec = [Node(random.random()) for _ in range(c)]
           weight_tensor.append(row_vec)
       return weight_tensor

   # converts int/float based vectors to Node-based vectors
   def tensorize(self, v):
       assert ( len(v) > 0 and isinstance(v[0], list) ), "invalid input, must have more than 1 dimension"
       for i in range(0, len(v)):
           for j in range(0, len(v[0])):
               v[i][j] = (Node(v[i][j]))


   def dot(self, v1, v2):
       assert len(v1) == len(v2), "dimension mismatch in dot product"
       result = v1[0] * v2[0]
       for i in range(1, len(v1)):
           result = result + (v1[i] * v2[i])
       return result

   def dot_prod(self, t1, t2):
       dp = []   
       if len(t1) == 1:
           # broadcast
           for i in range(len(t2)):
               dp.append(self.dot(t1[0],t2[i]))
       else:
           # matrix dot product 
           assert len(t1) == len(t2), "invalid input to dot_prod(), shape of both matrices don't match"
           for i in range(len(t1)):
               dp.append(self.dot(t1[i],t2[i]))
       return dp