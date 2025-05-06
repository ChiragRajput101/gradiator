from node import Node
import random

# initializes Node-based vector of dimensions = dims, random values in [0,1]
def init_weights(dims=()):
    assert len(dims) > 1, "can't initialize tensor with less than 2 dimensions"
    weight_tensor = []
    r = dims[0]; c = dims[1]
    for i in range(r):
        row_vec = [Node(random.random()) for _ in range(c)]
        weight_tensor.append(row_vec)
    return weight_tensor

# converts int/float based vectors to Node-based vectors
def tensorize(v):
    assert ( len(v) > 0 and isinstance(v[0], list) ), "invalid input, must have more than 1 dimension"
    for i in range(0, len(v)):
        for j in range(0, len(v[0])):
            v[i][j] = (Node(v[i][j]))

# calculates dot product of 2 Node-based vectors
def dot(v1, v2):
    assert (len(v1) == len(v2) and len(v1[0]) == len(v2[0])), "invalid, dims of both should be the same"
    ret = []
    for i in range(len(v1)):
        dp = [i * j for i,j in zip(v1[i],v2[i])]
        dummy = Node(0)
        for n in dp:
            dummy = dummy + n
        dp = dummy
        dp.op = 'dot_prod'
        ret.append(dp)
    return ret


h = 0.1 # sanity check: following the direction of grad to see if Loss increases

x = [[1.4, 2, 1.3, 0.4], [1.4, 2, 1.3, 0.4]]
tensorize(x)
# in = 4 nodes, hidden = 2 nodes, out = 1 node
w = init_weights((2,4))
'''
[ [w11 w21 w31 w41]    
  [w12 w22 w32 w42] ]  
'''

dp = [dot(x,w)]
w2 = init_weights((1,2))
a = dot(dp, w2)
z = a[0].sigmoid()
z.backward()

print('z: ', z)
print('a: ', a)
print('w2: ', w2)
print('dp: ', dp)
print('w: ', w)
