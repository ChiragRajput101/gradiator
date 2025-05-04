from node import Node

h = 0.1 # sanity check: following the direction of grad to see if Loss increases
 
w1 = Node(3.0); w1._label = 'w1'
w2 = Node(4.0); w2._label = 'w2'
x1 = Node(-2.0); x1._label = 'x1'
x2 = Node(12.0); x2._label = 'x2'

e = w1*x1; e._label = 'e'
f = w2*x2; f._label = 'f'

Loss = e+f; Loss._label = 'Loss'
print(Loss)
Loss.backward()