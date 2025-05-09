from network import Layer, Network

layers = [
    Layer(4,3),
    Layer(3,1)
]

x = [[0.2,0.5,1.2,1.3]]
y = [[3.25]]
NN = Network(x,y,iters=30)
model = NN.run(layers)