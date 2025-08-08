import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



def tt(num):
    return nn.Parameter(torch.randn(1, requires_grad=True))
    # return nn.Parameter(torch.tensor([float(num)], requires_grad=True))

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.g1 = tt(-3.79)
        self.g2 = tt(-3.32)
        self.g3 = tt(-2.26)
        self.g4 = tt(4.72)
        self.g5 = tt(1.57)
        self.b1 = tt(-5.24)
        self.b2 = tt(-0.07)


    def forward(self, x):
        x1= x[0]
        x2 = x[1]
        
        h1 = torch.sigmoid(4.9*(x1*self.g1 + x2*self.g2 + self.b1))

        o1 = torch.sigmoid(4.9*(x1*self.g3 + h1*self.g4 + x2*self.g5 + self.b2))

        return o1



net = Net()
print(net)

# optimizer = optim.SGD(net.parameters(), lr=0.01)
optimizer = optim.Adadelta(net.parameters())
# criterion = nn.MSELoss()
criterion = nn.BCELoss()





# 2-input XOR inputs and expected outputs.
xor_inputs = torch.tensor([(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)])
xor_outputs = torch.tensor([   (0.0,),     (1.0,),     (1.0,),     (0.0,)])

output = net(xor_inputs[0])
target = xor_outputs[0]


loss = criterion(output, target)

for gen in range(1000):
    
    for inX in range(4):
        optimizer.zero_grad()   # zero the gradient buffers

        output = net(xor_inputs[inX])
        target = xor_outputs[inX]
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    if gen%100 == 1:
        print(gen)
        for inX in range(4):

            output = net(xor_inputs[inX])
            target = xor_outputs[inX]

            print(output, target)


for p in net.parameters():
    print(p)
