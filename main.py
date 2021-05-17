from data.cifar import CIFAR10
from data.mnist import  MNIST 


train_data  = MNIST(
        root= './data',
        train = True,
        download= True,
        noise_type= 'symmetric'
        )

print(train_data)
