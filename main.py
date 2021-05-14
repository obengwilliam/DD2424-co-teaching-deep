from data.cifar import CIFAR10

train_data  = CIFAR10(
        root= './data',
        train = True,
        download= True,
        noise_type= 'symmetric'
        )

for i,target,ind in train_data:
    print(i, target, ind)
