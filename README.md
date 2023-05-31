# topk-subsample-LDP-FL

## Requirement
- Python
- PyTorch

## Dataset
- Mnist, Fashion Mnist and Cifar-10

## Model
- MLP and CNN

## Run
- NP-FL on Mnist MLP
  > python federated_learing.py --optimizer=sgd --model=MLP --dataset=MNIST --gpu=0 --iid=1 --epoch=20 

- LDP-FL on Cifar-10 CNN
  > python federated_learing.py --optimizer=sgd --model=CNN --dataset=CIFAR --gpu=0 --iid=1 --epoch=40

## Options

### Model Parameters

### Federated Parameters

### DP Parameters

# Result

