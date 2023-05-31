# topk-subsample-LDP-FL

## Requirement
- Python
- PyTorch
- torchvision

## Run
- NP-FL on Mnist MLP
  > python federated_learing.py --optimizer=sgd --model=MLP --dataset=MNIST --gpu=0 --iid=1 --epoch=20 

- LDP-FL on Cifar-10 CNN
  > python federated_learing.py --optimizer=sgd --model=CNN --dataset=CIFAR --gpu=0 --iid=1 --epoch=40

## Options

### Model Parameters
- `--model`: Default:mlp. Options:mlp,cnn,lm 
- `--dataset`: Default:mnist. Options:mnist,fmnist,cifar
- `--optimizer`: Default:sgd. Options:sgd,dpsgd,ldpsgd,kssldpsgd
- `--epochs`:global communication rounds. Default:20
- `--lr`:learning rate. Default:0.01
- `--seed`: Default:2023
- `--gpu`: Default: 0 
### Federated Parameters
- `--iid`:distribution of data. Default: 1
- `--num_users`:number of all users. Default:1000
- `--frac`:fractions of users tobe used. Default:0.1
- `--local_ep`:local epochs. Default:10
- `--local_bs`:local batch. Default:10

### DP Parameters
- `--mechanism`: Differential privacy mechanism. Default:laplace. Options:laplace, gaussian
- `--epsilon`: Privacy Budget. Default:1
- `--delta`: Privacy Parameter. Default:5e-6
- `--norm_clip`: Clip parameter. Default:0.15
### KSSLDP Parameters
- `--beta`: sample k/d. Default: 0.02
- `--np_rate`: n/np. Default: 1

# Result

