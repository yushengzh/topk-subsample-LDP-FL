
import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid
import privacy
import numpy as np

def l2norm_clipped(updates: torch.Tensor, threshold: float) -> torch.Tensor:
    l2norm = torch.linalg.norm(updates, ord=None)
    if l2norm > threshold:
        updates = updates * (threshold * 1.0 / l2norm)
    return updates


def l1norm_clipped(updates: torch.Tensor, clip_c: float) -> torch.Tensor:
    return torch.clip(updates, -clip_c, clip_c)


def transforming(vector, left, right, new_left, new_right):
    return new_left + (new_right - new_left) * (vector - left) / (right - left)


def transforming_v2(vector, clip_c):
    return (vector + clip_c) / (2 * clip_c)


def transform_back(vector, clip_c):
    return clip_c * (2 * vector - 1)


def setup_dim(dataset_name, model_name):
    # return dim_model, dim_x, dim_y
    if model_name == 'mlp':
        if dataset_name == 'mnist' or 'fmnist':
            return 784 * 64 + 64 * 10 + 64, 784, 10
        elif dataset_name == 'cifar':
            return 3072 * 64 + 64 * 10 + 64, 3072, 10
        else:
            raise "not found the dataset."
    if model_name == 'lm':
        if dataset_name == 'mnist' or 'fmnist':
            return 784 * 10, 784, 10
        elif dataset_name == 'cifar':
            return 3072 * 10, 3072, 10
        else:
            raise "not found the dataset."
    if model_name == 'cnn':
        if dataset_name == 'mnist':
            return 21840 - 10, 784, 10
        elif dataset_name == 'fmnist':
            return 29808 - 10, 784, 10
        elif dataset_name == 'cifar':
            return 61706 - 10, 3072, 10
        else:
            raise "not found the dataset."
    else:
        raise "not found the model."


def get_dataset(args):
    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=apply_transform)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=apply_transform)
        test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups


def average_weights(w):
    # print(type(w))
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def random_average_weights(w, epsilon, delta, clip_c, mechanism):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        if "weight" in key:
            # print(key)
            left = torch.min(w_avg[key])
            left_min = float(torch.tensor(left, dtype=torch.float32))
            right = torch.max(w_avg[key])
            right_max = float(torch.tensor(right, dtype=torch.float32))
            w_avg[key] = privacy.randomizer(w_avg[key], epsilon, delta, clip_c, left_min, right_max, mechanism)
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def dummy_align_average_weights(w, args, n_s, choice_list):
    w_avg = copy.deepcopy(w[0])
    dim_model, dim_x, dim_y = setup_dim(args.dataset, args.model)
    tensor_list = []
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] = w_avg[key] + w[i][key]
        tensor_list.append(torch.tensor(w_avg[key]))

    flattened = torch.cat([tensor.view(-1) for tensor in tensor_list])
    print("len of params:{}".format(flattened.size()[0]))
    n_p = int(args.num_users * args.frac / args.np_rate)  # padding size
    n_n = (torch.ones(n_s.size()[0]) * n_p).to('cuda') - n_s
    print("padding size n_n={}".format(n_n))
    # print("max_n_s={},max_n_n={}".format(max(n_s), max(n_n)))
    # print("min_n_s={},min_n_n={}".format(min(n_s), min(n_n)))
    en_s = args.num_users * args.frac * args.beta
    dummies = torch.zeros(n_n.size()[0])  # 每个维度的填充部分
    dum_flattened = []
    for idx, val in enumerate(n_n):
        dummies[idx] = sum(np.random.laplace(loc=0, scale=2 * args.norm_clip / args.epsilon, size=int(val)))
        noise_v = flattened[idx] + dummies[idx]
        dum_flattened.append(noise_v)

    # print(max(flattened), min(flattened))
    # print(max(dummies), min(dummies))
    # print(max(dum_flattened), min(dum_flattened))

    restored_tensors = []
    start_index = 0
    resl = torch.tensor(dum_flattened, dtype=float)

    # flattened还原
    for tensor in tensor_list:
        tensor_size = tensor.size()
        numel = tensor.numel()
        restored_tensor = resl[start_index:start_index + numel].view(tensor_size)
        restored_tensors.append(restored_tensor)
        start_index += numel
    i = 0
    for key in w_avg.keys():
        w_avg[key] = torch.div(restored_tensors[i], len(w))
        i += 1

    choice_list = torch.empty(choice_list.size())
    return w_avg


def randomkIdx(updates:torch.Tensor, samplek:int):
    dim = len(updates)
    return np.random.choice(dim, samplek)


def topkIdx(updates:torch.Tensor, topk:int):
    dim = len(updates)
    return torch.argsort(torch.abs(updates))[dim-topk:]


def sparsify(updates, topk):
    dim = len(updates)
    non_top_idx = torch.argsort(torch.abs(updates))[:dim-topk]
    updates[non_top_idx] = 0
    return updates


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning Rate : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')

    if 'dp' in args.optimizer:
        print('    Dp Parameters:')
        print(f'    Mechanism          : {args.mechanism}')
        print(f'    Epsilon(Budget)    : {args.epsilon}')
        print(f'    Delta              : {args.delta}')
        if 'k' in args.optimizer:
            print(f'    Beta               : {args.beta}')
            print(f'    n/np               : {args.np_rate}\n')
        else: print('\n')
    return
