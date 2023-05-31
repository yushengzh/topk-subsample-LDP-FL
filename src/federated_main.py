import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, SingleFc
from utils import get_dataset, average_weights, exp_details, random_average_weights
import utils
import matplotlib
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
matplotlib.use('Agg')


if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    dim_model, dim_x, dim_y = utils.setup_dim(args.dataset, args.model)
    k = int((dim_model + dim_y) * args.beta)
    n_p = args.num_users * args.frac / args.np_rate

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural network
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # MLP
        img_size, len_in = train_dataset[0][0].shape, 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    elif args.model == 'lm':
        img_size, len_in = train_dataset[0][0].shape, 1
        for x in img_size:
            len_in *= x
            global_model = SingleFc(dim_in=len_in, dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    torch.nn.DataParallel(global_model, device_ids=[0]).cuda()
    global_model.train()
    print(global_model)
    print("model parameters num={}.".format(dim_model + dim_y))
    if args.optimizer == 'kssldpsgd':
        print("--------KssLdp-SGD-----------")
        print("sample dim k={}.".format(k))
        print("align padding size np={}.".format(n_p))
        print("-------------------------------")
    # copy weights
    global_weights = global_model.state_dict()

    # testing
    test_loss, test_accuracy = [], []

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 1
    val_loss_pre, counter = 0, 0

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        choice_list = []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)

        # sample users
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        lot_size = len(idxs_users)
        # print("lot_size:{}".format(lot_size))
        # print("users are:{}".format(idxs_users))
        choice_list = []
        n_s = []
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            if args.optimizer == 'ldpsgd':
                update_weights = local_model.ldp_update_weight
            elif args.optimizer == 'sldpsgd':
                update_weights = local_model.sldp_update_weight
            elif args.optimizer == 'kssldpsgd':
                update_weights = local_model.kssldp_update_weight
            else: update_weights = local_model.update_weights


            w, loss = update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            choice_list.append(local_model.get_choice())


        # n_s = torch.bincount(choice_list, minlength=args)
        # update global weights
        if args.optimizer == 'sgd':
            global_weights = utils.average_weights(local_weights)
        elif args.optimizer == 'dpsgd':
            # cdp randomize
            global_weights = utils.random_average_weights(local_weights, args.epsilon, args.delta, args.norm_clip,
                                                              args.mechanism)
        elif args.optimizer == 'ldpsgd':
            global_weights = utils.average_weights(local_weights)
        elif args.optimizer == 'kssldpsgd' or 'sldpsgd':
            choice_list = torch.cat([torch.tensor(choice).view(-1) for choice in choice_list])
            print("{} users topk list:{}".format(int(args.num_users * args.frac), choice_list))
            n_s = torch.bincount(choice_list, minlength=(dim_model + dim_y))
            print("n_s:{}".format(n_s))
            global_weights = utils.dummy_align_average_weights(local_weights, args, n_s, choice_list)
            choice_list = torch.empty(choice_list.size())
        else:
            global_weights = utils.average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()

        # loop clients
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))
        test_acc, test_loss_avg = test_inference(args, global_model, test_dataset)
        test_accuracy.append(test_acc)
        test_loss.append(test_loss_avg)

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100 * train_accuracy[-1]))
            print('Test Accuracy: {:.2f}% \n'.format(100 * test_accuracy[-1]))

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
    print('Train Loss:')
    print(train_loss)
    print('Test Loss:')
    print(test_loss)
    print('Train Accuracy:')
    print(train_accuracy)
    print('Test Accuracy:')
    print(test_accuracy)
    # Saving the objects train_loss and train_accuracy:
    file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # PLOTTING (optional)


    # Plot Loss curve
    plt.figure()
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['SimSun']
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.title('Training Loss')
    plt.plot(range(len(train_loss)), train_loss, color='r', linewidth=2)
    plt.ylabel('Training loss', fontsize=20)
    plt.xlabel('global epochs', fontsize=20)
    plt.savefig('../save/fed_{}_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_{}_{}_{}_{}_loss.pdf'.
                format(args.optimizer, args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs, args.epsilon, args.delta, args.mechanism,
                       args.norm_clip))

    # # Plot Average Accuracy

    plt.figure()
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['SimSun']
    plt.rcParams['mathtext.fontset'] = 'stix'
    if args.optimizer == 'sgd': plt.title('分类准确率', fontsize=20)
    else: plt.title('$\epsilon$ = {}'.format(args.epsilon), fontsize=20)
    '''
    if args.mechanism == 'laplace':
        plt.title('{}::ε-{} {}模型的分类准确率; ε={}'
                  .format(args.dataset.upper(), args.optimizer.upper(), args.model.upper(), args.epsilon))
    elif args.mechanism == 'gaussian':
        plt.title('{}: Gaussian机制:(ε,δ)-{} {}模型的分类准确率; ε={},δ={}'
                  .format(args.dataset.upper(), args.optimizer.upper(), args.model.upper(), args.epsilon, args.delta))
    '''
    plt.plot(train_accuracy, marker="o",  linewidth=2, label='训练准确率')
    plt.plot(test_accuracy, marker="*",  linewidth=2, label='测试准确率')
    plt.grid()
    plt.legend(loc='best', fontsize=20, framealpha=0.5)
    plt.xticks(np.arange(0, args.epochs+1, args.epochs / (args.epochs/4)))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.ylabel('准确率', fontsize=20)
    plt.xlabel('通信轮数', fontsize=20)
    # plt.show()
    plt.savefig('../result/fed_{}_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_eps[{}]_del[{}]_mech[{}]_clip[{}]_acc.pdf'.
                 format(args.optimizer, args.dataset, args.model, args.epochs, args.frac,
                        args.iid, args.local_ep, args.local_bs,
                        args.epsilon, args.delta, args.mechanism, args.norm_clip))
