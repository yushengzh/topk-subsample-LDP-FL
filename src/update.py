import collections

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from optimizer import LdpSGD
import numpy as np
import privacy
from utils import setup_dim
import utils
class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.n_p = args.num_users * args.frac / args.np_rate
        self.em_s = args.num_users * args.beta
        self.dim_model, self.dim_x, self.dim_y = setup_dim(args.dataset, args.model)
        self.k = int((self.dim_model + self.dim_y) * args.beta)
        # print("Topk selecting or random sampling {} dimensions".format(self.k))
        self.choices = []

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.7*len(idxs))]
        idxs_val = idxs[int(0.7*len(idxs)):int(0.8*len(idxs))]
        idxs_test = idxs[int(0.8*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)), shuffle=False)
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd' or self.args.optimizer == 'dpsgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, weight_decay=1e-4, momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=1e-4)
        '''   
        elif self.args.optimizer == 'ldpsgd':
            optimizer = LdpSGD(model.parameters(), lr=self.args.lr, weight_decay=1e-4,
                               mechanism=self.args.mechanism, sensitivity=1, epsilon=self.args.epsilon)
        '''

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()
                '''
                if self.args.verbose and (batch_idx % 100 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                '''
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def ldp_update_weight(self, model, global_round):
        model.train()
        epoch_loss = []
        optimizer = LdpSGD(model.parameters(), lr=self.args.lr, weight_decay=1e-4,
                           mechanism=self.args.mechanism, sensitivity=1, epsilon=self.args.epsilon)
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):

                np.random.seed(2023)
                rng_state = np.random.get_state()
                np.random.shuffle(images)
                np.random.set_state(rng_state)
                np.random.shuffle(labels)

                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()
                '''
                if self.args.verbose and (batch_idx % 100 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))        
                '''

                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        # print(model.state_dict())
        params = model.state_dict().copy()
        for param, values in model.state_dict().items():
            new_values = values
            for i in range(len(values)):
                left = torch.min(values[i])
                left_min = float(torch.tensor(left, dtype=torch.float32))
                right = torch.max(values[i])
                right_max = float(torch.tensor(right, dtype=torch.float32))
                new_values[i] = privacy.randomizer(values[i], self.args.epsilon, self.args.delta,
                                                       self.args.norm_clip,
                                                       left_min, right_max, self.args.mechanism)
            params[param] = new_values
        return params, sum(epoch_loss) / len(epoch_loss)

    def sldp_update_weight(self, model, global_round):
        model.train()
        epoch_loss = []
        optimizer = LdpSGD(model.parameters(), lr=self.args.lr, weight_decay=1e-4,
                           mechanism=self.args.mechanism, sensitivity=1, epsilon=self.args.epsilon)
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                np.random.seed(2023)
                rng_state = np.random.get_state()
                np.random.shuffle(images)
                np.random.set_state(rng_state)
                np.random.shuffle(labels)

                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        # 参数处理
        params = model.state_dict().copy()
        tensor_list = []
        for param, values in model.state_dict().items():
            tensor_list.append(values)
        flattened = torch.cat([tensor.view(-1) for tensor in tensor_list])
        # subsample random sample k
        choices = utils.randomkIdx(flattened, self.k)
        new_flattened = privacy.sample_randomizer(flattened, choices, self.args.epsilon, self.args.delta,
                                                      self.args.norm_clip, 0, 1, self.args.mechanism)
        restored_tensors = []
        start_index = 0
        # flattened还原
        for tensor in tensor_list:
            tensor_size = tensor.size()
            numel = tensor.numel()
            restored_tensor = new_flattened[start_index:start_index + numel].view(tensor_size)
            restored_tensors.append(restored_tensor)
            start_index += numel
        i = 0
        for param, value in model.state_dict().items():
            params[param] = restored_tensors[i]
            # print(restored_tensors[i].size())
            # print(restored_tensors[i])
            i += 1

        return params, sum(epoch_loss) / len(epoch_loss)

    def kssldp_update_weight(self, model, global_round):
        model.train()
        epoch_loss = []
        optimizer = LdpSGD(model.parameters(), lr=self.args.lr, weight_decay=1e-4,
                           mechanism=self.args.mechanism, sensitivity=1, epsilon=self.args.epsilon)
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):

                np.random.seed(2023)
                rng_state = np.random.get_state()
                np.random.shuffle(images)
                np.random.set_state(rng_state)
                np.random.shuffle(labels)

                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        # 参数随机化（差分隐私化）处理
        # print(model.state_dict())
        params = model.state_dict().copy()
        tensor_list = []
        for param, values in model.state_dict().items():
            tensor_list.append(values)
        flattened = torch.cat([tensor.view(-1) for tensor in tensor_list])
        # subsample top-k
        self.choices = utils.topkIdx(flattened, self.k)
        # print("sampling top-k={} params of all {} model params".format(self.k, self.dim_model+self.dim_y))
        new_flattened = privacy.sample_randomizer(flattened, self.choices, self.args.epsilon, self.args.delta,
                                      self.args.norm_clip, 0, 1, self.args.mechanism)
        restored_tensors = []
        start_index = 0
        # flattened还原
        for tensor in tensor_list:
            tensor_size = tensor.size()
            numel = tensor.numel()
            restored_tensor = new_flattened[start_index:start_index + numel].view(tensor_size)
            restored_tensors.append(restored_tensor)
            start_index += numel
        i = 0
        for param, value in model.state_dict().items():
            params[param] = restored_tensors[i]
            # print(restored_tensors[i].size())
            # print(restored_tensors[i])
            i += 1

        return params, sum(epoch_loss) / len(epoch_loss)

    def get_choice(self):
        return self.choices

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss
