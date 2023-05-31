import torch
from torch.optim import SGD, Adam
from typing import List, Optional
import privacy


class LdpSGD(SGD):

    def __init__(self, params, lr=0.1, momentum=0, dampening=0, weight_decay=0, nesterov=False,
                 mechanism='laplace', sensitivity=1, epsilon=1, delta=10e-5, threshold=1):

        self.mechanism = mechanism
        self.sensitivity = sensitivity
        self.delta = delta
        self.epsilon = epsilon
        self.threshold = threshold
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(LdpSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def dp_sgd(self, params: List[torch.Tensor], d_p_list: List[torch.Tensor],
            momentum_buffer_list: List[Optional[torch.Tensor]], *,
            weight_decay: float, momentum: float, lr: float, dampening: float, nesterov: bool):

        for i, param in enumerate(params):
            d_p = d_p_list[i]
            if weight_decay != 0:
                d_p = d_p.add(param, alpha=weight_decay)

            if momentum != 0:
                buf = momentum_buffer_list[i]

                if buf is None:
                    buf = torch.clone(d_p).detach()
                    momentum_buffer_list[i] = buf
                else:
                    buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

                if nesterov:
                    d_p = d_p.add(buf, alpha=momentum)
                else:
                    d_p = buf
            # randomize
            '''
            left = torch.min(d_p)
            left_min = float(torch.tensor(left, dtype=torch.float32))
            right = torch.max(d_p)
            right_max = float(torch.tensor(right, dtype=torch.float32))
            d_p = privacy.randomizer(d_p, self.epsilon, self.delta, self.threshold, left_min, right_max, self.mechanism)
            print(d_p.size)
            '''
            # gradient update
            param.add_(d_p, alpha=-lr)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']

            # print(group['params'])

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    '''
                    left = torch.min(p.grad)
                    left_min = float(torch.tensor(left, dtype=torch.float32))
                    right = torch.max(p.grad)
                    right_max = float(torch.tensor(right, dtype=torch.float32))
                    p.grad = privacy.randomizer(p.grad,
                                                self.epsilon, self.delta, self.threshold,
                                                left_min, right_max, self.mechanism)
                    # p.grad = randomize(p.grad, self.sensitivity, self.epsilon)
                    '''
                    d_p_list.append(p.grad)
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            self.dp_sgd(params_with_grad,
                  d_p_list,
                  momentum_buffer_list,
                  weight_decay=weight_decay,
                  momentum=momentum,
                  lr=lr,
                  dampening=dampening,
                  nesterov=nesterov)

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss




class LDPAdam(Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    left = torch.min(p.grad)
                    left_min = float(torch.tensor(left, dtype=torch.float32))
                    right = torch.max(p.grad)
                    right_max = float(torch.tensor(right, dtype=torch.float32))
                    p.grad = privacy.randomizer(p.grad,
                                                self.epsilon, self.delta, self.threshold,
                                                left_min, right_max, self.mechanism)
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            F.adam(params_with_grad,
                   grads,
                   exp_avgs,
                   exp_avg_sqs,
                   max_exp_avg_sqs,
                   state_steps,
                   amsgrad=group['amsgrad'],
                   beta1=beta1,
                   beta2=beta2,
                   lr=group['lr'],
                   weight_decay=group['weight_decay'],
                   eps=group['eps'])
        return loss