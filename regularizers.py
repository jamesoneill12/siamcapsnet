import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class DropConnect(torch.nn.Module):
    def __init__(self, module, weights, dropout=0, variational=False):
        super(DropConnect, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.variational = variational
        self._setup()

    def blank(*args, **kwargs):
        # We need to replace flatten_parameters with a nothing function
        # It must be a function rather than a lambda as otherwise pickling explodes
        # We can't write boring code though, so ... blank!
        return

    def _setup(self):
        # Terrible temporary solution to an issue regarding compacting weights re: CUDNN RNN
        if issubclass(type(self.module), torch.nn.RNNBase):
            self.module.flatten_parameters = self.blank

        for name_w in self.weights:
            print('Applying weight drop of {} to {}'.format(self.dropout, name_w))
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(name_w + '_raw', nn.Parameter(w.data))

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            w = None
            if self.variational:
                mask = torch.autograd.Variable(torch.ones(raw_w.size(0), 1))
                if raw_w.is_cuda: mask = mask  # .cuda()
                mask = torch.nn.functional.dropout(mask, p=self.dropout, training=True)
                w = mask.expand_as(raw_w) * raw_w
            else:
                w = torch.nn.functional.dropout(raw_w, p=self.dropout, training=self.training)
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)


class VariationalDropout(nn.Module):
    def __init__(self, input_size, out_size, log_sigma2=-10, threshold=3):
        """
        :param input_size: An int of input size
        :param log_sigma2: Initial value of log sigma ^ 2.
               It is crusial for training since it determines initial value of alpha
        :param threshold: Value for thresholding of validation. If log_alpha > threshold, then weight is zeroed
        :param out_size: An int of output size
        """
        super(VariationalDropout, self).__init__()

        self.input_size = input_size
        self.out_size = out_size

        self.theta = Parameter(torch.FloatTensor(input_size, out_size))
        self.bias = Parameter(torch.Tensor(out_size))

        self.log_sigma2 = Parameter(torch.FloatTensor(input_size, out_size).fill_(log_sigma2))

        self.reset_parameters()

        self.k = [0.63576, 1.87320, 1.48695]

        self.threshold = threshold

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_size)

        self.theta.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    @staticmethod
    def clip(input, to=8):
        input = input.masked_fill(input < -to, -to)
        input = input.masked_fill(input > to, to)

        return input

    def kld(self, log_alpha):

        first_term = self.k[0] * F.sigmoid(self.k[1] + self.k[2] * log_alpha)
        second_term = 0.5 * torch.log(1 + t.exp(-log_alpha))

        return -(first_term - second_term - self.k[0]).sum() / (self.input_size * self.out_size)

    def forward(self, input):
        """
        :param input: An float tensor with shape of [batch_size, input_size]
        :return: An float tensor with shape of [batch_size, out_size] and negative layer-kld estimation
        """

        log_alpha = self.clip(self.log_sigma2 - torch.log(self.theta ** 2))
        kld = self.kld(log_alpha)

        if not self.training:
            mask = log_alpha > self.threshold
            return torch.addmm(self.bias, input, self.theta.masked_fill(mask, 0))

        mu = torch.mm(input, self.theta)
        std = torch.sqrt(torch.mm(input ** 2, self.log_sigma2.exp()) + 1e-6)

        eps = Variable(torch.randn(*mu.size()))
        if input.is_cuda:
            eps = eps.cuda()

        return std * eps + mu + self.bias, kld

    def max_alpha(self):
        log_alpha = self.log_sigma2 - self.theta ** 2
        return torch.max(log_alpha.exp())


# Gals implementation - https://github.com/yaringal/ConcreteDropout/blob/master/concrete-dropout-pytorch.ipynb
class Learn2Connect(nn.Module):

    def __init__(self, layer, input_shape, wr=1e-6, dr=1e-5, init_min=0.1, init_max=0.1):
        super(Learn2Connect, self).__init__()

        # Post dropout layer
        self.layer = layer
        # Input dim
        self.input_dim = np.prod(input_shape)
        # Regularisation hyper-params
        self.w_reg_param = wr
        self.d_reg_param = dr

        # Initialise p_logit
        init_min = np.log(init_min) - np.log(1. - init_min)
        init_max = np.log(init_max) - np.log(1. - init_max)

        self.p_logit = nn.Parameter(torch.Tensor(1))
        nn.init.uniform(self.p_logit, a=init_min, b=init_max)

    def sum_of_square(self):
        """
        For paramater regularisation
        """
        sum_of_square = 0
        for param in self.layer.parameters():
            sum_of_square += torch.sum(torch.pow(param, 2))
        return sum_of_square

    def regularisation(self):
        """
        Returns regularisation term, should be added to the loss
        """
        weights_regularizer = self.w_reg_param * self.sum_of_square() / (1 - self.p)
        dropout_regularizer = self.p * torch.log(self.p)
        dropout_regularizer += (1. - self.p) * torch.log(1. - self.p)
        dropout_regularizer *= self.d_reg_param * self.input_dim
        regularizer = weights_regularizer + dropout_regularizer
        return regularizer

    def forward(self, x):
        """
        Forward pass for dropout layer
        """
        eps = 1e-7
        temp = 0.1

        self.p = F.sigmoid(self.p_logit)

        unif_noise = Variable(torch.cuda.FloatTensor(np.random.uniform(size=tuple(x.size()))))

        drop_prob = (torch.log(self.p + eps)
                     - torch.log(1 - self.p + eps)
                     + torch.log(unif_noise + eps)
                     - torch.log(1 - unif_noise + eps))
        drop_prob = F.sigmoid(drop_prob / temp)
        random_tensor = 1 - drop_prob
        retain_prob = 1 - self.p

        x = torch.mul(x, random_tensor)
        x = torch.div(x, retain_prob)

        return self.layer(x)
