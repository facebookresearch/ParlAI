import torch
import itertools
from torch import nn
from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn.utils.rnn import PackedSequence
import torch.nn._functions.rnn as rnn
import math

# Used RNNBase and RNNCell as reference.
class VCRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=False, nonlinearity="tanh", initial_lambda = 0.1,
                 epsilon = 1e-5):
        super(VCRNN, self).__init__()
        assert num_layers == 1, 'VCRNN\'s behavior is only defined with single hidden layer'

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.nonlinearity = nonlinearity
        self.initial_lambda = self.lamda = initial_lambda
        self.epsilon = epsilon

        if nonlinearity == "tanh":
            self.base_unit = rnn.RNNTanhCell
        elif nonlinearity == "relu":
            self.base_unit = rnn.RNNReLUCell
        else:
            raise RuntimeError("Unknown nonlinearity: {}".format(nonlinearity))

        self.register_buffer('input_indices', torch.arange(1, input_size + 1))
        self.register_buffer('hidden_indices', torch.arange(1, hidden_size + 1))

        layer_weights = []

        for l in range(num_layers):
            layer_input_size = input_size if l == 0 else hidden_size
            m_model = nn.Sequential(
                nn.Linear(layer_input_size + hidden_size, 1),
                nn.Sigmoid(),
            )

            w_ih = Parameter(torch.Tensor(hidden_size, layer_input_size))
            w_hh = Parameter(torch.Tensor(hidden_size, hidden_size))
            layer_params = (m_model, w_ih, w_hh)
            param_names = ['m_model_l{}', 'weight_ih_l{}', 'weight_hh_l{}']
            if self.bias:
                b_ih = Parameter(torch.Tensor(hidden_size))
                b_hh = Parameter(torch.Tensor(hidden_size))
                layer_params += (b_ih, b_hh)
                param_names += ['bias_ih_l{}', 'bias_hh_l{}']
            else:
                self.register_parameter('bias_ih_l{}'.format(l), None)
                self.register_parameter('bias_hh_l{}'.format(l), None)

            for name, param in zip(param_names, layer_params):
                setattr(self, name.format(l), param)

            layer_weights.append(layer_params)

        self.layer_weights = layer_weights

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weights in self.layer_weights:
            for weight in weights[1:]:
                if weight is not None:
                    weight.data.uniform_(-stdv, stdv)
            for weight in weights[0].parameters():
                weight.data.zero_()

    def compute_softmask(self, m, size, indices):
        soft_mask = torch.sigmoid(self.lamda * (m * size - indices))
        soft_mask_clone = soft_mask.clone()
        soft_mask_clone[soft_mask > 1 - self.epsilon] = 1
        soft_mask_clone[soft_mask < self.epsilon] = 0
        return soft_mask_clone

    def cell_forward(self, input, hidden_ms, input_indices, hidden_indices, l):
        m_model, *layer_weight = self.layer_weights[l]
        h_prev, ms = hidden_ms
        m = m_model(torch.cat((input, h_prev), 1))
        ms.append(m)
        input_mask = self.compute_softmask(m, self.input_size, input_indices)
        if self.hidden_size == self.input_size:
            hidden_mask = input_mask
        else:
            hidden_mask = compute_softmask(m, self.hidden_size, hidden_indices)

        masked_input = input * input_mask
        masked_h_prev = h_prev * hidden_mask
        h = self.base_unit(masked_input, masked_h_prev, *layer_weight)
        h = hidden_mask * h + (h_prev - masked_h_prev)

        return h, ms

    # Inspired by torch.nn._functions.rnn.AutogradRNN and
    #             torch.nn._functions.rnn.StackedRNN
    # Only cares about and returns m values in training mode.
    def forward(self, input, hx=None):
        is_packed = isinstance(input, PackedSequence)
        if is_packed:
            # input: (\sum_{t in seq} #batch[t]) * #input
            input, batch_sizes = input
            max_batch_size = batch_sizes[0]
            net = rnn.variable_recurrent_factory(batch_sizes)(self.cell_forward)
        else:
            # transpose input to #seq first dim
            if self.batch_first:
                input = input.transpose(0, 1)
            # input: #seq * (\max_t #batch[t]) * #input
            batch_sizes = None
            max_batch_size = input.size(1)
            net = rnn.Recurrent(self.cell_forward)

        if self.training and type(hx) is tuple:
            prev_ms = hx[1]
            # transpose previous ms to #batch first dim
            if not self.batch_first:
                prev_ms = prev_ms.transpose(0, 1)
        else:
            prev_ms = None

        if hx is None:
            hx = Variable(input.data.new(self.num_layers, max_batch_size, self.hidden_size).zero_(), requires_grad=False)
        if type(hx) is not tuple:
            ms = None
        else:
            hx, ms = hx

        indices = Variable(self.input_indices), Variable(self.hidden_indices)

        next_hidden = []
        layer_ms = []

        for l in range(self.num_layers):

            (hidden, m), output = net(input, (hx[l], []), indices + (l,))
            next_hidden.append(hidden)
            layer_ms.append(torch.cat(m, 1).transpose(0, 1))

            input = output

        next_hidden = torch.stack(next_hidden, 0)
        layer_ms = torch.stack(layer_ms)

        if self.training:
            if ms is not None:
                ms = torch.cat((ms, layer_ms), 1)
            else:
                ms = layer_ms

            if self.batch_first:
                ms = ms.transpose(1, 2)
        # transpose back if necessary
        if self.batch_first:
            output = output.transpose(0, 1)

        if is_packed:
            output = PackedSequence(output, batch_sizes)

        if self.training:
            return output, (next_hidden, ms)
        else:
            return output, next_hidden

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size}, {num_layers}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'batch_first' in self.__dict__ and self.batch_first is not False:
            s += ', batch_first={batch_first}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != "tanh":
            s += ', nonlinearity={nonlinearity}'
        if 'initial_lambda' in self.__dict__ and self.initial_lambda != "0.1":
            s += ', initial_lambda={initial_lambda}'
        if 'lamda' in self.__dict__:
            s += ', lambda={lamda}'
        if 'epsilon' in self.__dict__ and self.epsilon != 0.05:
            s += ', epsilon={epsilon}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


