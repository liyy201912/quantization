import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.parameter import Parameter


class Round(Function):
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


class ScaleSign(Function):
    """take a real value x, output sign(x)*E(|x|)"""

    @staticmethod
    def forward(ctx, input):
        return torch.sign(input) * torch.mean(torch.abs(input))

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class Quantize(Function):
    @staticmethod
    def forward(ctx, input, bit, scheme='fp'):
        # I. fix point:
        if scheme == 'fp':
            scale = float(2 ** bit - 1)
            out = torch.round(input * scale) / scale

        # II. power of 2:
        elif scheme == 'po2':
            out = 2 ** torch.round(torch.log2(input)) * (input > 2 ** (-2 ** bit + 1)).float()

        # III. sp2:
        elif scheme == 'sp2':
            size = input.size()
            y = input.reshape(-1)

            centroids = torch.tensor(
                [0, 2 ** -4, 2 ** -3, 2 ** -4 + 2 ** -3, 2 ** -2, 2 ** -2 + 2 ** -3, 2 ** -1, 2 ** -1 + 2 ** -3, 1]).cuda()
            mag = y - centroids.reshape(-1, 1)

            minimum = torch.min(torch.abs(mag), dim=0)[1]
            out = centroids[minimum]
            out = out.reshape(size)
        else:
            raise NotImplementedError
        return out

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


def DorefaW(w, bit):
    mix = True
    if bit == 1:
        w = ScaleSign.apply(w)
    elif bit == 2:
        pass
    else:
        sign = torch.sign(w)
        scale = torch.max(torch.abs(w))
        # w = torch.tanh(w)
        w = torch.abs(w) / scale
        # w = w / (2 * scale) + 0.5
        if mix:
            weight = w.detach()
            size = weight.size()
            w2d = weight.reshape(weight.size(0), -1)
            var = torch.var(w2d, dim=1)
            ######
            out, idx = torch.topk(var, ((var.size(0) * 1) // 10))
            mid = out[-1]
            # mid = torch.median(var)
            # mix scheme:
            greater = w2d * (var >= mid).float().reshape(-1, 1)
            lower = w2d * (var < mid).float().reshape(-1, 1)
            greater_quant = Quantize.apply(greater, 2*bit-1, 'fp')
            lower_quant = Quantize.apply(lower, bit-1, 'sp2')
            residual = (greater_quant + lower_quant - w2d).reshape(size)
            # finally, a tricky method
            w = w + residual
            # print('quantized?', w)
        else:
            w = Quantize.apply(w, bit - 1, 'fp')

        w = w * scale * sign
    return w


class AQuantization(Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input, a, scheme='linear', k=4):
        ctx.save_for_backward(input, a)
        # output = input.sign()
        if scheme == 'linear':
            output = 0.5 * (torch.abs(input) - torch.abs(input - a) + a)
            output = torch.round(output * float(2 ** k - 1) / a) * (a / float(2 ** k - 1))
        else:
            raise NotImplementedError
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        input, a = ctx.saved_tensors

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.clone()
            grad_input = grad_input * (input < a).float()
            grad_input = grad_input * (input > 0).float()
            # grad_input[input.ge(a)] = 0.0
            # grad_input[input.le(0)] = 0.0
        if ctx.needs_input_grad[1]:
            grad_a = grad_output.clone()
            grad_a = grad_a * (input > a).float()
            # grad_a[input.le(a)] = 0

        return grad_input, grad_a, None, None


def AQ(a, bit):
    if bit == 1:
        a = ScaleSign.apply(a)
    elif bit == 2:
        pass
    else:
        # sign = torch.sign(a)
        # scale = torch.max(torch.abs(a))
        # a = torch.abs(a) / scale
        # w = torch.tanh(w)
        # a = torch.abs(a) / scale
        # w = w / (2 * scale) + 0.5
        a = torch.clamp(a, min=0.0, max=1.0)
        a = Quantize.apply(a, bit, 'fp')
        # a = a *
    return a 


class QConv(nn.Conv2d):
    """docstring for QuanConv"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=False, nbit_w=4, nbit_a=4):
        super(QConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias)
        # self.a = Parameter(torch.tensor(3.0))  # for PACT
        # self.scheme = 'linear'
        self.nbit_w = nbit_w
        self.nbit_a = nbit_a
        # self.quan_w = DorefaW()
        # self.quan_a = AQ(self.scheme, self.nbit_a)

    # @weak_script_method
    def forward(self, input):
        if self.nbit_w < 32:
            w = DorefaW(self.weight, self.nbit_w)
        else:
            w = self.weight

        if self.nbit_a < 32:
            x = AQ(input, self.nbit_a)
        else:
            x = input

        # print('weight', w)
        # print('input', x)

        output = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return output


class QLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, nbit_w=4,
                 nbit_a=4):
        super(QLinear, self).__init__(in_features, out_features, bias)

        # self.a = Parameter(torch.tensor(3.0))  # for PACT
        # self.scheme = 'linear'
        self.nbit_w = nbit_w
        self.nbit_a = nbit_a
        # self.quan_w = DorefaW()
        # self.quan_a = AQ(self.scheme, self.nbit_a)

    # @weak_script_method
    def forward(self, input):
        if self.nbit_w < 32:
            w = DorefaW(self.weight, self.nbit_w)
        else:
            w = self.weight

        if self.nbit_a < 32:
            x = AQ(input, self.nbit_a)
        else:
            x = input

        output = F.linear(x, w, self.bias)
        return output


def test():
    ts = torch.randn(1, 3, 32, 32).cuda()
    net = QConv(3, 10, 3, stride=1, padding=1).cuda()
    ot = net(ts)


# test()
