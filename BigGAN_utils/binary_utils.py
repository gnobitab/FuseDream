from torch.autograd import Function
from torch.optim import SGD


class BinaryActivation(Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x.sign() + 1.) / 2.

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()
