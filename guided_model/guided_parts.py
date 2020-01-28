import torch
from torch.autograd import Function
import numpy as np


class GuidedBackpropReLU(Function):
    @staticmethod
    def forward(ctx, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(
            torch.zeros(input.size()).type_as(input), input, positive_mask
        )
        ctx.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, output = ctx.saved_variables
        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(
            torch.zeros(input.size()).type_as(input),
            torch.addcmul(
                torch.zeros(input.size()).type_as(input), grad_output, positive_mask_1
            ),
            positive_mask_2,
        )
        return grad_input


class GuidedBackpropReLUSave(Function):
    @staticmethod
    def forward(ctx, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(
            torch.zeros(input.size()).type_as(input), input, positive_mask
        )
        ctx.save_for_backward(input, output)

        flag = np.load("test.npy")

        if flag > 0:
            mask = torch.load("test.pt")
            count = np.load("count.npy")
            output = torch.addcmul(
                torch.zeros(input.size()).type_as(input), output, mask[count]
            )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, output = ctx.saved_variables
        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(
            torch.zeros(input.size()).type_as(input),
            torch.addcmul(
                torch.zeros(input.size()).type_as(input), grad_output, positive_mask_1
            ),
            positive_mask_2,
        )
        # masks = torch.load("test.pt")
        # index = np.load("test.npy")
        # index += 1
        # # np.save("test.npy", index)
        # # if masks.nelement() == 0:
        # #     masks = grad_input.unsqueeze(0)
        # # else:
        # #     masks = torch.cat([grad_input.unsqueeze(0), masks])
        # # torch.save(masks, "test.pt")
        return grad_input


def guide_relu(self, input):
    output = GuidedBackpropReLU.apply(input)
    return output


def guide_relu_save(self, input):
    output = GuidedBackpropReLUSave.apply(input)
    return output
