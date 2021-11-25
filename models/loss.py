from torch.nn.modules.loss import _Loss, _WeightedLoss  # noqa
from torch.nn import BCELoss
from torch.nn import functional as F  # noqa
from . import functional as f  # use as small case to be separated with torch.nn.functional


class DiceLoss2d(_Loss):

    def forward(self, output, target):
        return f.dice_loss_nd(output, target, nd=2, reduction=self.reduction)


class IoULoss2d(_Loss):

    def forward(self, output, target):
        return f.iou_loss_nd(output, target, nd=2, reduction=self.reduction)


class BCEDiceIoULoss2d(_WeightedLoss):  # Use with sigmoid

    def __init__(self, dice_factor=1., bce_factor=2., iou_factor=2., bce_weight=None, reduction='mean'):
        super().__init__(weight=bce_weight, reduction=reduction)
        self.bce_factor = bce_factor
        self.dice_factor = dice_factor
        self.iou_factor = iou_factor

    def forward(self, probability, target):
        mul, add = probability * target, probability + target
        bce = F.binary_cross_entropy(probability, target, weight=self.weight, reduction=self.reduction) * \
            self.bce_factor
        dice = f._dice_loss(mul, add, nd=2, reduction=self.reduction) * self.dice_factor  # noqa
        iou = f._iou_loss(mul, add, nd=2, reduction=self.reduction) * self.iou_factor  # noqa
        return (bce + iou + dice) / (self.bce_factor + self.dice_factor + self.iou_factor)


class BCEDiceIoUWithLogitsLoss2d(BCEDiceIoULoss2d):

    # fastai.metrics.dice uses argmax() which is not differentiable, so it
    # can NOT be used in training, however it can be used in prediction.
    # see https://github.com/fastai/fastai/blob/master/fastai/metrics.py#L53
    def forward(self, logit, target):
        bce_input = logit.softmax(dim=-3)
        bce = F.binary_cross_entropy(bce_input, target, weight=self.weight, reduction=self.reduction) * \
            self.bce_factor
        if self.training:
            probability = bce_input
        else:
            probability = f.one_hot_nd(logit.argmax(dim=-3).long(), logit.size(dim=-3), nd=2).to(logit.dtype)
        mul, add = probability * target, probability + target
        dice = f._dice_loss(mul, add, nd=2, reduction=self.reduction) * self.dice_factor  # noqa
        iou = f._iou_loss(mul, add, nd=2, reduction=self.reduction) * self.iou_factor  # noqa
        return (bce + iou + dice) / (self.bce_factor + self.dice_factor + self.iou_factor)


del _Loss, _WeightedLoss

