from torch.nn.modules.loss import BCELoss
from torch.nn.modules.loss import _Loss  # noqa

from . import functional as f


class DiceLoss2d(_Loss):

    def forward(self, output, target):
        return f.dice_loss_2d(output, target, reduction=self.reduction)


class IoULoss2d(_Loss):

    def forward(self, output, target):
        return f.iou_loss_2d(output, target, reduction=self.reduction)


class BCEDiceIoULoss2d(_Loss):  # Use with sigmoid

    def __init__(self, dice_factor=1., bce_factor=2., iou_factor=2., bce_weight=None, reduction='mean'):
        super().__init__(reduction=reduction)
        self.bce_loss = BCELoss(weight=bce_weight, reduction=reduction)
        self.bce_factor = bce_factor
        self.dice_loss = DiceLoss2d(reduction=reduction)
        self.dice_factor = dice_factor
        self.iou_loss = IoULoss2d(reduction=reduction)
        self.iou_factor = iou_factor

    def forward(self, probability, target):
        bce = self.bce_loss(probability, target) * self.bce_factor
        dice = self.dice_loss(probability, target) * self.dice_factor
        iou = self.iou_loss(probability, target) * self.iou_factor
        return bce + dice + iou


class BCEDiceIoUWithLogitsLoss2d(BCEDiceIoULoss2d):

    # fastai.metrics.dice uses argmax() which is not differentiable, so it
    # can NOT be used in training, however it can be used in prediction.
    # see https://github.com/fastai/fastai/blob/master/fastai/metrics.py#L53
    def forward(self, logit, target):
        bce_input = logit.softmax(dim=-3)
        if self.trainig:
            probability = bce_input
        else:
            probability = f.one_hot_2d(logit.argmax(dim=-3).long(), logit.size(dim=-3)).to(logit.dtype)
        bce = self.bce_loss(bce_input, target) * self.bce_factor
        dice = self.dice_loss(probability, target) * self.dice_factor
        iou = self.iou_loss(probability, target) * self.iou_factor
        return bce + dice + iou


del _Loss
