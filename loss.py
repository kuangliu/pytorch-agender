from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class AGLoss(nn.Module):
    def __init__(self):
        super(AGLoss, self).__init__()

    def forward(self, age_preds, age_targets, gender_preds, gender_targets):
        '''Compute loss between (age_preds, age_targets) and (gender_preds, gender_targets).

        Args:
          age_preds: (tensor) predicted ages, sized [batch_size,100].
          age_targets: (tensor) target ages, sized [batch_size,].
          gender_preds: (tensor) predicted gender, sized [batch_size,].
          gender_targets: (tensor) target gender, sized [batch_size,].

        loss:
          (tensor) loss = SmoothL1Loss(age_preds, age_targets)
                        + BCEWithLogitsLoss(gender_preds, gender_targets)
        '''
        age_prob = F.softmax(age_preds)
        age_expect = torch.sum(Variable(torch.arange(1,101)).cuda()*age_prob, 1)
        age_loss = F.smooth_l1_loss(age_expect, age_targets.float())
        gender_loss = F.binary_cross_entropy_with_logits(gender_preds, gender_targets)
        print('age_loss: %.3f | gender_loss: %.3f' % (age_loss.data[0], gender_loss.data[0]), end=' | ')
        return age_loss + gender_loss
