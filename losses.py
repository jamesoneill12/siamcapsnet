from torch import nn

import torch
import torch.nn.functional as F


class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(size_average=False)

    def forward(self, images, labels, classes, reconstructions):
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2

        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum()

        reconstruction_loss = self.reconstruction_loss(reconstructions, images)

        return (margin_loss + 0.0005 * reconstruction_loss) / images.size(0)


# classes = predictions
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    # was 2.0 before using the sigmoid
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()

        self.reconstruction_loss = nn.MSELoss(size_average=False)
        # self.sigmoid = nn.Sigmoid()
        self.margin = margin

    # just for reconstruction_loss
    def forward_once(self, images, labels, classes, reconstructions):
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2
        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum()
        reconstruction_loss = self.reconstruction_loss(reconstructions, images)
        return (margin_loss + 0.0005 * reconstruction_loss) / images.size(0)

    # data = [img0, img1]
    def forward(self, output1, output2, labels, data=None, reconstruction=None, distance=False):
        euclidean_distance = F.pairwise_distance(output1, output2)
        # euclidean_distance = self.sigmoid(euclidean_distance)
        loss_contrastive = torch.mean((1 - labels) * torch.pow(euclidean_distance, 2) +
                                      (labels) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        if reconstruction != None and data != None:
            loss_contrastive += self.forward_once(F.pairwise_distance(data[0], data[1]), labels, output1,
                                                  reconstruction)
            # loss_contrastive += self.forward_once(data[1], labels, output2, reconstruction[1])

        return loss_contrastive, euclidean_distance


class OnlineContrastiveLoss(nn.Module):
    """
    Online Contrastive loss
    Takes a batch of embeddings and corresponding labels.
    Pairs are generated using pair_selector object that take embeddings and targets and return indices of positive
    and negative pairs
    """

    def __init__(self, pair_selector, margin=0.5):
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.pair_selector = pair_selector

    def forward(self, embeddings, target):
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
        if embeddings.is_cuda:
            positive_pairs = positive_pairs.cuda()
            negative_pairs = negative_pairs.cuda()
        positive_loss = (embeddings[positive_pairs[:, 0]] - embeddings[positive_pairs[:, 1]]).pow(2).sum(1)
        negative_loss = F.relu(
            self.margin - (embeddings[negative_pairs[:, 0]] - embeddings[negative_pairs[:, 1]]).pow(2).sum(
                1).sqrt()).pow(2)
        loss = torch.cat([positive_loss, negative_loss], dim=0)
        return loss.mean()
