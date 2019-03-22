"""
Dynamic Routing Between Capsules
https://arxiv.org/abs/1710.09829

PyTorch implementation by Kenta Iwasaki @ Gram.AI.
"""

import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.autograd.variable import Variable

from regularizers import *

BATCH_SIZE = 100
NUM_CLASSES = 5
NUM_EPOCHS = 500
NUM_ROUTING_ITERATIONS = 3


class Linear_relu(nn.Module):

    def __init__(self, inp, out):
        super(Linear_relu, self).__init__()
        self.model = nn.Sequential(nn.Linear(inp, out), nn.ReLU())

    def forward(self, x):
        return self.model(x)


class SigDrop(nn.Module):

    def __init__(self, inp, args):
        super(SigDrop, self).__init__()

        self.model = nn.Linear(inp, args.nhidden_caps)
        self.prior_probs = torch.rand(args.nhidden_caps, 1)
        self.drop = nn.Dropout()

    def forward(self, x, prob):
        x = self.drop(x, self.prior_probs)

        return self.model(x)


class SiameseNetwork(nn.Module):
    def __init__(self, args):
        super(SiameseNetwork, self).__init__()

        dims = [4, 8, 100, 500,
                args.nhidden_caps]  # if args.dataset == 'att' else [16, 32, 32, 73728, args.nhidden_caps]

        channels = 3 if args.dataset == 'wild' else 1

        self.concrete_dropout = args.concrete_dropout
        self.norm_dist = args.norm_dist

        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, dims[0], kernel_size=3),
            nn.Dropout2d(p=0.2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(dims[0]),

            nn.ReflectionPad2d(1),
            nn.Conv2d(dims[0], dims[1], kernel_size=3),
            nn.Dropout2d(p=0.2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(dims[1]),

            nn.ReflectionPad2d(1),
            nn.Conv2d(dims[1], dims[1], kernel_size=3),
            nn.Dropout2d(p=0.2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(dims[1]),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(dims[1] * dims[2] * dims[2], 500),
            nn.Dropout(p=0.2),
            nn.ReLU(inplace=True),
            nn.Linear(500, 500),
            nn.Dropout(p=0.2),
            nn.ReLU(inplace=True),
            nn.Linear(500, dims[4]))

        if args.concrete_dropout:
            l = 1e-4  # Lengthscale
            wr = l * 2. / 10000
            dr = 2. / 10000
            D = 1  # one mean, one logvar
            self.cd = Learn2Connect(Linear_relu(20, 1), input_shape=(args.train_batch_size, args.nhidden_caps), wr=wr,
                                    dr=dr)
        elif args.sig_dropout:
            l = 1e-4  # Lengthscale
            wr = l * 2. / 10000
            dr = 2. / 10000
            self.cd = SigmoidDrop(input_shape=(args.train_batch_size, args.nhidden_caps))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)

        if self.concrete_dropout:
            output = self.cd(output)
        if self.norm_dist:
            # cannot use output/= output.pow(...) in python 2.7 not supported
            output = torch.div(output, output.pow(2).sum(1, keepdim=True).sqrt())

        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


class SiameseAlexNet(nn.Module):

    def __init__(self, args):
        super(SiameseAlexNet, self).__init__()

        # should change based on dataset, last is num. of classes
        dims = [16, 32, 32, 73728,
                args.nhidden_caps]  # if args.dataset == 'att' else [16, 32, 32, 73728, args.nhidden_caps]

        channels = 3 if args.dataset == 'wild' else 1

        self.norm_dist = args.norm_dist
        self.concrete_dropout = args.concrete_dropout

        self.cnn1 = nn.Sequential(
            nn.Conv2d(channels, dims[0], kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Conv2d(dims[0], dims[1], kernel_size=3, padding=1),
            nn.Dropout2d(p=0.2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(dims[2], dims[2], kernel_size=3, padding=1),
            nn.Dropout2d(p=0.2),
            nn.ReLU(inplace=True)
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(dims[3], 500),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(500, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, dims[4]),
        )

        if args.concrete_dropout:
            l = 1e-4  # Lengthscale
            wr = l * 2. / 10000
            dr = 2. / 10000
            D = 1  # one mean, one logvar
            self.cd = Learn2Connect(Linear_relu(20, 1), input_shape=(args.train_batch_size, args.nhidden_caps), wr=wr,
                                    dr=dr)
        elif args.sig_dropout:
            l = 1e-4  # Lengthscale
            wr = l * 2. / 10000
            dr = 2. / 10000
            self.cd = SigmoidDrop(input_shape=(args.train_batch_size, args.nhidden_caps))

    def forward_once(self, x):
        x = self.cnn1(x)
        x = x.view(x.size(0), -1)
        # print(x.cpu().size())
        x = self.fc1(x)

        if self.norm_dist:
            x = torch.div(x, x.pow(2).sum(1, keepdim=True).sqrt())

        if self.concrete_dropout:
            x = self.cd(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)))
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)


def augmentation(x, max_shift=2):
    _, _, height, width = x.size()

    h_shift, w_shift = np.random.randint(-max_shift, max_shift + 1, size=2)
    source_height_slice = slice(max(0, h_shift), h_shift + height)
    source_width_slice = slice(max(0, w_shift), w_shift + width)
    target_height_slice = slice(max(0, -h_shift), -h_shift + height)
    target_width_slice = slice(max(0, -w_shift), -w_shift + width)

    shifted_image = torch.zeros(*x.size())
    shifted_image[:, :, source_height_slice, source_width_slice] = x[:, :, target_height_slice, target_width_slice]
    return shifted_image.float()


class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None,
                 num_iterations=NUM_ROUTING_ITERATIONS, cuda=False, squash=False):
        super(CapsuleLayer, self).__init__()

        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations
        self.cuda = cuda
        self.num_capsules = num_capsules
        self.squasher = squash

        if squash == False:
            self.logit = nn.Sigmoid()

        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))
        else:
            self.capsules = nn.ModuleList(
                [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0) for _ in
                 range(num_capsules)])

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        if self.num_route_nodes != -1:
            priors = torch.matmul(x[None, :, :, None, :], self.route_weights[:, None, :, :, :])
            logits = Variable(torch.zeros(*priors.size()))

            if self.cuda:
                logits = logits.cuda()

            for i in range(self.num_iterations):
                probs = softmax(logits, dim=2)

                if self.squasher:
                    outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))
                else:
                    outputs = self.logit((probs * priors).sum(dim=2, keepdim=True))

                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
        else:
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            outputs = torch.cat(outputs, dim=-1)
            outputs = self.squash(outputs)

        return outputs


class CapsuleNetwork(nn.Module):
    def __init__(self, args):
        super(CapsuleNetwork, self).__init__()

        self.nhidden_caps = args.nhidden_caps
        self.recon = args.recon
        self.linear = args.linear
        self.num_convs = args.num_convs
        self.batch_norm = args.batch_norm
        self.args = args.squash

        self.norm_dist = args.norm_dist
        self.concrete_dropout = args.concrete_dropout

        if args.dataset == 'att':
            dims = [1, 9, 12, 16]
            capsule_route = 12
        # if wild is transformed to 50 x 50
        elif args.dataset == 'wild':
            dims = [3, 9, 12, 16]
            capsule_route = 12
            # dims = [3, 6, 12, 16]
            # capsule_route = 5
        # youtube face datasets choose
        elif args.dataset == 'ytf':
            dims = [3, 6, 12, 16]
        else:
            dims = [1, 6]

        if args.batch_norm:
            if self.num_convs == 1:

                self.conv1 = nn.Sequential(
                    # nn.ReflectionPad2d(1),
                    nn.Conv2d(dims[0], 256, kernel_size=9, stride=3),
                    nn.Dropout2d(p=0.2),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(256)
                )

            elif self.num_convs == 2:
                self.conv1 = nn.Sequential(
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(dims[0], 16, kernel_size=3),
                    nn.Dropout2d(p=0.2),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(16),

                    nn.ReflectionPad2d(1),
                    nn.Conv2d(16, 32, kernel_size=3),
                    nn.Dropout2d(p=0.2),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(32),

                    nn.ReflectionPad2d(1),
                    nn.Conv2d(32, 32, kernel_size=3),
                    nn.Dropout2d(p=0.2),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(32),
                )
                self.conv2 = nn.Conv2d(in_channels=32,
                                       out_channels=32, kernel_size=2, stride=2)

        else:
            if self.num_convs > 1:
                self.conv1 = nn.Conv2d(in_channels=dims[0], out_channels=256, kernel_size=3, stride=3)
                self.conv2 = nn.Conv2d(in_channels=dims[0], out_channels=256, kernel_size=9, stride=3)
            else:
                self.conv1 = nn.Conv2d(in_channels=dims[0], out_channels=256, kernel_size=9, stride=3)

        if args.batch_norm and self.num_convs > 1:
            capsule_route = 21
            self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=32, out_channels=16,
                                                 kernel_size=dims[1], stride=2, cuda=True, squash=args.squash)
            if args.intermediate_convs:
                self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2)
            self.face_capsules = CapsuleLayer(num_capsules=args.nhidden_caps,
                                              num_route_nodes=16 * capsule_route * capsule_route, in_channels=8,
                                              out_channels=dims[3], cuda=True, squash=args.squash)

        else:
            self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=256, out_channels=32,
                                                 kernel_size=dims[1], stride=2, cuda=True, squash=args.squash)
            self.face_capsules = CapsuleLayer(num_capsules=args.nhidden_caps,
                                              num_route_nodes=32 * capsule_route * capsule_route, in_channels=8,
                                              out_channels=dims[3], cuda=True, squash=args.squash)

        # should connect face_capsules -> fc1
        self.fc1 = nn.Linear(dims[3] * args.nhidden_caps, args.nhidden_caps) if self.linear else None
        # self.fc1 = nn.Sequential(
        #    nn.Linear(dims[3]* args.nhidden_caps, 500),
        #    nn.ReLU(inplace=True),
        #    nn.Linear(500, 500),
        #    nn.ReLU(inplace=True),
        #    nn.Linear(500, self.nhidden_caps)
        #    )

        if self.recon:
            self.decoder = nn.Sequential(
                nn.Linear(dims[3] * args.nhidden_caps, 128),
                nn.Dropout(p=0.5),
                nn.ReLU(inplace=True),
                nn.Linear(128, 256),
                nn.Dropout(p=0.5),
                nn.ReLU(inplace=True),
                nn.Linear(256, 512),
                nn.Dropout(p=0.5),
                nn.ReLU(inplace=True),
                nn.Linear(512, 2048),
                nn.Dropout(p=0.5),
                nn.ReLU(inplace=True),
                nn.Linear(2048, 10304),
                nn.Sigmoid()
            )

        if args.concrete_dropout:
            l = 1e-4  # Lengthscale
            wr = l * 2. / 10000
            dr = 2. / 10000
            D = 1  # one mean, one logvar
            self.cd = Learn2Connect(Linear_relu(20, 1), input_shape=(args.train_batch_size, args.nhidden_caps), wr=wr,
                                    dr=dr)
        elif args.sig_dropout:
            l = 1e-4  # Lengthscale
            wr = l * 2. / 10000
            dr = 2. / 10000
            self.cd = SigmoidDrop(input_shape=(args.train_batch_size, args.nhidden_caps))

    def reconstruct(self, x1, x2):
        recon_1 = self.decoder((x1 * y[:, :, None]).view(x.size(0), -1))
        recon_2 = self.decoder((x2 * y[:, :, None]).view(x.size(0), -1))
        reconstructions = F.pairwise_distance(recon_1, recon_2)
        return (reconstructions)

    def forward_once(self, x):

        # print()
        # print(x.cpu().size())
        x = F.relu(self.conv1(x), inplace=True)

        # print()
        # print(x1.cpu().size())

        if self.num_convs > 1 and self.batch_norm:
            x = self.conv2(x)
            # print(x1.cpu().size())

        print("Conv Layer {}".format(x.cpu().size()))

        x = self.primary_capsules(x)

        # print()
        # print(x.cpu().size())
        print("Primary Capsule output {}".format(x.cpu().size()))
        x = self.face_capsules(x).squeeze().transpose(0, 1)
        print("Face capsule input {}".format(x.cpu().size()))
        print(x.cpu().size())

        # 16x20x7056x16 when sigmoid squasher
        # original squasher
        # print(x1.cpu().size())
        # print(x2.cpu().size())

        if self.fc1:
            classes = self.fc1(x.contiguous().view(-1, x.contiguous().size(1) * x.size(2)))
        if self.recon:
            classes = (x ** 2).sum(dim=-1) ** 0.5
        if self.norm_dist:
            classes = torch.div(classes, classes.pow(2).sum(1, keepdim=True).sqrt())
        if self.concrete_dropout:
            classes = self.cd(classes)

            # print(classes1.cpu().size())
        # print(classes2.cpu().size())
        # print()
        # print()
        # classes = F.softmax(classes1)
        # print("Class size ")
        # print(classes.size())

        return classes

    def forward(self, x1, x2, y=None):

        classes1 = self.forward_once(x1)
        classes2 = self.forward_once(x2)

        if y is None:
            # In all batches, get the most active capsule.
            _, max_length_indices = classes1.max(dim=1)

            if self.cuda:
                y = Variable(torch.sparse.torch.eye(self.nhidden_caps)).cuda().index_select(dim=0,
                                                                                            index=max_length_indices.data)
            else:
                y = Variable(torch.sparse.torch.eye(self.nhidden_caps))
                # print("Y1 size ")
                # print(y.size())
                y = y.index_select(dim=0, index=max_length_indices)  # apparently should be this .data)
                # print("Y2 size ")
                # print(y.size())

        reconstructions = self.reconstruct(x1, x2) if self.recon else None
        return classes1, classes2, reconstructions

    def _name(self):
        return "Capsule"


# to be used with CNNs or the outputs of the final capsule for <x1, x2> 
def HausforffDistance(x1, x2):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.directed_hausdorff.html
    from scipy.spatial.distance import directed_hausdorff
    if type(x1) == Variable:
        distance = directed_hausdorff(x1.cpu().data.numpy(), x2.cpu().data.numpy())
    elif type(x1) == torch.cuda.FloatTensor:
        distance = directed_hausdorff(x1.cpu().numpy(), x2.cpu().numpy())


########## INCEPTION MODEL WITH ONLY 1 CHANNEL ###############
model_urls = {
    # Inception v3 ported from TensorFlow
    'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
}


def inception_v3(pretrained=False, **kwargs):
    r"""Inception v3 model architecture from
    `"Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        if 'transform_input' not in kwargs:
            kwargs['transform_input'] = True
        model = Inception3(**kwargs)
        model.load_state_dict(model_zoo.load_url(model_urls['inception_v3_google']))
        return model

    return Inception3(**kwargs)


class Inception3(nn.Module):

    def __init__(self, args, aux_logits=False, transform_input=False):
        super(Inception3, self).__init__()
        self.aux_logits = aux_logits
        self.transform_input = transform_input

        num_classes = args.nhidden_caps
        self.norm_dist = args.norm_dist
        num_channels = 3 if args.dataset == 'wild' else 1
        self.num_channels = num_channels

        self.Conv2d_1a_3x3 = BasicConv2d(num_channels, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)

        if num_channels == 3:
            if aux_logits:
                self.AuxLogits = InceptionAux(768, num_classes)
            self.Mixed_7a = InceptionD(768)
            self.Mixed_7b = InceptionE(1280)
            self.Mixed_7c = InceptionE(2048)
            self.fc = nn.Linear(2048, num_classes)
        else:
            if aux_logits:
                self.AuxLogits = InceptionAux(768, num_classes)
                # moved outside now used for 3 channels also.
        self.pool = torch.nn.AvgPool2d(2, stride=0)
        self.fc = nn.Linear(3072, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.Tensor(X.rvs(m.weight.data.numel()))
                values = values.view(m.weight.data.size())
                m.weight.data.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        if self.transform_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768

        """
        if self.num_channels == 3:
            if self.training and self.aux_logits:
                aux = self.AuxLogits(x)
            # 17 x 17 x 768
            x = self.Mixed_7a(x)
            # 8 x 8 x 1280
            x = self.Mixed_7b(x)
            # 8 x 8 x 2048
            x = self.Mixed_7c(x)
            # 8 x 8 x 2048
            
            print(x.cpu().size())
            
            x = F.avg_pool2d(x, kernel_size=8)
            # 1 x 1 x 2048
            x = F.dropout(x, training=self.training)
            # 1 x 1 x 2048
            x = x.view(x.size(0), -1)
            # 2048
            x = self.fc(x)
            # 1000 (num_classes)
        else:
        """
        if self.training and self.aux_logits:
            aux = self.AuxLogits(x)
        x = self.pool(x)
        x = self.fc(x.view(x.size(0), -1))

        if self.norm_dist:
            x /= x.pow(2).sum(1, keepdim=True).sqrt()

        if self.training and self.aux_logits:
            return x, aux

        return x


class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):

    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):

    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):

    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv1 = BasicConv2d(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        # 17 x 17 x 768
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # 5 x 5 x 768
        x = self.conv0(x)
        # 5 x 5 x 128
        x = self.conv1(x)
        # 1 x 1 x 768
        x = x.view(x.size(0), -1)
        # 768
        x = self.fc(x)
        # 1000
        return x


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


# -----------------------------------------------------------------------------------------------------------------------------------------        
# ---------------------------------------------------------- RESNET -----------------------------------------------------------------------

# __all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
#           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, args):

        self.norm_dist = args.norm_dist
        self.sig_dropout = args.sig_dropout
        self.concrete_dropout = args.concrete_dropout
        num_classes = args.nhidden_caps
        channels = 3 if args.dataset == 'wild' else 1
        self.channels = channels

        self.inplanes = 64

        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if self.channels == 3:
            # was (7,1) when using 50x50 images
            self.avgpool = nn.AvgPool2d(4, stride=1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        else:
            self.avgpool = nn.AvgPool2d(4, stride=1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)

        self.norm_dist = args.norm_dist
        self.concrete_dropout = args.concrete_dropout or args.norm_dist

        if args.concrete_dropout:
            l = 1e-4  # Lengthscale
            wr = l * 2. / 10000
            dr = 2. / 10000
            D = 1  # one mean, one logvar
            self.cd = Learn2Connect(Linear_relu(20, 1), input_shape=(args.train_batch_size, args.nhidden_caps), wr=wr,
                                    dr=dr)
        elif args.sig_dropout:
            l = 1e-4  # Lengthscale
            wr = l * 2. / 10000
            dr = 2. / 10000
            self.cd = SigmoidDrop(input_shape=(args.train_batch_size, args.nhidden_caps))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        if self.norm_dist:
            x = torch.div(x, x.pow(2).sum(1, keepdim=True).sqrt())

        if self.concrete_dropout or self.sig_dropout:
            x = self.cd(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(args, pretrained=False):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], args)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
