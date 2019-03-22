import torchvision.datasets as dset
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils
from torch.autograd import Variable
from torch.utils.data import DataLoader

flatten = lambda l: [item for sublist in l for item in sublist]


def get_model(args):
    if args.model == 'standard':
        model = SiameseNetwork(args)
    if args.model == 'capsule':
        model = CapsuleNetwork(args)
    elif args.model == 'resnet':
        model = resnet34(args, pretrained=False)
        # norm_dist=args.norm_dist, concrete_dropout=args.concrete_dropout,
        #                    sigdrop = args.sig_dropout, num_classes= args.nhidden_caps,
        #                        channels=num_chans) #SiameseResNet(dataset= dset)
    elif args.model == 'alexnet':
        model = SiameseAlexNet(args)
    elif args.model == 'inception':
        model = Inception3(args)

    if str(args.criterion) == "contrastive":
        loss = ContrastiveLoss(margin=args.margin)
    elif str(args.criterion) == "ce":
        loss = torch.nn.BCEWithLogitsLoss()
    return (model, loss)


def get_dataset(args, dataset='att'):
    if dataset == 'att':
        train_dataloader, test_dataloader, data_iter = get_att(args)
    elif dataset == 'wild':
        train_dataloader, test_dataloader, data_iter = get_funneled_wild_new(args)

    return (train_dataloader, test_dataloader, data_iter)


def model_iter(model, args, img0, img1):
    if args.model == 'capsule':
        output1, output2, reconstructions = model(img0, img1)
        if args.recon != True:
            reconstruction = None
    elif args.model == 'inception' or args.model == 'resnet':
        output1 = model(img0)
        output2 = model(img1)
        reconstruction = None
    else:
        output1, output2 = model(img0, img1)
        reconstruction = None

    return (output1, output2, reconstruction)


def get_encoding_measure(h1, h2, measure):
    if measure == 'euclidean':
        dist = F.pairwise_distance(h1, h2)
    elif measure == 'manhattan':
        dist = torch.sum(torch.abs(h1 - h2))
        # below although wrong works surprisingly well
        # dist = torch.abs(torch.sum(h1,1)-torch.sum(h2,1))
    elif measure == 'cosine':
        dist = F.cosine_similarity(h1, h2, eps=1e-8)
    return (dist)


def get_loss(args, model, criterion, output1, output2, label, img=None, reconstruction=None, distance=False):
    # output1, output2, label, data=None recoconstruction1=None, recoconstruction2=None
    if args.model == 'capsule':
        # labels = torch.sparse.torch.eye(args.nclasses).index_select(dim=0, index=label)
        if args.criterion == 'contrastive':
            loss, dist = criterion(output1, output2, label, img, recoconstruction,
                                   distance=distance) if args.recon else criterion(output1, output2, label,
                                                                                   distance=distance)
        else:
            dist = get_encoding_measure(output1, output2, args.encoding_distance)
            if dist.dim() != label.dim() and args.criterion == 'ce':
                dist = dist.unsqueeze(1)

            loss = criterion(dist, label)
    else:
        if args.criterion == 'contrastive':
            loss, dist = criterion(output1, output2, label, distance=distance)
        else:
            dist = get_encoding_measure(output1, output2, args.encoding_distance)

            if dist.dim() != label.dim() and args.criterion == 'ce':
                dist = dist.unsqueeze(1)
            loss = criterion(dist, label)

    dist = dist if distance else None
    return (loss, dist)


def get_save_name(args):
    batch_norm = "batch_norm_" if args.batch_norm else ""
    intermediate_convs = "intermediate_conv_" if args.intermediate_convs else ""
    hsize = str(args.nhidden_caps) + "hidden_"
    loss_name = str(args.criterion) + "_"
    num_epochs = str(args.train_number_epochs) + "epochs_"
    dist_measure = str(args.encoding_distance) + "_"
    squash = "sigmoid_squash_" if args.squash else ""
    concrete_dropout = 'concrete_dropout_' if args.concrete_dropout else ""
    name = args.dataset + '_' + batch_norm + hsize + loss_name + dist_measure + concrete_dropout + intermediate_convs + num_epochs + squash + args.model
    return (name)


def train(args):
    results = {'epochs': [], 'loss': [], 'accuracies': [], 'accuracy_conf': [],
               'distance': [], 'std': [], 'dist_std': [], 'y': [], 'y_hat': []}

    train_dataloader, test_dataloader, dataiter = get_dataset(args, args.dataset)
    model, criterion = get_model(args)
    save_name = get_save_name(args)

    if args.gpu:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    counter = []
    loss_history = []
    iteration_number = 0
    t = torch.Tensor([args.margin])  # threshold
    best_loss = 100.0

    save_per_iter = 50 if args.train_batch_size < 20 else 10

    for epoch in range(0, args.train_number_epochs):
        for i, data in enumerate(train_dataloader, 0):

            img0, img1, label = data
            img0, img1, label = Variable(img0), Variable(img1), Variable(label)

            if args.gpu:
                img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()

            output1, output2, reconstruction = model_iter(model, args, img0, img1)

            optimizer.zero_grad()

            loss, _ = get_loss(args, model, criterion, output1, output2, label)

            loss.backward()
            optimizer.step()

            if i % save_per_iter == 0:
                print("Epoch number {}\n Current loss {}\n".format(epoch, loss.data[0]))

                iteration_number += 10
                counter.append(iteration_number)
                loss_history.append(loss.data[0])
                losses = [];
                dist = [];
                acc = [];
                acc_conf = [];
                total = 0;
                correct = 0;

                for i, data in enumerate(test_dataloader, 0):
                    x0, x1, label = data  # next(dataiter)
                    # label = label.type(torch.LongTensor) if args.criterion!= 'contrastive' else label

                    concatenated = torch.cat((x0.cpu(), x1.cpu()), 0)
                    x0, x1, label = Variable(x0), Variable(x1), Variable(label)
                    if args.gpu:
                        x0, x1, label = x0.cuda(), x1.cuda(), label.cuda()

                    output1, output2, reconstruction = model_iter(model, args, x0, x1)

                    loss, euclidean_distance = get_loss(args, model, criterion, output1, output2, label, distance=True)

                    losses.append(float(loss.data[0]))

                    if args.accuracy:
                        predicted = euclidean_distance.data.cpu()  # .numpy()
                        prediction = (predicted > args.margin) * 1
                        total += label.size(0)
                        accur = sum(prediction.numpy() == label.data.cpu().numpy())
                        correct += accur
                        batch_acc = (100 * float(correct) / float(total))
                        acc.append(batch_acc)
                        acc_conf.append(accur)
                        results['y_hat'].append(predicted.numpy())
                        results['y'].append(label.data.cpu().numpy())
                        dist.append(predicted.numpy())

                av_loss = sum(losses) / float(len(losses))
                av_dist = sum(dist) / float(len(dist))
                # loss_std = np.std(np.array(losses))
                loss_std = np.std(losses)
                dist_std = np.std(dist)

                results['epochs'].append(epoch)
                results['loss'].append(av_loss)
                results['distance'].append(dist)
                results['std'].append(loss_std)
                results['dist_std'].append(dist_std)
                results['accuracies'].append(acc)
                results['accuracy_conf'].append(acc_conf)

                if av_loss < best_loss:
                    best_loss = av_loss
                    best_dist = av_dist
                    if args.save_model and args.model != "inception":
                        if args.checkpoint:
                            save_checkpoint({
                                'epoch': args.train_number_epochs + 1,
                                'arch': args.model,
                                'state_dict': model.state_dict(),
                                'best_loss': best_loss,
                                'optimizer': optimizer.state_dict(),
                            }, save_name)
                        else:
                            save_model(save_name, model)

            del img0, img1, label

    show_plot(counter, loss_history)
    save_obj(results, save_name)

    return model, best_loss


def test(net, args):
    folder_dataset_test = dset.ImageFolder(root=args.testing_dir)
    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                            transform=transforms.Compose([transforms.Scale((100, 100)),
                                                                          transforms.ToTensor()
                                                                          ])
                                            , should_invert=False)

    test_dataloader = DataLoader(siamese_dataset, num_workers=6, batch_size=1, shuffle=True)
    dataiter = iter(test_dataloader)
    x0, _, _ = next(dataiter)

    for i in range(10):
        _, x1, label2 = next(dataiter)
        concatenated = torch.cat((x0.cpu(), x1.cpu()), 0)
        x0, x1 = Variable(x0), Variable(x1)
        if args.gpu:
            x0, x1 = x0.cuda(), x1.cuda()
        output1, output2 = net(x0, x1)
        euclidean_distance = F.pairwise_distance(output1, output2)
        imshow(torchvision.utils.make_grid(concatenated),
               'Dissimilarity: {:.2f}'.format(euclidean_distance.cpu().data.numpy()[0][0]))


if __name__ == "__main__":

    import argparse
    from helpers import *
    from models import *
    from losses import *
    from loaders import *

    parse = argparse.ArgumentParser()

    root = "./data/att_faces/"
    lfw_path = root + 'LFW_DIR/lfw-deepfunneled'
    training_path = root + "training/"
    testing_path = root + "testing/"

    parse.add_argument("--gpu", default=True, type=bool)
    parse.add_argument("--root", default=root, type=str)
    parse.add_argument("--training_dir", default=training_path, type=str)
    parse.add_argument("--testing_dir", default=testing_path, type=str)

    # 10 batch_size is the limit for 3 channels 
    parse.add_argument("--train_batch_size", default=10, type=int)
    parse.add_argument("--train_number_epochs", default=100, type=int)

    parse.add_argument("--model", default='standard', type=str,
                       help='options - standard, alexnet, inception, resnet or capsule.')

    parse.add_argument("--dataset", default='wild', type=str, help='options - att, wild or ...')
    parse.add_argument("--funneled", default=True, type=str, help='options - deep_funneled, funneled, original')
    parse.add_argument('-d', '--dataset_path', default=lfw_path)
    parse.add_argument('--fold', type=int, default=10, choices=[0, 10])

    parse.add_argument("--encoding_distance", default='euclidean', type=str,
                       help='options - euclidean, manhattan, cosine ')
    parse.add_argument("--color", default=True, type=bool)
    parse.add_argument("--criterion", default='contrastive', type=str, help='options - contrastive, ce')
    parse.add_argument("--nhidden_caps", default=20, type=int)
    parse.add_argument("--start_epoch", default=10, type=int)
    parse.add_argument("--best_loss", default=None, type=float)
    parse.add_argument("--batch_norm", default=True, type=bool)

    # weight and drop rates for concrete dropout
    parse.add_argument("--wr", default=0.001, type=float)
    parse.add_argument("--dr", default=0.1, type=float)
    parse.add_argument("--concrete_dropout", default=False, type=bool)
    parse.add_argument("--sig_dropout", default=False, type=bool)
    # l2 normalizes the features before distance measure
    parse.add_argument("--norm_dist", default=False, type=bool)

    parse.add_argument("--recon", default=False, type=bool)
    parse.add_argument("--linear", default=True, type=bool)
    parse.add_argument("--num_convs", default=2, type=int)
    parse.add_argument("--intermediate_convs", default=True, type=int)
    parse.add_argument("--test_all", default=True, type=bool)
    parse.add_argument("--checkpoint", default=True, type=bool)
    parse.add_argument("--save_model", default=False, type=bool)
    parse.add_argument("--squash", default=False, type=bool)
    parse.add_argument("--accuracy", default=True, type=bool)

    # contrastive margin
    parse.add_argument("--margin", default=0.5, type=float)
    parse.add_argument("--accuracy_thresh", default=0.2, type=float)

    args = parse.parse_args()

    # if l2-normalized features then threhsold is set to
    if args.norm_dist:
        args.margin = 1.0

    # done - ,'standard','resnet','inception','alexnet',
    params = {'models': ['capsule'],
              'criterions': ['contrastive'],  # 'ce'],
              'datasets': ['wild'],  # 'att'
              'dist_measure': ['manhattan', 'cosine', 'euclidean']}

    # resnet cannot change number of input channels from 3 to 1 so cannot use on att.

    cnt = 0

    # resnet, ce, euclidean

    if args.test_all:
        for dataset in params['datasets']:
            for model in params['models']:
                for measure in params['dist_measure']:
                    for criterion in params['criterions']:
                        # try:
                        print("\n Testing dataset: {0}  \t model: {1} \t distance: {2} \t criterion: {3} \n".format(
                            dataset, model, measure, criterion))

                        args.dataset = dataset
                        args.model = model
                        args.encoding_distance = measure
                        args.criterion = criterion

                        if model == 'caspule':
                            args.train_batch_size = 10
                            # args.squash = True
                            net, best_loss = train(args)
                            # args.squash = False
                        else:
                            net, best_loss = train(args)
                            cnt += 1

                        # except:
                        #    pass

        print ("\n {0} \% passed the test \n".format(str(100 * (cnt / float(sum(flatten(len(params.values()))))))))

    else:
        net, best_1oss = train(args)

    # test(net, args)
