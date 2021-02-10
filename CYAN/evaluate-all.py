#test.py
#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset
of a model

author baiyu
"""

import argparse

from matplotlib import pyplot as plt

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchsummary import summary

from conf import settings
from utils import get_network, get_test_dataloader

def evaluate(model,net):
    net.load_state_dict(torch.load(model))
    #print(net)
    net.eval()

    correct_1 = 0.0
    correct_5 = 0.0
    total = 0

    with torch.no_grad():
        for n_iter, (image, label) in enumerate(cifar100_test_loader):
            #print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(cifar100_test_loader)))

            if args.gpu:
                image = image.cuda()
                label = label.cuda()
                print('GPU INFO.....')
                print(torch.cuda.memory_summary(), end='')


            output = net(image)
            _, pred = output.topk(5, 1, largest=True, sorted=True)

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()
            _,predicted = torch.max(output,1)
            #print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(1)))
            #print(output,"\n******\n",predicted,"\n******\n",pred,"\n******\n",label)
            #compute top 5
            print(correct,"************",correct[:, :1])
            correct_5 += correct[:, :5].sum()

            #compute top1
            correct_1 += correct[:, :1].sum()

    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')

    print()
    print("Top 1 accuracy: ", 1 - correct_1 / len(cifar100_test_loader.dataset))
    print("Top 5 accuracy: ", 1 - correct_5 / len(cifar100_test_loader.dataset))
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    args = parser.parse_args()


    cifar100_test_loader,classes = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        #settings.CIFAR100_PATH,
        num_workers=1,
        batch_size=1,
    )
    models = args.weights.split(" ")
    networks = args.net.split(" ")
    print(models, networks, classes)
    for i in range(len(models)):
        args.net = networks[i]
        net = get_network(args)
        #print(net)
        #print(summary(net,batch_size=-1, device='cuda'))
        model = models[i]
        evaluate(model,net)


