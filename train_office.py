from silly_funcs import *
import torch.optim as optim
from torchvision import transforms
import torch.nn as nn
import torch
import pdb
import torch.nn.functional as F
from torchvision import models
import torchvision

from dataset_loader_office31 import get_train_test_loaders 
resnet_model = models.resnet18(pretrained=True)
device = torch.device('cuda')

resnet_model = resnet_model.to(device)
num_ftrs = resnet_model.fc.in_features
# print(f'Last FC layer of ResNet-18: {resnet_model.fc}')
# print(f'Output layer dim for ImageNet: {resnet_model.fc.out_features}')
# print(f'Trainable ResNet-18 params before freezing all layers: {sum(p.numel() for p in resnet_model.parameters() if p.requires_grad)}')
# for param in resnet_model.parameters():
#     param.requires_grad = False
# print(f'Trainable params after freezing all layers: {sum(p.numel() for p in resnet_model.parameters() if p.requires_grad)}')
resnet_model.fc = nn.Linear(num_ftrs, 31)
# print(f'Trainable params after changing last FC layer: {sum(p.numel() for p in resnet_model.parameters() if p.requires_grad)}')
# print(f'Last FC layer (changed for CIFAR-10): {resnet_model.fc}')
# print(f'Output layer dim for CIFAR-10: {resnet_model.fc.out_features}')
# print(resnet_model)

resnet_model = resnet_model.to(device)
# test_losses = []
# test(resnet_model, 1, testloader, if_resnet=True)
# print(f'As the last layer is not trained, we can see the accuracy of randomly picking a class. \ni.e. 100/#classes : ~10%')

def train(classifier, epoch, train_loader, if_resnet=False, verbose = True, show_less = False):

    classifier.train() # we need to set the mode for our model
    correct = 0
    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        output = classifier(images)
        if if_resnet:
            output = nn.LogSoftmax(dim=1)(output)
        loss = F.nll_loss(output, targets) # Here is a typical loss function (negative log likelihood)
        pred = output.data.max(1, keepdim=True)[1] # we get the estimate of our result by look at the largest class value
        correct += pred.eq(targets.data.view_as(pred)).sum() # sum up the corrected samples        
        loss.backward()
        # print(loss)
        optimizer.step()

        if batch_idx % 10 == 0: # We record our output every 10 batches
            train_losses.append(loss.item()) # item() is to get the value of the tensor directly
            train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
        print_after_batch = 100
        # if show_less:
        #     print_after_batch = 250
        # if batch_idx % print_after_batch == 0: # We visulize our output every 10 batches
        #     if verbose:
        #         print(f'Epoch {epoch}: [{batch_idx*len(images)}/{len(train_loader.dataset)}] Loss: {loss.item()}')
    return 100.*correct/len(train_loader.dataset)

def test(classifier, epoch, test_loader, if_resnet=False):

    classifier.eval() # we need to set the mode for our model

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device)        
            output = classifier(images)
            if if_resnet:
                output = nn.LogSoftmax(dim=1)(output)      
            test_loss += F.nll_loss(output, targets, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1] # we get the estimate of our result by look at the largest class value
            correct += pred.eq(targets.data.view_as(pred)).sum() # sum up the corrected samples

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    test_counter.append(len(test_loader.dataset)*epoch)
    return test_loss, 100.*correct/len(test_loader.dataset)
    # print(f'Test result on epoch {epoch}: Avg loss is {test_loss}, Accuracy: {100.*correct/len(test_loader.dataset)}%')

transforms = transforms.Compose([transforms.ToTensor(), transforms.Resize((224,224))])
train_dataloader, test_dataloader = {}, {}
for domain in ['amazon', 'dslr', 'webcam']:
    train_dataloader[domain], test_dataloader[domain] = get_train_test_loaders(folder_path='datasets/office31', batch_size=32, domain = domain, transforms=transforms)

train_domain = 'amazon'
max_epoch = 10
train_losses = []
train_counter = []
test_losses = []
test_counter = []
optimizer = optim.SGD(resnet_model.parameters(), lr=0.01, momentum=0.8)

save_dir = f"checkpoints_office31/{train_domain}/"
mkdir_func(save_dir)
for epoch in range(1, max_epoch+1):
    train_acc = train(resnet_model, epoch, train_dataloader[train_domain], if_resnet=True, verbose=True, show_less = True)
    print(f'Epoch {epoch}: Domain:{train_domain} Train Acc: {train_acc:.3f}')

    loss, test_acc = test(resnet_model, epoch, test_dataloader['amazon'], if_resnet=True)
    print(f"Acc on amazon: {test_acc:.2f}%")

    loss, acc = test(resnet_model, epoch, test_dataloader['dslr'], if_resnet=True)
    print(f"Acc on dslr: {acc:.2f}%")

    loss, acc = test(resnet_model, epoch, test_dataloader['webcam'], if_resnet=True)
    print(f"Acc on webcam: {acc:.2f}%")
    torch.save(resnet_model.state_dict(), f"{save_dir}epoch_{epoch}_{test_acc:.2f}.pth")