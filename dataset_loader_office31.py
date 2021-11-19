import os
import pdb
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import glob
import random

class Dataset_from_folder(Dataset):
    def __init__(self, sample_list = [], domain = None, transforms=None):
        self.transforms = transforms
        # self.images_path = glob.glob(f'{folder_path}/{domain}/*/*/*.jpg')
        self.images_path = sample_list
        
        # self.transforms = transforms.Compose([transforms.ToTensor()])
        # print(images)
        classes = list(set([path.split('/')[-2] for path in self.images_path]))
        classes.sort()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        # self.img_target = [(image, class_to_idx[image.split('/')[-2]]) for image in images]
        
    def __len__(self):
        return len(self.images_path)
    # def __classes__(self):
    #     return self.class_to_idx
    def __getitem__(self, index):
        # pdb.set_trace()
        path = self.images_path[index]
        # img = Image.open(path.split(',')[0])
        img = Image.open(path)

        # exit()
        img = img.convert('RGB')
        img = self.transforms(img)
        target_names = path.split('/')[-2]
        targets = self.class_to_idx[path.split('/')[-2]]
        targets = torch.tensor(targets)
        # return img, targets, target_names, path
        return img, targets

def get_train_test_loaders(folder_path='datasets/office31', domain = 'amazon', batch_size=32, transforms=transforms):
    images_path = glob.glob(f'{folder_path}/{domain}/*/*/*.jpg')
    # pdb.set_trace()
    random.shuffle(images_path)
    train_list = images_path[:int(len(images_path)*0.8)]
    test_list = images_path[int(len(images_path)*0.8):]
    train_dataset = Dataset_from_folder(train_list, domain, transforms)
    test_dataset = Dataset_from_folder(test_list, domain, transforms)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=1)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=True,num_workers=1)

    return train_dataloader, test_dataloader

if __name__ == '__main__':

    transforms = transforms.Compose([transforms.ToTensor(), transforms.Resize((224,224))])

    train_dataloader, test_dataloader = get_train_test_loaders(folder_path='datasets/office31', batch_size=32, domain = 'amazon', transforms=transforms)

    # train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=4,shuffle=True,num_workers=1)

    for inputs, targets in train_dataloader:
        print(inputs.shape)
        print(targets.shape)
        # print(path)
        pdb.set_trace()
        exit()