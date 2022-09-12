from models import inception_resnet_v1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import os
import sys
import losses
import sampler
import time
import copy
import train_options
import customdataset
import math
import triplet
import torchvision
import quadruplet

class ToTensorWithScaling(object):
    def __call__(self, image):

        return torch.ByteTensor(np.array(image)).permute(2, 0, 1).float().div(255).mul(2).add(-1)

def getTPLoss(anchor, positive, negative, num_image_per_class, num_people_per_batch, alpha, device):
    loss  = losses.triplet_loss(anchor, positive, negative, num_image_per_class, num_people_per_batch, alpha, device)
    
    return loss

def updateLR(optimizer, epoch, decay_start_epoch, org_lr, cur_lr):

    decay_epoch = epoch - decay_start_epoch
    lr = cur_lr - org_lr / decay_epoch

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def updateLRFactor(optimizer, cur_lr, factor):
    
    lr = cur_lr * factor

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def column(matrix, i):
    return [row[i] for row in matrix]


def checkLabel(label_list):
    
    num_for_class = []
    old = -1
    new = -1
    unique_class = np.unique(label_list.cpu())

    for u_label in unique_class:
        
        num_for_class.append((label_list == u_label).sum().item())

    
    return len(unique_class), num_for_class




def train_model():
    since = time.time()
    
    options = train_options.TrainOptions(True)
    opt = options.parse()

    print(opt.gpu_ids)
    device = torch.device("cuda:{}".format(opt.gpu_ids) if torch.cuda.is_available() else "cpu")

    model_save_path = './saved_model/{}.pt'.format(options.opt.name)

    lr = opt.lr
    num_epochs = opt.epoch
    decay_start_epoch = opt.decay_start_epoch

    weight_update_step = options.opt.update_step
    print_step = 10
    cur_step = 0
    step_loss = 0.0
    block_classes = opt.class_per_block
    class_samples = opt.image_per_class

    cur_epoch = 0
    cur_lr = lr
    
    dataset = datasets.ImageFolder(opt.dataroot, transform=test_transform)
    dataset_folder = customdataset.ImageFolderWithPaths(opt.dataroot)
    
    label_loader = DataLoader(dataset_folder, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)

    path_list = []

    alpha = opt.margin
    alpha2 = opt.margin2
    

    for idx, (labels, paths) in enumerate(label_loader):

        if idx == 0:
            label_list = labels
        else:
            label_list = torch.cat((label_list, labels), 0)


    train_batch_sampler = sampler.BalancedBatchSampler(label_list, n_classes=block_classes, n_samples=class_samples)
    
    blockloader = DataLoader(dataset_folder, batch_sampler=train_batch_sampler)

    steps_for_epoch = math.ceil(label_list.shape[0] / (block_classes * class_samples))
        
    if opt.model_type == "mobilenetv2":
        model = torchvision.models.mobilenet_v2(pretrained=False, num_classes=512)
    elif opt.model_type == "inceptionresnetv1":
        model = inception_resnet_v1.InceptionResnetV1(pretrained=None)
    else:
        print("Please specify the proper model names")
        sys.exit(0)


    model.to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(opt.beta1, opt.beta2), weight_decay=0.0005, eps=0.1)

    criterion = getTPLoss
    
    step_size = opt.epoch_size
    batch_cnt = 0
    adapt_start_epoch = opt.adapt_start_epoch

    dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)


        for phase in ['train']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            print('Learning Rate = {}'.format(cur_lr))

            valid_block = False
            total_trip_num = 0

                      
            for idx, (labels, paths) in enumerate(blockloader):
                
                class_per_batch, num_for_class = checkLabel(labels)

                
                block = customdataset.ImageFolderBlocks(dataset=paths, transform=test_transform)
                tripletloader = DataLoader(block, batch_size=opt.loading_batch_size, shuffle=False)

            
                with torch.set_grad_enabled(False):
                    
                    for idx2, data in enumerate(tripletloader):
                        

                        data = data.to(device)
                        
                        

                        if idx2 == 0:
                            embeddings  = model(data)
                        else:
                            embeddings = torch.cat((embeddings, model(data)), axis=0)


                    
                    triplet_list = triplet.select_triplets(embeddings, paths, num_for_class, class_per_batch, alpha, device)       

                    trip_num = len(triplet_list)


                    print('block_num: {}'.format(idx + 1))

                    del embeddings
                    torch.cuda.empty_cache()


                

                if trip_num == 0:
                    continue
                
                valid_block = True

                total_trip_num += trip_num

                batchset = customdataset.ImageFolderTriplets(triplet=triplet_list, transform=test_transform)
                batchloader = DataLoader(batchset, batch_size=opt.tuple_batch_size, shuffle=True, drop_last=True, num_workers=4)
   
                update_count = 0
                cur_step = 0

                                                                
                with torch.set_grad_enabled(phase == 'train'):

                    for idx, (anchor_image, positive_image, negative_image) in enumerate(batchloader):
                    
                        if len(anchor_image) == 1:
                            del anchor_image
                            del positive_image
                            del negative_image
                            torch.cuda.empty_cache()
                            break

                        
                        anchor_image = anchor_image.to(device)
                        positive_image = positive_image.to(device)
                        negative_image = negative_image.to(device)
                                                
                        anchors = model(anchor_image)
                        positives = model(positive_image)
                        negatives = model(negative_image)
                        

                        loss  = criterion(anchors, positives, negatives, num_for_class, class_per_batch, alpha, device)
                                                
                            
                        step_loss += loss.item()
                        running_loss += loss.item()
                        
                        loss.backward()

                        cur_step += 1
                        
                         
                        if cur_step >= weight_update_step:
                            print("step loss: {}".format(loss.item()))
                            print("-" * 5)
                            print()

                            optimizer.step()

                            cur_step = 0
                            
                            optimizer.zero_grad()
                            update_count += 1

                            if update_count >= step_size:
                                optimizer.zero_grad()
                                update_count = 0

                                break

                        
                        
            

            if valid_block:
                epoch_loss = running_loss / steps_for_epoch
                valid_block = False
                print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            else:
                print('No Triplets for Epoch')

            
            if epoch >= decay_start_epoch:

                cur_lr = updateLRFactor(optimizer, cur_lr, opt.decay_factor)

            epoch_time_elap = time.time() - epoch_start_time

            print('Epoch complete in {:.0f}m {:.0f}s'.format( epoch_time_elap // 60, epoch_time_elap % 60))

            
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    torch.save(model.state_dict(), model_save_path)
    

    print('saved models')

    


np.set_printoptions(threshold=sys.maxsize)
pd.options.display.max_columns = 4000

workers = 0 if os.name == 'nt' else 4

if __name__ == '__main__':

    test_transform = transforms.Compose([
        ToTensorWithScaling()
        ])

    
    train_model()


