import torch
from torchvision import datasets
from PIL import Image
from torch.utils.data import Dataset

class ImageFolderWithPaths(datasets.ImageFolder):
    
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        (image, label) = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]

        
        # make a new tuple that includes original and the path
        tuple_with_path = (label ,path)
        return tuple_with_path



class ImageFolderBlocks(Dataset):
    def __init__(self, dataset, transform):
        self.block_dataset = dataset
        self.transform = transform
                

    def __len__(self):
        return len(self.block_dataset)

    def __getitem__(self, idx):
        #img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(self.block_dataset[idx]).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image


class ImageFolderTriplets(Dataset):
    def __init__(self, triplet, transform):
        self.triplet_dataset = triplet
        self.transform = transform
                

    def __len__(self):
        return len(self.triplet_dataset)

    def __getitem__(self, idx):
        #img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        anchor = Image.open(self.triplet_dataset[idx][0]).convert("RGB")
        positive = Image.open(self.triplet_dataset[idx][1]).convert("RGB")
        negative = Image.open(self.triplet_dataset[idx][2]).convert("RGB")

        anchor = self.transform(anchor)
        positive = self.transform(positive)
        negative = self.transform(negative)

        return anchor, positive, negative

class ImageFolderQuads(Dataset):
    def __init__(self, quadruplet, transform):
        self.quad_dataset = quadruplet
        self.transform = transform
                

    def __len__(self):
        return len(self.quad_dataset)

    def __getitem__(self, idx):
        #img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        anchor = Image.open(self.quad_dataset[idx][0]).convert("RGB")
        positive = Image.open(self.quad_dataset[idx][1]).convert("RGB")
        negative = Image.open(self.quad_dataset[idx][2]).convert("RGB")
        negative2 = Image.open(self.quad_dataset[idx][3]).convert("RGB")



        
        anchor = self.transform(anchor)
        positive = self.transform(positive)
        negative = self.transform(negative)
        negative2 = self.transform(negative2)

        return anchor, positive, negative, negative2
