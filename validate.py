from models import inception_resnet_v1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.nn.functional import interpolate
import numpy as np
import pandas as pd
import os
import sys
import torchvision
import PIL
import test_options

np.set_printoptions(threshold=sys.maxsize)
pd.options.display.max_columns = 4000
#torch.multiprocessing.set_sharing_strategy('file_system')

workers = 0 if os.name == 'nt' else 4

class ToTensorWithScaling(object):
    """H x W x C -> C x H x W"""
    def __call__(self, image):

        #print(np.array(image))

        return torch.ByteTensor(np.array(image)).permute(2, 0, 1).float().div(255).mul(2).add(-1)
    
def rejectFace(distances, labels, threshold):
    
    min_distance = sys.maxsize

    labels_unique = np.unique(labels)
    #print(labels_unique)

    is_rejected = False

    for class_value in labels_unique:
        cluster_indices = np.where(labels == class_value)[0]
        avg_dist = 0
        #print(class_value)
        #print(cluster_indices)



        for dist_index in cluster_indices:
            #print(distances[dist_index])
            avg_dist += distances[dist_index]
        
        
        avg_dist /= cluster_indices.size

        if avg_dist < min_distance:
            min_distance = avg_dist

    if min_distance > threshold:
        is_rejected = True


    return is_rejected

    
    
    

def predictFace(ref_embeddings, ref_labels, test_face, k, rejection=False, threshold=1.0):
    
    all_dists = np.sum(np.square(ref_embeddings-test_face), axis=1)
    #print(all_dists)

    
    is_rejected = False
    if rejection:
        is_rejected = rejectFace(all_dists, ref_labels, threshold)

    if not is_rejected:
        k_top_indices = np.argsort(all_dists)[:5]

        #k_top_dists = np.sort(all_dists[-5:])
        #print(k_top_indices)
        k_top_labels = ref_labels[k_top_indices]
        #print(k_top_labels)
        label_counts = np.bincount(k_top_labels)

        #print(label_counts)

        max_label = np.argmax(label_counts)

        #print(max_label)

    else:
        max_label = -1

    return max_label
        
        
    
    
    


if __name__ == '__main__':

    options = test_options.TestOptions()
    opt = options.parse()


    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(device)
    print('Running on device: {}'.format(device))

    model_load_path = opt.param_path
 
    test_transform = transforms.Compose([
        ToTensorWithScaling()
        ])
    
    ref_dataset = datasets.ImageFolder(opt.ref_path, transform=test_transform)

    ref_loader = DataLoader(ref_dataset, batch_size=1,  num_workers=4)
    ref_dataset.idx_to_class = {i:c for c, i in ref_dataset.class_to_idx.items()}

    
    if opt.model_type == "mobilenetv2":
        model = torchvision.models.mobilenet_v2(pretrained=False, num_classes=512)
    elif opt.model_type == "inceptionresnetv1":
        model = inception_resnet_v1.InceptionResnetV1(pretrained=None)
    else:
        print("Please specify the proper model names")
        sys.exit(0)


    model.to(device)

    model.load_state_dict(torch.load(model_load_path), strict=False)

    model.eval()
    
    data_count = 1
    names = []
    inputs_list = []
    label_list = []
    

    stop_count = 0


    with torch.no_grad():

        for idx, (inputs, labels)  in enumerate(ref_loader):
            if inputs is not None:
                

                for label in labels:
                    names.append(ref_dataset.idx_to_class[label.item()])
                    label_list.append(label.item())
                
                            
                inputs = inputs.to(device)
                
                
                embeddings = model(inputs)



                emb_np = embeddings.cpu().detach().numpy()

                if idx == 0:
                    embedding_list = emb_np
                else:
                    embedding_list = np.concatenate((embedding_list, emb_np), axis=0)

                stop_count += 1
               


        labels_np = np.array(label_list)

        test_face = PIL.Image.open(opt.test_image)
        test_face = test_transform(test_face)
        test_face = test_face.unsqueeze(0)
        test_face = test_face.to(device)


        test_face_emb = model(test_face)

        test_face_emb = test_face_emb.squeeze(0)
        test_face_emb = test_face_emb.cpu().detach().numpy()
        
        predicted_label = predictFace(embedding_list, labels_np, test_face_emb, opt.rejection, 5)


        if predicted_label >= 0:
            print("predicted_label: {}".format(ref_dataset.idx_to_class[predicted_label]))
        else:
            print("rejected")


       

        
