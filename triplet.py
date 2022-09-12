from datetime import datetime
import os.path
import time
import sys
import numpy as np
import importlib
import itertools
import argparse
import torch



def select_triplets(embeddings, paths, nrof_images_per_class, people_per_batch, alpha, device):
    """ Select the triplets for training
    """
    emb_start_idx = 0
    anchor = torch.zeros(1, embeddings.shape[1]).to(device)
    positive = torch.zeros(1, embeddings.shape[1]).to(device)
    negative = torch.zeros(1, embeddings.shape[1]).to(device)
    
    
    triplets = []
    

    for i in range(people_per_batch):
        nrof_images = int(nrof_images_per_class[i])
        for j in range(1,nrof_images):
            a_idx = emb_start_idx + j - 1
            neg_dists_sqr = torch.sum(torch.pow(embeddings[a_idx] - embeddings, 2), 1)
            neg_dists_sqr[emb_start_idx:emb_start_idx+nrof_images] = np.NaN

            for pair in range(j, nrof_images):
                p_idx = emb_start_idx + pair
                pos_dist_sqr = torch.sum(torch.pow(embeddings[a_idx]-embeddings[p_idx], 2))
                
                all_neg = torch.where(neg_dists_sqr-pos_dist_sqr<alpha)[0] # VGG Face selecction
                

                nrof_random_negs = all_neg.shape[0]
                if nrof_random_negs>0:
                    
                    rnd_idx = np.random.randint(nrof_random_negs)
                    n_idx = all_neg[rnd_idx]
                    
                    triplets.append((paths[a_idx], paths[p_idx], paths[n_idx]))

                    

        emb_start_idx += nrof_images

    
    return triplets


