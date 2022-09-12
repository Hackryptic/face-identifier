from datetime import datetime
import os.path
import time
import sys
import numpy as np
import importlib
import itertools
import argparse
import torch

def select_quads(embeddings, paths, nrof_images_per_class, people_per_batch, alpha, alpha2, device):
    
    emb_start_idx = 0
    anchor = torch.zeros(1, embeddings.shape[1]).to(device)
    positive = torch.zeros(1, embeddings.shape[1]).to(device)
    negative = torch.zeros(1, embeddings.shape[1]).to(device)
    
    
    quadruplets = []
    
    print("alpha: {}".format(alpha))
    print("alpha2: {}".format(alpha2))

    for i in range(people_per_batch):
        nrof_images = int(nrof_images_per_class[i])
        for j in range(1,nrof_images):
            a_idx = emb_start_idx + j - 1
            neg_dists_sqr = torch.sum(torch.pow(embeddings[a_idx] - embeddings, 2), 1)
            neg_dists_sqr[emb_start_idx:emb_start_idx+nrof_images] = np.NaN

            
            for pair in range(j, nrof_images):                 
                p_idx = emb_start_idx + pair
                pos_dist_sqr = torch.sum(torch.pow(embeddings[a_idx]-embeddings[p_idx], 2))
                
                
                all_neg = torch.where(neg_dists_sqr-pos_dist_sqr<alpha)[0]
                
                
                nrof_random_negs = all_neg.shape[0]
                if nrof_random_negs>0:
                    
                    rnd_idx = np.random.randint(nrof_random_negs)
                    n_idx = all_neg[rnd_idx]

                    n2_start_idx = int(n_idx / 5)
                    n2_end_idx = min(n2_start_idx + 4, embeddings.shape[0])

                    neg2_dists_sqr = torch.sum(torch.pow(embeddings[n_idx] - embeddings, 2), 1)
                    neg2_dists_sqr[emb_start_idx:emb_start_idx+nrof_images] = np.NaN
                    neg2_dists_sqr[n2_start_idx:n2_end_idx] = np.NaN

                    all_neg2_alpha = torch.where(neg2_dists_sqr-pos_dist_sqr<alpha)[0]
                    
                    num_all_negs2_alpha = all_neg2_alpha.shape[0]

                    if num_all_negs2_alpha > 0:
                        rnd_idx_neg2 = np.random.randint(num_all_negs2_alpha)
                        n2_idx = all_neg2_alpha[rnd_idx_neg2]

                    else:
                        all_neg2 = torch.where(neg2_dists_sqr != np.NaN)[0]
                        num_all_negs2 = all_neg2.shape[0]
                        rnd_idx_neg2 = np.random.randint(num_all_negs2)
                        n2_idx = all_neg2[rnd_idx_neg2]
                    
                    
                    quadruplets.append((paths[a_idx], paths[p_idx], paths[n_idx], paths[n2_idx]))


        emb_start_idx += nrof_images


    return quadruplets



def select_Adaquads(embeddings, paths, nrof_images_per_class, people_per_batch, alpha, alpha2, device):
    """ Select the adaptie quads
    """
    emb_start_idx = 0
    anchor = torch.zeros(1, embeddings.shape[1]).to(device)
    positive = torch.zeros(1, embeddings.shape[1]).to(device)
    negative = torch.zeros(1, embeddings.shape[1]).to(device)
    
    quadruplets = []

    sum_pos = 0.0
    sum_neg = 0.0

    num_pos = nrof_images_per_class[0] * (nrof_images_per_class[0] - 1) / 2 * people_per_batch
    num_neg = (people_per_batch - 1) * nrof_images_per_class[0] * (embeddings.shape[0]) 

    for i in range(people_per_batch):
        nrof_images = int(nrof_images_per_class[i])
        for j in range(1,nrof_images):
            a_idx = emb_start_idx + j - 1
            neg_dists_sqr = torch.sum(torch.pow(embeddings[a_idx] - embeddings, 2), 1)
            neg_dists_sqr[emb_start_idx:emb_start_idx+nrof_images] = np.NaN

            sum_neg += torch.nansum(neg_dists_sqr, 0).item()
            
            for pair in range(j, nrof_images):
                p_idx = emb_start_idx + pair
                pos_dist_sqr = torch.sum(torch.pow(embeddings[a_idx]-embeddings[p_idx], 2))

                
                sum_pos += pos_dist_sqr.item()
        
        emb_start_idx += nrof_images
    
    mean_pos = (sum_pos / num_pos)
    mean_neg = (sum_neg / num_neg)
    mean_diff = mean_neg - mean_pos

    if mean_diff > 0:
        alpha = mean_diff
        alpha2 = mean_diff * 0.5

    
    print("alpha: {}".format(alpha))
    print("alpha2: {}".format(alpha2))

    emb_start_idx = 0

    for i in range(people_per_batch):
        nrof_images = int(nrof_images_per_class[i])
        for j in range(1,nrof_images):
            a_idx = emb_start_idx + j - 1
            neg_dists_sqr = torch.sum(torch.pow(embeddings[a_idx] - embeddings, 2), 1)
            neg_dists_sqr[emb_start_idx:emb_start_idx+nrof_images] = np.NaN

            
            for pair in range(j, nrof_images):                 
                p_idx = emb_start_idx + pair
                pos_dist_sqr = torch.sum(torch.pow(embeddings[a_idx]-embeddings[p_idx], 2))
                
                sum_pos += pos_dist_sqr.item()
                num_pos += 1

                
                
                all_neg = torch.where(neg_dists_sqr-pos_dist_sqr<alpha)[0]
                

                nrof_random_negs = all_neg.shape[0]
                if nrof_random_negs>0:
                    
                    rnd_idx = np.random.randint(nrof_random_negs)
                    n_idx = all_neg[rnd_idx]

                    n2_start_idx = int(n_idx / 5)
                    n2_end_idx = min(n2_start_idx + 4, embeddings.shape[0])

                    neg2_dists_sqr = torch.sum(torch.pow(embeddings[n_idx] - embeddings, 2), 1)
                    neg2_dists_sqr[emb_start_idx:emb_start_idx+nrof_images] = np.NaN
                    neg2_dists_sqr[n2_start_idx:n2_end_idx] = np.NaN

                    all_neg2_alpha = torch.where(neg2_dists_sqr-pos_dist_sqr<alpha)[0]
                    
                    num_all_negs2_alpha = all_neg2_alpha.shape[0]

                    if num_all_negs2_alpha > 0:
                        rnd_idx_neg2 = np.random.randint(num_all_negs2_alpha)
                        n2_idx = all_neg2_alpha[rnd_idx_neg2]

                    else:
                        all_neg2 = torch.where(neg2_dists_sqr != np.NaN)[0]
                        num_all_negs2 = all_neg2.shape[0]
                        rnd_idx_neg2 = np.random.randint(num_all_negs2)
                        n2_idx = all_neg2[rnd_idx_neg2]
                    
                    
                    quadruplets.append((paths[a_idx], paths[p_idx], paths[n_idx], paths[n2_idx]))


                    


        emb_start_idx += nrof_images

    

    return alpha, alpha2, quadruplets


    
