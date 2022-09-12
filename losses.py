import torch
import numpy as np
import triplet



def triplet_loss(anchor, positive, negative, nrof_images_per_class, people_per_batch, alpha, device):
    """Calculate the triplet loss according to the FaceNet paper
    
    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.
  
    Returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """

    #anchor, positive, negative = triplet.select_triplets(embeddings, nrof_images_per_class, people_per_batch, alpha, device)

    

    #triplets_t = triplets.T
    #anchor = triplets_t[0]
    #positive = triplets_t[1]
    #negative = triplets_t[2]
    

    #print(anchor)
    #print(positive)
    #print(negative)

    
    pos_dist = torch.sum(torch.pow(torch.sub(anchor, positive), 2), 1)
    neg_dist = torch.sum(torch.pow(torch.sub(anchor, negative), 2), 1)

    #print(pos_dist)
    #print(neg_dist)

    pos_mean = torch.mean(pos_dist)
    neg_mean = torch.mean(neg_dist)

    #print('pos mean {}'.format(pos_mean.item()))
    #print('neg mean {}'.format(neg_mean.item()))
  

    basic_loss = torch.add(torch.sub(pos_dist,neg_dist), alpha)

    #print('basic : {}'.format(basic_loss))

    #print('basic: {}'.format(basic_loss))

    clamped_loss = torch.clamp(basic_loss, min=0.0)
    #print('clamped: {}'.format(clamped_loss))

    loss = torch.mean(clamped_loss, 0)
    
    
     
    return loss


def quadruplet_loss(anchor, positive, negative, negative_2, nrof_images_per_class, people_per_batch, alpha1, alpha2, device):
    """Calculate the quadruplet loss 
    
    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.
      nagative_2: the embeddings for the second negative images.
  
    Returns:
      the quadruplet loss
    """

    # calculate d1, d2, d3
    pos_dist = torch.sum(torch.pow(torch.sub(anchor, positive), 2), 1)
    neg_dist = torch.sum(torch.pow(torch.sub(anchor, negative), 2), 1)
    neg_dist_2 = torch.sum(torch.pow(torch.sub(negative, negative_2), 2), 1)

    # print(pos_dist)
    # print(neg_dist)
    # print(neg_dist_2)

    pos_neg = torch.add(torch.sub(pos_dist, neg_dist), alpha1)
    pos_neg_2 = torch.add(torch.sub(pos_dist, neg_dist_2), alpha2)
    
    loss1 = torch.clamp(pos_neg, min = 0)
    loss2 = torch.clamp(pos_neg_2, min = 0)

    quadruplet_loss = torch.mean(torch.add(loss1, loss2), 0)

    return quadruplet_loss
