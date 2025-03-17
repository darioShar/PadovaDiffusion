import os
import numpy as np
import torch
import scipy

import torch.nn as nn
import torch.optim as optim
import shutil


from .DiffusionBlocks import DiffusionBlockConditioned
from .Embeddings import SinusoidalPositionalEmbedding

    

#Can predict gaussian_noise, stable_noise, anterior_mean
class MLPModel(nn.Module):

    def __init__(self, 
                 nfeatures = 2,
                 time_emb_size = 32,
                 nblocks = 2,
                 nunits = 32,
                 skip_connection = True,
                 layer_norm = True,
                 dropout_rate = 0.1,
                 learn_variance = False,
                 ):
        super(MLPModel, self).__init__()

        # extract from param dict
        self.nfeatures =        nfeatures
        self.time_emb_size =    time_emb_size
        self.nblocks =          nblocks
        self.nunits =           nunits
        self.skip_connection =  skip_connection
        self.layer_norm =       layer_norm
        self.dropout_rate =     dropout_rate
        self.learn_variance =   learn_variance
        
        
        
        # for dropout and group norm.
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.layer_norm_in = nn.LayerNorm([self.nunits]) if self.layer_norm else nn.Identity()
        self.act = nn.SiLU(inplace=False)
        
        
        
        self.time_emb = nn.Linear(1, self.time_emb_size) #Embedding.LearnableEmbedding(1, self.time_emb_size, self.device)
        self.time_mlp = nn.Sequential(self.time_emb,
                                      self.act,
                                      nn.Linear(self.time_emb_size, self.time_emb_size), 
                                      self.act)
        

        self.linear_in =  nn.Linear(self.nfeatures + self.additional_dim, self.nunits)
        
        self.inblock = nn.Sequential(self.linear_in,
                                     self.layer_norm_in, 
                                     self.act)
        
        self.midblocks = nn.ModuleList([DiffusionBlockConditioned(
                                            self.nunits, 
                                            self.dropout_rate, 
                                            self.skip_connection, 
                                            self.layer_norm,
                                            time_emb_size = self.time_emb_size,
                                            activation = nn.SiLU)
                                        for _ in range(self.nblocks)])
        
        # add one conditioned block and one MLP for both mean and variance computation
        self.outblocks_mean_cond_mlp = DiffusionBlockConditioned(
                                                self.nunits, 
                                                self.dropout_rate, 
                                                self.skip_connection, 
                                                self.layer_norm,
                                                time_emb_size = self.time_emb_size,
                                                activation = nn.SiLU)
        self.outblocks_mean_ff = nn.Linear(self.nunits, self.nfeatures)
            
        if self.learn_variance:
            self.outblocks_var_mlp =  DiffusionBlockConditioned(
                                                self.nunits, 
                                                self.dropout_rate, 
                                                self.skip_connection, 
                                                self.layer_norm,
                                                time_emb_size = self.time_emb_size,
                                                activation = nn.SiLU)
            self.outblocks_mvar_ff = nn.Linear(self.nunits, self.nfeatures)
        
        
    def forward(self, x, timestep):
    
        t = self.time_mlp(timestep.to(torch.float32))
        
        # input
        val = x
        
        # input block
        val = self.inblock(val)
        
        # midblocks
        for midblock in self.midblocks:
            val = midblock(val, t)
        
        # output blocks
        val_mean = self.outblocks_mean_cond_mlp(val, t)
        val_mean = self.outblocks_mean_ff(val_mean)
        
        # if we learn variance
        if not self.learn_variance:
            return val_mean

        val_var = self.outblocks_var[0](val, t)
        val_var = self.outblocks_var[1](val_var)
        
        return torch.concat([val_mean, val_var], dim = 1) # concat on channels dim
        

    
    