#!/usr/bin/env python
# coding: utf-8

"""
Keras implementation of DNGR model. Generate embeddings for 3NG 
of 20NewsGroup dataset. Also visualizing embeddings with t-SNE.

Author: Apoorva Vinod Gorur
"""

import sys
import numpy as np
import warnings
import DNGR_utils as ut
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, noise
from keras.models import Model
from argparse import ArgumentParser


#Stage 1 -  Random Surfing
def random_surf(cosine_sim_matrix, num_hops, alpha):
    
    num_nodes = len(cosine_sim_matrix)
    
    adj_matrix = ut.scale_sim_matrix(cosine_sim_matrix)
    P0 = np.eye(num_nodes, dtype='float32')
    P = np.eye(num_nodes, dtype='float32')
    A = np.zeros((num_nodes,num_nodes),dtype='float32')
    
    for i in range(num_hops):
        P = (alpha*np.dot(P,adj_matrix)) + ((1-alpha)*P0)
        A = A + P

    return A



#Stage 2 - PPMI Matrix
def PPMI_matrix(A):
    
    num_nodes = len(A)
    A = ut.scale_sim_matrix(A)
    
    row_sum = np.sum(A, axis=1).reshape(num_nodes,1)
    col_sum = np.sum(A, axis=0).reshape(1,num_nodes)
    
    D = np.sum(col_sum)
    PPMI = np.log(np.divide(np.multiply(D,A),np.dot(row_sum,col_sum)))
    #Gotta use numpy for division, else it runs into divide by zero error, now it'll store inf or -inf
    #All Diag elements will have either inf or -inf.
    #Get PPMI by making negative values to 0
    PPMI[np.isinf(PPMI)] = 0.0
    PPMI[np.isneginf(PPMI)] = 0.0
    PPMI[PPMI<0.0] = 0.0
    
    return PPMI



def sdae(PPMI, hidden_neurons):
    
    inp = Input(shape=(PPMI.shape[1],))
    enc = noise.GaussianNoise(0.5)(inp)
    
    for neurons in hidden_neurons:
        enc = Dense(neurons, activation = 'relu')(enc)
    
    dec = Dense(hidden_neurons[-2], activation = 'relu')(enc)
    for neurons in hidden_neurons[:-3][::-1]:
        dec = Dense(neurons, activation = 'relu')(dec)
    dec = Dense(PPMI.shape[1], activation='relu')(dec)
    
    auto_enc = Model(inputs=inp, outputs=dec)
    auto_enc.compile(optimizer='adam', loss='mse')
    
    auto_enc.fit(x=PPMI, y=PPMI, batch_size=10, epochs=5)
    
    encoder = Model(inputs=inp, outputs=enc)
    encoder.compile(optimizer='adam', loss='mse')
    embeddings = encoder.predict(PPMI)
    
    return embeddings



def process(args):
    
    num_hops = args.hops
    alpha = args.alpha
    hidden_neurons = args.hidden_neurons
    
    if num_hops < 1:
        sys.exit("DNGR: error: argument --hops: Max hops should be a positive natural number")
        
    if alpha < 0.0 or alpha > 1.0:
        sys.exit("DNGR: error: argument --alpha: Alpha's range is 0-1")
    
    if len(hidden_neurons) < 3:
        sys.exit("DNGR: error: argument --hidden_neurons: Need a minimum of 3 hidden layers")
    
    #Read groups
    text_corpus, file_names, target = ut.read_data()
    
    #Compute Cosine Similarity Matrix. This acts as Adjacency matrix for the graph.
    cosine_sim_matrix = ut.get_cosine_sim_matrix(text_corpus)
    
    #Stage 1 - Compute Transition Matrix A by random surfing model
    A = random_surf(cosine_sim_matrix, num_hops, alpha)
    
    #Stage 2 - Compute PPMI matrix 
    PPMI = PPMI_matrix(A)

    #Stage 3 - Generate Embeddings using Auto-Encoder
    embeddings = sdae(PPMI, hidden_neurons)

    #Evaluation 
    ut.compute_metrics(embeddings, target)

    #Visualize embeddings using t-SNE
    ut.visualize_TSNE(embeddings, target)
    plt.show()
    return



def main():
    
    parser = ArgumentParser('DNGR',description="This is a Keras implementaion of DNGR evaluating the 20NewsGroup dataset.")
    
    parser.add_argument('--hops', default=5, type=int, 
                       help='Maximum number of hops for Transition Matrix in Random surfing')

    parser.add_argument('--alpha', default=0.98,
                       help='Probability of (alpha) that surfing will go to next node, probability of (1-alpha) that surfing  will return to original vertex and restart. Range 0-1')
    
    parser.add_argument('--hidden_neurons', default=[512,256,128], type=int, nargs = '+',
                       help='Eg: \'512 256 128\' or \'256 128 64 32\'.  Number of hidden neurons in auto-encoder layers. Make sure there are 3 or more layers')
    
    warnings.filterwarnings("ignore")
    args = parser.parse_args()
    
    process(args)




if __name__ == '__main__':
    main()

