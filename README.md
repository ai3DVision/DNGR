# DNGR : Deep Neural Networks for Graph Representations


*This is a python implementation of DNGR model using TensorFlow*


The DNGR model uses the following steps to compute node embeddings for a given graph


1. Use Random Surfing Model to construct transition matricies for all vertices

2. Construct PPMI matrices for each transition matrix

3. Feed them to Stacked Denoising Autoencoder to learn non-linear relationships and generate embeddings

Dataset chosen: 20NewsGroup

It can be downloaded [here](https://archive.ics.uci.edu/ml/datasets/Twenty+Newsgroups) or loaded directly using [sklearn](http://scikit-learn.org/stable/datasets/twenty_newsgroups.html)



**Implementation under progress**


*03/09/17 - Problem Conceptualization*
- Developed a high-level plan for execution for each step. 

*03/09/17 - Graph construction*
- Coded helper functions to create the graph adjacency matrix.

*03/13/17 - PPMI, auto-encoder, TSNE*
- Computed PPMI matrix, node embedding generation using auto-encoder and visualization of embedding using TSNE

References:

[DNGR: Deep Neural Network for Graphical Representations](https://pdfs.semanticscholar.org/1a37/f07606d60df365d74752857e8ce909f700b3.pdf)
