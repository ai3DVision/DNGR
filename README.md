# DNGR : Deep Neural Networks for Graph Representations


*This is a python implementation of DNGR model using TensorFlow*


The DNGR model uses the following steps to compute node embeddings for a given graph


1. Use Random Surfing Model to construct transition matricies for all vertices

2. Construct PPMI matrices for each transition matrix

3. Feed them to Stacked Denoising Autoencoder to learn non-linear relationships and generate embeddings


**Implementation under progress**


*03/09/17 - Problem Conceptualization*
- Developed a high-level plan for execution for each step. 
