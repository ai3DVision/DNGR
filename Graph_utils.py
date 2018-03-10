
# coding: utf-8

# In[ ]:


"""
Note: Following conventions from 
Tian, F.; Gao, B.; Cui, Q.; Chen, E.; and Liu, T.-Y. 2014. Learning deep representations for graph clustering.
They have 3 types of classifications, a 3 group, 6 group and a 9 group classification.
The 3 group classification has 200 artices sampled at random from each of the groups listed in NG3 below.
Each article is converted into a TF-IDF vector from the whole corpus. 
The graph construction is done by taking the TF-IDF vectors as nodes and the cosine similarity between them as edge weights.
"""


# In[125]:


import numpy as np
import networkx as nx
import random
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[184]:


#newsgroups_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'),categories=categories)
#Try removing headers, footers and quotes because classifiers tend to overfit and learn only those parts. Remove them
#and let it try learning from body

def read_data():
    
    NG3 = ['comp.graphics','rec.sport.baseball','talk.politics.guns']
    text_corpus = []
    file_names = []
    target = np.arange(0,len(NG3)).tolist()*200
    target.sort()
    permutation = np.arange(200)
    for i,category in enumerate(NG3):
        np.random.seed(i+42)
        np.random.shuffle(permutation)
        news = fetch_20newsgroups(subset='train',categories=[category])
        randomtext_200 = np.asarray(news.data)[permutation]
        files_200 = news.filenames[permutation]
        text_corpus = text_corpus + randomtext_200.tolist()
        file_names = file_names + files_200.tolist()
        
    return text_corpus, file_names, target


# In[146]:


def get_cosine_sim_matrix(text_corpus):
    
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(text_corpus)
    cosine_sim_matrix = cosine_similarity(vectors)
    
    return cosine_sim_matrix


# In[167]:


def get_edge_list(cosine_sim_matrix,file_names):
    
    edges = []
    for file,weights in zip(file_names,cosine_sim_matrix):
        [edges.append((file,file_names[i],w)) for i,w in enumerate(weights)]
    
    return edges 


# In[177]:


def remove_self_loops(G):
    
    loops = []
    
    #Get loops
    for i,j in G.edges():
        if i==j:
            loops.append((i,j))
    
    G.remove_edges_from(loops)
    return G

