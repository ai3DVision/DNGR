
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


# In[6]:


import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn import metrics


# In[2]:


#newsgroups_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'),categories=categories)
#Try removing headers, footers and quotes because classifiers tend to overfit and learn only those parts. Remove them
#and let it try learning from body

def read_data():
    
    NG3 = ['comp.graphics','rec.sport.baseball','talk.politics.guns']
    text_corpus = []
    file_names = []
    target = np.arange(0,len(NG3)).tolist()*200
    target.sort()
    for i,category in enumerate(NG3):
        np.random.seed(i+42)
        news = fetch_20newsgroups(subset='train',categories=[category])
        permutation = np.arange(len(news.data)).tolist()
        np.random.shuffle(permutation)
        permutation = random.sample(permutation,200)
        randomtext_200 = np.asarray(news.data)[permutation]
        files_200 = news.filenames[permutation]
        text_corpus = text_corpus + randomtext_200.tolist()
        file_names = file_names + files_200.tolist()
        
    return text_corpus, file_names, target


# In[3]:


def get_cosine_sim_matrix(text_corpus):
    
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(text_corpus)
    cosine_sim_matrix = cosine_similarity(vectors)
    
    return cosine_sim_matrix


# In[ ]:


def scale_sim_matrix(mat):
    #Row-wise sacling of matrix
    mat = mat - np.diag(np.diag(mat)) #Make diag elements zero
    D_inv = np.diag(np.reciprocal(np.sum(mat,axis=0)))
    mat = np.dot(D_inv, mat)
    return mat


# In[7]:


def compute_metrics(embeddings, target):
    
    clf = MultinomialNB(alpha=0.1)
    clf.fit(embeddings, target)
    preds = clf.predict(embeddings)
    f1 = metrics.f1_score(target, preds, average='macro')
    nmi = normalized_mutual_info_score(target,preds)
    
    print("\nEvaluated embeddings using Multinomial Naive Bayes")
    print("F1 - score(Macro) : ",f1)
    print("NMI : ",nmi)
    
    return


# In[ ]:


def visualize_TSNE(embeddings,target):
    tsne = TSNE(n_components=2, init='pca',
                         random_state=0, perplexity=30)
    data = tsne.fit_transform(embeddings)
    plt.figure(figsize=(12, 6))
    plt.title("TSNE visualization of the embeddings")
    plt.scatter(data[:,0],data[:,1],c=target)

    return

