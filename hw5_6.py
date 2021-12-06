"""
Author： YuYue Yang
Date：2021/12/02~
Assignment 5 Kaggle score 0.43413
Achieve PRF(1.Other's method used to reference、2. twice BM25(by myself))
hw5_6.py、twice BM25、top 7 file be new Query、with stopwords
DataSet：20000 Documents, 50 Queries(.txt)
Reference：
https://www.kaggle.com/jerrykuo7727/rocchio/notebook
https://gist.github.com/koreyou/f3a8a0470d32aa56b32f198f49a9f2b8
Running time: 0.03 hours
"""
import math

import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Variables
# Config
root_path = 'D:/Python/NLP_DataSet/q_100_d_20000_random/'  # Relative path of homework data

# TF-IDF
max_df = 0.95  # Ignore words with high df. (Similar effect to stopword filtering)
min_df = 5  # Ignore words with low df.
smooth_idf = True  # Smooth idf weights by adding 1 to df.
sublinear_tf = True  # Replace tf with 1 + log(tf).

# Rocchio (Below is a param set called Ide Dec-Hi)
alpha = 0.99
beta = 0.75
gamma = 0.15
rel_count = 7  # Use top-5 relevant documents to update query vector.
nrel_count = 1  # Use only the most non-relevant document to update query vector.
iters = 7

from scipy import sparse


class BM25(object):
    def __init__(self, b=0.85, k1=1.8):
        self.vectorizer = TfidfVectorizer(norm=None, max_df=max_df, min_df=min_df, smooth_idf=smooth_idf, sublinear_tf=sublinear_tf)
        self.b = b
        self.k1 = k1
        self.Doc_IDFList = None
        self.sortIDF = None
        self.BM25_TFvec = None
        self.dlen = None
        self.avgLength = None
        self.featurelist = None
        # self.rankings = None

    # Build IDF list
    def Cale_TFandIDF(self, newDoc):
        # Total number of files  -> int
        N = 20000
        Ni = {}
        self.Doc_IDFList = {}
        self.dlen = []
        for subDoc in newDoc:
            context = subDoc.split(' ')
            self.dlen.append(len(context))
            for word in set(context):
                Ni[word] = Ni.get(word, 0) + 1
        # IDF calculation
        for word, value in Ni.items():
            self.Doc_IDFList[word] = math.log10(1 + ((N - value + 0.5) / (value + 0.5)))
        self.sortIDF = sorted(self.Doc_IDFList.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        self.avgLength = sum(self.dlen) / 20000
        print("Doc_IDFList：" + str(len(self.Doc_IDFList)))
        print("BM25 IDF End")
        vectorizer = CountVectorizer(max_df=max_df, min_df=min_df)
        self.BM25_TFvec = vectorizer.fit_transform(newDoc).toarray()
        self.featurelist = vectorizer.get_feature_names()
        print("BM25 TF Size：" + str(len(self.BM25_TFvec)))
        print("Feature：" + str(len(vectorizer.get_feature_names())))
        print("BM25 TF End")

    def fit(self, X):
        """ Fit IDF to documents X """
        self.vectorizer.fit(X)
        y = super(TfidfVectorizer, self.vectorizer).transform(X)
        self.avdl = y.sum(1).mean()

    def transform(self, q, X):
        """ Calculate BM25 between query q and documents X """
        b, k1, avdl = self.b, self.k1, self.avdl

        # apply CountVectorizer
        X = super(TfidfVectorizer, self.vectorizer).transform(X)
        len_X = X.sum(1).A1
        q, = super(TfidfVectorizer, self.vectorizer).transform([q])
        assert sparse.isspmatrix_csr(q)

        # convert to csc for better column slicing
        X = X.tocsc()[:, q.indices]
        denom = X + (k1 * (1 - b + b * len_X / avdl))[:, None]
        # idf(t) = log [ n / df(t) ] + 1 in sklearn, so it need to be coneverted
        # to idf(t) = log [ n / df(t) ] with minus 1
        idf = self.vectorizer._tfidf.idf_[None, q.indices] - 1.
        numer = X.multiply(np.broadcast_to(idf, X.shape)) * (k1 + 1)
        return (numer / denom).sum(1).A1

    def transform2(self, q, X):
        """ Calculate BM25 between query q and documents X """
        b, k1, avdl = 0.85, 1.8, self.avdl

        # apply CountVectorizer
        X = super(TfidfVectorizer, self.vectorizer).transform(X)
        len_X = X.sum(1).A1
        q, = super(TfidfVectorizer, self.vectorizer).transform([q])
        assert sparse.isspmatrix_csr(q)

        # convert to csc for better column slicing
        X = X.tocsc()[:, q.indices]
        denom = X + (k1 * (1 - b + b * len_X / avdl))[:, None]
        # idf(t) = log [ n / df(t) ] + 1 in sklearn, so it need to be coneverted
        # to idf(t) = log [ n / df(t) ] with minus 1
        idf = self.vectorizer._tfidf.idf_[None, q.indices] - 1.
        numer = X.multiply(np.broadcast_to(idf, X.shape)) * (k1 + 1)
        return (numer / denom).sum(1).A1


def read_filelist():
    with open(root_path + 'doc_list.txt') as file:
        doc_list = [line.rstrip() for line in file]
    with open(root_path + 'query_list.txt') as file:
        query_list = [line.rstrip() for line in file]


# Process files into string list
def process_file(doc_list, query_list):
    print("In process_file step---")
    for doc_name in doc_list:
        with open(root_path + 'docs/' + doc_name + '.txt') as file:
            segDoc = file.read().split(' ')
            doc = ""
            for word in segDoc:
                if word not in stopwords and word.isalpha():
                    doc += (word + " ")
            # print(doc)
            documents.append(doc)
    print("document size：" + str(len(documents)))
    for query_name in query_list:
        with open(root_path + 'queries/' + query_name + '.txt') as file:
            query = ' '.join(word for word in file.read().split(' '))
            # query = file.read().split(' ')
            queries.append(query)


def TFIDF_vector():
    print("In TFIDF_vector step---")
    global doc_tfidfs, query_vecs
    global rankings
    # Build TF-IDF vectors of docs and queries
    vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df,
                                 smooth_idf=smooth_idf, sublinear_tf=sublinear_tf)
    doc_tfidfs = vectorizer.fit_transform(documents).toarray()
    print(vectorizer.get_feature_names())
    query_vecs = vectorizer.transform(queries).toarray()
    # Rank documents based on cosine similarity
    cos_sim = cosine_similarity(query_vecs, doc_tfidfs)  # 20000 * 100
    rankings = np.flip(cos_sim.argsort(), axis=1)  # 20000 * 100


# Rocchio Algorithm
def RocchioALG():
    print("In RocchioALG step---")
    global rankings, query_vecs
    for _ in range(iters):
        # Update query vectors with Rocchio algorithm
        rel_vecs = doc_tfidfs[rankings[:, :rel_count]].mean(axis=1)
        nrel_vecs = doc_tfidfs[rankings[:, -nrel_count:]].mean(axis=1)
        query_vecs = alpha * query_vecs + beta * rel_vecs - gamma * nrel_vecs

        # Rerank documents based on cosine similarity
        cos_sim = cosine_similarity(query_vecs, doc_tfidfs)
        rankings = np.flip(cos_sim.argsort(axis=1), axis=1)
    # print(cos_sim)


def write2res():
    with open('D:/Python/NLP_DataSet/q_100_d_20000_random/result_5_6T.txt', mode='w') as file:
        file.write('Query,RetrievedDocuments\n')
        for query_name, ranking in zip(query_list, rankings):
            count = 0
            ranked_docs = ""
            for idx in ranking:
                ranked_docs += (doc_list[idx] + " ")
                count += 1
                if count >= 999:
                    break
            file.write(query_name.replace('.txt', '') + ',')
            file.write('%s\n' % ranked_docs)


def pre_write2res():
    write_path = 'D:/Python/NLP_DataSet/q_100_d_20000_random/result_5_63.txt'
    f = open(write_path, 'w')
    f.write("Query,RetrievedDocuments" + '\n')
    f.close()


def write2res_(quefile, sort_rank):
    write_path = 'D:/Python/NLP_DataSet/q_100_d_20000_random/result_5_63.txt'
    f = open(write_path, 'a')
    f.write(quefile.replace('.txt', '') + ',')
    for i in range(0, 999):
        f.write(doc_list[sort_rank[i]].replace('.txt', ''))
        if i <= 998:
            f.write(" ")
    f.write('\n')
    f.close()


if __name__ == '__main__':
    global stopwords
    global doc_list, query_list
    global documents, queries
    print(str(datetime.now().strftime("%Y-%m-%d %I:%M:%S")))
    Q_string = 'D:/Python/NLP_DataSet/q_100_d_20000_random/queries/'
    D_string = 'D:/Python/NLP_DataSet/q_100_d_20000_random/docs/'
    stopwords = ['by', 'come', 'did', 'of', 'go', 'the', 'in', 'just', 'to', 'that', 'for', 'is', 'it', 'wa', 'a',
                 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
                 'v', 'w', 'x', 'y', 'z', 'and',
                 'be', 'he', 'are', 'on', 'or', 'you', 'we', 'have', 'with', 'had', 'mr', 'vw', 'at', 'as', 'mar', 'ro',
                 'ha', 'thi', 'her', 'she', 'not', 'hi', 'imo', 'been', 'who', 'one', 'if', 'un', 'up', 'were', 'no',
                 'us', 'by', 'an', 'but', 'use', 'from', 'their', 'they', 'will', 'said', 'which']
    with open(root_path + 'doc_list.txt') as file:
        doc_list = [line.rstrip() for line in file]
    with open(root_path + 'query_list.txt') as file:
        query_list = [line.rstrip() for line in file]
    documents, queries = [], []

    read_filelist()
    process_file(doc_list, query_list)
    TFIDF_vector()
    RocchioALG()
    write2res()

    # ----------------------------------BM25
    pre_write2res()
    bm25_Doc = BM25()
    bm25_Doc.fit(documents[0:])
    for i in range(0, len(query_list)):
        first_scList = bm25_Doc.transform(queries[i], documents)
        rank = sorted(range(len(first_scList)), reverse=True, key=lambda x: first_scList[x])
        relQuerycontext = ""
        for j in range(0, rel_count):
            # print(doc_list[rank[j]])
            relQuerycontext += (documents[doc_list.index(doc_list[rank[j]])]) + " "
        second_scLis = bm25_Doc.transform2(relQuerycontext.strip(), documents)
        rank = sorted(range(len(second_scLis)), reverse=True, key=lambda x: first_scList[x])
        write2res_(query_list[i], rank)
        print(str(query_list[i]) + " -> " + str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
