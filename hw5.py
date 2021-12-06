"""
Author： YuYue Yang
Date：2021/12/02~
Assignment 5 Kaggle score 0.36862
Achieve PRF
Reference：https://github.com/ruisizhang123/Pseudo-Relevance-Feedback/blob/master/EE448.ipynb
with stopwords
DataSet：20000 Documents, 50 Queries(.txt)
Running time:  hours
"""
# -*- coding: utf-8 -*-
import math
import os
from collections import defaultdict
from datetime import datetime

from math import sqrt


# ------------------------------------Score Start------------------------------------
def read_file(file):
    FileList = os.listdir(file)
    output = defaultdict(list)
    for quefile in FileList:
        document = open(file + quefile)
        segDoc = document.read().split(' ')
        for word in set(segDoc):
            if word not in Lexicon and word not in stopwords:
                Lexicon.append(word)
            output[quefile].append(word)
    # print(output)
    print("Lex Size：" + str(len(Lexicon)))
    return output


def read_file2(file):
    FileList = os.listdir(file)
    output = defaultdict(list)
    for docfile in FileList:
        document = open(file + docfile)
        segDoc = document.read().split(' ')
        for word in set(segDoc):
            if word in Lexicon and word not in stopwords:
                output[docfile].append(word)
    # print(output)
    print("Doc Size：" + str(len(output)))
    return output


def example_list2dict(input):
    output = dict()
    for word in input:
        if output.get(word) is None:
            output[word] = 0
        output[word] += 1
    return output


def cal_idf(doc_dict):
    doc_num = len(doc_dict)
    idf = dict()
    for doc_id in doc_dict:
        doc_text = list(set(doc_dict[doc_id]))
        for word in doc_text:
            if idf.get(word) is None:
                idf[word] = 0
            idf[word] += 1
    for word in idf:
        idf[word] = math.log(1 + ((doc_num - idf[word] + 0.5) / (idf[word] + 0.5)))
    print("IDF Size：" + str(len(idf)))
    return idf


def bm25(query, doc, idf, avg_doc_len=374):
    k1 = 1.8
    k2 = 1
    b = 0.845
    score = 0.0
    avg_doc_len = avg_length
    for word in query:
        if doc.get(word) == None:
            continue
        W_i = idf[word]
        f_i = doc[word]
        qf_i = query[word]
        doc_len = sum(doc.values())
        K = k1 * (1 - b + b * doc_len / avg_doc_len)
        R1 = f_i * (k1 + 1) / (f_i + K)
        R2 = qf_i * (k2 + 1) / (qf_i + k2)
        R = R1 * R2
        score += W_i * R
    return score


def GetScore(query, doc_name, doc_dict, idf):
    query = example_list2dict(query)
    doc = example_list2dict(doc_dict[doc_name])
    score = bm25(query, doc, idf)
    return score


# ------------------------------------Score End------------------------------------
ALPHA = 1
BETA = 0.75
GAMMA = 0.15


# ------------------------------------Util Start------------------------------------
def generateInvertedIndex():
    global avg_length
    invertedIndex = {}
    # tokenDict = {}
    dlen = 0
    FileList = os.listdir(D_string)
    for docfile in FileList:
        document = open(D_string + docfile)
        doc_text = document.read().split(' ')
        length = len(doc_text)
        # tokenDict[docfile] = length
        dlen += length
        for word in doc_text:
            if word.isalpha() and word not in stopwords:
                if word not in invertedIndex.keys():
                    docIDCount = {docfile: 1}
                    invertedIndex[word] = docIDCount
                elif docfile in invertedIndex[word].keys():
                    invertedIndex[word][docfile] += 1
                else:
                    docIDCount = {docfile: 1}
                    invertedIndex[word].update(docIDCount)

    print("IIx Size：" + str(len(invertedIndex)))
    avg_length = dlen / len(FileList)
    # print(invertedIndex)
    return invertedIndex


def queryFrequency(query, invertedIndex):
    queryFreq = {}
    for term in query:
        if term in queryFreq.keys():
            queryFreq[term] += 1
        else:
            queryFreq[term] = 1
    for term in invertedIndex:
        if term not in queryFreq.keys():
            queryFreq[term] = 0
    # print(queryFreq)
    return queryFreq


def calculateDocsCount(doc, docIndex):
    # doc_dict = open('D:/Python/NLP_DataSet/q_100_d_20000_random/docs/')
    FileList = os.listdir(D_string)
    for docfile in FileList:
        if docfile == doc:
            document = open(D_string + docfile)
            segDoc = document.read().split(' ')
            for term in segDoc:
                if term in docIndex.keys():
                    docIndex[term] += 1
                else:
                    docIndex[term] = 1
    return docIndex


def findDocs(k, sortedBM25Score, invertedIndex, relevancy):
    relIndex = {}
    nonRelIndex = {}
    if relevancy == "Relevant":
        for i in range(0, k):
            doc, doc_score = sortedBM25Score[i]
            relIndex = calculateDocsCount(doc, relIndex)
        for term in invertedIndex:
            if term not in relIndex.keys():
                relIndex[term] = 0
        return relIndex
    elif relevancy == "Non-Relevant":
        for i in range(k + 1, len(sortedBM25Score)):
            doc, doc_score = sortedBM25Score[i]
            nonRelIndex = calculateDocsCount(doc, nonRelIndex)
        for term in invertedIndex:
            if term not in nonRelIndex.keys():
                nonRelIndex[term] = 0
        return nonRelIndex


def findRelDocMagnitude(docIndex):
    mag = 0
    for term in docIndex:
        mag += float(docIndex[term] ** 2)
        mag = float(sqrt(mag))
    return mag


def findNonRelDocMagnitude(docIndex):
    mag = 0
    for term in docIndex:
        mag += float(docIndex[term] ** 2)
    mag = float(sqrt(mag))
    return mag


def findRocchioScore(term, queryFreq, relDocMag, relIndex, nonRelMag, nonRelIndex):
    Q1 = ALPHA * queryFreq[term]
    Q2 = (BETA / relDocMag) * relIndex[term]
    Q3 = (GAMMA / nonRelMag) * nonRelIndex[term]
    rocchioScore = ALPHA * queryFreq[term] + (BETA / relDocMag) * relIndex[term] - (GAMMA / nonRelMag) * nonRelIndex[
        term]
    return rocchioScore


def findNewQuery(query, k, sortedBM25Score, invertedIndex):
    queryFreq = queryFrequency(query, invertedIndex)
    relIndex = findDocs(k, sortedBM25Score, invertedIndex, "Relevant")
    relDocMag = findRelDocMagnitude(relIndex)
    nonRelIndex = findDocs(k, sortedBM25Score, invertedIndex, "Non-Relevant")
    nonRelMag = findNonRelDocMagnitude(nonRelIndex)
    updatedQuery = {}
    newQuery = query
    for term in invertedIndex:
        updatedQuery[term] = findRocchioScore(term, queryFreq, relDocMag, relIndex, nonRelMag, nonRelIndex)
    sortedUpdatedQuery = sorted(updatedQuery.items(), key=lambda x: x[1], reverse=True)
    if len(sortedUpdatedQuery) < 5:
        loopRange = len(sortedUpdatedQuery)
    else:
        loopRange = 5
    for i in range(loopRange):
        term, frequency = sortedUpdatedQuery[i]
        # print("term, frequency", term, frequency)
        if term not in query:
            newQuery.append(term)
    return newQuery


# invertedIndex = generateInvertedIndex()
# print(invertedIndex)

def getReduceIndex(query, invertedIndex):
    query_term_freq = {}
    query_term_list = query.split()
    reduced_inverted_index = {}
    for term in query.split():
        if term in query_term_freq.keys():
            query_term_freq[term] += 1
        else:
            query_term_freq[term] = 1

    for term in query_term_freq:
        if term in invertedIndex:
            reduced_inverted_index.update({term: invertedIndex[term]})
        else:
            reduced_inverted_index.update({term: {}})
    return reduced_inverted_index


# ------------------------------------Util End------------------------------------
def pre_write2res():
    write_path = 'D:/Python/NLP_DataSet/q_100_d_20000_random/result_52.txt'
    f = open(write_path, 'a')
    f.write("Query,RetrievedDocuments" + '\n')
    f.close()


def write2res(quefile, sort_res):
    write_path = 'D:/Python/NLP_DataSet/q_100_d_20000_random/result_52.txt'
    f = open(write_path, 'a')
    f.write(quefile.replace('.txt', '') + ',')
    for i in range(0, 999):
        f.write(sort_res[i][0].replace('.txt', ''))
        if i <= 998:
            f.write(" ")
    f.write('\n')
    f.close()


if __name__ == '__main__':
    global Lexicon, avg_length
    global D_string, Q_string
    global stopwords
    Q_string = 'D:/Python/NLP_DataSet/q_100_d_20000_random/queries/'
    D_string = 'D:/Python/NLP_DataSet/q_100_d_20000_random/docs/'
    stopwords = ['by', 'come', 'did', 'of', 'go', 'the', 'in', 'just', 'to', 'that', 'for', 'is', 'it', 'wa', 'a',
                 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
                 'v', 'w', 'x', 'y', 'z', 'and',
                 'be', 'he', 'are', 'on', 'or', 'you', 'we', 'have', 'with', 'had', 'mr', 'vw', 'at', 'as', 'mar', 'ro',
                 'ha', 'thi', 'her', 'she', 'not', 'hi', 'imo', 'been', 'who', 'one', 'if', 'un', 'up', 'were', 'no',
                 'us', 'by', 'an', 'but', 'use', 'from', 'their', 'they', 'will', 'said', 'which']
    Lexicon = []
    query_dict = read_file(Q_string)
    doc_dict = read_file2(D_string)
    idf = cal_idf(doc_dict=doc_dict)
    invertedIndex = generateInvertedIndex()
    queries = Lexicon
    feedbackFlag = 1


    def pseudoRelevanceFeedbackScores(sortedBM25Score, query, queryID):
        global feedbackFlag
        feedbackFlag += 1
        k = 6
        # reducedIndex = getReduceIndex(query, invertedIndex)
        # print(reducedIndex)
        newQuery = findNewQuery(query, k, sortedBM25Score, invertedIndex)
        # print(query)
        print(newQuery)
        PseudoRelevanceScoreList = RankDoc(newQuery, queryID)
        return PseudoRelevanceScoreList


    def RankDoc(query, queryID):
        # queryID = query filename(ex:301.txt)
        BM25ScoreList = {}
        global feedbackFlag
        FileList = os.listdir(D_string)
        for docfile in FileList:
            docID = docfile
            BM25 = GetScore(query, docID, doc_dict, idf)
            BM25ScoreList[docID] = BM25
        sortedBM25Score = sorted(BM25ScoreList.items(), key=lambda x: x[1], reverse=True)
        # return sortedBM25Score
        if feedbackFlag == 1:
            return pseudoRelevanceFeedbackScores(sortedBM25Score, query, queryID)
        elif feedbackFlag == 2:
            feedbackFlag = 1
            return sortedBM25Score


    pre_write2res()
    FileList = os.listdir(Q_string)
    # for query in queries:
    for quefile in FileList:
        feedbackFlag = 1
        query = query_dict[str(quefile)]
        print(query)
        sortedScoreList = RankDoc(query, quefile)
        write2res(str(quefile), sortedScoreList)
        print(quefile + " -> " + str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
