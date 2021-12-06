# Pseudo-Relevance-Feedback
2021 IR學習_功課5，實做PRF\
Pseudo-Relevance-Feedback

# Run hw5.py
Have Remove stop words\
參考其他作者的\
是使用BM25計算cos，選前7篇score較高的，當作New Queries\
在計算RocchioScore -> sorted -> write in result.txt

# Run hw5_6.py
Have Remove stop words\
參考其他作者的\
此檔案有兩個方法(two way)
1. 用TFIDF概念，生成doc & query各自的vector -> cosin similarity -> sorted -> Rocchio Algorithm -> write2result
2. 使用BM25計算cos，選前7篇score較高的，當作New Queries，再用BM25 score -> sorted -> write in result.txt
