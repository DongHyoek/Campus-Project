
### Imports
import pandas as pd
import numpy as np
import os


### Read data
train = pd.read_csv(os.path.abspath("../input")+"/X_train.csv" , encoding = 'cp949')
test = pd.read_csv(os.path.abspath("../input")+"/X_test.csv" , encoding = 'cp949')


### Make corpus
p_level = 'part_nm'  # 상품 분류 수준

# W2V 학습데이터가 부족하여 구매한 상품 목록으로부터 n배 oversampling을 수행
def oversample(x, n, seed=0):
    if n == 0:
        return list(x)
    uw = np.unique(x)
    bs = np.array([])
    np.random.seed(seed)
    for j in range(n):
        bs = np.append(bs, np.random.choice(uw, len(uw), replace=True))
    return list(bs)

train_corpus = list(train.groupby('custid')[p_level].agg(oversample, 20))
test_corpus = list(test.groupby('custid')[p_level].agg(oversample, 20))


### Training the Word2Vec model
num_features = 100 # 단어 벡터 차원 수
min_word_count = 1 # 최소 단어 수
context = 5 # 학습 윈도우(인접한 단어 리스트) 크기

# 초기화 및 모델 학습
from gensim.models import word2vec

# 모델 학습
w2v = word2vec.Word2Vec(train_corpus, 
                        vector_size=num_features, 
                        min_count=min_word_count,
                        window=context,
                        seed=0, workers=1)
# 필요없는 메모리 unload
w2v.init_sims(replace=True)


### Make features
# 구매상품에 해당하는 벡터의 평균/최소/최대 벡터를 feature로 만드는 전처리기
class EmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = num_features
    def fit(self, X):
        return self
    def transform(self, X):
        return np.array([
            np.hstack([
                np.max([self.word2vec[w] for w in words if w in self.word2vec] or [np.zeros(self.dim)], axis=0),
                np.min([self.word2vec[w] for w in words if w in self.word2vec] or [np.zeros(self.dim)], axis=0),
                np.mean([self.word2vec[w] for w in words if w in self.word2vec] or [np.zeros(self.dim)], axis=0),                
                #np.std([self.word2vec[w] for w in words if w in self.word2vec] or [np.zeros(self.dim)], axis=0)                
            ]) 
            for words in X
        ]) 

# W2V 기반 feature 생성
train_features = pd.DataFrame(EmbeddingVectorizer(w2v.wv).fit(train_corpus).transform(train_corpus))
test_features = pd.DataFrame(EmbeddingVectorizer(w2v.wv).transform(test_corpus))

train_features.columns = ['v'+f'{c+1:03d}' for c in train_features.columns]
test_features.columns = ['v'+f'{c+1:03d}' for c in test_features.columns]

# 학습용과 제출용 데이터로 분리
X_train_part = pd.concat([pd.DataFrame({'custid': np.sort(train['custid'].unique())}), train_features], axis=1)#.to_csv('X_train_buyer.csv', index=False)
X_test_part = pd.concat([pd.DataFrame({'custid': np.sort(test['custid'].unique())}), test_features], axis=1)#.to_csv('X_test_buyer.csv', index=False)
