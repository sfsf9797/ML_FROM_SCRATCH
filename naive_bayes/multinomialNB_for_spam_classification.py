import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict
from collections import Counter



from naive_bayes import multinomialNB

df = pd.read_csv('data\spam_ham_dataset.csv')

def feature_extractor(column):
    unique_word = set()
    column.str.lower().str.split().apply(unique_word.update)
    w2i = defaultdict(lambda: len(w2i))
    for word in unique_word:
        w2i[word]

    X = np.zeros((len(column),len(unique_word)+1))
    dic = column.str.split().apply(Counter).to_dict()
    for i, j in dic.items():
        for key,value in j.items():
            X[i,w2i[key]] = value

    return X , w2i

X , w2i = feature_extractor(df['text'])
y = df['label_num']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)


_multinomialNB = multinomialNB()
_multinomialNB.fit(X_train,y_train)

probs = _multinomialNB.predict(X_test)
y_pred=np.argmax(probs, 1)
print(f"Accuracy: {sum(y_pred==y_test)/X_test.shape[0]}")
