from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


class NLPModel(object):

    def __init__(self):
       
        self.clf = MultinomialNB()
        self.vectorizer = TfidfVectorizer(lowercase=False)

    def vectorizer_fit(self, X):
    
        self.vectorizer.fit(X)

    def vectorizer_transform(self, X):
        
        X_transformed = self.vectorizer.transform(X)
        return X_transformed

    def train(self, X, y):

        self.clf.fit(X, y)

    def predict_proba(self, X):

        y_proba = self.clf.predict_proba(X)
        return y_proba[:, 1]

    def predict(self, X):

        y_pred = self.clf.predict(X)
        return y_pred

    def pickle_vectorizer(self, path='models/TFIDFVectorizer.pkl'):

        with open(path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
            print("Pickled vectorizer at {}".format(path))

    def pickle_clf(self, path='models/Classifier.pkl'):
      
        with open(path, 'wb') as f:
            pickle.dump(self.clf, f)
            print("Pickled classifier at {}".format(path))


