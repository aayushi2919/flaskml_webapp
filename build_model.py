from model import NLPModel
import pandas as pd
from sklearn.model_selection import train_test_split


def build_model():
    model = NLPModel()


    data = pd.read_csv('extract_combined.csv')
    data2 = pd.read_csv('labels.csv', error_bad_lines = False)
    merged= pd.merge(data,data2)
    yn = {'Yes': 1,'No': 0} 
   
    merged.is_fitara = [yn[i] for i in merged.is_fitara] 


    model.vectorizer_fit(data.loc[:, 'text'])
    print('Vectorizer fit complete')

    X = model.vectorizer_transform(data.loc[:, 'text'])
    print('Vectorizer transform complete')
    y = merged.loc[:, 'is_fitara']

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model.train(X_train, y_train)
    print('Model training complete')

    model.pickle_clf()
    model.pickle_vectorizer()

    


if __name__ == "__main__":
    build_model()
