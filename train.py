import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score # Accuracy metrics 
import pickle

def train(csvName = "coords.csv", modelName = "model.pkl"):
    df = pd.read_csv(csvName)
    x = df.drop("class", axis=1)
    y = df["class"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1234)

    pipelines = {
        'lr':make_pipeline(StandardScaler(), LogisticRegression()),
        'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
        'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
        'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
    }

    fit_models = {}
    for algo, pipeline in pipelines.items():
        model = pipeline.fit(x_train.values, y_train.values)
        fit_models[algo] = model

    fit_models['rc'].predict(x_test)

    for algo, model in fit_models.items():
        yhat = model.predict(x_test)
        print(algo, accuracy_score(y_test, yhat))

    fit_models['rf'].predict(x_test)

    with open(modelName, 'wb') as f:
        pickle.dump(fit_models['rf'], f)

if __name__ == "__main__":
    try:
        train('coordsl.csv','modell.pkl')
    except:
        print("Please Provide coordsl.csv")
    try:
        train('coordsr.csv','modelr.pkl')
    except:
        print("Please Provide coordsr.csv")