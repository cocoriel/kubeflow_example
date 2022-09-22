import argparse
import json
from io import StringIO

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_data(data):
    d = StringIO(data)
    iris = pd.read_csv(d, sep=',')
    print(iris.shape)
    print(iris.head(10))
    return iris


def get_train_test_data(iris):
    encode = LabelEncoder()
    iris.Species = encode.fit_transform(iris.Species)

    train, test = train_test_split(iris, test_size=0.2, random_state=0)
    print('shape of training data : ', train.shape)
    print('shape of testing data', test.shape)

    X_train = train.drop(columns=['Species'], axis=1)
    y_train = train['Species']
    X_test = test.drop(columns=['Species'], axis=1)
    y_test = test['Species']

    print("success split data")

    return X_train, X_test, y_train, y_test


def evaluation(y_test, predict):
    accuracy = accuracy_score(y_test, predict)
    print("accuracy: {}".format(accuracy))

    metrics = {
        "metrics": [{
            "name": "accuracy-score",
            "numberValue": accuracy,
            "format": "PERCENTAGE"
        }]
    }

    with open("./accuracy.json", "w") as f:
        json.dump(accuracy, f)
    with open("./mlpipeline-metrics.json", "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()

    argument_parser.add_argument(
        '--data',
        type=str,
        help="Input data csv"
    )

    args = argument_parser.parse_args()
    # iris = args.data
    # pd.set_option('display.max_columns', None)
    iris = load_data(args.data)

    X_train, X_test, y_train, y_test = get_train_test_data(iris)

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)
    predict = model.predict(X_test)
    evaluation(y_test, predict)
