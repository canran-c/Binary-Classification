import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 读取数据
def read_data(mode='Train'):
    df = pd.read_csv(f"{mode}.txt", delim_whitespace=True)
    if mode == 'Train':
        y = df["PregnancyStatus"]
        X = df.drop(columns=["PregnancyStatus"])
    else:
        X = df
        y = pd.read_csv("Label.txt", delim_whitespace=True)
        y = y['x']

    # 复制重要特征
    important_cols = X.columns[:6]
    for col in important_cols:
        X[f"{col}_dup1"] = X[col]
    return X,y


# 识别离散和连续特征
def recognize_features(X):
    discrete_features = []
    continuous_features = []

    for col in X.columns:
        unique_vals = X[col].nunique()

        if pd.api.types.is_integer_dtype(X[col]):
            discrete_features.append(col)
        else:
            continuous_features.append(col)

    return continuous_features, discrete_features

def main():
    X_train, y_train = read_data()
    X_test, y_test = read_data('Test')

    continuous_features, discrete_features = recognize_features(X_train)
    print(f"识别出的离散特征：{discrete_features}", len(discrete_features))
    print(f"识别出的连续特征：{continuous_features}", len(continuous_features))
    
    # preproces
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), discrete_features),
            ("num", StandardScaler(), continuous_features)
        ]
    )

    # pipeline
    clf = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42))
    ])

    # train
    clf.fit(X_train, y_train)

    # evaluate
    y_pred = clf.predict(X_test)
    print('predictions: \n', y_pred)
    print("分类报告：\n", classification_report(y_test, y_pred))

main()