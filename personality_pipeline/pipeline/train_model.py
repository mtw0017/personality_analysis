from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

def train_model(df, model_path="model/personality_rf.pkl"):
    X = df.drop('Personality', axis=1)
    y = df['Personality']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    joblib.dump(clf, model_path)
    return clf
