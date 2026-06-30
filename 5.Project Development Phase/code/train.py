import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans


def train_and_evaluate():

    # Load Dataset
    dataset_path = "data/Crop_recommendation.csv"

    if not os.path.exists(dataset_path):
        print("Dataset not found!")
        return

    df = pd.read_csv(dataset_path)

    print("Dataset loaded successfully!")

    
    # Missing Value Handling
    for col in df.columns[:-1]:
        df[col] = df[col].fillna(df[col].median())

    # Features & Labels
    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Feature Scaling
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1500, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    }

    best_model = None
    best_model_name = ""
    best_accuracy = 0
    requires_scaling = False

    print("\n========== MODEL COMPARISON ==========\n")

    for name, model in models.items():

        if name in ["Logistic Regression", "K-Nearest Neighbors"]:

            model.fit(X_train_scaled, y_train)
            predictions = model.predict(X_test_scaled)
            scaling = True

        else:

            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            scaling = False

        accuracy = accuracy_score(y_test, predictions)

        print(f"{name}: {accuracy*100:.2f}%")

        if accuracy > best_accuracy:

            best_accuracy = accuracy
            best_model = model
            best_model_name = name
            requires_scaling = scaling

    print("\nSelected Model :", best_model_name)
    print(f"Accuracy : {best_accuracy*100:.2f}%")

    # Evaluation
    if requires_scaling:
        best_predictions = best_model.predict(X_test_scaled)
    else:
        best_predictions = best_model.predict(X_test)

    print("\nClassification Report\n")
    print(classification_report(y_test, best_predictions))
          
    # KMeans (Optional)
    kmeans = KMeans(
        n_clusters=5,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_train_scaled)

    print("\nK-Means clustering completed.")

    # Save Files
    os.makedirs("models", exist_ok=True)

    with open("models/crop_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open("models/model_info.pkl", "wb") as f:
        pickle.dump(
            {
                "model_name": best_model_name,
                "requires_scaling": requires_scaling
            },
            f
        )

    print("\nFiles Saved Successfully")
    print("crop_model.pkl")
    print("scaler.pkl")
    print("model_info.pkl")


if __name__ == "__main__":
    train_and_evaluate()