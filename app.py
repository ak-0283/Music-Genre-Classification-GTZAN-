import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.title("🎵 Music Genre Classification")

# Upload Dataset
uploaded_file = st.file_uploader("Upload CSV", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success(f"✅ Dataset Loaded: {data.shape[0]} rows, {data.shape[1]} columns")

    # Preprocess
    X = data.drop(['label', 'filename'], axis=1)
    y = data['label']

    # Encode labels
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train Model
    @st.cache_resource
    def train_model(X_train, y_train):
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        return model

    model = train_model(X_train, y_train)

    # Model Evaluation
    if st.checkbox("Show Model Evaluation"):
        y_test_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_test_pred) * 100
        st.write(f"📊 **Testing Accuracy:** {test_acc:.2f}%")
        
        st.text("Classification Report (Test Data):")
        st.text(classification_report(y_test, y_test_pred, target_names=encoder.classes_))

        # Confusion Matrix
        st.write("Confusion Matrix:")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(confusion_matrix(y_test, y_test_pred),
                    annot=True, fmt="d",
                    xticklabels=encoder.classes_,
                    yticklabels=encoder.classes_,
                    cmap="Blues", ax=ax)
        st.pyplot(fig)

    # -------------------------------
    # Sample Prediction Section (Main Page)
    st.header("Test a Sample from Test Set")
    sample_index = st.number_input(
        "Pick Sample Index", min_value=0, max_value=len(X_test)-1, value=0
    )

    if st.button("Predict Sample Genre"):
        sample = X_test[sample_index].reshape(1, -1)
        pred_genre = encoder.inverse_transform(model.predict(sample))
        st.markdown("### 🎵 Predicted Genre")
        st.success(f"**{pred_genre[0]}**")

else:
    st.warning("⚠️ Please upload the features CSV to start.")