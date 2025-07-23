import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from streamlit_chat import message
import matplotlib.pyplot as plt
import seaborn as sns

# Page setup
st.set_page_config(page_title="🤖 ML Chatbot", layout="centered")

# Initialize session states
if "step" not in st.session_state:
    st.session_state.step = 0
if "dataset" not in st.session_state:
    st.session_state.dataset = None
if "target_column" not in st.session_state:
    st.session_state.target_column = None
if "model" not in st.session_state:
    st.session_state.model = None
if "task" not in st.session_state:
    st.session_state.task = None
if "show_results" not in st.session_state:
    st.session_state.show_results = False
if "chat" not in st.session_state:
    st.session_state.chat = []

# Chat rendering
def bot(message_text):
    st.session_state.chat.append({"role": "bot", "content": message_text})
    message(message_text, is_user=False, key=f"bot_{len(st.session_state.chat)}")

def user(message_text):
    st.session_state.chat.append({"role": "user", "content": message_text})
    message(message_text, is_user=True, key=f"user_{len(st.session_state.chat)}")

# Title
st.title("🤖 ML Chatbot")

# Step 0: Upload CSV
if st.session_state.step == 0:
    bot("Hi! Please upload a CSV file so I can help you build a predictive model.")
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv"], key="file_upload")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.dataset = df
        user("Here is my dataset!")
        st.session_state.step = 1
        st.rerun()

# Step 1: Select target column
elif st.session_state.step == 1:
    bot("Here is a preview of your dataset:")
    st.dataframe(st.session_state.dataset.head(), use_container_width=True)

    bot("Which column would you like me to predict?")
    target_column = st.selectbox(
        "🎯 Select Target Column", 
        st.session_state.dataset.columns, 
        key="target_column_select"
    )

    if target_column:
        user(f"I want to predict '{target_column}'")
        st.session_state.target_column = target_column
        st.session_state.step = 2
        st.rerun()

# Step 2: Train model
elif st.session_state.step == 2:
    df = st.session_state.dataset
    target = st.session_state.target_column

    X = df.drop(columns=[target])
    y = df[target]

    # Encode categorical
    X = pd.get_dummies(X)

    # Task type check
    if y.nunique() <= 10 and y.dtype != 'float64':
        task = "classification"
        y = y.astype(str)
        model = RandomForestClassifier()
    else:
        task = "regression"
        model = RandomForestRegressor()

    st.session_state.task = task
    st.session_state.model = model

    user("Start training!")
    bot(f"Got it! This is a **{task}** problem. Training a Random Forest model...")

    # Train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Save results
    st.session_state.X_test = X_test
    st.session_state.y_test = y_test
    st.session_state.y_pred = y_pred
    st.session_state.show_results = True
    st.session_state.step = 3
    st.rerun()

# Step 3: Show results
elif st.session_state.step == 3 and st.session_state.show_results:
    task = st.session_state.task
    model = st.session_state.model
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    y_pred = st.session_state.y_pred

    bot("✅ Model training complete!")

    if task == "classification":
        acc = accuracy_score(y_test, y_pred)
        bot(f"📈 Model Accuracy: **{acc:.2f}**")
        st.text("📋 Classification Report:")
        st.text(classification_report(y_test, y_pred))
    else:
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        bot(f"📉 RMSE: **{rmse:.2f}**, R² Score: **{r2:.2f}**")

    # Feature importance
    bot("📌 Top Feature Importances:")
    importance_df = pd.DataFrame({
        'Feature': X_test.columns,
        'Importance': model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=importance_df.head(10), x="Importance", y="Feature", ax=ax)
    st.pyplot(fig)

    # Predictions
    bot("🔮 Here are some sample predictions:")
    result_df = pd.DataFrame({
        'Actual': y_test.values[:10],
        'Predicted': y_pred[:10]
    })
    st.dataframe(result_df)

    st.success("🎉 Done! Upload another file to try again.")
    if st.button("🔁 Restart", key="restart_btn"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Sidebar Chat History (safe keys)
st.sidebar.title("📝 Conversation")
for i, chat in enumerate(st.session_state.chat):
    message(chat["content"], is_user=(chat["role"] == "user"), key=f"{chat['role']}_{i}")
