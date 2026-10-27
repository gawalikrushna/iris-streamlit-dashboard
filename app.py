import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# --- Page Configuration ---
st.set_page_config(
    page_title="Iris Species Classifier",
    page_icon="🌸",
    layout="wide"
)

# --- Title and Header ---
st.title("🌸 Iris Flower Species Classifier")
st.write(
    "An interactive Machine Learning dashboard built with **Streamlit** and **Decision Tree Classifier**."
)

# --- Helper Function: Load Dataset ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("iris.csv")
    except FileNotFoundError:
        # Fallback to sklearn load_iris if iris.csv is missing locally
        from sklearn.datasets import load_iris
        raw_data = load_iris()
        df = pd.DataFrame(raw_data.data, columns=["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"])
        df['species'] = [raw_data.target_names[i] for i in raw_data.target]
    return df

df = load_data()

# Features and Target
feature_cols = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
]
X = df[feature_cols]
Y = df["species"]

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Model Parameters")
max_depth = st.sidebar.slider("Decision Tree Max Depth:", min_value=1, max_value=10, value=3, step=1)
test_size = st.sidebar.slider("Test Set Size (%):", min_value=10, max_value=50, value=20, step=5) / 100

# --- Train Model ---
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=test_size, random_state=42, stratify=Y
)

model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)

# --- Tabs Navigation ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Real-time Prediction", 
    "📊 EDA & Data Exploration", 
    "📈 Model Performance", 
    "🌳 Decision Tree Visualization"
])

# ==========================================
# TAB 1: Real-time Prediction
# ==========================================
with tab1:
    st.subheader("📥 Enter Flower Measurements")
    st.write("Adjust the sliders below to predict the species of an Iris flower:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        sepal_length = st.slider("Sepal Length (cm)", float(df["sepal length (cm)"].min()), float(df["sepal length (cm)"].max()), 5.8, 0.1)
        sepal_width = st.slider("Sepal Width (cm)", float(df["sepal width (cm)"].min()), float(df["sepal width (cm)"].max()), 3.0, 0.1)
        
    with col2:
        petal_length = st.slider("Petal Length (cm)", float(df["petal length (cm)"].min()), float(df["petal length (cm)"].max()), 4.3, 0.1)
        petal_width = st.slider("Petal Width (cm)", float(df["petal width (cm)"].min()), float(df["petal width (cm)"].max()), 1.3, 0.1)

    if st.button("Predict Species 🚀", type="primary", use_container_width=True):
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        
        st.divider()
        st.success(f"🌺 **Predicted Species:** `{prediction.capitalize()}`")
        
        # Display Prediction Probabilities
        st.write("**Prediction Probabilities:**")
        prob_df = pd.DataFrame({
            "Species": model.classes_,
            "Probability": [f"{p * 100:.2f}%" for p in prediction_proba]
        })
        st.dataframe(prob_df, use_container_width=True)

# ==========================================
# TAB 2: Exploratory Data Analysis (EDA)
# ==========================================
with tab2:
    st.subheader("🔍 Exploratory Data Analysis")
    
    col_a, col_b = st.columns([1, 1])
    
    with col_a:
        st.write("**Dataset Overview:**")
        st.dataframe(df.head(10), use_container_width=True)
        st.write(f"**Total Samples:** {df.shape[0]} | **Columns:** {df.shape[1]}")
        
    with col_b:
        st.write("**Statistical Summary:**")
        st.dataframe(df.describe().T, use_container_width=True)

    st.divider()
    st.subheader("🎨 Feature Scatter Plot")
    
    col_x, col_y = st.columns(2)
    with col_x:
        x_axis = st.selectbox("Select X-axis Feature:", feature_cols, index=0)
    with col_y:
        y_axis = st.selectbox("Select Y-axis Feature:", feature_cols, index=1)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(data=df, x=x_axis, y=y_axis, hue="species", palette="Set1", s=70, ax=ax)
    plt.title(f"{x_axis} vs {y_axis}")
    plt.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig)

# ==========================================
# TAB 3: Model Performance
# ==========================================
with tab3:
    st.subheader("🎯 Model Evaluation Metrics")
    
    st.metric(label="Model Accuracy", value=f"{accuracy * 100:.2f}%")
    
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        st.write("**Confusion Matrix:**")
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(Y_test, Y_pred, ax=ax_cm, cmap="Blues", colorbar=False)
        st.pyplot(fig_cm)
        
    with col_m2:
        st.write("**Classification Report:**")
        report_dict = classification_report(Y_test, Y_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        st.dataframe(report_df.style.highlight_max(axis=0), use_container_width=True)

# ==========================================
# TAB 4: Decision Tree Visualization
# ==========================================
with tab4:
    st.subheader("🌳 Decision Tree Rules")
    
    fig_tree, ax_tree = plt.subplots(figsize=(14, 8))
    plot_tree(
        model, 
        feature_names=feature_cols, 
        class_names=model.classes_, 
        filled=True, 
        rounded=True,
        ax=ax_tree
    )
    st.pyplot(fig_tree)