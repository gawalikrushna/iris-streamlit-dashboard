# 🌸 Iris Species Classifier Dashboard

An interactive Machine Learning web application built using **Streamlit** and **scikit-learn**. This app performs Exploratory Data Analysis (EDA), trains a Decision Tree Classifier, evaluates model performance, and provides real-time predictions for Iris flower species based on physical measurements.

---

## 🚀 Features

- **🔮 Real-Time Prediction:** Input custom measurements (sepal length/width, petal length/width) via sliders to predict flower species alongside prediction probabilities.
- **📊 Exploratory Data Analysis (EDA):** View raw data, statistical summaries, and dynamic scatter plots to analyze feature relationships.
- **📈 Model Performance Metrics:** Evaluate performance via calculated accuracy scores, interactive confusion matrices, and detailed classification reports.
- **🌳 Decision Tree Visualization:** Render the trained decision tree structure to understand feature splits and classification rules.
- **⚙️ Interactive Controls:** Adjust parameters like Decision Tree `max_depth` and train/test split ratio directly from the sidebar.

---

## 🛠️ Project Structure

```text
├── app.py              # Main Streamlit application file
├── iris.csv            # Iris dataset (optional, auto-fallback to sklearn datasets)
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

---

## 📦 Installation & Setup

### 1. Clone or Download the Repository
```bash
git clone https://github.com/your-username/iris-streamlit-dashboard.git
cd iris-streamlit-dashboard
```

### 2. Create and Activate a Virtual Environment (Optional but Recommended)
- **Windows:**
  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```
- **macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt` file yet, install the libraries directly:
```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn
```

---

## 🏃 Running the Application

Launch the Streamlit app using the following command:

```bash
streamlit run app.py
```

After running the command, open your browser and navigate to `http://localhost:8501`.

---

## 📄 Dataset Overview

The dataset contains physical measurements of three Iris species:
- **Setosa** (`iris-setosa`)
- **Versicolor** (`iris-versicolor`)
- **Virginica** (`iris-virginica`)

### Features:
1. `sepal length (cm)`
2. `sepal width (cm)`
3. `petal length (cm)`
4. `petal width (cm)`

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page or open a pull request.

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).