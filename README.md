# Human Activity Recognition - Level 0 🚶‍♂️📱

A Machine Learning project to recognize human activities using smartphone sensor data. This project is part of the **Mastering MLOps - Development to Deployment** journey, with model training, real-time and batch predictions, and deployment via **Streamlit**.

---

## 🚀 Project Structure

```
📁 HumanActivityRecognition-Level0
│
├── 📁 Data
│   ├── new_data.csv
│   ├── Drift/
│   └── Train_data.gzip
│
├── 📁 Notebook
│   ├── EDA/
│   ├── model_features/
│   └── model_registry/
│
├── 📁 Streamlit
│   └── Activity_recognition.py
│
├── .gitattributes
├── requirements.txt
└── README.md
```

---

## 🔍 Problem Statement

Smartphones equipped with accelerometers and gyroscopes collect data from users performing various physical activities (e.g., walking, sitting, laying). This data is used to train a classifier to **predict the type of activity** a person is engaged in.

---

## 📊 Features Used

- `tGravityAcc-energy()-X`
- `tGravityAcc-mean()-X`
- `tGravityAcc-max()-X`
- `tGravityAcc-min()-X`
- `angle(X,gravityMean)`
- `tGravityAcc-min()-Y`
- `tGravityAcc-mean()-Y`
- `tGravityAcc-max()-Y`
- `angle(Y,gravityMean)`
- `tBodyAcc-max()-X`
- `tGravityAcc-energy()-Y`
- `fBodyAcc-entropy()-X`

---

## 🧠 Model

- Algorithm: **Random Forest Classifier (tuned)**
- Frameworks: `scikit-learn`, `joblib`
- Training: Done on extracted sensor features
- Label Encoding: Activity names to class IDs
- Deployment UI: **Streamlit**

---

## 🧪 How to Run

1. **Clone the Repository**
   ```bash
   git clone https://github.com/manola1109/MLops-Model-Deployment-level-0.git
   cd HumanActivityRecognition-Level0
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv myenv
   myenv\Scripts\activate  # On Windows
   ```

3. **Install requirements**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit App**
   ```bash
   streamlit run Streamlit/Activity_recognition.py
   ```

---

## 🔁 Functionality

### ✅ Batch Prediction
- Upload `.csv` with sensor data
- Returns predicted activity labels
- Displays distribution as a bar chart

### ✅ Real-Time Prediction
- Randomly select feature vector
- Manual override of sensor values possible
- One-click prediction and display

---

## 🧰 Tools & Tech

- Python 🐍
- scikit-learn
- pandas & numpy
- Streamlit
- Joblib
- Git LFS (for large model files)

---

## 📦 Deployment Note

⚠️ Files > 100MB are tracked via [Git LFS](https://git-lfs.github.com). Ensure `git lfs install` before cloning.

---

## 📬 Author

**Deepak Singh**  
📧 manola1109  
🔗 [LinkedIn](https://www.linkedin.com/in/deepak-singh1109/)

---

## 📄 License

MIT License — use it freely!
