# Cross-Platform Mental Health Classifier — Streamlit Demo App

This application provides an interactive interface to evaluate the performance of five machine-learning models trained on Reddit and Twitter datasets for binary mental-health classification.  
It showcases the models' predictions, performance visualizations, and cross-platform robustness results.

This README explains how to install dependencies, how to correctly place the model files, and how to run the Streamlit application locally.

---

# 1. Project Overview

This Streamlit tool allows users to:

- Perform live prediction using **five models**:
  - Logistic Regression  
  - Support Vector Machine  
  - Random Forest  
  - BiLSTM  
  - BERT
- Display all evaluation graphs from the project (confusion matrices, ROC curves, PR curves).
- View performance comparison across:
  - Reddit → Reddit  
  - Reddit → Twitter  
  - Mixed-domain training (80% Reddit + 20% Twitter)

Only the following directories are needed to run the app:

```
app/
output/
models/ #(downloaded separately)
```
---

# 2. Folder Structure

Your project must follow this exact structure:
```
crossplatform-mentalhealth-ml/  
  app/  
    app.py  
    README.md  
  output/  
    (all PNG plots already included in this repository)  
  models/  
    (download separately; see next section)
```
---

# 3. Required Model Files (Download Required)

Because the `models/` folder contains large files, it is not pushed to GitHub.

Download it from the following link:

**OneDrive Download Link:**  
```
https://liveconcordia-my.sharepoint.com/:f:/g/personal/h_belamr_live_concordia_ca/IgBAtEToFM5bRa545fCF_aEVAdbLtArq-_hPuO94UrGxndo?e=2MeAmz
```
After downloading, place the folder exactly here:

crossplatform-mentalhealth-ml/models/

Inside the `models/` folder, you must have:
```
logreg/  
  logreg_model.pkl  
  tfidf_reddit_vectorizer.pkl  
  logreg_model.json  

svm/  
  svm_model.pkl  
  tfidf_vectorizer.pkl  
  svm_tuning_log.json  

rf/  
  rf_model.pkl  
  tfidf_rf.pkl  

bilstm/  
  bilstm_model.h5  
  tokenizer.pkl  
  bilstm_config.json  

bert/  
  model.safetensors  
  config.json  
  training_args.json  
  label_space.json  
```
---

# 4. Installation Instructions

Follow these steps:

### Step 1 — Create a virtual environment  
In your terminal, run the command to create a Python virtual environment.

### Step 2 — Activate the environment  
Use the correct activation command for your OS:
- On Windows PowerShell: run the activation command for venv\Scripts\activate  
- On macOS/Linux: run the activation command for source venv/bin/activate

### Step 3 — Install required libraries  
Run the following installation command in your terminal:

Install all required libraries including:
- streamlit  
- scikit-learn  
- pandas  
- numpy  
- joblib  
- matplotlib  
- tensorflow  
- torch  
- transformers  
- pillow  

If BERT fails to load, also install:
- accelerate  
- sentencepiece  

You may create a requirements.txt if preferred, but manual installation works fine.

---

# 5. Running the Application

Make sure you are inside the project root:

crossplatform-mentalhealth-ml/

Then run Streamlit by executing the following command in your terminal:

```
py -m streamlit run app/app.py
```

After running, open the generated localhost link in your browser.

---

# 6. How to Use the App

### Home Page
- Summary of the project  
- Team information  
- Explanation of models, datasets, and preprocessing  
- Accuracy comparison table  

### Model Comparison Page
- Choose a model  
- View confusion matrices  
- View ROC curves  
- View PR curves  
(All graphs are loaded from the `output/` folder)

### Live Prediction Page
- Select any model  
- Enter text  
- Obtain prediction and confidence score  
- See which model performs best on each type of text  

---

# 7. Team Members

Hala Belamri  
Electrical and Computer Engineering  
Concordia University, Montreal, Canada  
ID: 40221036

Minhtu Chau  
Electrical and Computer Engineering  
Concordia University, Montreal, Canada  
ID: 40232883

Suomi Seki  
Electrical and Computer Engineering  
Concordia University, Montreal, Canada  
ID: 40328157

James Johnson  
Electrical and Computer Engineering  
Concordia University, Montreal, Canada  
ID: 40253492

Arefe Emrani  
Electrical and Computer Engineering  
Concordia University, Montreal, Canada  
ID: 40259157

---

# 8. Notes and Limitations

- This app demonstrates **binary** mental-health classification  
  - 1 = mental-health-related  
  - 0 = control/non-mental-health
- Performance varies across platforms (Reddit vs Twitter)
- BERT may run slower due to model size
- Because of GitHub limits, `models/` must be downloaded manually

---

# End of README

