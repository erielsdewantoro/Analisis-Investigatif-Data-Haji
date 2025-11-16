# 🕋 Hajj Pilgrim Satisfaction: A Predictive Analysis

An interactive web dashboard built with Streamlit to analyze and predict factors influencing pilgrim satisfaction during the Hajj.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://yutetmurpzgpnqov6kzyp9.streamlit.app/)

**[➡️ Visit the Live App Here!](https://yutetmurpzgpnqov6kzyp9.streamlit.app/)**

---

![Hajj Pilgrim Satisfaction Predictive Analysis Preview](Overview.png)

---

## 🚀 Project Overview

Crowd management during the Hajj and Umrah is one of the world's most complex logistical challenges. This project aims to analyze a crowd management dataset to identify key factors influencing pilgrim satisfaction.

The main goal is to transform raw data analysis (from a Jupyter Notebook) into an interactive and actionable data product.

---

## ✨ Key Dashboard Features

The application is divided into four main pages:

1.  **Project Summary:**
    * Displays key performance indicators (KPIs) from the dataset, such as total records, average queue time, and average safety rating.
    * Provides a general overview of the project's objectives and methodology.

2.  **Interactive EDA (Exploratory Data Analysis):**
    * Allows users to dynamically filter data based on **Age Group** and **Pilgrim Experience**.
    * All visualizations (rating distributions, density, etc.) update automatically based on the selected filters.

3.  **Model Simulation & Performance:**
    * **Satisfaction Prediction Calculator:** A live feature where users can input a scenario (e.g., queue time, safety rating, crowd density) and get an instant prediction (`Pleased`/`Not Pleased`) from the XGBoost model.
    * Transparently displays model performance evaluations, including a Classification Report and Confusion Matrix.

4.  **Insights & Recommendations:**
    * Shows the top 5 most important factors the model uses for decision-making (Feature Importance).
    * Translates data insights into clear, **actionable operational recommendations** to improve pilgrim satisfaction.

---

## 🛠️ Technology Stack (Tech Stack)

* **Data Analysis:** Python, Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn (Preprocessing), XGBoost (Modeling)
* **Web App & Deployment:** Streamlit, Streamlit Community Cloud

---

## 💡 Key Insights & Main Findings

From the analysis and model, we found two main factors that significantly impact pilgrim satisfaction:

1.  **Perception of Safety is #1:** `Perceived_Safety_Rating` is the most important factor. Pilgrims who *feel* safe are far more likely to be satisfied.
    * **Recommendation:** Increase the *visibility* of security personnel and ensure adequate lighting, not just the *number* of personnel.

2.  **Queue Time is a "Satisfaction Killer":** `Queue_Time_minutes` and `Security_Checkpoint_Wait_Time` are the strongest negative factors.
    * **Recommendation:** Implement real-time queue monitoring systems and dynamically open additional checkpoints when crowd density is detected as 'High'.

---

## 📊 Data Source & Limitations

* **Data Source:** The dataset used is the "Hajj & Umrah Crowd Management" dataset obtained from [Kaggle](https://www.kaggle.com/datasets/saidakd/hajj-umrah-crowd-management-dataset).
* **Limitation:** Analysis indicates this dataset is likely **synthetic** (artificially generated), marked by its perfect data distribution and lack of missing values. Therefore, the model's performance on real-world data may differ.
