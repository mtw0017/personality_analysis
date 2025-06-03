# 🧠 Personality Prediction from Behavioral Data

This project explores the relationship between behavioral traits and personality types (Introvert vs Extrovert). It uses both **supervised** and **unsupervised** machine learning techniques to classify personality types based on self-reported social behaviors.

---

## 📂 Dataset

- **Source**: `personality_dataset.csv`  
- **Size**: 2,900 entries × 8 columns  
- **Target Variable**: `Personality` (Introvert or Extrovert)

### Features
- `Time_spent_Alone`: Time spent alone per week
- `Stage_fear`: Fear of public speaking (Yes/No)
- `Social_event_attendance`: Frequency of attending social events
- `Going_outside`: General outdoor activity
- `Drained_after_socializing`: Whether the person feels tired after socializing (Yes/No)
- `Friends_circle_size`: Approximate number of friends
- `Post_frequency`: Frequency of social media posting

---

## 🧼 Data Preprocessing

- Handled missing values (median for numerics, mode for categoricals)
- Label encoded categorical features
- Normalized data for clustering

---

## 📊 Exploratory Data Analysis (EDA)

Key insights visualized:
- Extroverts tend to **feel energized** after socializing.
- Introverts typically have **smaller friend circles** and **higher social fatigue**.
- `Drained_after_socializing` alone is highly indicative of personality type.

---

## 🤖 Modeling

### Supervised Learning
- **Model**: Random Forest Classifier  
- **Accuracy**: ~90% on test data  
- **Feature Importance**: Top features were `Drained_after_socializing`, `Friends_circle_size`, and `Time_spent_Alone`

### Unsupervised Learning
- **Clustering**: KMeans with 2 clusters  
- **Dimensionality Reduction**: PCA for 2D visualization  
- **Adjusted Rand Index**: 0.76 — strong agreement with actual personality labels

---

## 🔎 Example Prediction

Use the trained model to classify personality based on new behavioral input:

```python
# Example input
{
    "Time_spent_Alone": 7,
    "Stage_fear": 1,  # Yes
    "Social_event_attendance": 2,
    "Going_outside": 1,
    "Drained_after_socializing": 1,  # Yes
    "Friends_circle_size": 3,
    "Post_frequency": 1
}

# Output
Predicted Personality: Introvert
