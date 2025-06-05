============================================================
                  Personality Prediction Pipeline
============================================================

This project uses behavioral data to predict whether someone is an Introvert or Extrovert using machine learning.

------------------------------------------------------------
PROJECT STRUCTURE
------------------------------------------------------------
personality_pipeline/
├── data/
│   └── personality_dataset.csv
├── model/
│   └── personality_rf.pkl             # Trained model
├── pipeline/
│   ├── __init__.py                    # Marks pipeline folder as a package
│   ├── load_data.py                   # Loads raw dataset
│   ├── preprocess.py                  # Cleans and encodes data
│   ├── train_model.py                 # Trains and saves RandomForest model
│   └── predict.py                     # Makes predictions from input data
├── main.py                            # Runs full pipeline & tests predictions
└── README.txt                         # This file

------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------
1. Make sure you have Python 3.7+ installed.

2. Install dependencies:
   pip install pandas scikit-learn joblib

3. Run the pipeline:
   python main.py

   This will:
   - Load and clean the dataset
   - Train a Random Forest model
   - Save the model to disk
   - Predict personality type for 5 sample cases

------------------------------------------------------------
SAMPLE OUTPUT
------------------------------------------------------------
🧠 Personality Predictions:

Example 1: Introvert
Example 2: Extrovert
Example 3: Introvert
Example 4: Extrovert
Example 5: Introvert

------------------------------------------------------------
NOTES
------------------------------------------------------------
- Personality types are predicted using traits like:
  - Time spent alone
  - Stage fear
  - Social event attendance
  - Feeling drained after socializing
