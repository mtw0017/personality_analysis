from pipeline.load_data import load_dataset
from pipeline.preprocess import clean_and_encode
from pipeline.train_model import train_model
from pipeline.predict import predict_personality

# Step 1: Load and clean dataset
df = load_dataset("data/personality_dataset.csv")
df_clean = clean_and_encode(df)

# Step 2: Train model and save
train_model(df_clean)

# Step 3: Define test cases (mix of introvert/extrovert behaviors)
test_inputs = [
    {
        "Time_spent_Alone": 8,
        "Stage_fear": 1,
        "Social_event_attendance": 1,
        "Going_outside": 1,
        "Drained_after_socializing": 1,
        "Friends_circle_size": 2,
        "Post_frequency": 1
    },
    {
        "Time_spent_Alone": 2,
        "Stage_fear": 0,
        "Social_event_attendance": 8,
        "Going_outside": 6,
        "Drained_after_socializing": 0,
        "Friends_circle_size": 12,
        "Post_frequency": 6
    },
    {
        "Time_spent_Alone": 5,
        "Stage_fear": 1,
        "Social_event_attendance": 4,
        "Going_outside": 3,
        "Drained_after_socializing": 1,
        "Friends_circle_size": 5,
        "Post_frequency": 2
    },
    {
        "Time_spent_Alone": 1,
        "Stage_fear": 0,
        "Social_event_attendance": 10,
        "Going_outside": 7,
        "Drained_after_socializing": 0,
        "Friends_circle_size": 15,
        "Post_frequency": 8
    },
    {
        "Time_spent_Alone": 6,
        "Stage_fear": 1,
        "Social_event_attendance": 2,
        "Going_outside": 2,
        "Drained_after_socializing": 1,
        "Friends_circle_size": 3,
        "Post_frequency": 2
    }
]

# Step 4: Run predictions
print("\n🧠 Personality Predictions:\n")
for i, entry in enumerate(test_inputs, start=1):
    result = predict_personality(entry)
    print(f"Example {i}: {result}")
