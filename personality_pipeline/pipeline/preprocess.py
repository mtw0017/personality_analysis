from sklearn.preprocessing import LabelEncoder

def clean_and_encode(df):
    df = df.copy()

    # Fill missing values
    for col in df.select_dtypes(include='number'):
        df[col].fillna(df[col].median(), inplace=True)
    for col in ['Stage_fear', 'Drained_after_socializing']:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Encode categoricals
    le = LabelEncoder()
    df['Stage_fear'] = le.fit_transform(df['Stage_fear'])
    df['Drained_after_socializing'] = le.fit_transform(df['Drained_after_socializing'])
    df['Personality'] = le.fit_transform(df['Personality'])

    return df
