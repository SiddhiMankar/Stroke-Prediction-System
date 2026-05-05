import pickle
with open('processed_data/preprocessed_data.pkl', 'rb') as f:
    meta = pickle.load(f)
with open('cleaned_features.txt', 'w', encoding='utf-8') as f:
    for feat in meta.get('feature_names', []):
        f.write(f"{feat}\n")
