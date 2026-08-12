import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('ml_dataset.csv')

# Remove any NaN labels
df = df.dropna(subset=['label'])

features = ['bicep_envelope', 'tricep_envelope', 'ratio_b_to_t', 'co_activation']
X = StandardScaler().fit_transform(df[features].values)
le = LabelEncoder()
y = le.fit_transform(df['label'].values)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

print(f'Dataset size: {len(df)} rows')
print(f'Test accuracy: {acc:.4f} ({acc*100:.2f}%)')
print(f'\nLabels: {list(le.classes_)}')
print(f'\nPer-class performance:')
print(classification_report(y_test, pred, target_names=list(le.classes_)))

# Save the cleaned dataset
df.to_csv('ml_dataset.csv', index=False)
print(f'\nDataset cleaned and saved with {len(df)} valid rows')
