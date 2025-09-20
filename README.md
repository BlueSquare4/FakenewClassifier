# FakeNewsClassifier — Full project (train + API)

This repo contains a minimal, copy-paste-ready Fake News Classifier you can drop into your GitHub repo. It includes:

* `requirements.txt` — Python deps
* `data/` — (you add) `train.csv` with columns `text` and `label`
* `train.py` — train/save a scikit-learn pipeline (TF-IDF + LogisticRegression)
* `app.py` — small Flask API to serve predictions
* `models/` — trained model saved as `models/fakenews_model.joblib`
* `README.md` — quick usage

---

## FILE: requirements.txt

```
pandas
scikit-learn
joblib
flask
gunicorn
```

---

## FILE: README.md

````
# FakeNewsClassifier

## Setup

1. Create venv and install deps:

```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
````

2. Add your dataset to `data/train.csv` with two columns: `text` and `label`.

   * `label` should be 0/1 or `FAKE`/`REAL`. If using strings, `train.py` will map them to 0/1.

3. Train:

```bash
python train.py --data data/train.csv --out models/fakenews_model.joblib
```

4. Run API locally:

```bash
python app.py
# or production: gunicorn -w 4 app:app
```

5. Test:

```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"text":"This is a sample news text"}'
```

````

---

## FILE: train.py

```python
import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib


def load_data(path):
    df = pd.read_csv(path)
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns")
    X = df['text'].fillna('')
    y = df['label']
    # Map string labels to 0/1 if needed
    if y.dtype == object:
        y = y.map(lambda v: 1 if str(v).lower() in ('real', 'true', '1') else 0)
    return X, y


def build_pipeline():
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1,2), stop_words='english')),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    return pipe


def main(args):
    X, y = load_data(args.data)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipe = build_pipeline()
    print('Training...')
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_val)
    probs = pipe.predict_proba(X_val)[:, 1]

    print('Accuracy:', accuracy_score(y_val, preds))
    print(classification_report(y_val, preds))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump(pipe, args.out)
    print('Saved model to', args.out)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--out', default='models/fakenews_model.joblib')
    args = p.parse_args()
    main(args)
````

---

## FILE: app.py

```python
from flask import Flask, request, jsonify
import joblib
import os

MODEL_PATH = os.environ.get('MODEL_PATH', 'models/fakenews_model.joblib')

app = Flask(__name__)

# Load model at startup
model = None


def load_model():
    global model
    model = joblib.load(MODEL_PATH)


@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'model not loaded'}), 500
    data = request.get_json(force=True)
    text = data.get('text')
    if not text:
        return jsonify({'error': 'no text provided'}), 400
    pred = model.predict([text])[0]
    proba = float(model.predict_proba([text])[0][1])
    return jsonify({'prediction': int(pred), 'probability_real': proba})


if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5000, debug=False)
```

---

## NOTES & next steps

* This is a simple baseline (TF-IDF + Logistic Regression). If you want transformer-based models (BERT), I can add a `train_transformer.py` that uses `transformers` and `datasets` — but that needs more dependencies and a GPU for reasonable speed.

* You can easily add a small React UI later and call `/predict`.

* If your dataset uses different column names, rename or update `train.py` accordingly.

---

Good luck — paste these files into your repo (`train.py`, `app.py`, `requirements.txt`, `README.md`) and run the steps in the README.
