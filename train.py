"""

1. Cleans raw data and merges "above average", "average", "below average" -> "medium"
2. Uses lemmatization for text
3. Tunes LightGBM with an expanded parameter grid (including class_weight, colsample_bytree, subsample)
4. 5-fold CV for better generalization
5. Saves the best pipeline components
"""

import pandas as pd
import ast
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier

from scipy.sparse import hstack, csr_matrix

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Ensure NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

#################################
# 1) HELPER FUNCTIONS
#################################

def categorize_revenue(rev):
    """Categorize revenue into success labels (original thresholds)."""
    if rev >= 500_000_000:
        return 'blockbuster'
    elif rev >= 100_000_000:
        return 'hit'
    elif rev >= 50_000_000:
        return 'above average'
    elif rev >= 25_000_000:
        return 'average'
    elif rev >= 10_000_000:
        return 'below average'
    else:
        return 'flop'

def merge_similar_labels(label):
    """
    Merge "above average", "average", "below average" into "medium".
    Keep "flop", "hit", "blockbuster" as is.
    """
    if label in ["above average", "average", "below average"]:
        return "medium"
    return label

def get_main_genre(genres_str):
    """Extract the first genre from the 'genres' JSON-like string."""
    try:
        genres = ast.literal_eval(genres_str)
        if isinstance(genres, list) and len(genres) > 0:
            return genres[0].get('name', 'Unknown')
    except (ValueError, SyntaxError):
        pass
    return 'Unknown'

def get_director(crew_str):
    """Extract the first Director from the 'crew' JSON-like string."""
    try:
        crew = ast.literal_eval(crew_str)
        if isinstance(crew, list):
            for member in crew:
                if member.get('job') == 'Director':
                    return member.get('name', 'Unknown')
    except (ValueError, SyntaxError):
        pass
    return 'Unknown'

#################################
# 2) CLEANING + MERGING CLASSES
#################################

def clean_data(original_csv="data/tmdb-box-office-prediction/train.csv", output_csv="data/train_cleaned_LightGBM.csv"):
    """
    Cleans raw data -> 'train_cleaned.csv'.
    Also merges 'above average', 'average', 'below average' into 'medium'.
    """
    # 1) Load
    try:
        df = pd.read_csv(original_csv)
        print(f"[Data Cleaning] Loaded {len(df)} rows from '{original_csv}'.")
    except FileNotFoundError:
        print(f"[Data Cleaning] File not found: {original_csv}")
        return
    except Exception as e:
        print(f"[Data Cleaning] Error loading '{original_csv}': {e}")
        return

    # 2) Combine overview + Keywords -> description
    df['overview'] = df.get('overview', '').fillna('')
    df['Keywords'] = df.get('Keywords', '').fillna('')
    df['description'] = df['overview'] + ' ' + df['Keywords']

    # 3) budget
    if 'budget' not in df.columns:
        df['budget'] = 0.0
    df['budget'] = pd.to_numeric(df['budget'], errors='coerce').fillna(0.0)

    # 4) average_cast_pop
    if 'avg_cast_pop_score' in df.columns:
        df.rename(columns={'avg_cast_pop_score': 'average_cast_pop'}, inplace=True)
    if 'average_cast_pop' not in df.columns:
        df['average_cast_pop'] = 0.0
    df['average_cast_pop'] = pd.to_numeric(df['average_cast_pop'], errors='coerce').fillna(0.0)

    # 5) genre
    if 'genres' in df.columns:
        df['genre'] = df['genres'].apply(get_main_genre)
    else:
        df['genre'] = 'Unknown'

    # 6) director
    if 'crew' in df.columns:
        df['director'] = df['crew'].apply(get_director)
    else:
        df['director'] = 'Unknown'

    # 7) success_label
    if 'success_label' not in df.columns:
        if 'revenue' not in df.columns:
            raise ValueError("[Data Cleaning] Missing 'success_label' & 'revenue'. Cannot proceed.")
        df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce').fillna(0.0)
        df['success_label'] = df['revenue'].apply(categorize_revenue)

    # 7b) Merge "above average", "average", "below average" -> "medium"
    df['success_label'] = df['success_label'].apply(merge_similar_labels)

    # 8) Keep columns
    needed_cols = ["description", "budget", "average_cast_pop", "genre", "director", "success_label"]
    df_final = df[needed_cols].copy()
    df_final.dropna(subset=needed_cols, inplace=True)

    # 9) Save
    df_final.to_csv(output_csv, index=False)
    print(f"[Data Cleaning] Saved {len(df_final)} rows -> '{output_csv}'")

#################################
# 3) LEMMATIZATION PREPROCESS
#################################
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """
    - Lowercase
    - Tokenize
    - Remove stopwords & non-alpha tokens
    - Lemmatize each token (verb & noun)
    """
    if not isinstance(text, str):
        return ""
    tokens = word_tokenize(text.lower())
    sw = set(stopwords.words('english'))
    final_tokens = []
    for w in tokens:
        if w.isalpha() and w not in sw:
            # Lemmatize as verb, then as noun
            w_verb = lemmatizer.lemmatize(w, pos='v')
            w_noun = lemmatizer.lemmatize(w_verb, pos='n')
            final_tokens.append(w_noun)
    return " ".join(final_tokens)

#################################
# 4) TRAIN & TUNE LIGHTGBM
#################################

def train_and_tune_lgbm(cleaned_csv="data/train_cleaned_LightGBM.csv"):
    """
    1. Reads cleaned CSV (with merged classes)
    2. TF-IDF w/ custom lemmatizer
    3. Label-encodes genre/director + numeric
    4. SMOTE
    5. RandomizedSearchCV on LightGBM w/ class_weight, subsample, etc.
    6. 5-Fold CV
    7. Evaluate on test
    8. Save best pipeline
    """
    # A) Load
    df = pd.read_csv(cleaned_csv)
    print(f"[Model Training] Loaded: {df.shape}")

    # B) Target
    le_success = LabelEncoder()
    y = le_success.fit_transform(df["success_label"])  
    # Now we have classes [0..3 or 4, depending on how many remain]

    # C) TF-IDF on description
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1,2),
        stop_words=None,  # we do manual stopwords
        preprocessor=preprocess_text
    )
    X_desc = tfidf.fit_transform(df["description"].fillna(""))

    # D) Categorical features
    df["genre"] = df["genre"].fillna("Unknown")
    le_genre = LabelEncoder()
    df["genre_enc"] = le_genre.fit_transform(df["genre"])
    X_genre = df["genre_enc"].values.reshape(-1, 1)

    df["director"] = df["director"].fillna("Unknown")
    le_director = LabelEncoder()
    df["director_enc"] = le_director.fit_transform(df["director"])
    X_director = df["director_enc"].values.reshape(-1, 1)

    # E) Numeric
    X_budget = df["budget"].values.reshape(-1, 1)
    X_castpop = df["average_cast_pop"].values.reshape(-1, 1)

    # Combine
    X_numcat = np.hstack([X_genre, X_director, X_budget, X_castpop])
    X_numcat_sparse = csr_matrix(X_numcat)
    X_final = hstack([X_desc, X_numcat_sparse])

    # F) Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )
    print(f"[Model Training] Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # G) SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print("[Model Training] After SMOTE:", X_train_res.shape)

    # H) LightGBM
    lgbm = LGBMClassifier(random_state=42)

    # Parameter grid: more extensive to handle new merged classes
    param_dist_lgbm = {
        "n_estimators": [100, 300, 500],
        "max_depth": [-1, 5, 10],
        "learning_rate": [0.01, 0.05, 0.1],
        "num_leaves": [31, 63, 127],
        "colsample_bytree": [0.8, 1.0],
        "subsample": [0.8, 1.0],
        "class_weight": [None, "balanced"],  # test if built-in weighting helps
    }

    # I) RandomizedSearchCV
    from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = "f1_macro"

    rnd_search_lgbm = RandomizedSearchCV(
        estimator=lgbm,
        param_distributions=param_dist_lgbm,
        n_iter=10,  # can increase for a more thorough search
        scoring=scoring,
        cv=cv,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    print("\n[Model Training] Tuning LightGBM with merged classes ...")
    rnd_search_lgbm.fit(X_train_res, y_train_res)
    print("[Model Training] Best params:", rnd_search_lgbm.best_params_)
    print("[Model Training] Best CV Macro-F1:", rnd_search_lgbm.best_score_)

    best_model = rnd_search_lgbm.best_estimator_

    # J) Evaluate on test
    y_pred_test = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test, average='macro')

    print(f"\n[TEST] Accuracy: {acc:.4f}")
    print(f"[TEST] Macro-F1: {test_f1:.4f}")
    print("\n[TEST] Classification Report:")
    print(classification_report(
        y_test, y_pred_test,
        target_names=le_success.classes_
    ))

    # K) Save
    model_pipeline = {
        "model": best_model,
        "label_encoder_success": le_success,
        "tfidf_vectorizer": tfidf,
        "label_encoder_genre": le_genre,
        "label_encoder_director": le_director
    }
    joblib.dump(model_pipeline, "models/best_lgbm_pipeline_merged.pkl")
    print("\n[INFO] Saved best model + encoders -> 'best_lgbm_pipeline_merged.pkl'.")

#################################
# 5) MAIN
#################################

def main():
    # 1) Clean & merge classes
    clean_data(
        original_csv="data/tmdb-box-office-prediction/train.csv",  # adjust your CSV
        output_csv="data/train_cleaned_LightGBM.csv"
    )
    # 2) Train & tune LightGBM
    train_and_tune_lgbm(cleaned_csv="data/train_cleaned_LightGBM.csv")

if __name__ == "__main__":
    main()
