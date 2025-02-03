###############################
# app.py
###############################

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

import matplotlib.pyplot as plt
from wordcloud import WordCloud
import io
import base64
import joblib
import numpy as np

# --- (A) DEFINE preprocess_text AT TOP LEVEL ---
# So the unpickling of TfidfVectorizer(preprocessor=preprocess_text) can find it here
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

def preprocess_text(text):
    """
    Lemmatize, remove stopwords & non-alpha tokens, etc.
    EXACT same code used when training/pickling your pipeline/dict.
    """
    if not isinstance(text, str):
        return ""
    tokens = word_tokenize(text.lower())
    sw = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for w in tokens:
        if w.isalpha() and w not in sw:
            w_verb = lemmatizer.lemmatize(w, pos='v')
            w_noun = lemmatizer.lemmatize(w_verb, pos='n')
            clean_tokens.append(w_noun)
    return " ".join(clean_tokens)

###############################
# 1. LOAD DATA & MODEL
###############################

df = pd.read_csv("data/train_cleaned_LightGBM.csv")

required_cols = ["description", "budget", "average_cast_pop", "director", "genre"]
for col in required_cols:
    if col not in df.columns:
        print(f"WARNING: Missing column '{col}'. Creating fallback.")
        if col in ["description", "director", "genre"]:
            df[col] = ""
        else:
            df[col] = 0.0

df[required_cols] = df[required_cols].fillna({
    "description": "",
    "budget": 0.0,
    "average_cast_pop": 0.0,
    "director": "",
    "genre": ""
})

# Load your dictionary-based model
model_dict = None
try:
    model_dict = joblib.load("models/best_lgbm_pipeline_merged.pkl")  
    # ^ Adjust name if needed

    best_model = model_dict.get("model", None)
    tfidf_vec = model_dict.get("tfidf_vectorizer", None)
    le_genre = model_dict.get("label_encoder_genre", None)
    le_director = model_dict.get("label_encoder_director", None)
    le_success = model_dict.get("label_encoder_success", None)

    if best_model is None:
        print("ERROR: 'model' key not found in your .pkl file.")
        model_dict = None
except Exception as e:
    print(f"Error loading model dictionary: {e}")
    model_dict = None
    best_model = None
    tfidf_vec = None
    le_genre = None
    le_director = None
    le_success = None

###############################
# 2. CREATE OPTIONS
###############################

full_directors = df["director"].dropna().unique().tolist()
unique_directors = sorted([d for d in full_directors if d.strip()])

full_genres = df["genre"].dropna().unique().tolist()
unique_genres = sorted([g for g in full_genres if g.strip()])

if not unique_directors:
    unique_directors = ["No Director Found"]
if not unique_genres:
    unique_genres = ["No Genre Found"]

###############################
# 3. DASH LAYOUT
###############################
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "🎬 Movie Success Classifier"

# We keep only 4 final classes in the UI: flop, medium, hit, blockbuster
# merging above/below/average => medium
label_descriptions = {
    "flop": "A movie that performs far below cost expectations.",
    "medium": "A moderate-performance range (includes above/below/average).",
    "hit": "A movie anticipated to perform very well.",
    "blockbuster": "A movie expected to achieve phenomenal success."
}

app.layout = html.Div([
    html.Div([
        dcc.RadioItems(
            id='nav-radio',
            options=[
                {'label': 'Prediction', 'value': 'prediction'},
                {'label': 'Exploratory Data Analysis (EDA)', 'value': 'eda'}
            ],
            value='prediction',
            labelStyle={'display': 'inline-block', 'marginRight': '20px'},
            style={
                'textAlign': 'center',
                'padding': '20px',
                'fontSize': '20px',
                'backgroundColor': '#f8f9fa'
            }
        )
    ]),

    ##########################################
    # PREDICTION PAGE
    ##########################################
    html.Div([
        html.H1("🎬 Movie Success Classifier", style={'textAlign': 'center'}),

        html.Div([
            html.Div([
                html.Label('Budget (USD):'),
                dcc.Input(
                    id='inp-budget',
                    type='number',
                    value=50_000_000,
                    min=0.0,
                    step=1.0,
                    style={'width': '100%'}
                )
            ], style={'padding': 10, 'width': '30%'}),

            html.Div([
                html.Label('Director:'),
                dcc.Dropdown(
                    id='inp-director',
                    options=[{'label': d, 'value': d} for d in unique_directors],
                    value=unique_directors[0],
                    clearable=True,
                    placeholder="Select a Director"
                )
            ], style={'padding': 10, 'width': '30%'})
        ], style={'display': 'flex', 'justifyContent': 'center'}),

        html.Div([
            html.Div([
                html.Label('Genre:'),
                dcc.Dropdown(
                    id='inp-genre',
                    options=[{'label': g, 'value': g} for g in unique_genres],
                    value=unique_genres[0],
                    clearable=True,
                    placeholder="Select a Genre"
                )
            ], style={'padding': 10, 'width': '30%'}),

            html.Div([
                html.Label('Average Cast Popularity (0-10):'),
                dcc.Slider(
                    id='inp-castpop',
                    min=0,
                    max=10,
                    step=0.1,
                    value=5.0,
                    marks={i: str(i) for i in range(0, 11)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'padding': 10, 'width': '60%'})
        ], style={'display': 'flex', 'justifyContent': 'center'}),

        html.Div([
            html.Label("Plot / Overview / Keywords:"),
            dcc.Textarea(
                id='inp-description',
                style={'width': '100%', 'height': '100px'},
                placeholder="Enter a brief overview or relevant keywords..."
            )
        ], style={'padding': 10, 'width': '60%', 'margin': 'auto'}),

        html.Div([
            dbc.Button(
                'Predict',
                id='predict-button',
                n_clicks=0,
                color='primary',
                size='lg',
                style={'marginTop': 20, 'fontSize': '18px', 'padding': '10px 20px'}
            )
        ], style={'textAlign': 'center'}),

        html.Div([
            html.H2("🔍 Prediction Result", style={'textAlign': 'center', 'marginTop': 30}),
            html.Div(
                id='prediction-output',
                style={
                    'textAlign': 'center',
                    'fontSize': '24px',
                    'fontWeight': 'bold',
                    'color': '#28B463'
                }
            )
        ]),

        html.Div([
            dbc.Button("Info", id="info-button", n_clicks=0, color="info")
        ], style={'textAlign': 'center', 'marginTop': '20px'}),

        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Prediction Labels Information")),
                dbc.ModalBody([
                    html.P(f"**blockbuster**: {label_descriptions['blockbuster']}"),
                    html.P(f"**hit**: {label_descriptions['hit']}"),
                    html.P(f"**medium**: {label_descriptions['medium']}"),
                    html.P(f"**flop**: {label_descriptions['flop']}")
                ]),
                dbc.ModalFooter(
                    dbc.Button("Close", id="close-modal", className="ml-auto")
                ),
            ],
            id="info-modal",
            is_open=False,
        ),
    ], id='prediction-page'),

    ##########################################
    # EDA PAGE
    ##########################################
    html.Div([
        html.H1("📊 Exploratory Data Analysis (EDA)", style={'textAlign': 'center'}),

        # Only keeping the "Budget Distribution" plot here
        html.Div([
            html.Div([
                html.H3("Budget Distribution", style={'textAlign': 'center'}),
                dcc.Graph(id='fig-budget-hist')
            ], style={'width': '48%', 'padding': '10px'}),
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center'}),

        # Only keeping the "Top Directors by Frequency" table
        html.Div([
            html.Div([
                html.H3("Top Directors by Frequency", style={'textAlign': 'center'}),
                html.Div(id='director-table', style={'textAlign': 'center'})
            ], style={'width': '48%', 'padding': '10px'}),
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center'}),

        # Only keeping the "Word Cloud: Genre"
        html.Div([
            html.H3("Word Cloud: Genre", style={'textAlign': 'center'}),
            html.Img(
                id='genre-wordcloud',
                style={'width': '60%', 'display': 'block', 'margin': 'auto'}
            )
        ], style={'padding': 20}),

    ], id='eda-page', style={'display': 'none'})
])

###############################
# 4. CALLBACKS
###############################

@app.callback(
    [Output('prediction-page', 'style'),
     Output('eda-page', 'style')],
    [Input('nav-radio', 'value')]
)
def display_page(selected_page):
    if selected_page == 'prediction':
        return {'display': 'block'}, {'display': 'none'}
    return {'display': 'none'}, {'display': 'block'}

@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [
        State('inp-budget', 'value'),
        State('inp-director', 'value'),
        State('inp-genre', 'value'),
        State('inp-castpop', 'value'),
        State('inp-description', 'value')
    ]
)
def predict_label(n_clicks, budget, director, genre, cast_pop, description):
    if not n_clicks:
        return ""

    if model_dict is None or best_model is None:
        return "No valid pipeline loaded. Cannot predict."

    row_data = {
        'description': [description if description else ""],
        'budget': [budget if budget else 0.0],
        'average_cast_pop': [cast_pop if cast_pop else 0.0],
        'director': [director if director else "Unknown"],
        'genre': [genre if genre else "Unknown"]
    }
    df_input = pd.DataFrame(row_data)

    try:
        # 1) TF-IDF
        X_desc = tfidf_vec.transform(df_input["description"])

        # 2) Label encode
        from scipy.sparse import csr_matrix, hstack

        try:
            df_input["genre_enc"] = le_genre.transform(df_input["genre"])
        except:
            df_input["genre_enc"] = [0]

        try:
            df_input["director_enc"] = le_director.transform(df_input["director"])
        except:
            df_input["director_enc"] = [0]

        X_budget = df_input["budget"].values.reshape(-1, 1)
        X_castpop = df_input["average_cast_pop"].values.reshape(-1, 1)

        X_cat = np.hstack([
            df_input["genre_enc"].values.reshape(-1,1),
            df_input["director_enc"].values.reshape(-1,1),
            X_budget,
            X_castpop
        ])
        X_cat_sparse = csr_matrix(X_cat)
        X_final = hstack([X_desc, X_cat_sparse])

        # 3) Predict
        raw_pred = best_model.predict(X_final)[0]  # might be string or numeric

        # If numeric => decode with label_encoder_success if present
        pred_label = raw_pred
        if isinstance(raw_pred, (int, np.integer)) and le_success:
            pred_label = le_success.inverse_transform([raw_pred])[0]

        # 4) Merge above/below/average => medium for final display
        if pred_label in ["above average", "average", "below average"]:
            pred_label = "medium"

        # 5) Get label description from our final 4 keys
        desc_label = label_descriptions.get(pred_label, "")

        return f"Predicted: {pred_label}\n\n{desc_label}"
    except Exception as e:
        return f"Prediction error: {str(e)}"

@app.callback(
    [
        Output('fig-budget-hist', 'figure'),
        Output('director-table', 'children'),
        Output('genre-wordcloud', 'src')
    ],
    [Input('nav-radio', 'value')]
)
def update_eda(page):
    if page != 'eda':
        return {}, "", ""

    # Budget Distribution
    fig_budget = px.histogram(df, x='budget', nbins=30, title='Distribution of Budget')
    fig_budget.update_layout(xaxis_title='Budget (USD)', yaxis_title='Count')

    # Top Directors by Frequency
    top_dir = df['director'].value_counts().head(10).reset_index()
    top_dir.columns = ['Director', 'Count']
    table_header = [html.Thead(html.Tr([html.Th("Director"), html.Th("Count")]))]
    rows = [html.Tr([html.Td(r['Director']), html.Td(r['Count'])]) for _, r in top_dir.iterrows()]
    table_body = [html.Tbody(rows)]
    director_table = dbc.Table(table_header + table_body, bordered=True, striped=True, hover=True)

    # Word Cloud for Genre
    genre_text = " ".join([str(x) for x in df['genre'] if x.strip().lower() != 'unknown'])
    if genre_text:
        wc = WordCloud(width=800, height=400, background_color='white').generate(genre_text)
        img = io.BytesIO()
        plt.figure(figsize=(8, 4))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        base64_img = base64.b64encode(img.getvalue()).decode()
        wordcloud_src = f"data:image/png;base64,{base64_img}"
    else:
        wordcloud_src = ""

    return fig_budget, director_table, wordcloud_src

@app.callback(
    Output("info-modal", "is_open"),
    [Input("info-button", "n_clicks"), Input("close-modal", "n_clicks")],
    [State("info-modal", "is_open")]
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)
