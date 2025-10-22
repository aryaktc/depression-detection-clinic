# streamlit_depression_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import datetime
import sqlite3
from fpdf import FPDF
from io import BytesIO
import matplotlib.pyplot as plt
import re
import json

# optional sklearn imports used for demo model creation
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

st.set_page_config(page_title="Depression Detection — Clinical Assistant", layout="wide")

# -------------------------
# Configuration & Constants
# -------------------------
DB_PATH = "app_data.db"
MODEL_FILENAMES = [
    "project1_model.pkl",
    "project1_model.joblib",
    "model.pkl",
    "model.joblib",
    "project1_model.sav",
]

PHQ9_QUESTIONS = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Trouble falling or staying asleep, or sleeping too much",
    "Feeling tired or having little energy",
    "Poor appetite or overeating",
    "Feeling bad about yourself — or that you are a failure or have let yourself or your family down",
    "Trouble concentrating on things, such as reading the newspaper or watching television",
    "Moving or speaking so slowly that other people could have noticed or the opposite — being so fidgety or restless that you have been moving a lot more than usual",
    "Thoughts that you would be better off dead or of hurting yourself in some way"
]

PHQ9_SEVERITY = [
    (0, 4, 'None-minimal'),
    (5, 9, 'Mild'),
    (10, 14, 'Moderate'),
    (15, 19, 'Moderately severe'),
    (20, 27, 'Severe')
]

# Default per-channel feature order used when no explicit model feature list
DEFAULT_CHANNEL_FEATURE_ORDER = [
    'mean','std','var','min','max','median','skew','kurtosis','rms',
    'delta','theta','alpha','beta','delta_rel','theta_rel','alpha_rel','beta_rel'
]

# -------------------------
# Database helpers (SQLite)
# -------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            name TEXT,
            age INTEGER,
            gender TEXT,
            contact TEXT,
            phq_score INTEGER,
            phq_severity TEXT,
            suicidality_flag INTEGER,
            substance_flag INTEGER,
            previous_dx INTEGER,
            model_result TEXT,
            notes TEXT
        )
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS followups (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            timestamp TEXT,
            next_steps TEXT,
            plan_notes TEXT
        )
    ''')
    conn.commit()
    conn.close()

# -------------------------
# Model loading & wrapper
# -------------------------
@st.cache_resource
def load_model_from_file(path=None):
    if path is not None and os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            st.warning(f"Failed to load model at {path}: {e}")
    for fname in MODEL_FILENAMES:
        if os.path.exists(fname):
            try:
                return joblib.load(fname)
            except Exception as e:
                st.warning(f"Found {fname} but failed to load it: {e}")
    # fallback: try project1_model module
    try:
        import importlib
        m = importlib.import_module('project1_model')
        if hasattr(m, 'model'):
            return getattr(m, 'model')
    except Exception:
        pass
    return None

def model_predict(model, user_text, phq_score, input_mode="text", numeric_features=None):
    """Unified wrapper for several input modes, including numeric EEG features."""
    if model is None:
        return None
    try:
        # numeric EEG features
        if input_mode == "eeg" and numeric_features is not None:
            try:
                arr = np.array([numeric_features])
                if hasattr(model, 'predict_proba'):
                    out = model.predict_proba(arr)
                    if out.shape[1] >= 2:
                        p = float(out[0, 1])
                        label = model.classes_[1] if hasattr(model, 'classes_') else (1 if p > 0.5 else 0)
                        return {"label": label, "probability": p, "method": "eeg-features"}
                pred = model.predict(arr)
                return {"label": pred[0], "probability": None, "method": "eeg-features"}
            except Exception as e:
                st.info(f"Model could not accept EEG numeric features directly: {e}")

        # text mode
        if input_mode == "text":
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba([user_text])
                    if proba.shape[1] >= 2:
                        p = float(proba[0, 1])
                        label = model.classes_[1] if hasattr(model, 'classes_') else (1 if p > 0.5 else 0)
                        return {"label": label, "probability": p, "method": "text-pipeline"}
            except Exception:
                pass
            try:
                pred = model.predict([user_text])
                return {"label": pred[0], "probability": None, "method": "text-raw"}
            except Exception:
                pass

        # phq mode
        if input_mode == "phq":
            try:
                out = model.predict_proba(np.array([[phq_score]]))
                if out.shape[1] >= 2:
                    p = float(out[0, 1])
                    label = model.classes_[1] if hasattr(model, 'classes_') else (1 if p > 0.5 else 0)
                    return {"label": label, "probability": p, "method": "phq-only"}
            except Exception:
                pass

        # dict mode
        if input_mode == "dict":
            try:
                feat = {"text": user_text, "phq": phq_score}
                pred = model.predict([feat])
                prob = None
                if hasattr(model, 'predict_proba'):
                    prob = float(model.predict_proba([feat])[0, 1])
                return {"label": pred[0], "probability": prob, "method": "dict-features"}
            except Exception:
                pass

        # fallback attempt
        try:
            pred = model.predict(np.array([[phq_score]]))
            return {"label": pred[0], "probability": None, "method": "fallback"}
        except Exception as e:
            st.info(f"Model present but couldn't run predictions automatically: {e}")
            return None
    except Exception as e:
        st.warning(f"Unexpected error while using model: {e}")
        return None

# -------------------------
# Notebook parsing: extract training feature names
# -------------------------
def parse_notebook_for_feature_list(nb_path):
    """Attempt to extract a feature-name list literal from a Jupyter notebook (heuristic)."""
    if not os.path.exists(nb_path):
        return None
    try:
        with open(nb_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except Exception:
        return None
    text = ''
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            text += '\n'.join(cell.get('source', [])) + '\n'
    # patterns to find list literals assigned to variables
    patterns = [
        r"(?m)^(?P<var>\w*feature\w*)\s*=\s*(\[(?:.|\n)*?\])",
        r"(?m)^(?P<var>\w*features\w*)\s*=\s*(\[(?:.|\n)*?\])",
        r"(?m)(?:\w+\.columns)\s*=\s*(\[(?:.|\n)*?\])"
    ]
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.S):
            lit = m.group(2)
            try:
                arr = eval(lit, {'__builtins__': None}, {})
                if isinstance(arr, (list, tuple)) and all(isinstance(x, str) for x in arr):
                    return [x.strip() for x in arr]
            except Exception:
                continue
    return None

# -------------------------
# EEG feature extraction
# -------------------------
def detect_time_column(df):
    for c in df.columns:
        name = str(c).lower()
        if 'time' in name or name == 't':
            return c
    return None

def extract_band_powers(signal, sr):
    n = len(signal)
    if n < 4:
        return {'delta': 0, 'theta': 0, 'alpha': 0, 'beta': 0, 'delta_rel': 0, 'theta_rel': 0, 'alpha_rel': 0, 'beta_rel': 0}
    freqs = np.fft.rfftfreq(n, d=1.0/sr)
    fft_vals = np.fft.rfft(signal - np.mean(signal))
    psd = (np.abs(fft_vals) ** 2) / n
    bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30)}
    band_powers = {}
    total_power = psd.sum()
    for b, (lo, hi) in bands.items():
        idx = np.where((freqs >= lo) & (freqs < hi))[0]
        band_powers[b] = float(psd[idx].sum()) if idx.size > 0 else 0.0
    if total_power > 0:
        for b in ['delta','theta','alpha','beta']:
            band_powers[f'{b}_rel'] = float(band_powers[b] / total_power)
    else:
        for b in ['delta','theta','alpha','beta']:
            band_powers[f'{b}_rel'] = 0.0
    return band_powers

def extract_eeg_features(df, sr=256.0, time_col=None):
    df_proc = df.copy()
    if time_col is not None and time_col in df_proc.columns:
        df_proc = df_proc.drop(columns=[time_col])
    numeric_cols = [c for c in df_proc.columns if pd.api.types.is_numeric_dtype(df_proc[c])]
    if len(numeric_cols) == 0:
        raise ValueError("No numeric EEG channels found in uploaded file.")
    feats = []
    for c in numeric_cols:
        sig = df_proc[c].dropna().values.astype(float)
        if sig.size == 0:
            continue
        mean = float(np.mean(sig))
        std = float(np.std(sig))
        var = float(np.var(sig))
        mn = float(np.min(sig))
        mx = float(np.max(sig))
        median = float(np.median(sig))
        skew = float(pd.Series(sig).skew())
        kurt = float(pd.Series(sig).kurt())
        rms = float(np.sqrt(np.mean(sig**2)))
        band = extract_band_powers(sig, sr)
        row = {
            'channel': c,
            'n_samples': int(len(sig)),
            'mean': mean,
            'std': std,
            'var': var,
            'min': mn,
            'max': mx,
            'median': median,
            'skew': skew,
            'kurtosis': kurt,
            'rms': rms,
            'delta': band.get('delta', 0.0),
            'theta': band.get('theta', 0.0),
            'alpha': band.get('alpha', 0.0),
            'beta': band.get('beta', 0.0),
            'delta_rel': band.get('delta_rel', 0.0),
            'theta_rel': band.get('theta_rel', 0.0),
            'alpha_rel': band.get('alpha_rel', 0.0),
            'beta_rel': band.get('beta_rel', 0.0),
        }
        feats.append(row)
    features_by_channel = pd.DataFrame(feats).set_index('channel')
    numeric_features = features_by_channel.select_dtypes(include=[np.number]).mean(axis=0).to_dict()
    features_agg = {f'agg_{k}': float(v) for k, v in numeric_features.items()}
    return features_agg, features_by_channel

# -------------------------
# Flattening / matching utilities
# -------------------------
def get_model_feature_names(model):
    try:
        if hasattr(model, 'feature_names_in_'):
            return list(model.feature_names_in_)
    except Exception:
        pass
    try:
        if hasattr(model, 'named_steps'):
            for name, step in model.named_steps.items():
                if hasattr(step, 'get_feature_names_out'):
                    try:
                        return list(step.get_feature_names_out())
                    except Exception:
                        pass
    except Exception:
        pass
    try:
        if hasattr(model, 'feature_names'):
            return list(model.feature_names)
    except Exception:
        pass
    return None

def flatten_channel_features(features_by_channel, model_feature_names=None, channel_feature_order=DEFAULT_CHANNEL_FEATURE_ORDER):
    ordered_keys = []
    vector = []
    missing = {}
    channels = list(features_by_channel.index)
    cols = list(features_by_channel.columns)
    def get_val(channel, feat):
        try:
            return float(features_by_channel.at[channel, feat])
        except Exception:
            return np.nan
    if model_feature_names:
        for mf in model_feature_names:
            mapped = False
            parts = re.split(r'[\W_]+', mf)
            parts = [p for p in parts if p]
            # pattern channel_feature
            if len(parts) >= 2:
                ch_try = parts[0]
                feat_try = '_'.join(parts[1:])
                for ch in channels:
                    if ch.lower() == ch_try.lower():
                        for col in cols:
                            if col.lower() == feat_try.lower():
                                val = get_val(ch, col)
                                ordered_keys.append(mf)
                                vector.append(val)
                                mapped = True
                                break
                        if mapped:
                            break
            if mapped:
                continue
            # reverse: feature_channel
            if len(parts) >= 2:
                feat_try = parts[0]
                ch_try = '_'.join(parts[1:])
                for ch in channels:
                    if ch.lower() == ch_try.lower():
                        for col in cols:
                            if col.lower() == feat_try.lower():
                                val = get_val(ch, col)
                                ordered_keys.append(mf)
                                vector.append(val)
                                mapped = True
                                break
                        if mapped:
                            break
            if mapped:
                continue
            # fuzzy match
            for ch in channels:
                if ch.lower() in mf.lower():
                    for col in cols:
                        if col.lower() in mf.lower() or mf.lower() in col.lower():
                            val = get_val(ch, col)
                            ordered_keys.append(mf)
                            vector.append(val)
                            mapped = True
                            break
                if mapped:
                    break
            if mapped:
                continue
            ordered_keys.append(mf)
            vector.append(np.nan)
            missing[mf] = 'unmapped'
        return ordered_keys, vector, missing
    else:
        channels_sorted = sorted(channels)
        for ch in channels_sorted:
            for feat in channel_feature_order:
                key = f"{ch}_{feat}"
                ordered_keys.append(key)
                val = get_val(ch, feat)
                vector.append(val)
        return ordered_keys, vector, missing

# -------------------------
# Utilities
# -------------------------
def phq9_score(answers):
    s = sum(answers)
    severity = next((label for lo, hi, label in PHQ9_SEVERITY if lo <= s <= hi), 'Unknown')
    return s, severity

def save_session_to_db(summary):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('''INSERT INTO sessions (timestamp, name, age, gender, contact, phq_score, phq_severity, suicidality_flag, substance_flag, previous_dx, model_result, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
        summary['timestamp'], summary['name'], summary['age'], summary['gender'], summary['contact'], summary['phq_score'], summary['phq_severity'], int(summary['suicidality_flag']), int(summary['substance_flag']), int(summary['previous_dx']), summary['model_result'], summary['notes']
    ))
    conn.commit()
    session_id = cur.lastrowid
    conn.close()
    return session_id

def save_followup_to_db(session_id, plan):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('''INSERT INTO followups (session_id, timestamp, next_steps, plan_notes) VALUES (?, ?, ?, ?)''', (
        session_id, plan['timestamp'], plan['next_steps'], plan['plan_notes']
    ))
    conn.commit()
    conn.close()

def create_pdf_report(summary, next_steps_list=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Depression Screening Report', ln=True, align='C')
    pdf.ln(4)
    pdf.set_font('Arial', '', 11)
    for k in ['timestamp', 'name', 'age', 'gender', 'contact', 'phq_score', 'phq_severity', 'model_result']:
        pdf.cell(0, 8, f"{k.replace('_', ' ').title()}: {summary.get(k, '')}", ln=True)
    pdf.ln(4)
    notes_text = f"Notes:\n{summary.get('notes', '')}"
    pdf.multi_cell(0, 7, notes_text)
    if next_steps_list:
        pdf.ln(4)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, 'Recommended Next Steps:', ln=True)
        pdf.set_font('Arial', '', 11)
        for s in next_steps_list:
            pdf.cell(0, 7, f"- {s}", ln=True)
    bio = pdf.output(dest='S').encode('latin-1')
    return bio

# -------------------------
# Demo model creation helper
# -------------------------
def create_demo_model_and_features(n_channels=4):
    """Create a small demo classifier and write feature_order.txt and model file.
       n_channels: create channel names like C1, C2, ... and features for each.
    """
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn not available in this environment.")
    # create synthetic feature names in the form Channel_featureName
    channels = [f"C{i+1}" for i in range(n_channels)]
    feature_names = []
    for ch in channels:
        for feat in DEFAULT_CHANNEL_FEATURE_ORDER:
            feature_names.append(f"{ch}_{feat}")
    # synthetic training data
    n_features = len(feature_names)
    X = np.random.randn(200, n_features)
    # create a toy binary label related to sum of first channel's alpha and beta
    y = (X[:, 0:5].sum(axis=1) + np.random.randn(200)*0.1 > 0).astype(int)
    clf = Pipeline([('scaler', StandardScaler()), ('rf', RandomForestClassifier(n_estimators=50, random_state=42))])
    clf.fit(X, y)
    # attach feature names attribute so get_model_feature_names can find them
    try:
        clf.feature_names_in_ = np.array(feature_names)
    except Exception:
        pass
    # save model and feature list
    joblib.dump(clf, 'project1_model.joblib')
    with open('feature_order.txt', 'w', encoding='utf-8') as f:
        for fn in feature_names:
            f.write(fn + '\n')
    return 'project1_model.joblib', feature_names

# -------------------------
# App initialization
# -------------------------
init_db()
model = load_model_from_file()

# Try notebook parsing for feature order (will be None if not found)
NOTEBOOK_FEATURE_LIST = None
nb_path = os.path.join(os.getcwd(), 'project1_model.ipynb')
try:
    NOTEBOOK_FEATURE_LIST = parse_notebook_for_feature_list(nb_path)
except Exception:
    NOTEBOOK_FEATURE_LIST = None

# -------------------------
# Authentication (simple)
# -------------------------
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'doctorpass')
try:
    if hasattr(st, 'secrets') and 'admin_password' in st.secrets:
        ADMIN_PASSWORD = st.secrets.get('admin_password', ADMIN_PASSWORD)
except Exception:
    pass

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# Top layout
st.title("Depression Detection — Clinical Assistant")
st.write("A clinical screening tool with PHQ-9, optional ML model integration, EEG Excel upload + feature extraction, PDF export, and session management.")

# Sidebar: model uploader + demo model button + settings
with st.sidebar:
    st.header('Model & App settings')
    uploaded_model = st.file_uploader('Upload a trained model file (joblib/pkl)', type=['pkl', 'joblib', 'sav'])
    if uploaded_model is not None:
        save_path = os.path.join(os.getcwd(), uploaded_model.name)
        with open(save_path, 'wb') as f:
            f.write(uploaded_model.getbuffer())
        model = load_model_from_file(save_path)
        st.success('Uploaded model loaded (attempted).')
    st.markdown('---')
    st.write('Model input mode: choose how the model expects input data')
    input_mode = st.selectbox('Model input mode', options=['text', 'phq', 'dict', 'eeg'])
    st.markdown('---')
    st.write('Export / storage')
    use_db = st.checkbox('Enable database storage (SQLite)', value=True)
    st.write('Note: This demo stores data locally on the server.')

    st.markdown('---')
    st.write('Developer / demo tools')
    if SKLEARN_AVAILABLE:
        if st.button('Create demo model & feature_order.txt'):
            try:
                path, fns = create_demo_model_and_features(n_channels=4)
                st.success(f"Demo model saved to {path} and feature_order.txt generated ({len(fns)} features).")
            except Exception as e:
                st.error(f"Failed to create demo model: {e}")
    else:
        st.info('scikit-learn not installed — demo model creation disabled.')

# Main UI tabs
tabs = st.tabs(['Screening', 'Patient History', 'Model & Diagnostics', 'EEG Upload & Analysis', 'Settings & Deploy'])

# -------------------------
# Tab: Screening
# -------------------------
with tabs[0]:
    col1, col2 = st.columns((2, 1))
    with col1:
        with st.form(key='screen_form'):
            st.header('Patient / Session Details')
            name = st.text_input('Full name (optional)')
            age = st.number_input('Age', min_value=0, max_value=120, value=30)
            gender = st.selectbox('Gender', ['Prefer not to say', 'Female', 'Male', 'Other'])
            contact = st.text_input('Contact (optional)')
            st.markdown('---')
            st.subheader('PHQ-9 Questionnaire')
            phq_answers = []
            cols = st.columns(3)
            for i, q in enumerate(PHQ9_QUESTIONS):
                with cols[i % 3]:
                    val = st.radio(f"{i+1}. {q}", [0,1,2,3], index=0, key=f"phq_{i}", horizontal=False)
                    phq_answers.append(val)
            st.markdown('---')
            st.subheader('Free-text clinical notes / patient description (optional)')
            free_text = st.text_area("Describe the patient's symptoms, history, or a short excerpt of conversation", height=150)
            st.markdown('---')
            st.subheader('Other risk flags (check any that apply)')
            suicidality_flag = st.checkbox('Recent suicidal thoughts or plans')
            substance_flag = st.checkbox('Substance misuse')
            previous_dx = st.checkbox('Previous diagnosis of depression')

            submitted = st.form_submit_button('Run assessment')

        if submitted:
            timestamp = datetime.datetime.utcnow().isoformat()
            phq_sum, phq_sev = phq9_score(phq_answers)
            st.metric('PHQ-9 score', f"{phq_sum} / 27")
            st.write(f"Severity: **{phq_sev}**")

            # Visualization
            fig, ax = plt.subplots(figsize=(6, 1.5))
            ax.barh([0], [phq_sum], height=0.6)
            ax.set_xlim(0, 27)
            ax.set_xlabel('PHQ-9 score')
            ax.set_yticks([])
            for lo, hi, label in PHQ9_SEVERITY:
                ax.axvspan(lo, hi, alpha=0.08)
                ax.text((lo+hi)/2, 0, label, va='center', ha='center', fontsize=8)
            st.pyplot(fig)

            # Run model
            model_result = model_predict(model, free_text or "", phq_sum, input_mode=input_mode)
            st.markdown('### Model prediction')
            if model_result is None:
                st.info('No usable model found — displaying PHQ-9 based risk and recommendations only.')
                model_out_text = 'No model'
            else:
                lab = model_result.get('label')
                prob = model_result.get('probability')
                method = model_result.get('method')
                if prob is not None:
                    st.write(f"Predicted probability of depression: **{prob:.2f}** (method: {method})")
                    model_out_text = f"Prob={prob:.3f}, method={method}, label={lab}"
                else:
                    st.write(f"Predicted label: **{lab}** (method: {method})")
                    model_out_text = f"Label={lab}, method={method}"

            # Clinical suggestions
            st.markdown('### Clinical suggestions')
            if suicidality_flag or phq_sum >= 20:
                st.error('High risk flagged — immediate clinical evaluation recommended. If imminent danger, contact emergency services.')
            elif phq_sum >= 15:
                st.warning('Moderately severe to severe depressive symptoms — consider urgent psychiatric referral and medication evaluation.')
            elif phq_sum >= 10:
                st.info('Moderate depressive symptoms — consider psychotherapy and close monitoring; evaluation for medication as needed.')
            else:
                st.success('None to mild symptoms — monitor and offer psychoeducation; consider brief psychotherapy if needed.')

            st.markdown('---')
            st.subheader('Session summary & export')
            summary = {
                'timestamp': timestamp,
                'name': name,
                'age': age,
                'gender': gender,
                'contact': contact,
                'phq_score': phq_sum,
                'phq_severity': phq_sev,
                'suicidality_flag': suicidality_flag,
                'substance_flag': substance_flag,
                'previous_dx': previous_dx,
                'model_result': model_out_text,
                'notes': free_text[:2000]
            }
            st.dataframe(pd.DataFrame([summary]))

            sid = None
            if st.session_state.authenticated and use_db:
                sid = save_session_to_db(summary)
                st.success(f'Saved session to local DB (session id: {sid})')
            elif use_db and not st.session_state.authenticated:
                st.info('Enable saving by logging in as clinician.')

            csv_bytes = pd.DataFrame([summary]).to_csv(index=False).encode('utf-8')
            st.download_button('Download session CSV', data=csv_bytes, file_name=f'session_{timestamp}.csv', mime='text/csv')

            pdf_next = st.multiselect('Select recommended next steps:', [
                'Schedule psychotherapy appointment', 'Refer to psychiatry for medication consideration', 'Safety planning / crisis resources', 'Routine monitoring in 2 weeks', 'Substance misuse referral'
            ])
            plan_notes = st.text_area('Add clinician notes or plan details (optional)')
            if st.button('Save follow-up plan'):
                if st.session_state.authenticated and use_db:
                    plan = {'timestamp': timestamp, 'next_steps': '; '.join(pdf_next), 'plan_notes': plan_notes}
                    save_followup_to_db(sid, plan)
                    st.success('Saved follow-up plan')
                else:
                    st.info('Log in to save follow-up plan to DB')

            if st.button('Generate PDF report'):
                pdf_bytes = create_pdf_report(summary, pdf_next)
                st.download_button('Download PDF report', data=pdf_bytes, file_name=f'report_{timestamp}.pdf', mime='application/pdf')

# -------------------------
# Tab: Patient History
# -------------------------
with tabs[1]:
    st.header('Patient History & Records')
    if not os.path.exists(DB_PATH):
        st.info('No database yet — run screening and save a session first.')
    else:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql('SELECT * FROM sessions ORDER BY id DESC LIMIT 500', conn)
        conn.close()
        if df.empty:
            st.info('No saved sessions yet.')
        else:
            st.dataframe(df)
            name_filter = st.text_input('Filter by patient name (substring)')
            if name_filter:
                st.dataframe(df[df['name'].str.contains(name_filter, na=False, case=False)])

# -------------------------
# Tab: Model & Diagnostics
# -------------------------
with tabs[2]:
    st.header('Model & Diagnostics')
    st.write('Model status:')
    if model is None:
        st.warning('No model loaded — the app will function using PHQ-9 only.')
    else:
        st.success('Model object loaded (attempt).')
        st.write('Model repr:')
        st.code(repr(model)[:1000])
        mf = get_model_feature_names(model)
        if mf is not None:
            st.markdown('**Detected model feature names (first 200 shown):**')
            st.write(mf[:200])

    st.markdown('---')
    st.subheader('Quick manual test')
    sample_text = st.text_area('Enter sample clinical text to test model (if text mode)')
    test_score = st.slider('Simulated PHQ-9 score for test', 0, 27, 8)
    if st.button('Run quick test'):
        res = model_predict(model, sample_text or '', test_score, input_mode=input_mode)
        st.write(res)

# -------------------------
# Tab: EEG Upload & Analysis
# -------------------------
with tabs[3]:
    st.header("EEG Excel Upload & Analysis")
    st.write("Upload an Excel/CSV where each channel is a column. If file has a time column name it 'Time' or similar; otherwise set sampling rate below.")
    uploaded = st.file_uploader("Upload EEG Excel (xlsx/csv)", type=['xlsx', 'xls', 'csv'])
    sr = st.number_input("Sampling rate (Hz)", min_value=1.0, max_value=5000.0, value=256.0, step=1.0)
    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith('.csv'):
                df = pd.read_csv(uploaded)
            else:
                # require openpyxl for xlsx
                df = pd.read_excel(uploaded)
        except Exception as e:
            st.error(f"Could not read uploaded file: {e}\nInstall openpyxl: pip install openpyxl")
            df = None
        if df is not None:
            st.subheader("Preview (first 10 rows)")
            st.dataframe(df.head(10))
            time_col = detect_time_column(df)
            if time_col is not None:
                st.info(f"Detected time column: {time_col}")
            else:
                st.info("No time column detected — assuming regular sampling at given sampling rate.")

            channel_cols = df.columns.tolist()
            if time_col is not None:
                channel_cols = [c for c in channel_cols if c != time_col]
            st.write(f"Detected channels ({len(channel_cols)}):")
            st.write(channel_cols)

            if len(channel_cols) > 0:
                sel_ch = st.selectbox("Select channel to visualize", options=channel_cols)
                sig = df[sel_ch].dropna().values.astype(float)
                fig, axs = plt.subplots(2, 1, figsize=(8, 4))
                if time_col is not None:
                    t = df[time_col].iloc[:len(sig)].values
                else:
                    t = np.arange(len(sig)) / float(sr)
                axs[0].plot(t[:min(len(t), int(sr*5))], sig[:min(len(sig), int(sr*5))])
                axs[0].set_title(f"Signal preview: {sel_ch} (first 5 sec)")
                axs[0].set_xlabel("Time (s)")
                axs[0].set_ylabel("Amplitude")
                n = len(sig)
                if n >= 4:
                    freqs = np.fft.rfftfreq(n, d=1.0/sr)
                    fft_vals = np.fft.rfft(sig - np.mean(sig))
                    psd = (np.abs(fft_vals) ** 2) / n
                    axs[1].semilogy(freqs, psd)
                    axs[1].set_xlim(0, min(60, sr/2))
                    axs[1].set_xlabel("Frequency (Hz)")
                    axs[1].set_ylabel("PSD")
                else:
                    axs[1].text(0.1, 0.5, "Signal too short for PSD", transform=axs[1].transAxes)
                st.pyplot(fig)

            if st.button("Extract EEG features and run model"):
                try:
                    features_agg, features_by_channel = extract_eeg_features(df, sr=sr, time_col=time_col)
                except Exception as e:
                    st.error(f"Feature extraction failed: {e}")
                    features_agg = None
                    features_by_channel = None
                if features_by_channel is not None:
                    st.subheader("Per-channel features (first 20 channels shown)")
                    st.dataframe(features_by_channel.head(20))
                    st.markdown("### Aggregated features (mean across channels)")
                    st.json(features_agg)

                    csv_feat = features_by_channel.reset_index().to_csv(index=False).encode('utf-8')
                    st.download_button("Download per-channel features CSV", data=csv_feat, file_name=f"eeg_features_{datetime.datetime.utcnow().isoformat()}.csv", mime='text/csv')

                    # Flattening options
                    st.subheader('Flatten features to match model')

                    # --- NEW: parse training notebook button ---
                    st.markdown("If your model was trained in `project1_model.ipynb`, you can extract the exact feature order here:")
                    if st.button("Extract feature list from training notebook"):
                        nb_path = os.path.join(os.getcwd(), 'project1_model.ipynb')
                        if os.path.exists(nb_path):
                            features = parse_notebook_for_feature_list(nb_path)
                            if features:
                                out_path = os.path.join(os.getcwd(), 'extracted_feature_order.txt')
                                try:
                                    with open(out_path, 'w', encoding='utf-8') as f:
                                        for feat in features:
                                            f.write(feat + '\n')
                                    st.success(f"Extracted {len(features)} feature names and saved to {out_path}")
                                    st.download_button("Download extracted feature list", data='\n'.join(features),
                                                       file_name="extracted_feature_order.txt", mime="text/plain")
                                except Exception as e:
                                    st.error(f"Failed to write extracted_feature_order.txt: {e}")
                            else:
                                st.warning("No feature list found in the notebook — check that it defines a list or X.columns.")
                        else:
                            st.error("project1_model.ipynb not found in the app directory.")
                    st.markdown('---')

                    model_feat_names = None
                    # priority: notebook list -> model object -> user upload -> fallback
                    if NOTEBOOK_FEATURE_LIST is not None:
                        model_feat_names = NOTEBOOK_FEATURE_LIST
                        st.info(f"Using feature list discovered in notebook ({len(model_feat_names)} entries).")
                    else:
                        # also allow user to pick an extracted file if it exists
                        extracted_path = os.path.join(os.getcwd(), 'extracted_feature_order.txt')
                        if os.path.exists(extracted_path):
                            try:
                                mf = open(extracted_path, 'r', encoding='utf-8').read().splitlines()
                                if len(mf) > 0:
                                    model_feat_names = mf
                                    st.info(f"Using feature list from {extracted_path}")
                            except Exception:
                                pass
                        if model is not None and model_feat_names is None:
                            model_feat_names = get_model_feature_names(model)
                            if model_feat_names is not None:
                                st.info("Detected model feature names from loaded model.")
                    st.write('Options:')
                    choice = st.radio('Choose flattening strategy', options=['Auto-match to model (if available)', 'Upload feature-name-order file', 'Flatten by channel order (fallback)'])
                    uploaded_featlist = None
                    if choice == 'Upload feature-name-order file':
                        uploaded_featlist = st.file_uploader('Upload a text/CSV file with one feature name per line in the order your model expects', type=['txt','csv'])
                        if uploaded_featlist is not None:
                            try:
                                fl = pd.read_csv(uploaded_featlist, header=None).iloc[:,0].astype(str).tolist()
                                model_feat_names = fl
                                st.success(f'Loaded {len(fl)} feature names from uploaded file')
                            except Exception as e:
                                st.error(f'Could not read feature list: {e}')

                    if choice == 'Auto-match to model (if available)' and model_feat_names is None:
                        st.info('Auto-match selected but no feature names detected. Falling back to flattened channel order when creating vector.')

                    if st.button('Create flattened vector'):
                        ordered_keys, vector, missing = flatten_channel_features(features_by_channel, model_feature_names=model_feat_names)
                        flat_df = pd.DataFrame([vector], columns=ordered_keys)
                        st.subheader('Flattened feature vector (first 200 shown)')
                        st.dataframe(flat_df.iloc[:, :200])

                        csv_flat = flat_df.to_csv(index=False).encode('utf-8')
                        st.download_button('Download flattened feature CSV', data=csv_flat, file_name=f'flat_features_{datetime.datetime.utcnow().isoformat()}.csv', mime='text/csv')

                        # run model if available
                        if model is not None:
                            st.markdown('### Model prediction (using flattened vector)')
                            res = model_predict(model, '', 0, input_mode='eeg', numeric_features=vector)
                            st.write(res)
                        else:
                            st.info('No model loaded — only feature extraction available.')

# -------------------------
# Tab: Settings & Deploy
# -------------------------
with tabs[4]:
    st.header('Settings, Security & Deployment Tips')
    st.subheader('Security notes')
    st.write("""- This demo stores data locally and is not HIPAA-compliant by default. For production use encrypt data at rest, use secure servers, and add authentication & audit logs.""")
    st.subheader('Deployment options')
    st.write("1) Streamlit Cloud: easiest for public or team demos. Add model file to repo or implement secure upload flow.")
    st.write("2) Docker + VPS: containerize the app and run behind HTTPS with a reverse proxy (nginx). Use environment variables or Vault for secrets.")
    st.write("3) Managed platforms: Heroku / Render / AWS Elastic Beanstalk — ensure large model files are stored in object storage (S3) and loaded securely.")
    st.subheader('Suggested next features')
    st.write("""- User roles & RBAC (doctors, admins)
- Audit logs and export to EHR (FHIR)
- Better model versioning and A/B testing UI
- Encrypted database and access controls
- Telemetry and usage dashboards""")

# Footer / credits
st.markdown('---')
st.caption('This app is a demo clinical assistant. Not for standalone diagnostic use. Always combine with clinical judgement and local guidelines.')
