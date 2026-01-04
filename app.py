import joblib
import pandas as pd
from flask import Flask, request, render_template, jsonify
import shap
import numpy as np
import os

app = Flask(__name__)

# --- configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, '..', 'pipeline')

MODELS = {
    'active': os.path.join(MODELS_DIR, 'ensemble_model.pkl'),
    'retired': os.path.join(MODELS_DIR, 'xgb_retired_optimized.pkl')
}
ACTIVE_THRESHOLD = 0.5  # default, will be loaded


RECOMMENDATION_RULES = {
    'CLAIM3YEARS': 'Review recent claim history and offer claims assistance.',
    'P1_EMP_STATUS': 'Update employment details and check for eligible discounts.',
    'BUS_USE': 'Verify commercial usage terms and offer business policy add-ons.',
    'AD_BUILDINGS': 'Review building accidental damage coverage limits.',
    'RISK_RATED_AREA_B': 'Assess regional risk factors and suggest mitigation measures.',
    'MTA_FAP': 'Review mid-term adjustments and premium impacts.',
    'LAST_ANN_PREM_GROSS': 'Offer loyalty discount or flexible payment plan.',
    'P1_AGE': 'Check age-related policy benefits (e.g., senior discounts).',
    'PROP_AGE': 'Suggest property maintenance review or updated valuation.',
    'DAYS_TO_BIND': 'Streamline binding process for future renewals.',
    'PAYMENT_PureDD': 'Suggest switching to automated payments for convenience.',
    'NCD_GRANTED_YEARS_B': 'Highlight No Claims Discount benefits.',
    'LEGAL_ADDON_POST_REN': 'Highlight value of legal protection in case of disputes.',
    'LEGAL_ADDON_PRE_REN': 'Highlight value of legal protection in case of disputes.',
    'MAX_DAYS_UNOCC': 'Discuss occupancy terms and range of unoccupancy cover.',
    'SUM_INSURED_CONTENTS': 'Review coverage limits to ensure adequate protection.',
    'RISK_RATED_AREA_B': 'Explain regional risk factors affecting premium.',
    'RISK_RATED_AREA_C': 'Explain regional risk factors affecting premium.',
}


global_data_cache = None
global_filename = None
PROCESSED_DATA_CACHE = {}  # {filename: {'df': dataframe, 'ranges': ranges}}



loaded_models = {}

def load_models():
    """Load models into memory on startup."""
    for key, path in MODELS.items():
        try:
            loaded_models[key] = joblib.load(path)
            print(f"Loaded {key} model successfully.")
        except Exception as e:
            print(f"Error loading {key} model: {e}")

    # load active threshold
    global ACTIVE_THRESHOLD
    try:
        thresh_path = os.path.join(MODELS_DIR, 'ensemble_threshold.txt')
        with open(thresh_path, 'r') as f:
            ACTIVE_THRESHOLD = float(f.read().strip())
        print(f"Loaded Active Threshold: {ACTIVE_THRESHOLD}")
    except Exception as e:
        print(f"Error loading threshold, using default 0.5: {e}")


FEATURE_RANGES = {}

def compute_ranges(df):
    """Dynamically compute min/max ranges for the provided dataframe."""
    ranges = {}
    try:
        for col in FEATURE_ORDER:
            if col not in df.columns:
                continue
            
            # check for binary
            unique_vals = df[col].dropna().unique()
            is_binary = False
            if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                is_binary = True
            
            if is_binary:
                ranges[col] = {'type': 'Binary', 'text': 'Binary (0/1)'}
            elif pd.api.types.is_numeric_dtype(df[col]):
                # numeric range
                vmin = df[col].min()
                vmax = df[col].max()
                ranges[col] = {
                    'type': 'Numeric', 
                    'text': f"Range: {vmin:.1f} - {vmax:.1f}"
                }
            else:
                # categorical or other
                ranges[col] = {'type': 'Categorical', 'text': 'Categorical'}
    except Exception as e:
        print(f"Error computing ranges: {e}")
    return ranges

FEATURE_MAPPING = {}

def load_feature_mapping():
    """load variable descriptions from text file."""
    global FEATURE_MAPPING
    try:
        # Assuming the file is in the parent directory of Web/
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base_dir, '..', 'variable_descriptions.txt')
        
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or ':' not in line:
                    continue
                # split only on the first colon
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    val = parts[1].strip()
                    FEATURE_MAPPING[key] = val
        print(f"Loaded {len(FEATURE_MAPPING)} feature descriptions.")
    except Exception as e:
        print(f"Error loading feature mapping: {e}")


FEATURE_ORDER = [
    'CLAIM3YEARS', 'BUS_USE', 'CLERICAL', 'AD_BUILDINGS', 'RISK_RATED_AREA_B',
    'SUM_INSURED_BUILDINGS', 'NCD_GRANTED_YEARS_B', 'AD_CONTENTS', 'RISK_RATED_AREA_C',
    'SUM_INSURED_CONTENTS', 'NCD_GRANTED_YEARS_C', 'CONTENTS_COVER', 'BUILDINGS_COVER',
    'SPEC_SUM_INSURED', 'SPEC_ITEM_PREM', 'UNSPEC_HRP_PREM', 'P1_POLICY_REFUSED',
    'P1_SEX', 'APPR_ALARM', 'APPR_LOCKS', 'BEDROOMS', 'ROOF_CONSTRUCTION',
    'WALL_CONSTRUCTION', 'FLOODING', 'LISTED', 'MAX_DAYS_UNOCC', 'NEIGH_WATCH',
    'OWNERSHIP_TYPE', 'PAYING_GUESTS', 'PROP_TYPE', 'SAFE_INSTALLED', 'SEC_DISC_REQ',
    'SUBSIDENCE', 'YEARBUILT', 'PAYMENT_FREQUENCY',
    'LEGAL_ADDON_PRE_REN', 'LEGAL_ADDON_POST_REN', 'HOME_EM_ADDON_PRE_REN',
    'HOME_EM_ADDON_POST_REN', 'GARDEN_ADDON_PRE_REN', 'GARDEN_ADDON_POST_REN',
    'KEYCARE_ADDON_PRE_REN', 'KEYCARE_ADDON_POST_REN', 'HP1_ADDON_PRE_REN',
    'HP1_ADDON_POST_REN', 'HP2_ADDON_PRE_REN', 'HP2_ADDON_POST_REN',
    'HP3_ADDON_PRE_REN', 'HP3_ADDON_POST_REN', 'MTA_FLAG', 'MTA_FAP', 'MTA_APRP',
    'LAST_ANN_PREM_GROSS', 'POL_STATUS', 'HAS_MTA', 'HAS_ADJUSTED', 'P1_AGE', 'PROP_AGE', 'DAYS_TO_BIND',
    'IS_STANDARD_OCCUPANCY', 'HAS_PT_EMP', 'MAR_STATUS_Couple', 'MAR_STATUS_Solo',
    'DAYS_SINCE_MTA', 'QUOTE_MONTH', 'QUOTE_WEEKDAY',
    'PAYMENT_METHOD',
    'EMP_STATUS_Carer', 'EMP_STATUS_Disabled', 'EMP_STATUS_Employed', 
    'EMP_STATUS_Houseperson', 'EMP_STATUS_None', 'EMP_STATUS_Other', 
    'EMP_STATUS_Student', 'EMP_STATUS_Unemployed', 'EMP_STATUS_Voluntary'
]

def shap_to_delta_percent(shap_value, base_value):
    # force conversion to float
    base_prob = 1.0 / (1.0 + np.exp(-float(base_value)))
    feature_prob = 1.0 / (1.0 + np.exp(-(float(base_value) + float(shap_value))))
    delta_percent = (feature_prob - base_prob) * 100
    return round(delta_percent, 2)

def process_dataset_internal(local_filename):
    """Helper to process a dataset and return the df + ranges, using cache if available."""
    if local_filename in PROCESSED_DATA_CACHE:
        print(f"accessing cache for {local_filename}")
        return PROCESSED_DATA_CACHE[local_filename]

    print(f"Processing {local_filename} (Cold Start)...")
    
    # Determine model type
    if 'active' in local_filename.lower():
        user_type = 'active'
    elif 'retired' in local_filename.lower():
        user_type = 'retired'
    else:
        raise ValueError("Filename must contain 'active' or 'retired'")

    model = loaded_models.get(user_type)
    if not model:
        raise ValueError(f"Model {user_type} not loaded")

    file_path = os.path.join(DATA_DIR, local_filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found")

    df = pd.read_csv(file_path)
    df.columns = df.columns.astype(str)

    # ... Prediction Logic ...
    # Re-using the logic from the previous analyze function, but cleaner
    
    # Feature Alignment
    feature_names = None
    if hasattr(model, 'feature_names_in_'):
        feature_names = model.feature_names_in_
    elif hasattr(model, 'get_booster'):
         try: feature_names = model.get_booster().feature_names
         except: pass

    if feature_names is not None:
        feature_names = [str(x) for x in feature_names]
        missing_cols = [c for c in feature_names if c not in df.columns]
        for c in missing_cols: df[c] = 0
        X = df[feature_names]
    else:
        cols_to_use = [c for c in FEATURE_ORDER if c in df.columns]
        X = df[cols_to_use]
    
    X.columns = X.columns.astype(str)

    # Predict
    predictions = model.predict(X)
    try:
        probs = model.predict_proba(X)[:, 1]
        df['Churn Probability Raw'] = [float(p)*100 for p in probs]
    except:
        df['Churn Probability Raw'] = 0.0

    # Status Classification
    thresh = ACTIVE_THRESHOLD if user_type == 'active' else 0.5
    def classify(p):
        prob = p / 100.0
        if prob < thresh: return 'Retain'
        elif prob < 0.85: return 'Likely Churn'
        else: return 'Certain Churn'

    df['Status'] = df['Churn Probability Raw'].apply(classify)
    df['Prediction'] = predictions.astype(int)
    df['Churn Probability'] = df['Churn Probability Raw'].apply(lambda p: f"{p:.2f}%")

    # SHAP
    explainer_model = model
    if hasattr(model, 'estimators_'):
        try:
             if 'xgb' in model.named_estimators_: explainer_model = model.named_estimators_['xgb']
             else: explainer_model = model.estimators_[0]
        except: pass
            
    explainer = shap.TreeExplainer(explainer_model)
    # limit shap calculation for speed if very large? No, user wants full data.
    # But for 'retired' (30k rows), it takes time. 
    shap_values = explainer.shap_values(X)
    base_value = explainer.expected_value
    
    is_list = isinstance(shap_values, list)
    
    # vectorized/fast-loop construction of top features
    all_shap_data = [] # top n
    all_shap_full = [] # full dict
    
    # We can optimize this loop, but keeping it robust for now
    top_n = 5
    
    for i in range(len(X)):
        if is_list: current_vals = shap_values[1][i]
        else: current_vals = shap_values[i]
        
        feats = X.columns.tolist()
        contributions = dict(zip(feats, current_vals))
        
        # Sort and take top 5
        sorted_feats = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        top_list = [[str(f), shap_to_delta_percent(v, base_value)] for f, v in sorted_feats]
        
        all_shap_data.append(top_list)
        all_shap_full.append({f: shap_to_delta_percent(v, base_value) for f, v in contributions.items()})

    df['shap_top_features'] = all_shap_data
    df['shap_full'] = all_shap_full

    ranges = compute_ranges(df)
    
    result = {'df': df, 'ranges': ranges}
    PROCESSED_DATA_CACHE[local_filename] = result
    print(f"Finished processing {local_filename}")
    return result

def warmup_cache():
    """Pre-load critical datasets."""
    print("--- WARMING UP CACHE (This may take a minute) ---")
    datasets = ['active_clean.csv', 'retired_clean.csv']
    for ds in datasets:
        try:
            process_dataset_internal(ds)
        except Exception as e:
            print(f"Skipping {ds}: {e}")
    print("--- CACHE WARMUP COMPLETE ---")


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/get_local_datasets')
def get_local_datasets():
    """Return specific CSV files available in the modeling directory."""
    try:
        if not os.path.exists(DATA_DIR):
            return jsonify({"files": [], "error": f"Directory not found: {DATA_DIR}"})
            
        # Only allow specific files as requested
        allowed_files = ['active_clean.csv', 'retired_clean.csv']
        files = [f for f in os.listdir(DATA_DIR) if f in allowed_files]
        files.sort()
        return jsonify({"files": files})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/analyze', methods=['POST'])
def analyze():
    # Only accepts local_file now, effectively just a switch command
    local_filename = request.form.get('local_file')
    if not local_filename:
        return jsonify({"error": "No file specified"}), 400

    try:
        # Get from cache (instant) or process (fallback)
        result = process_dataset_internal(local_filename)
        
        global global_data_cache, global_filename, FEATURE_RANGES
        global_data_cache = result['df']
        global_filename = local_filename
        FEATURE_RANGES = result['ranges']
        
        return jsonify({"success": True, "filename": local_filename})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_data', methods=['POST'])
def get_data():
    global global_data_cache, global_filename
    
    if global_data_cache is None:
        return jsonify({"error": "no data loaded"}), 400
        
    req = request.json
    
    sort_by = req.get('sort_by', 'id_asc')
    status_filter = req.get('status_filter', 'all')
    
    try:
        limit = int(req.get('limit', 50))
    except:
        limit = 50
    
    # filter
    df_filtered = global_data_cache.copy()
    if status_filter != 'all':
        df_filtered = df_filtered[df_filtered['Status'] == status_filter]
        
    # sort
    if sort_by == 'prob_desc':
        df_filtered = df_filtered.sort_values(by='Churn Probability Raw', ascending=False)
    elif sort_by == 'prob_asc':
        df_filtered = df_filtered.sort_values(by='Churn Probability Raw', ascending=True)
    elif sort_by == 'id_desc':
        df_filtered = df_filtered.sort_index(ascending=False)
    else:
        # default: id ascending
        df_filtered = df_filtered.sort_index(ascending=True)

    # pagination: return limit rows
    df_page = df_filtered.head(limit).copy()
    
    # ensure id matches original row number
    df_page['original_id'] = df_page.index + 1
    
    # on-demand shap calculation removed (reverted to pre-computed)
    # the columns 'shap_full' and 'shap_top_features' already exist from analyze()

    # convert to dictionary
    result = df_page.to_dict(orient='records')
    
    # send metadata to frontend instead of mapping keys here
    meta = {
        'descriptions': FEATURE_MAPPING,
        'ranges': FEATURE_RANGES
    }
    
    # insert recommendation logic
    for r in result:
        # find top positive shap feature from the now-computed shap_top_features
        # just grab the first positive one from top 5
        top_driver = None
        if 'shap_top_features' in r and r['shap_top_features']:
            for name, val_str in r['shap_top_features']:
                # values are strings like "+15.2" or "-5.0"
                try:
                    val = float(val_str)
                    if val > 0:
                        top_driver = name
                        break
                except: continue
        
        if top_driver and top_driver in RECOMMENDATION_RULES:
            r['Recommendation'] = RECOMMENDATION_RULES[top_driver]
        else:
            r['Recommendation'] = "Standard Retention Check"

    return jsonify({"rows": result, "meta": meta, "filename": global_filename})

@app.route('/reset', methods=['POST'])
def reset_analysis():
    global global_data_cache, global_filename
    global_data_cache = None
    global_filename = None
    return jsonify({"success": True})

@app.route('/get_dashboard_stats', methods=['GET'])
def get_dashboard_stats():
    global global_data_cache
    if global_data_cache is None:
        return jsonify({"error": "No data analyzed yet"}), 400

    df = global_data_cache
    
    # 1. summary metrics
    total = len(df)
    churn_count = len(df[df['Status'] == 'Certain Churn'])
    churn_rate = round((churn_count / total) * 100, 1)
    
    # financial impact (value at risk)
    # using LAST_ANN_PREM_GROSS as proxy for value
    risk_likely = 0.0
    risk_certain = 0.0
    value_at_risk = 0.0
    
    if 'LAST_ANN_PREM_GROSS' in df.columns:
        likely_mask = df['Status'] == 'Likely Churn'
        certain_mask = df['Status'] == 'Certain Churn'
        
        risk_likely = df[likely_mask]['LAST_ANN_PREM_GROSS'].sum()
        risk_certain = df[certain_mask]['LAST_ANN_PREM_GROSS'].sum()
        value_at_risk = risk_likely + risk_certain
    
    # approx revenue saved if we save 20% of at-risk
    revenue_saved = value_at_risk * 0.20
    
    # 2. aggregated shap insights
    # we need to aggregate the 'shap_full' column which contains dicts
    # extracting this is a bit heavy, let's optimize if possible.
    # actually 'shap_full' is a list of dicts.
    
    shap_series = df['shap_full']
    
    # logic: sum up shap values per feature across all rows
    feature_sums = {}
    feature_counts = {}
    
    # initialize with 0
    for feat in FEATURE_ORDER:
        feature_sums[feat] = 0.0
        feature_counts[feat] = 0
        
    # python loop might be slow for huge files, but ok for demo <10k rows
    # vectorized approach is hard because shap_full is json/dict in dataframe
    # but we can reconstruct if we had the raw matrix. 
    # for now, let's just sample if it's too big, or just iterate.
    
    limit = 2000 # sample size for dashboard speed if large
    if total > limit:
         sample_indices = np.random.choice(total, limit, replace=False)
         subset = shap_series.iloc[sample_indices]
    else:
         subset = shap_series

    for row_dict in subset:
        for feat, val in row_dict.items():
            feature_sums[feat] += val

    # calculate averages
    n = len(subset)
    avg_impact = {k: v/n for k, v in feature_sums.items()}
    
    # sort for drivers
    # top churn drivers: largest positive values
    sorted_features = sorted(avg_impact.items(), key=lambda x: x[1], reverse=True)
    
    top_churn = []
    for f, val in sorted_features:
        if val > 0:
            name = FEATURE_MAPPING.get(f, f)
            top_churn.append({'feature': name, 'impact': round(val, 2)})
        if len(top_churn) >= 5: break
            
    # top retention drivers: largest negative values (smallest numbers)
    sorted_retention = sorted(avg_impact.items(), key=lambda x: x[1])
    
    top_retention = []
    for f, val in sorted_retention:
        if val < 0:
            name = FEATURE_MAPPING.get(f, f)
            # make positive for the chart (amount of retention force)
            top_retention.append({'feature': name, 'impact': round(abs(val), 2)})
        if len(top_retention) >= 5: break

    return jsonify({
        "summary": {
            "total_customers": total,
            "churn_rate": churn_rate,
            "risk_likely": round(risk_likely, 2),
            "risk_certain": round(risk_certain, 2),
            "value_at_risk": round(value_at_risk, 2),
            "revenue_saved": round(revenue_saved, 2)
        },
        "churn_drivers": top_churn,
        "retention_drivers": top_retention
    })





if __name__ == "__main__":
    load_models()
    load_feature_mapping()
    warmup_cache()
    app.run(debug=True, use_reloader=False) # Access reloader false to avoid double warmup

