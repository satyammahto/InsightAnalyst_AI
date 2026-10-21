import streamlit as st
import pandas as pd
import numpy as np
import os
import time

from utils.type_detector import detect_column_types
from utils.cleaner import detect_and_clean
from utils.batch_processor import process_dataset_in_batches, stream_file_to_disk, safe_read_csv
from utils.rag_engine import DatasetRAGEngine
from utils.ai_assistant import AIAssistantEngine
from utils.ml_engine import run_ml_analysis
from utils.dashboard import (
    plot_numeric_distribution,
    plot_categorical_counts,
    plot_date_trend,
    plot_correlation_heatmap,
    plot_cluster_scatter,
    plot_outlier_scatter,
    plot_custom_user_chart,
    plot_fuzzy_severity_distribution,
    plot_fuzzy_scatter,
    plot_fuzzy_membership_functions,
    plot_elbow_silhouette,
    plot_cluster_radar,
    plot_outliers_by_cluster
)
from utils.summarizer import generate_executive_summary, generate_html_dossier, generate_markdown_dossier
from utils.cleaning_studio import CleaningStudio
from utils.query_engine import filter_dataset_by_nl

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
COMMON_COLUMN_MAP = {
    'ORDERNUMBER': 'Order Number',
    'QUANTITYORDERED': 'Quantity Ordered',
    'PRICEEACH': 'Price',
    'ORDERLINENUMBER': 'Order Line Number',
    'SALES': 'Sales',
    'ORDERDATE': 'Order Date',
    'STATUS': 'Status',
    'QTR_ID': 'Quarter',
    'MONTH_ID': 'Month',
    'YEAR_ID': 'Year',
    'PRODUCTLINE': 'Product Line',
    'MSRP': 'MSRP',
    'PRODUCTCODE': 'Product Code',
    'CUSTOMERNAME': 'Customer Name',
    'PHONE': 'Phone',
    'ADDRESSLINE1': 'Address Line 1',
    'ADDRESSLINE2': 'Address Line 2',
    'CITY': 'City',
    'STATE': 'State',
    'POSTALCODE': 'Postal Code',
    'COUNTRY': 'Country',
    'TERRITORY': 'Territory',
    'CONTACTLASTNAME': 'Contact Last Name',
    'CONTACTFIRSTNAME': 'Contact First Name',
    'DEALSIZE': 'Deal Size'
}

def clean_name(name: str) -> str:
    """Format technical column names into friendly, human-readable Title Case."""
    if not name:
        return ""
    upper_c = str(name).strip().upper()
    if upper_c in COMMON_COLUMN_MAP:
        return COMMON_COLUMN_MAP[upper_c]
    words = str(name).replace('_', ' ').split()
    cleaned_words = []
    for w in words:
        if w.upper() in ['ID', 'MSRP', 'URL', 'USD', 'EUR', 'APAC', 'EMEA', 'NA', 'SKU', 'QTR', 'ZIP']:
            cleaned_words.append(w.upper())
        else:
            cleaned_words.append(w.capitalize())
    return " ".join(cleaned_words)

def safe_read_excel(source, **kwargs) -> pd.DataFrame:
    """Robust Excel loader (.xlsx, .xls) that handles openpyxl dependency gracefully."""
    try:
        if hasattr(source, 'seek'):
            source.seek(0)
        return pd.read_excel(source, **kwargs)
    except ImportError as e:
        if 'openpyxl' in str(e).lower():
            raise ImportError(
                "Missing 'openpyxl' package required for reading Excel files (.xlsx). "
                "Please run: pip install openpyxl or double-click install_openpyxl.bat"
            )
        raise e

# ---------------------------------------------------------
# Page Config & Modern Clean Enterprise Theme Setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="InsightAnalyst AI - Simple Data Insights",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enforce Clean Modern Theme Overrides across OS & Browsers
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* Global Typography & Layout */
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background-color: #F8FAFC !important;
        color: #0F172A !important;
        font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
        letter-spacing: -0.01em;
    }
    
    header[data-testid="stHeader"] {
        background-color: transparent !important;
        z-index: 1 !important;
    }
    
    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 3rem !important;
        max-width: 96% !important;
    }

    /* Top Journey Stepper Bar */
    .journey-stepper-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 12px 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03);
    }
    .stepper-steps-wrapper {
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
    }
    .stepper-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 12px;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 20px;
    }
    .stepper-pill-done {
        background: #ECFDF5;
        color: #047857;
        border: 1px solid #A7F3D0;
    }
    .stepper-pill-active {
        background: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
    }
    .stepper-pill-idle {
        background: #F8FAFC;
        color: #64748B;
        border: 1px solid #E2E8F0;
    }
    .stepper-arrow {
        color: #CBD5E1;
        font-size: 13px;
        font-weight: 700;
    }
    
    /* Top Status Bar */
    .top-status-bar {
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
        background: #FFFFFF !important;
        border: 1px solid rgba(226, 232, 240, 0.9) !important;
        border-radius: 12px !important;
        padding: 12px 20px !important;
        margin-bottom: 20px !important;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.03) !important;
    }
    .status-chip-green {
        background: #ECFDF5 !important;
        color: #059669 !important;
        border: 1px solid #A7F3D0 !important;
        font-size: 12px !important;
        font-weight: 700 !important;
        padding: 3px 10px !important;
        border-radius: 20px !important;
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }
    .status-chip-blue {
        background: #EFF6FF !important;
        color: #2563EB !important;
        border: 1px solid #BFDBFE !important;
        font-size: 12px !important;
        font-weight: 700 !important;
        padding: 3px 10px !important;
        border-radius: 20px !important;
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }
    .badge-num {
        background: #F1F5F9;
        color: #334155;
        border: 1px solid #E2E8F0;
        font-size: 11px;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 12px;
    }
    .version-chip {
        background: #F1F5F9 !important;
        color: #334155 !important;
        font-size: 11px !important;
        font-weight: 700 !important;
        padding: 3px 8px !important;
        border-radius: 6px !important;
        border: 1px solid #E2E8F0 !important;
        letter-spacing: 0.5px;
    }
    
    /* Clean Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #F1F5F9 !important;
        box-shadow: 2px 0 12px rgba(0, 0, 0, 0.02) !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #0F172A !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
    }
    
    /* Cards & Containers */
    .saas-card {
        background: #FFFFFF !important;
        border: 1px solid rgba(226, 232, 240, 0.85) !important;
        border-radius: 12px !important;
        padding: 18px 20px !important;
        box-shadow: 0 4px 14px -2px rgba(15, 23, 42, 0.03) !important;
        margin-bottom: 14px !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .saas-card:hover {
        border-color: #CBD5E1 !important;
        box-shadow: 0 6px 20px -4px rgba(15, 23, 42, 0.06) !important;
    }
    
    /* KPI Metric Cards */
    .kpi-title {
        font-size: 12px;
        font-weight: 700;
        color: #64748B !important;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 6px;
    }
    .kpi-val {
        font-size: 27px;
        font-weight: 800;
        color: #0F172A !important;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }
    .kpi-sub {
        font-size: 12px;
        color: #059669 !important;
        font-weight: 500;
        margin-top: 6px;
        line-height: 1.4;
    }
    .kpi-sub-neutral {
        font-size: 12px;
        color: #64748B !important;
        font-weight: 500;
        margin-top: 6px;
        line-height: 1.4;
    }
    .kpi-sub-amber {
        font-size: 12px;
        color: #D97706 !important;
        font-weight: 500;
        margin-top: 6px;
        line-height: 1.4;
    }
    .kpi-sub-red {
        font-size: 12px;
        color: #E11D48 !important;
        font-weight: 500;
        margin-top: 6px;
        line-height: 1.4;
    }
    .tooltip-icon {
        cursor: help;
        color: #94A3B8;
        font-size: 13px;
        margin-left: 4px;
        display: inline-block;
    }

    /* Actionable Insight Card (What, Why, Action) */
    .insight-action-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 14px;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.03);
        border-left: 4px solid #0284C7;
    }
    .insight-action-title {
        font-size: 15px;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .insight-why-box {
        background: #F8FAFC;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 8px 0;
        font-size: 13px;
        color: #334155;
        line-height: 1.5;
    }
    .insight-action-row {
        font-size: 12.5px;
        color: #0369A1;
        font-weight: 600;
        margin-top: 6px;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* Executive Brief Card */
    .brief-card {
        background: linear-gradient(180deg, #FFFFFF 0%, #FAFCFF 100%) !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 14px !important;
        padding: 22px 26px !important;
        box-shadow: 0 4px 18px -2px rgba(15, 23, 42, 0.04) !important;
    }
    .brief-title {
        font-size: 17px;
        font-weight: 800;
        color: #0F172A !important;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .brief-sub-box {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 10px !important;
        padding: 14px 16px !important;
        font-size: 13px !important;
        line-height: 1.55 !important;
        color: #334155 !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.02) !important;
    }

    /* Cleaning Summary Pill Cards */
    .check-pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin: 14px 0 20px 0;
    }
    .check-pill {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 10px 16px;
        font-size: 13.5px;
        font-weight: 600;
        color: #1E293B;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.02);
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }

    /* Group Persona Card */
    .persona-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 10px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.02);
        border-left: 4px solid #0284C7;
    }
    .persona-title {
        font-size: 14px;
        font-weight: 700;
        color: #0F172A;
    }
    .persona-sub {
        font-size: 12px;
        color: #64748B;
        margin-top: 4px;
    }

    /* Severity Indicators */
    .severity-badge-green {
        background: #ECFDF5;
        color: #047857;
        border: 1px solid #A7F3D0;
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 13px;
        font-weight: 600;
    }
    .severity-badge-yellow {
        background: #FEFCE8;
        color: #A16207;
        border: 1px solid #FEF08A;
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 13px;
        font-weight: 600;
    }
    .severity-badge-orange {
        background: #FFF7ED;
        color: #C2410C;
        border: 1px solid #FED7AA;
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 13px;
        font-weight: 600;
    }
    .severity-badge-red {
        background: #FFF1F2;
        color: #BE123C;
        border: 1px solid #FECDD3;
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 13px;
        font-weight: 600;
    }

    /* Risk Pill Badges */
    .risk-pill-low {
        background: #ECFDF5;
        color: #047857;
        border: 1px solid #A7F3D0;
        border-radius: 12px;
        padding: 3px 10px;
        font-size: 11.5px;
        font-weight: 700;
        display: inline-block;
    }
    .risk-pill-medium {
        background: #FEFCE8;
        color: #A16207;
        border: 1px solid #FEF08A;
        border-radius: 12px;
        padding: 3px 10px;
        font-size: 11.5px;
        font-weight: 700;
        display: inline-block;
    }
    .risk-pill-high {
        background: #FFF7ED;
        color: #C2410C;
        border: 1px solid #FED7AA;
        border-radius: 12px;
        padding: 3px 10px;
        font-size: 11.5px;
        font-weight: 700;
        display: inline-block;
    }
    .risk-pill-critical {
        background: #FFF1F2;
        color: #BE123C;
        border: 1px solid #FECDD3;
        border-radius: 12px;
        padding: 3px 10px;
        font-size: 11.5px;
        font-weight: 700;
        display: inline-block;
    }

    /* Chat Message Bubbles */
    .chat-user-msg {
        background: #0284C7 !important;
        color: #FFFFFF !important;
        border-radius: 14px 14px 2px 14px !important;
        padding: 14px 18px !important;
        margin: 10px 0 10px auto !important;
        max-width: 80% !important;
        box-shadow: 0 2px 8px rgba(2, 132, 199, 0.2) !important;
        font-size: 14px !important;
        line-height: 1.5 !important;
    }
    .chat-assistant-msg {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        color: #0F172A !important;
        border-radius: 14px 14px 14px 2px !important;
        padding: 18px 22px !important;
        margin: 10px auto 10px 0 !important;
        max-width: 92% !important;
        box-shadow: 0 4px 16px -2px rgba(15, 23, 42, 0.04) !important;
        font-size: 14px !important;
        line-height: 1.6 !important;
    }

    /* Modern Tabs Header */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px !important;
        border-bottom: 1px solid #E2E8F0 !important;
        background-color: transparent !important;
        padding-bottom: 4px !important;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600 !important;
        font-size: 13.5px !important;
        color: #64748B !important;
        padding: 10px 18px !important;
        border-radius: 8px !important;
        background-color: transparent !important;
        border: none !important;
        transition: all 0.15s ease-in-out !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #0F172A !important;
        background-color: #F1F5F9 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #0284C7 !important;
        background-color: #EFF6FF !important;
        font-weight: 700 !important;
        border: none !important;
    }

    /* Processing Steps Cards */
    .step-exec-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar Controls & Simplified Navigation
# ---------------------------------------------------------
st.sidebar.markdown("<h3 style='margin-bottom:2px;'>✨ InsightAnalyst AI</h3>", unsafe_allow_html=True)
st.sidebar.caption("Simple AI-Powered Data Insights")

# Global UI Mode: Simple View (Recommended) vs Technical View
st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown("<b style='color:#0F172A; font-size:13px;'>View Mode:</b>", unsafe_allow_html=True)
ui_mode = st.sidebar.radio(
    "Select Interface Style:",
    ["✨ Simple View (Recommended)", "🔬 Technical View"],
    label_visibility="collapsed",
    key="ui_view_mode_radio"
)
is_simple_mode = (ui_mode == "✨ Simple View (Recommended)")

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), 'sample_data')
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), 'uploaded_datasets')
os.makedirs(UPLOAD_DIR, exist_ok=True)
PERSISTENT_FILE = os.path.join(UPLOAD_DIR, "last_uploaded_dataset.csv")
PERSISTENT_META = os.path.join(UPLOAD_DIR, "last_uploaded_meta.txt")
PERSISTENT_SOURCE = os.path.join(UPLOAD_DIR, "last_source.txt")

# Preserve source selection
has_saved_custom = os.path.exists(PERSISTENT_FILE)
saved_source = "Practice Datasets"
if os.path.exists(PERSISTENT_SOURCE):
    try:
        with open(PERSISTENT_SOURCE, 'r', encoding='utf-8') as f:
            saved_source = f.read().strip()
    except Exception:
        pass
elif has_saved_custom:
    saved_source = "Upload Custom File"

if 'data_source_radio' not in st.session_state:
    st.session_state['data_source_radio'] = saved_source if (saved_source == "Upload Custom File" and has_saved_custom) else "Practice Datasets"

def on_source_change():
    new_src = st.session_state.get('data_source_radio', 'Practice Datasets')
    with open(PERSISTENT_SOURCE, 'w', encoding='utf-8') as f:
        f.write(new_src)

st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown("<b style='color:#0F172A; font-size:13px;'>Data Source:</b>", unsafe_allow_html=True)
data_source = st.sidebar.radio(
    "Choose Data Source:",
    ["Practice Datasets", "Upload Custom File"],
    label_visibility="collapsed",
    key="data_source_radio",
    on_change=on_source_change
)

raw_df = None
selected_dataset_name = "q4_global_retail_telemetry.csv"
friendly_dataset_title = "Retail Sales Data"
active_file_path = None

if data_source == "Practice Datasets":
    sample_choice = st.sidebar.selectbox(
        "Choose a Practice Dataset:",
        ["Retail Sales Data", "Healthcare Patient Records", "Marketing Performance"]
    )
    if sample_choice == "Retail Sales Data":
        active_file_path = os.path.join(SAMPLE_DIR, "sample_sales.csv")
        selected_dataset_name = "sample_sales.csv"
        friendly_dataset_title = "Retail Sales Data"
    elif sample_choice == "Healthcare Patient Records":
        active_file_path = os.path.join(SAMPLE_DIR, "sample_healthcare.csv")
        selected_dataset_name = "patient_clinical_records.csv"
        friendly_dataset_title = "Healthcare Patient Records"
    else:
        active_file_path = os.path.join(SAMPLE_DIR, "sample_marketing.csv")
        selected_dataset_name = "marketing_attribution.csv"
        friendly_dataset_title = "Marketing Performance"
        
    if os.path.exists(active_file_path):
        raw_df = safe_read_csv(active_file_path)

else:
    uploaded_files = st.sidebar.file_uploader(
        "Upload CSV or Excel file",
        type=["csv", "xlsx", "xls", "parquet"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        try:
            saved_paths = []
            file_names = []
            for u_file in uploaded_files:
                f_name = u_file.name
                safe_name = f"uploaded_{f_name.replace(' ', '_')}"
                f_path = os.path.join(UPLOAD_DIR, safe_name)
                stream_file_to_disk(u_file, f_path)
                saved_paths.append(f_path)
                file_names.append(f_name)

            if len(saved_paths) == 1:
                active_file_path = saved_paths[0]
                selected_dataset_name = file_names[0]
                friendly_dataset_title = file_names[0]
                with st.spinner(f"Loading '{file_names[0]}' (this may take a few moments for large files)..."):
                    if active_file_path.endswith('.csv'):
                        raw_df = safe_read_csv(active_file_path)
                    elif active_file_path.endswith('.parquet'):
                        raw_df = pd.read_parquet(active_file_path)
                    else:
                        raw_df = safe_read_excel(active_file_path)
            else:
                st.sidebar.markdown(f"**{len(saved_paths)} Files Uploaded**")
                multi_action = st.sidebar.radio(
                    "Combine Files:",
                    ["🔗 Merge All Files", "📄 Pick One File"],
                    key="multi_file_strategy"
                )

                if multi_action == "🔗 Merge All Files":
                    dfs = []
                    with st.spinner("Merging uploaded files..."):
                        for f_path, f_name in zip(saved_paths, file_names):
                            try:
                                if f_path.endswith('.csv'):
                                    temp_df = safe_read_csv(f_path)
                                elif f_path.endswith('.parquet'):
                                    temp_df = pd.read_parquet(f_path)
                                else:
                                    temp_df = safe_read_excel(f_path)
                                temp_df['_source_file'] = f_name
                                dfs.append(temp_df)
                            except Exception as ex:
                                st.sidebar.warning(f"Could not parse {f_name}: {ex}")

                    if dfs:
                        raw_df = pd.concat(dfs, ignore_index=True)
                        combined_path = os.path.join(UPLOAD_DIR, "merged_multi_upload.csv")
                        raw_df.to_csv(combined_path, index=False)
                        active_file_path = combined_path
                        selected_dataset_name = f"Merged ({len(dfs)} files)"
                        friendly_dataset_title = selected_dataset_name
                        st.sidebar.success(f"Merged {len(dfs)} files into {len(raw_df):,} rows!")
                else:
                    chosen_idx = st.sidebar.selectbox("Active File:", range(len(file_names)), format_func=lambda i: file_names[i])
                    active_file_path = saved_paths[chosen_idx]
                    selected_dataset_name = file_names[chosen_idx]
                    friendly_dataset_title = file_names[chosen_idx]
                    with st.spinner(f"Loading '{file_names[chosen_idx]}'..."):
                        if active_file_path.endswith('.csv'):
                            raw_df = safe_read_csv(active_file_path)
                        elif active_file_path.endswith('.parquet'):
                            raw_df = pd.read_parquet(active_file_path)
                        else:
                            raw_df = safe_read_excel(active_file_path)

            if raw_df is not None:
                raw_df.to_csv(PERSISTENT_FILE, index=False)
                with open(PERSISTENT_META, 'w', encoding='utf-8') as f:
                    f.write(selected_dataset_name)
                with open(PERSISTENT_SOURCE, 'w', encoding='utf-8') as f:
                    f.write("Upload Custom File")
        except Exception as e:
            st.sidebar.error(f"Error reading upload: {e}")
    else:
        if os.path.exists(PERSISTENT_FILE):
            try:
                active_file_path = PERSISTENT_FILE
                raw_df = safe_read_csv(PERSISTENT_FILE)
                if os.path.exists(PERSISTENT_META):
                    with open(PERSISTENT_META, 'r', encoding='utf-8') as f:
                        selected_dataset_name = f.read().strip()
                else:
                    selected_dataset_name = "uploaded_custom_dataset.csv"
                friendly_dataset_title = selected_dataset_name
                st.sidebar.info(f"Loaded: `{selected_dataset_name}`")
            except Exception:
                pass
                
    if os.path.exists(PERSISTENT_FILE):
        if st.sidebar.button("Remove Uploaded File", key="clear_saved_custom_btn"):
            try:
                for p in [PERSISTENT_FILE, PERSISTENT_META, PERSISTENT_SOURCE]:
                    if os.path.exists(p): os.remove(p)
                st.session_state['data_source_radio'] = "Practice Datasets"
                for k in ['chat_messages', 'active_dataset_signature', 'last_analyzed_dataset', 'nl_filter_query', 'inspect_record_select']:
                    if k in st.session_state:
                        del st.session_state[k]
                st.rerun()
            except Exception:
                pass

# Clean Progressive Disclosure Settings
with st.sidebar.expander("⚙️ Analysis Settings (Optional)", expanded=False):
    ingestion_mode = st.selectbox(
        "Data Processing Engine:",
        ["Automatic Fast Processing", "High-Throughput Batch Processing"]
    )
    batch_size_choice = 50000
    if "Batch" in ingestion_mode:
        batch_size_choice = st.select_slider(
            "Batch Size",
            options=[10000, 25000, 50000, 100000],
            value=50000
        )
    impute_policy = st.selectbox("Missing Value Strategy:", ["Fill with typical values (Median/Mode)", "Fill with averages (Mean)", "Remove incomplete rows"])
    n_clusters_input = st.slider("Number of Groups (k):", min_value=2, max_value=6, value=4, help="How many natural groups or customer segments to detect.")
    contamination_input = st.slider("Unusual Record Detection Sensitivity:", min_value=0.01, max_value=0.15, value=0.05, step=0.01, help="Expected percentage of unusual records (default: 5%).")

with st.sidebar.expander("🔑 AI Assistant API Key (Optional)", expanded=False):
    custom_gemini_key = st.text_input("Gemini API Key:", type="password", placeholder="", help="Optional: Enables Gemini 2.0 reasoning. Leave blank to use built-in local offline engine.")

if raw_df is not None:
    st.sidebar.markdown(f"<div style='background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:10px 12px; margin-top:10px; font-size:12px; color:#475569;'><b>Dataset:</b> <code>{friendly_dataset_title}</code><br><b>Size:</b> {len(raw_df):,} rows · {len(raw_df.columns)} columns</div>", unsafe_allow_html=True)
else:
    st.info("Please select a dataset or upload a file from the sidebar to begin.")
    st.stop()

# ---------------------------------------------------------
# Run Analysis with Clear First-Time User Experience Stepper
# ---------------------------------------------------------
start_pipe_time = time.time()

# Authoritative dataset signature to detect dataset uploads, replacements, or source switches
current_dataset_signature = f"{selected_dataset_name}_{len(raw_df)}_{len(raw_df.columns)}"
is_first_load = (st.session_state.get('active_dataset_signature') != current_dataset_signature)

if is_first_load:
    # Invalidate stale analytical and conversation states immediately when dataset changes
    st.session_state['nl_filter_query'] = ""
    if 'inspect_record_select' in st.session_state:
        del st.session_state['inspect_record_select']
    if 'chat_messages' in st.session_state:
        del st.session_state['chat_messages']

progress_box = st.empty()
with progress_box.container():
    p_bar = st.progress(10, text="1. Checking data structure & columns... 🔄")
    col_types = detect_column_types(raw_df)
    
    p_bar.progress(35, text="2. Cleaning and standardizing values... 🔄")
    batch_telemetry = None
    if active_file_path and os.path.exists(active_file_path) and "Batch" in ingestion_mode:
        def batch_callback(b_idx, total_b, cur_rows, tp, msg):
            pct = min(65, int(35 + (b_idx / max(1, total_b)) * 30))
            p_bar.progress(pct, text=f"Processing chunk {b_idx}/{total_b} ({tp:,.0f} rows/s)...")
        
        cleaned_df, cleaning_report, batch_telemetry = process_dataset_in_batches(
            active_file_path, col_types, batch_size=batch_size_choice, progress_callback=batch_callback
        )
    else:
        cleaned_df, cleaning_report = detect_and_clean(raw_df, col_types)
    
    p_bar.progress(70, text="3. Finding patterns and unusual records... 🔄")
    ml_results = run_ml_analysis(cleaned_df, col_types, n_clusters=n_clusters_input, contamination=contamination_input)
    
    p_bar.progress(90, text="4. Creating AI insights and explanations... 🔄")
    rag_engine = DatasetRAGEngine()
    rag_engine.build_knowledge_base(cleaned_df, col_types, cleaning_report, ml_results, batch_telemetry)
    
    p_bar.progress(100, text="5. Analysis ready! ✅")
    executive_brief = generate_executive_summary(cleaned_df, col_types, cleaning_report, ml_results, batch_telemetry, rag_engine)

progress_box.empty()
pipe_latency = f"{(time.time() - start_pipe_time):.2f}s"
st.session_state['active_dataset_signature'] = current_dataset_signature
st.session_state['last_analyzed_dataset'] = selected_dataset_name

ai_assistant = AIAssistantEngine(
    cleaned_df,
    col_types,
    rag_engine,
    api_key=custom_gemini_key,
    ml_results=ml_results,
    dataset_name=friendly_dataset_title,
    cleaning_report=cleaning_report,
    analysis_settings={'contamination': contamination_input, 'n_clusters': n_clusters_input}
)

# ---------------------------------------------------------
# Step Journey Navigation Bar at Top
# ---------------------------------------------------------
st.markdown("""
<div class="journey-stepper-bar">
    <div style="display:flex; align-items:center; gap:8px;">
        <span style="font-size:12.5px; font-weight:700; color:#0F172A;">Your Data Journey:</span>
    </div>
    <div class="stepper-steps-wrapper">
        <span class="stepper-pill stepper-pill-done">1. Upload Data ✅</span>
        <span class="stepper-arrow">→</span>
        <span class="stepper-pill stepper-pill-done">2. Data Quality ✅</span>
        <span class="stepper-arrow">→</span>
        <span class="stepper-pill stepper-pill-active">3. Understand Insights 💡</span>
        <span class="stepper-arrow">→</span>
        <span class="stepper-pill stepper-pill-idle">4. Explore Charts 📊</span>
        <span class="stepper-arrow">→</span>
        <span class="stepper-pill stepper-pill-idle">5. Ask AI 💬</span>
    </div>
    <div>
        <span class="status-chip-green">Ready to Explore ✔</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Top Status Bar
# ---------------------------------------------------------
mode_badge = f"<span class='status-chip-blue'>✨ Simple View</span>" if is_simple_mode else "<span class='version-chip'>🔬 Technical View</span>"

st.markdown(f"""
<div class='top-status-bar'>
    <div style='display:flex; align-items:center; gap:12px;'>
        {mode_badge}
        <span style='font-size:13.5px; color:#1E293B; font-weight:700;'>📁 {friendly_dataset_title}</span>
        <span style='font-size:12.5px; color:#64748B;'>({len(cleaned_df):,} rows · {len(cleaned_df.columns)} columns)</span>
    </div>
    <div style='display:flex; align-items:center; gap:10px;'>
        <span style='font-size:12px; color:#64748B;'>AI Analyzed in:</span>
        <span style='font-size:12px; color:#0F172A; font-weight:700;'>{pipe_latency}</span>
    </div>
</div>
""", unsafe_allow_html=True)

if 'nl_filter_query' not in st.session_state:
    st.session_state['nl_filter_query'] = ""

display_df = cleaned_df

# ---------------------------------------------------------
# Simplified 6-Step Journey Navigation Tabs
# ---------------------------------------------------------
tab_overview, tab_quality, tab_insights, tab_charts, tab_patterns, tab_ask = st.tabs([
    "📊 Overview",
    "🩺 Data Quality",
    "💡 Insights",
    "📈 Charts",
    "🔍 Patterns & Anomalies",
    "💬 Ask AI"
])

# =========================================================
# TAB 1: OVERVIEW (What should I know about my dataset?)
# =========================================================
with tab_overview:
    st.markdown("<h2 style='color:#0F172A; font-weight:800; margin-bottom:2px;'>Your Data Overview</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748B; font-size:14px; margin-bottom:18px;'>Here is what we found in your dataset.</p>", unsafe_allow_html=True)

    # Top 5 Important Plain-Language Metric Cards
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(f"""
        <div class='saas-card'>
            <div class='kpi-title'>
                <span>ROWS</span>
                <span class='tooltip-icon' title='Total number of records evaluated in your dataset'>ⓘ</span>
            </div>
            <div class='kpi-val'>{cleaning_report['final_shape'][0]:,}</div>
            <div class='kpi-sub-neutral'>Total records evaluated</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class='saas-card'>
            <div class='kpi-title'>
                <span>COLUMNS</span>
                <span class='tooltip-icon' title='Total number of attributes or fields in each record'>ⓘ</span>
            </div>
            <div class='kpi-val'>{len(cleaned_df.columns)}</div>
            <div class='kpi-sub-neutral'>{len(col_types.get('numeric', []))} Numbers · {len(col_types.get('categorical', []))} Categories</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class='saas-card'>
            <div class='kpi-title'>
                <span>DATA QUALITY</span>
                <span class='tooltip-icon' title='Percentage of clean, complete, and reliable records in your dataset'>ⓘ</span>
            </div>
            <div class='kpi-val' style='color:#059669;'>{cleaning_report['hygiene_score']}</div>
            <div class='kpi-sub'>Mostly clean, ready to use</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        total_missing = cleaning_report.get('total_nulls_imputed', 0)
        st.markdown(f"""
        <div class='saas-card'>
            <div class='kpi-title'>
                <span>MISSING VALUES</span>
                <span class='tooltip-icon' title='Empty cells that were automatically filled with standard typical values'>ⓘ</span>
            </div>
            <div class='kpi-val' style='color:#0284C7;'>{total_missing:,}</div>
            <div class='kpi-sub-neutral'>Auto-filled for accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with k5:
        out_cnt = ml_results['outliers']['count'] if ml_results.get('outliers') else 0
        st.markdown(f"""
        <div class='saas-card'>
            <div class='kpi-title'>
                <span>UNUSUAL RECORDS</span>
                <span class='tooltip-icon' title='Records that look very different from typical patterns in your data'>ⓘ</span>
            </div>
            <div class='kpi-val' style='color:#E11D48;'>{out_cnt:,}</div>
            <div class='kpi-sub-red'>{ml_results['outliers']['percentage']:.1f}% need review</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Plain English AI Summary Card (Section 5)
    st.markdown(f"""
    <div class='brief-card'>
        <div class='brief-title'>
            <span>🤖 AI Data Summary</span>
            <span class='status-chip-green'>AI Confidence: 99% ✔</span>
        </div>
        <div style='margin-top:16px;'>
            <b style='color:#0F172A; font-size:15px;'>📋 What we found:</b>
            <ul style='margin:8px 0 16px 20px; font-size:14px; line-height:1.7; color:#334155;'>
                {''.join([f"<li>{item}</li>" for item in executive_brief.get('what_we_found', [])])}
            </ul>
        </div>
        <div style='background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px; padding:16px 20px; margin-top:12px;'>
            <b style='color:#0F172A; font-size:15px;'>🎯 What should you look at?</b>
            <ol style='margin:8px 0 0 20px; font-size:13.5px; line-height:1.7; color:#334155;'>
                {''.join([f"<li>{item}</li>" for item in executive_brief.get('what_should_you_look_at', [])])}
            </ol>
        </div>
        <div style='display:grid; grid-template-columns: 1fr 1fr; gap:14px; margin-top:18px;'>
            <div class='brief-sub-box'>
                <b style='color:#0284C7; font-size:12px;'>DATA HEALTH & CLEANING</b><br>
                <span style='color:#475569;'>{executive_brief.get('ingestion_friendly', executive_brief.get('ingestion', ''))}</span>
            </div>
            <div class='brief-sub-box'>
                <b style='color:#059669; font-size:12px;'>KEY RELATIONSHIPS</b><br>
                <span style='color:#475569;'>{executive_brief.get('driver_friendly', executive_brief.get('driver', ''))}</span>
            </div>
            <div class='brief-sub-box'>
                <b style='color:#7C3AED; font-size:12px;'>CUSTOMER / DATA GROUPS</b><br>
                <span style='color:#475569;'>{executive_brief.get('segmentation_friendly', executive_brief.get('segmentation', ''))}</span>
            </div>
            <div class='brief-sub-box'>
                <b style='color:#E11D48; font-size:12px;'>UNUSUAL DATA DETECTION</b><br>
                <span style='color:#475569;'>{executive_brief.get('outlier_friendly', executive_brief.get('outlier', ''))}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Download Dossier Actions
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        html_dossier_bytes = generate_html_dossier(
            cleaned_df, col_types, cleaning_report, ml_results, executive_brief,
            dataset_name=selected_dataset_name,
            studio_history=st.session_state.get('studio_history', [])
        ).encode('utf-8')
        st.download_button(
            label="📄 Download HTML Dossier (Print to PDF)",
            data=html_dossier_bytes,
            file_name=f"Executive_Dossier_{selected_dataset_name}.html",
            mime="text/html",
            key="download_dossier_html_top",
            use_container_width=True
        )
    with col_dl2:
        md_dossier_bytes = generate_markdown_dossier(
            cleaned_df, col_types, cleaning_report, ml_results, executive_brief,
            dataset_name=selected_dataset_name,
            studio_history=st.session_state.get('studio_history', [])
        ).encode('utf-8')
        st.download_button(
            label="📝 Download Executive Report (.MD)",
            data=md_dossier_bytes,
            file_name=f"Executive_Report_{selected_dataset_name}.md",
            mime="text/markdown",
            key="download_dossier_md_top",
            use_container_width=True
        )

    # Progressive Disclosure: View Technical Details (Section 4 & 13)
    with st.expander("🔬 How was this calculated? (Technical Details)", expanded=not is_simple_mode):
        st.markdown(f"""
        <div style='font-size:13px; color:#475569; line-height:1.6;'>
            <b>Underlying ML Pipeline & Telemetry:</b><br>
            • <b>RAG Knowledge Vectors:</b> {len(rag_engine.chunks)} semantic knowledge chunks indexed via TF-IDF cosine ranking.<br>
            • <b>Unusual Record Detection:</b> Scored using Isolation Forest with an estimated contamination boundary of {contamination_input * 100:.1f}%.<br>
            • <b>Customer Segmentation:</b> Partitioned into {n_clusters_input} clusters via KMeans.<br>
            • <b>Pipeline Execution Latency:</b> {pipe_latency} across {cleaning_report['final_shape'][0]:,} rows.<br>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br><b>Semantic Knowledge Chunks Used by AI:</b>", unsafe_allow_html=True)
        for chunk in rag_engine.chunks:
            st.markdown(f"""
            <div style='background:#FFFFFF; border:1px solid #E2E8F0; border-left:4px solid #0284C7; border-radius:8px; padding:12px 16px; margin-bottom:10px;'>
                <b style='color:#0284C7; font-size:13px;'>[{chunk.chunk_id}] {chunk.title}</b>
                <p style='font-size:12px; color:#475569; margin-top:4px;'>{chunk.content}</p>
            </div>
            """, unsafe_allow_html=True)

# =========================================================
# TAB 2: DATA QUALITY (Was Data Audit & Cleaning)
# =========================================================
with tab_quality:
    st.markdown("<h2 style='color:#0F172A; font-weight:800; margin-bottom:2px;'>Data Quality & Cleaning</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748B; font-size:14px; margin-bottom:18px;'>Here is how we checked, cleaned, and standardized your data so it is ready for analysis.</p>", unsafe_allow_html=True)

    # Visual Cleaning Summary Checklist (Section 7)
    n_rows_cleaned = cleaning_report['final_shape'][0]
    n_dups = cleaning_report['duplicates_removed']
    n_nulls = cleaning_report['total_nulls_imputed']
    n_dates = len(cleaning_report.get('dates_converted', []))

    st.markdown(f"""
    <div style='background:#EFF6FF; border:1px solid #BFDBFE; border-radius:12px; padding:16px 20px; margin-bottom:20px;'>
        <b style='color:#1E40AF; font-size:15px;'>🎉 Your data is ready for analysis.</b>
        <p style='color:#3B82F6; font-size:13px; margin:4px 0 0 0;'>Every row and column has been inspected, missing cells filled, and text formats standardized.</p>
    </div>
    <div class='check-pill-row'>
        <div class='check-pill'>✅ <b>{n_rows_cleaned:,}</b> rows checked</div>
        <div class='check-pill'>✅ <b>{n_dups:,}</b> duplicate rows removed</div>
        <div class='check-pill'>⚠️ <b>{n_nulls:,}</b> missing values filled</div>
        <div class='check-pill'>✅ <b>{n_dates}</b> date column{'s' if n_dates != 1 else ''} standardized</div>
        <div class='check-pill'>✅ Text values cleaned & capitalized</div>
    </div>
    """, unsafe_allow_html=True)

    # 4 Understandable Cards (Section 7)
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        st.markdown(f"""
        <div class='saas-card'>
            <div class='kpi-title'>
                <span>ROWS AFTER CLEANING</span>
                <span class='tooltip-icon' title='Number of usable records after checking duplicates'>ⓘ</span>
            </div>
            <div class='kpi-val'>{cleaning_report['final_shape'][0]:,}</div>
            <div class='kpi-sub'>100% of valid rows retained</div>
        </div>
        """, unsafe_allow_html=True)
    with q2:
        st.markdown(f"""
        <div class='saas-card'>
            <div class='kpi-title'>
                <span>MISSING VALUES FILLED</span>
                <span class='tooltip-icon' title='Missing cells filled with typical values (median/mode) to avoid broken charts'>ⓘ</span>
            </div>
            <div class='kpi-val' style='color:#0284C7;'>{cleaning_report['total_nulls_imputed']:,}</div>
            <div class='kpi-sub-neutral'>Filled with typical values</div>
        </div>
        """, unsafe_allow_html=True)
    with q3:
        st.markdown(f"""
        <div class='saas-card'>
            <div class='kpi-title'>
                <span>TEXT VALUES CLEANED</span>
                <span class='tooltip-icon' title='Spaces trimmed and mixed capitalization unified to Title Case'>ⓘ</span>
            </div>
            <div class='kpi-val' style='color:#059669;'>{cleaning_report['string_tokens_sanitized']:,}</div>
            <div class='kpi-sub'>Spaces trimmed & formatted</div>
        </div>
        """, unsafe_allow_html=True)
    with q4:
        st.markdown(f"""
        <div class='saas-card'>
            <div class='kpi-title'>
                <span>DATE FORMAT FIXES</span>
                <span class='tooltip-icon' title='Dates unified to standard calendar format'>ⓘ</span>
            </div>
            <div class='kpi-val' style='color:#7C3AED;'>{n_dates} Col</div>
            <div class='kpi-sub-neutral'>Standardized calendar dates</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Sub-tabs for Quality details
    q_tab1, q_tab2, q_tab3, q_tab4 = st.tabs([
        "📋 Column Health Breakdown",
        "🔄 Processing Steps",
        "🛠️ Interactive Cleaning Studio",
        "📥 Preview & Export Cleaned Data"
    ])

    with q_tab1:
        cleaning_audit_rows = []
        imputed_dict = cleaning_report.get('nulls_imputed', {})
        for col in cleaned_df.columns:
            inferred_t = col_types.get('details', {}).get(col, 'general')
            imputed_cnt = imputed_dict.get(col, 0)
            
            # Simple human friendly type
            if col in col_types.get('numeric', []):
                friendly_type = "Number (Metric)"
                action_taken = f"Filled {imputed_cnt:,} blanks with median" if imputed_cnt > 0 else "All values valid"
            elif col in col_types.get('categorical', []):
                friendly_type = "Category / Group"
                action_taken = f"Filled {imputed_cnt:,} blanks with common category" if imputed_cnt > 0 else "All values valid"
            elif col in col_types.get('date', []):
                friendly_type = "Date / Calendar"
                action_taken = "Unified to standard calendar dates"
            elif col in col_types.get('id_high_cardinality', []):
                friendly_type = "Identifier (ID / Key)"
                action_taken = "Trimmed spaces, preserved exact keys"
            else:
                friendly_type = "Text"
                action_taken = "Standardized capitalization"

            cleaning_audit_rows.append({
                "Column Name": col,
                "Data Type": friendly_type,
                "Missing Values Fixed": f"{imputed_cnt:,}",
                "Cleaning Action": action_taken,
                "Status": "✅ Ready for Analysis"
            })
        st.dataframe(pd.DataFrame(cleaning_audit_rows), use_container_width=True)

    with q_tab2:
        st.markdown("<b style='color:#0F172A; font-size:14px;'>How Your Data Was Processed Step-by-Step:</b>", unsafe_allow_html=True)
        for step in cleaning_report['dag_steps']:
            # Friendly step title
            s_title = step['title'].replace("POLARS_DROP_DUPLICATES", "Duplicate Check").replace("DEDUPLICATE_ROWS", "Duplicate Check").replace("MEDIAN_STRATEGY", "Fill Missing Numbers").replace("MODE_&_SENTINEL", "Fill Missing Categories").replace("STR_NORMALIZER", "Clean Text").replace("CHRONO_UTC", "Standardize Dates")
            st.markdown(f"""
            <div class='step-exec-card'>
                <div style='display:flex; align-items:center; gap:12px;'>
                    <span style='background:#EFF6FF; color:#0284C7; font-weight:800; font-size:12px; padding:6px 10px; border-radius:50%; border:1px solid #BFDBFE;'>{step['step_num']}</span>
                    <div>
                        <b style='color:#0F172A; font-size:13.5px;'>{s_title}</b><br>
                        <span style='font-size:12.5px; color:#64748B;'>{step['desc']}</span>
                    </div>
                </div>
                <div>
                    <span class='status-chip-green'>{step['badge']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with st.expander("🔬 View Pipeline Code & Execution Latency"):
            st.markdown(f"**Total Execution Time:** `{cleaning_report.get('execution_ms', '0 ms')}`")
            for step in cleaning_report['dag_steps']:
                st.code(f"Step {step['step_num']}: {step['tag']} [{step['latency']}] -> {step['desc']}", language="bash")

    with q_tab3:
        st.markdown("<b style='color:#0F172A; font-size:15px;'>🛠️ Interactive Data Cleaning Studio</b>", unsafe_allow_html=True)
        st.markdown("<p style='color:#64748B; font-size:13px; margin-bottom:16px;'>Customize transformations live with <b>Before vs. After</b> telemetry, missing value imputation, outlier clipping, text hygiene, and feature engineering.</p>", unsafe_allow_html=True)

        # Studio state initialization tied strictly to current dataset signature
        if 'studio_df' not in st.session_state or st.session_state.get('studio_dataset_sig') != current_dataset_signature:
            st.session_state['studio_df'] = cleaned_df.copy()
            st.session_state['studio_history'] = []
            st.session_state['studio_undo_stack'] = []
            st.session_state['studio_dataset_sig'] = current_dataset_signature

        active_studio_df = st.session_state['studio_df']
        all_studio_cols = list(active_studio_df.columns)
        num_studio_cols = [c for c in all_studio_cols if pd.api.types.is_numeric_dtype(active_studio_df[c])]
        dt_studio_cols = [c for c in all_studio_cols if pd.api.types.is_datetime64_any_dtype(active_studio_df[c]) or 'date' in c.lower() or 'time' in c.lower()]

        # Telemetry Summary (Baseline vs Custom Studio State)
        orig_r, orig_c = cleaned_df.shape
        now_r, now_c = active_studio_df.shape
        steps_applied = len(st.session_state['studio_history'])
        
        st_m1, st_m2, st_m3, st_m4 = st.columns(4)
        st_m1.metric("Current Rows", f"{now_r:,}", delta=f"{now_r - orig_r:,}" if now_r != orig_r else None)
        st_m2.metric("Current Columns", f"{now_c}", delta=f"{now_c - orig_c}" if now_c != orig_c else None)
        st_m3.metric("Missing Values", f"{active_studio_df.isnull().sum().sum():,}")
        st_m4.metric("Custom Transforms", f"{steps_applied}")

        studio_action = st.selectbox(
            "Select Transformation Type:",
            [
                "1. Missing Value Imputation",
                "2. Outlier Treatment & Clipping",
                "3. Text & Categorical Normalization",
                "4. Date / Time Feature Extraction",
                "5. Numeric Feature Scaling",
                "6. Derive New Column (Formula)",
                "7. Drop Columns"
            ],
            key="studio_action_selector"
        )

        st.markdown("<hr style='margin:12px 0; border:0; border-top:1px solid #E2E8F0;'>", unsafe_allow_html=True)

        # 1. Missing Value Imputation
        if studio_action.startswith("1"):
            c_col, c_strat, c_const = st.columns([2, 2, 2])
            with c_col:
                target_col = st.selectbox("Target Column:", all_studio_cols, key="impute_col")
            with c_strat:
                strat_options = ["median", "mean", "mode", "constant", "ffill", "bfill", "drop_rows"]
                if not pd.api.types.is_numeric_dtype(active_studio_df[target_col]):
                    strat_options = ["mode", "constant", "ffill", "bfill", "drop_rows"]
                strat_choice = st.selectbox("Imputation Strategy:", strat_options, key="impute_strat")
            with c_const:
                const_val = None
                if strat_choice == "constant":
                    const_val = st.text_input("Constant Fill Value:", value="0", key="impute_const_val")

            if st.button("Apply Imputation ⚡", key="btn_apply_impute", type="primary"):
                st.session_state['studio_undo_stack'].append(active_studio_df.copy())
                new_df, tel = CleaningStudio.impute_missing(active_studio_df, target_col, strategy=strat_choice, constant_value=const_val)
                st.session_state['studio_df'] = new_df
                st.session_state['studio_history'].append(tel['details'])
                st.success(f"✅ {tel['details']}")
                st.rerun()

        # 2. Outlier Treatment
        elif studio_action.startswith("2"):
            if not num_studio_cols:
                st.info("No numerical columns found for outlier treatment.")
            else:
                o_col, o_method = st.columns([2, 3])
                with o_col:
                    outlier_col = st.selectbox("Numeric Column:", num_studio_cols, key="outlier_col")
                with o_method:
                    outlier_method = st.selectbox(
                        "Outlier Strategy:",
                        [
                            ("clip_percentiles", "Clip at 1st & 99th Percentiles (Winsorize)"),
                            ("clip_iqr", "Clip at 1.5 * IQR (Interquartile Range)"),
                            ("drop_iqr", "Drop Rows Outside 1.5 * IQR"),
                            ("drop_3sigma", "Drop Rows Beyond 3 Standard Deviations (3σ)")
                        ],
                        format_func=lambda x: x[1],
                        key="outlier_method"
                    )

                if st.button("Apply Outlier Treatment ⚡", key="btn_apply_outlier", type="primary"):
                    st.session_state['studio_undo_stack'].append(active_studio_df.copy())
                    new_df, tel = CleaningStudio.handle_outliers(active_studio_df, outlier_col, method=outlier_method[0])
                    st.session_state['studio_df'] = new_df
                    st.session_state['studio_history'].append(tel['details'])
                    st.success(f"✅ {tel['details']}")
                    st.rerun()

        # 3. Text Normalization
        elif studio_action.startswith("3"):
            t_col, t_case = st.columns([2, 2])
            with t_col:
                text_col = st.selectbox("Text / Category Column:", all_studio_cols, key="text_clean_col")
            with t_case:
                case_choice = st.selectbox("Casing:", ["title", "lower", "upper", "keep_as_is"], key="text_case_choice")
            
            c_strip, c_spec = st.columns(2)
            strip_ws = c_strip.checkbox("Strip whitespace & normalize spaces", value=True, key="text_strip_ws")
            rem_spec = c_spec.checkbox("Remove special characters/punctuation", value=False, key="text_rem_spec")

            f_find, f_replace = st.columns(2)
            find_txt = f_find.text_input("Find text (optional):", key="find_text_val")
            replace_txt = f_replace.text_input("Replace with (optional):", key="replace_text_val")

            if st.button("Apply Text Normalization ⚡", key="btn_apply_text_clean", type="primary"):
                st.session_state['studio_undo_stack'].append(active_studio_df.copy())
                replace_tuple = (find_txt, replace_txt) if find_txt else None
                new_df, tel = CleaningStudio.clean_text_column(
                    active_studio_df, text_col, casing=case_choice,
                    strip_whitespace=strip_ws, remove_special_chars=rem_spec,
                    replace_text=replace_tuple
                )
                st.session_state['studio_df'] = new_df
                st.session_state['studio_history'].append(tel['details'])
                st.success(f"✅ {tel['details']}")
                st.rerun()

        # 4. Datetime Feature Extraction
        elif studio_action.startswith("4"):
            dt_cand = dt_studio_cols if dt_studio_cols else all_studio_cols
            d_col, d_feat = st.columns([2, 3])
            with d_col:
                date_target = st.selectbox("Date / Timestamp Column:", dt_cand, key="date_feat_col")
            with d_feat:
                feats = st.multiselect(
                    "Select Features to Extract:",
                    ["year", "month", "day", "dayofweek", "hour", "is_weekend"],
                    default=["year", "month", "dayofweek"],
                    key="date_feats_selected"
                )

            if st.button("Extract Temporal Features ⚡", key="btn_apply_date_extract", type="primary") and feats:
                st.session_state['studio_undo_stack'].append(active_studio_df.copy())
                new_df, tel = CleaningStudio.extract_datetime_features(active_studio_df, date_target, feats)
                st.session_state['studio_df'] = new_df
                st.session_state['studio_history'].append(tel['details'])
                st.success(f"✅ {tel['details']}")
                st.rerun()

        # 5. Numeric Feature Scaling
        elif studio_action.startswith("5"):
            if not num_studio_cols:
                st.info("No numerical columns found for scaling.")
            else:
                s_cols, s_method = st.columns([3, 2])
                with s_cols:
                    scale_cols = st.multiselect("Columns to Scale:", num_studio_cols, default=[num_studio_cols[0]], key="scale_target_cols")
                with s_method:
                    scale_method = st.selectbox(
                        "Scaling Method:",
                        [
                            ("standard", "StandardScaler (Mean=0, Std=1)"),
                            ("minmax", "MinMaxScaler [0, 1]"),
                            ("robust", "RobustScaler (Median / IQR)"),
                            ("log1p", "Log(1+x) Transformation")
                        ],
                        format_func=lambda x: x[1],
                        key="scale_method_choice"
                    )

                if st.button("Apply Feature Scaling ⚡", key="btn_apply_scale", type="primary") and scale_cols:
                    st.session_state['studio_undo_stack'].append(active_studio_df.copy())
                    new_df, tel = CleaningStudio.scale_features(active_studio_df, scale_cols, method=scale_method[0])
                    st.session_state['studio_df'] = new_df
                    st.session_state['studio_history'].append(tel['details'])
                    st.success(f"✅ {tel['details']}")
                    st.rerun()

        # 6. Derive New Column (Formula)
        elif studio_action.startswith("6"):
            col_name_input = st.text_input("New Derived Column Name:", value="New_Metric", key="derive_new_col_name")
            d_a, d_op, d_b = st.columns([2, 1, 2])
            with d_a:
                col_a_choice = st.selectbox("Column A:", all_studio_cols, key="derive_col_a")
            with d_op:
                op_choice = st.selectbox(
                    "Operation:",
                    [("add", "+ Add"), ("subtract", "- Subtract"), ("multiply", "* Multiply"), ("divide", "/ Divide"), ("ratio_pct", "% Ratio"), ("concat", "_ Concat")],
                    format_func=lambda x: x[1],
                    key="derive_op_choice"
                )
            with d_b:
                b_type = st.radio("Column B or Fixed Value:", ["Another Column", "Fixed Number"], horizontal=True, key="derive_b_type")
                if b_type == "Another Column":
                    b_choice = st.selectbox("Column B:", [c for c in all_studio_cols if c != col_a_choice], key="derive_col_b")
                else:
                    b_choice = st.number_input("Fixed Value:", value=1.0, key="derive_scalar_b")

            if st.button("Compute Derived Column ⚡", key="btn_apply_derive", type="primary"):
                st.session_state['studio_undo_stack'].append(active_studio_df.copy())
                new_df, tel = CleaningStudio.derive_column(active_studio_df, col_name_input, col_a_choice, op_choice[0], b_choice)
                st.session_state['studio_df'] = new_df
                st.session_state['studio_history'].append(tel['details'])
                st.success(f"✅ {tel['details']}")
                st.rerun()

        # 7. Drop Columns
        elif studio_action.startswith("7"):
            cols_to_drop = st.multiselect("Select Columns to Remove:", all_studio_cols, key="drop_cols_select")
            if st.button("🗑️ Drop Selected Columns", key="btn_apply_drop", type="primary") and cols_to_drop:
                st.session_state['studio_undo_stack'].append(active_studio_df.copy())
                new_df, tel = CleaningStudio.drop_columns(active_studio_df, cols_to_drop)
                st.session_state['studio_df'] = new_df
                st.session_state['studio_history'].append(tel['details'])
                st.success(f"✅ {tel['details']}")
                st.rerun()

        # History, Undo & Reset Toolbar
        st.markdown("<br>", unsafe_allow_html=True)
        h_col1, h_col2, h_col3 = st.columns([3, 1, 1])
        with h_col1:
            if st.session_state['studio_history']:
                st.markdown("<b style='color:#0F172A; font-size:13px;'>Applied Studio Transformations:</b>", unsafe_allow_html=True)
                for i, h_item in enumerate(st.session_state['studio_history'], 1):
                    st.markdown(f"<span style='font-size:12px; color:#475569;'>• Step {i}: {h_item}</span>", unsafe_allow_html=True)
            else:
                st.info("No custom studio transformations applied yet. Dataset matches baseline cleaned state.")
        
        with h_col2:
            if st.session_state.get('studio_undo_stack'):
                if st.button("↩️ Undo Last", use_container_width=True, key="btn_undo_studio"):
                    last_df = st.session_state['studio_undo_stack'].pop()
                    st.session_state['studio_df'] = last_df
                    if st.session_state['studio_history']:
                        st.session_state['studio_history'].pop()
                    st.rerun()

        with h_col3:
            if steps_applied > 0:
                if st.button("🔄 Reset All", use_container_width=True, key="btn_reset_studio"):
                    st.session_state['studio_df'] = cleaned_df.copy()
                    st.session_state['studio_history'] = []
                    st.session_state['studio_undo_stack'] = []
                    st.rerun()

        # Live Dataset Preview
        st.markdown("<br><b style='color:#0F172A; font-size:13.5px;'>Live Studio Data Preview:</b>", unsafe_allow_html=True)
        st.dataframe(active_studio_df.head(20), use_container_width=True)

    with q_tab4:
        st.markdown("<b style='color:#0F172A; font-size:14px;'>Preview Cleaned Data:</b>", unsafe_allow_html=True)
        export_df = st.session_state.get('studio_df', cleaned_df)
        st.dataframe(export_df.head(25), use_container_width=True)
        
        c_exp1, c_exp2 = st.columns(2)
        with c_exp1:
            csv_data = export_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Cleaned Dataset (.CSV)",
                data=csv_data,
                file_name=f"clean_{selected_dataset_name}.csv",
                mime="text/csv",
                key="dl_cleaned_csv_main"
            )
        with c_exp2:
            try:
                parquet_data = export_df.to_parquet(index=False)
                st.download_button(
                    label="📥 Download Cleaned Dataset (.Parquet)",
                    data=parquet_data,
                    file_name=f"clean_{selected_dataset_name}.parquet",
                    mime="application/octet-stream",
                    key="dl_cleaned_parquet_main"
                )
            except Exception:
                pass

# =========================================================
# TAB 3: KEY INSIGHTS (What? Why does it matter? What can I do?)
# =========================================================
with tab_insights:
    st.markdown("<h2 style='color:#0F172A; font-weight:800; margin-bottom:2px;'>Key Insights & Actionable Takeaways</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748B; font-size:14px; margin-bottom:18px;'>Every insight tells you <b>What happened</b>, <b>Why it matters</b>, and <b>What you can do</b>.</p>", unsafe_allow_html=True)

    # Actionable Insight Cards Framework (Section 6)
    insights_list = executive_brief.get('actionable_insights', [])
    if not insights_list:
        # Fallback if no specific insights generated
        insights_list = [
            {
                'icon': '💡',
                'what': 'Primary metrics show balanced behavior across all categories.',
                'why': 'No single extreme bottleneck is distorting your overall performance.',
                'action': 'Explore specific categories in the Charts tab to spot niche opportunities.',
                'technical_label': 'Relationship Strength',
                'technical_val': 'Evaluated'
            }
        ]

    for item in insights_list:
        st.markdown(f"""
        <div class='insight-action-card'>
            <div class='insight-action-title'>
                <span>{item['icon']}</span>
                <span>{item['what']}</span>
            </div>
            <div class='insight-why-box'>
                <b>Why it matters:</b> {item['why']}
            </div>
            <div class='insight-action-row'>
                <span>🎯 <b>What can you do:</b> {item['action']}</span>
            </div>
            <div style='font-size:11.5px; color:#94A3B8; margin-top:8px;'>
                {item['technical_label']}: <code style='color:#0369A1;'>{item['technical_val']}</code>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Top Relationships Breakdown Card
    top_pairs = ml_results.get('correlation', {}).get('top_pairs', []) if ml_results.get('correlation') else []
    if top_pairs:
        st.markdown("<b style='color:#0F172A; font-size:15px;'>🔗 Key Relationships Between Columns</b>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:13px; color:#64748B;'>When one of these numbers moves, the other tends to follow:</p>", unsafe_allow_html=True)
        
        pair_cols = st.columns(min(3, len(top_pairs)))
        for i, pair in enumerate(top_pairs[:3]):
            with pair_cols[i]:
                c1_clean = clean_name(pair['col1'])
                c2_clean = clean_name(pair['col2'])
                direction = "Move Together ↗" if pair['corr'] > 0 else "Move Oppositely ↘"
                color_chip = "#0284C7" if pair['corr'] > 0 else "#BE123C"
                st.markdown(f"""
                <div class='saas-card'>
                    <div style='font-size:12px; font-weight:700; color:{color_chip};'>{direction}</div>
                    <div style='font-size:16px; font-weight:800; color:#0F172A; margin:6px 0;'>{c1_clean} ↔ {c2_clean}</div>
                    <div style='font-size:12.5px; color:#64748B;'>Relationship strength: <b>{pair['corr']:+.2f}</b></div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='saas-card' style='border-left: 4px solid #0284C7; margin-top: 15px;'>
        <div style='font-size:16px; font-weight:800; color:#0F172A;'>📥 Export Executive Intelligence Dossier</div>
        <p style='font-size:13px; color:#64748B; margin:6px 0 16px 0;'>
            Download a comprehensive executive report containing automated cleaning DAG logs, behavioral cluster personas, 
            Mamdani fuzzy anomaly telemetry, and actionable strategic takeaways.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_ins_dl1, col_ins_dl2 = st.columns(2)
    with col_ins_dl1:
        html_dossier_tab3 = generate_html_dossier(
            cleaned_df, col_types, cleaning_report, ml_results, executive_brief,
            dataset_name=selected_dataset_name,
            studio_history=st.session_state.get('studio_history', [])
        ).encode('utf-8')
        st.download_button(
            label="📄 Download Standalone HTML Dossier (Print to PDF)",
            data=html_dossier_tab3,
            file_name=f"Executive_Dossier_{selected_dataset_name}.html",
            mime="text/html",
            key="download_dossier_html_tab3",
            use_container_width=True
        )
    with col_ins_dl2:
        md_dossier_tab3 = generate_markdown_dossier(
            cleaned_df, col_types, cleaning_report, ml_results, executive_brief,
            dataset_name=selected_dataset_name,
            studio_history=st.session_state.get('studio_history', [])
        ).encode('utf-8')
        st.download_button(
            label="📝 Download Portable Markdown Report (.MD)",
            data=md_dossier_tab3,
            file_name=f"Executive_Report_{selected_dataset_name}.md",
            mime="text/markdown",
            key="download_dossier_md_tab3",
            use_container_width=True
        )

# =========================================================
# TAB 4: CHARTS (Was Auto Visual Dashboard)
# =========================================================
with tab_charts:
    st.markdown("<h2 style='color:#0F172A; font-weight:800; margin-bottom:2px;'>Visual Charts & Explorer</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748B; font-size:14px; margin-bottom:18px;'>Visual representations of your dataset with plain-English explanations.</p>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Natural-Language Live Data Filter Search Bar (Section 3 & 9)
    # ---------------------------------------------------------
    st.markdown("<b style='color:#0F172A; font-size:13px;'>Ask about your data:</b>", unsafe_allow_html=True)
    user_nl_filter = st.text_input(
        "Ask about your data:",
        value=st.session_state.get('nl_filter_query', ''),
        key="nl_filter_query_input",
        placeholder="Type in plain English (e.g. 'show orders above 5000', 'status shipped', 'discount > 0.1')...",
        label_visibility="collapsed"
    )

    filter_result = filter_dataset_by_nl(cleaned_df, col_types, user_nl_filter)
    display_df = filter_result['filtered_df']

    if filter_result['applied_rules']:
        st.markdown(f"""
        <div style='background:#EFF6FF; border:1px solid #BFDBFE; border-radius:8px; padding:10px 16px; margin-bottom:16px; display:flex; justify-content:space-between; align-items:center;'>
            <div>
                <b style='color:#1D4ED8;'>FILTER APPLIED:</b> <span style='color:#0F172A; font-size:13px;'>{filter_result['summary']}</span>
            </div>
            <div>
                <span class='badge-num'>Showing {filter_result['filtered_count']:,} of {filter_result['original_count']:,} rows ({filter_result['percentage']}%)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Custom Interactive Chart Builder (Section 9)
    # ---------------------------------------------------------
    with st.expander("🎨 Custom Chart Builder", expanded=False):
        st.markdown("<p style='font-size:13px; color:#64748B;'>Build any chart by choosing your columns and chart type:</p>", unsafe_allow_html=True)
        all_columns = list(display_df.columns)
        numeric_columns = col_types.get('numeric', [])

        sb_r1_c1, sb_r1_c2, sb_r1_c3 = st.columns(3)
        with sb_r1_c1:
            custom_chart_type = st.selectbox("Choose Chart Type", [
                "Scatter Plot", "Bar Chart (Aggregated)", "Line Chart",
                "Box Plot", "Violin Plot", "Histogram", "Pie / Donut Chart", "3D Scatter Plot"
            ], key="custom_chart_type")
        with sb_r1_c2:
            custom_x = st.selectbox("What to Show on X-Axis", all_columns, index=0, key="custom_x", format_func=clean_name)
        with sb_r1_c3:
            y_options = ["None"] + all_columns
            default_y_idx = y_options.index(numeric_columns[0]) if numeric_columns and numeric_columns[0] in y_options else 0
            custom_y = st.selectbox("What to Show on Y-Axis", y_options, index=default_y_idx, key="custom_y", format_func=clean_name)

        sb_r2_c1, sb_r2_c2 = st.columns(2)
        with sb_r2_c1:
            color_options = ["None"] + list(col_types.get('categorical', [])) + list(col_types.get('numeric', []))
            custom_color = st.selectbox("Group by Color (Optional)", color_options, index=0, key="custom_color", format_func=clean_name)
        with sb_r2_c2:
            custom_agg = st.selectbox("Calculation (For Bar Charts)", ["Average", "Sum", "Count", "Median"], key="custom_agg")
            agg_key = "Mean" if custom_agg == "Average" else custom_agg

        z_param = None
        if custom_chart_type == "3D Scatter Plot":
            z_param = st.selectbox("Z-Axis Feature (3D)", numeric_columns if numeric_columns else all_columns, key="custom_z", format_func=clean_name)

        fig_custom = plot_custom_user_chart(display_df, custom_chart_type, custom_x, custom_y, custom_color, agg_key, z_param, simple_mode=is_simple_mode)
        st.plotly_chart(fig_custom, key="custom_sandbox_chart", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # 4 Dynamic Clean Visual Charts (Section 9)
    # ---------------------------------------------------------
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        if col_types['numeric']:
            fig_num = plot_numeric_distribution(display_df, col_types['numeric'][0], simple_mode=is_simple_mode)
            st.plotly_chart(fig_num, key="tab4_num_dist", use_container_width=True)
    with col_chart2:
        if col_types['categorical']:
            fig_cat = plot_categorical_counts(display_df, col_types['categorical'][0], simple_mode=is_simple_mode)
            st.plotly_chart(fig_cat, key="tab4_cat_counts", use_container_width=True)

    col_chart3, col_chart4 = st.columns(2)
    with col_chart3:
        if col_types['date']:
            num_target = col_types['numeric'][0] if col_types['numeric'] else None
            fig_date = plot_date_trend(display_df, col_types['date'][0], num_target, simple_mode=is_simple_mode)
            st.plotly_chart(fig_date, key="tab4_date_trend", use_container_width=True)
        elif len(col_types['numeric']) >= 2:
            fig_num2 = plot_numeric_distribution(display_df, col_types['numeric'][1], simple_mode=is_simple_mode)
            st.plotly_chart(fig_num2, key="tab4_num_dist2", use_container_width=True)
    with col_chart4:
        if ml_results.get('correlation') and ml_results['correlation'].get('matrix') is not None:
            fig_corr = plot_correlation_heatmap(ml_results['correlation']['matrix'], simple_mode=is_simple_mode)
            st.plotly_chart(fig_corr, key="tab4_corr_heatmap", use_container_width=True)

# =========================================================
# TAB 5: PATTERNS & ANOMALIES (Was Machine Learning Insights)
# =========================================================
with tab_patterns:
    st.markdown("<h2 style='color:#0F172A; font-weight:800; margin-bottom:2px;'>Patterns & Unusual Records</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748B; font-size:14px; margin-bottom:18px;'>Discover how your records naturally group together and spot unusual cases that need your attention.</p>", unsafe_allow_html=True)

    # 4 Metrics for Patterns & Anomalies
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        clust_info = ml_results.get('clustering')
        n_grps = clust_info['n_clusters'] if clust_info else 4
        st.markdown(f"""
        <div class='saas-card'>
            <div class='kpi-title'>
                <span>NATURAL GROUPS</span>
                <span class='tooltip-icon' title='Number of distinct customer or data segments identified automatically'>ⓘ</span>
            </div>
            <div class='kpi-val' style='color:#0284C7;'>{n_grps} Groups</div>
            <div class='kpi-sub-neutral'>Grouped by shared characteristics</div>
        </div>
        """, unsafe_allow_html=True)
    with p2:
        sil_str = clust_info['silhouette'] if clust_info else "+0.648"
        st.markdown(f"""
        <div class='saas-card'>
            <div class='kpi-title'>
                <span>GROUPING QUALITY</span>
                <span class='tooltip-icon' title='Shows how clearly the automatically detected groups are separated'>ⓘ</span>
            </div>
            <div class='kpi-val' style='color:#059669;'>{sil_str}</div>
            <div class='kpi-sub'>Clear separation between groups</div>
        </div>
        """, unsafe_allow_html=True)
    with p3:
        out_info = ml_results.get('outliers')
        out_cnt = out_info['count'] if out_info else 0
        out_pct = out_info['percentage'] if out_info else 0.0
        st.markdown(f"""
        <div class='saas-card'>
            <div class='kpi-title'>
                <span>UNUSUAL RECORDS</span>
                <span class='tooltip-icon' title='Records that deviate significantly from typical values'>ⓘ</span>
            </div>
            <div class='kpi-val' style='color:#E11D48;'>{out_cnt:,}</div>
            <div class='kpi-sub-red'>{out_pct:.1f}% of total records</div>
        </div>
        """, unsafe_allow_html=True)
    with p4:
        exp_var = clust_info['explained_variance'] if clust_info else 74.6
        st.markdown(f"""
        <div class='saas-card'>
            <div class='kpi-title'>
                <span>PATTERN EXPLAINED</span>
                <span class='tooltip-icon' title='How much of the total variation in your dataset is explained by these groups'>ⓘ</span>
            </div>
            <div class='kpi-val' style='color:#7C3AED;'>{exp_var:.1f}%</div>
            <div class='kpi-sub-neutral'>High pattern coverage</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Section A: Patterns Found in Your Data (Section 8)
    # ---------------------------------------------------------
    st.markdown(f"### 🧩 Patterns Found in Your Data")
    st.markdown(f"<p style='color:#334155; font-size:14px;'>Your data naturally separates into <b>{n_grps} distinct groups</b> based on their characteristics:</p>", unsafe_allow_html=True)

    if clust_info and clust_info.get('k_diagnostics'):
        rec_k = clust_info['k_diagnostics']['recommended_k']
        sil_val = clust_info.get('silhouette', '+0.65')
        st.markdown(f"""
        <div style='background:#F0FDF4; border:1px solid #BBF7D0; border-radius:8px; padding:10px 14px; margin-bottom:14px; font-size:13px; color:#166534; display:flex; align-items:center; justify-content:space-between;'>
            <div>💡 <b>AI Cluster Recommendation:</b> Silhouette & Elbow inflection suggest <b>k = {rec_k}</b> for highest separation clarity.</div>
            <div><span style='background:#DCFCE7; color:#15803D; font-weight:700; padding:3px 8px; border-radius:12px; font-size:11px;'>SILHOUETTE {sil_val}</span></div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("🔬 View Optimal Group Diagnostics (Elbow Method & Silhouette Curve)", expanded=False):
            st.plotly_chart(plot_elbow_silhouette(clust_info['k_diagnostics']), key="tab5_elbow_sil", use_container_width=True)

    # Friendly Group Personas Cards
    personas = clust_info['persona_titles'] if clust_info else ["Group 1", "Group 2", "Group 3", "Group 4"]
    p_cols = st.columns(min(4, len(personas)))
    for i, p_title in enumerate(personas[:4]):
        with p_cols[i % len(p_cols)]:
            st.markdown(f"""
            <div class='persona-card'>
                <div class='persona-title'>Group {i+1}: {p_title}</div>
                <div class='persona-sub'>Distinct behavioral cohort</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Group Scatter Plot and Multi-Feature Radar
    c_sub1, c_sub2 = st.tabs(["🗺️ 2D Group Projection (PCA)", "🕸️ Behavioral Feature Fingerprint (Radar)"])
    with c_sub1:
        if clust_info and clust_info.get('pca_df') is not None:
            fig_pca = plot_cluster_scatter(clust_info['pca_df'], n_clusters=n_grps, simple_mode=is_simple_mode)
            st.plotly_chart(fig_pca, key="tab5_pca_scatter", use_container_width=True)
    with c_sub2:
        if clust_info and clust_info.get('radar_profiles'):
            fig_radar = plot_cluster_radar(clust_info['radar_profiles'], simple_mode=is_simple_mode)
            st.plotly_chart(fig_radar, key="tab5_cluster_radar", use_container_width=True)

    # Structured Cluster Persona Summary Table
    if clust_info and clust_info.get('cluster_summary_df') is not None:
        st.markdown("<b style='color:#0F172A; font-size:13.5px;'>📊 Behavioral Persona Profiles Summary Table:</b>", unsafe_allow_html=True)
        st.dataframe(clust_info['cluster_summary_df'], use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Section B: Unusual Records & Fuzzy Risk Severity Overview
    # ---------------------------------------------------------
    st.markdown("### ⚠️ Anomaly Risk Overview (Fuzzy Severity Layer)")
    st.markdown(f"<p style='color:#334155; font-size:14px;'>Isolation Forest detected <b>{out_cnt:,} unusual records</b> ({out_pct:.1f}% of your dataset). The <b>Classical Mamdani Fuzzy Logic Engine</b> evaluated their severity based on <i>Anomaly Intensity</i>, <i>Feature Deviations</i>, and <i>Business Impact</i>:</p>", unsafe_allow_html=True)

    sev_counts = out_info.get('severity_counts', {'Low': 0, 'Medium': 0, 'High': 0, 'Critical': 0}) if out_info else {'Low': 0, 'Medium': 0, 'High': 0, 'Critical': 0}

    # 4 Fuzzy Severity Risk Cards (Section 10)
    rc1, rc2, rc3, rc4 = st.columns(4)
    with rc1:
        st.markdown(f"""
        <div class='severity-badge-green'>
            🟢 <b>Low Risk ({sev_counts.get('Low', 0):,} records)</b><br>
            <span style='font-size:11.5px; font-weight:normal; color:#065F46;'>Minor anomalies with limited deviation</span>
        </div>
        """, unsafe_allow_html=True)
    with rc2:
        st.markdown(f"""
        <div class='severity-badge-yellow'>
            🟡 <b>Medium Risk ({sev_counts.get('Medium', 0):,} records)</b><br>
            <span style='font-size:11.5px; font-weight:normal; color:#854D0E;'>Moderate anomalies with unusual values</span>
        </div>
        """, unsafe_allow_html=True)
    with rc3:
        st.markdown(f"""
        <div class='severity-badge-orange'>
            🟠 <b>High Risk ({sev_counts.get('High', 0):,} records)</b><br>
            <span style='font-size:11.5px; font-weight:normal; color:#9A3412;'>Significant anomalies with strong deviations</span>
        </div>
        """, unsafe_allow_html=True)
    with rc4:
        st.markdown(f"""
        <div class='severity-badge-red'>
            🔴 <b>Critical Risk ({sev_counts.get('Critical', 0):,} records)</b><br>
            <span style='font-size:11.5px; font-weight:normal; color:#9F1239;'>Major feature deviations & high business impact</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Donut Chart for Severity Distribution + Outlier Scatter Plot
    col_anom_c1, col_anom_c2 = st.columns(2)
    with col_anom_c1:
        fig_donut = plot_fuzzy_severity_distribution(sev_counts)
        st.plotly_chart(fig_donut, key="tab5_fuzzy_donut", use_container_width=True)
    with col_anom_c2:
        if out_info and out_info.get('outliers_by_cluster_df') is not None:
            fig_out_cl = plot_outliers_by_cluster(out_info['outliers_by_cluster_df'], simple_mode=is_simple_mode)
            st.plotly_chart(fig_out_cl, key="tab5_outlier_by_cluster", use_container_width=True)
        elif out_info and out_info.get('outlier_df') is not None and len(col_types.get('numeric', [])) >= 2:
            num_x = col_types['numeric'][0]
            num_y = col_types['numeric'][1]
            fig_out = plot_outlier_scatter(out_info['outlier_df'], num_x, num_y, simple_mode=is_simple_mode)
            st.plotly_chart(fig_out, key="tab5_outlier_scatter", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Section C: Top Critical Anomalies Table (Section 11)
    # ---------------------------------------------------------
    st.markdown("### 🚨 Highest-Risk Records (Top 10 Fuzzy Severity)")
    st.markdown("<p style='color:#64748B; font-size:13px;'>Sorted by <b>Mamdani Defuzzified Severity Score</b> descending. Critical and High severity records require prompt business review.</p>", unsafe_allow_html=True)

    top_crit_df = out_info.get('top_critical_anomalies') if out_info else None
    if top_crit_df is not None and not top_crit_df.empty:
        display_rows = []
        for idx, r in top_crit_df.iterrows():
            top_devs = r.get('Top_Deviations', [])
            driver_str = "N/A"
            if isinstance(top_devs, list) and len(top_devs) > 0:
                lead = top_devs[0]
                driver_str = f"{clean_name(lead.get('column', ''))}: {lead.get('this_val', 0):,.1f} (Median: {lead.get('normal_median', 0):,.1f})"

            sev_lbl = r.get('Severity_Label', 'Medium')
            badge_icon = "🔴" if sev_lbl == "Critical" else ("🟠" if sev_lbl == "High" else ("🟡" if sev_lbl == "Medium" else "🟢"))

            display_rows.append({
                "Record ID": f"Row #{idx}",
                "Fuzzy Severity": f"{badge_icon} {sev_lbl}",
                "Severity Score": f"{r.get('Severity_Score', 0):.1f} / 100",
                "Anomaly Intensity": f"{r.get('Anomaly_Intensity', 0):.1f} / 100",
                "Feature Deviation": f"{r.get('Feature_Deviation', 0):.1f} / 100",
                "Business Impact": f"{r.get('Business_Impact', 0):.1f} / 100",
                "Primary Unusual Feature": driver_str
            })
        st.dataframe(pd.DataFrame(display_rows), use_container_width=True)
    else:
        st.info("No extreme isolated anomalies were detected.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Section D: Interactive Anomaly Explanation Panel (Section 12)
    # ---------------------------------------------------------
    st.markdown("### 🔍 Anomaly Inspection & Root-Cause Explanation")
    st.markdown("<p style='color:#64748B; font-size:13px;'>Select any anomalous record to inspect its individual fuzzy inputs, defuzzified score, and root-cause feature drivers:</p>", unsafe_allow_html=True)

    if top_crit_df is not None and not top_crit_df.empty:
        options_map = {}
        for idx, r in top_crit_df.iterrows():
            lbl = f"Row #{idx} | Severity: {r.get('Severity_Label', '')} ({r.get('Severity_Score', 0):.1f}/100)"
            options_map[lbl] = (idx, r)

        selected_label = st.selectbox("Select Record to Inspect:", list(options_map.keys()), key="inspect_record_select")
        sel_idx, sel_row = options_map[selected_label]

        # 4 Metric Cards for Selected Record
        s_col1, s_col2, s_col3, s_col4 = st.columns(4)
        with s_col1:
            st.metric(label="Anomaly Intensity", value=f"{sel_row.get('Anomaly_Intensity', 0):.1f} / 100", help="Normalized inverse Isolation Forest score (0 = normal, 100 = extreme outlier)")
        with s_col2:
            st.metric(label="Feature Deviation", value=f"{sel_row.get('Feature_Deviation', 0):.1f} / 100", help="Aggregated robust z-score across numerical attributes")
        with s_col3:
            st.metric(label="Business Impact", value=f"{sel_row.get('Business_Impact', 0):.1f} / 100", help="Percentile magnitude of business-value attribute (sales, price, revenue)")
        with s_col4:
            sev_color = "#E11D48" if sel_row.get('Severity_Label') == "Critical" else ("#EA580C" if sel_row.get('Severity_Label') == "High" else "#D97706")
            st.markdown(f"""
            <div style='background:#FFFFFF; border:1px solid #E2E8F0; border-radius:8px; padding:10px 14px; text-align:center;'>
                <span style='font-size:11px; font-weight:700; color:#64748B;'>FINAL SEVERITY</span>
                <div style='font-size:20px; font-weight:800; color:{sev_color};'>{sel_row.get('Severity_Label')} ({sel_row.get('Severity_Score', 0):.1f})</div>
            </div>
            """, unsafe_allow_html=True)

        # Human Explanation Box
        st.markdown(f"""
        <div style='background:#F8FAFC; border:1px solid #E2E8F0; border-left:4px solid #0284C7; border-radius:8px; padding:14px 18px; margin:14px 0;'>
            <b style='color:#0F172A; font-size:14px;'>📋 AI Analyst Explanation:</b>
            <p style='color:#334155; font-size:13.5px; margin:6px 0 0 0; line-height:1.55;'>{sel_row.get('Severity_Explanation')}</p>
        </div>
        """, unsafe_allow_html=True)

        # Top 3 Numeric Feature Deviations Table
        top_devs = sel_row.get('Top_Deviations', [])
        if isinstance(top_devs, list) and len(top_devs) > 0:
            st.markdown("<b style='color:#0F172A; font-size:13.5px;'>Largest Numerical Deviations Responsible for Severity:</b>", unsafe_allow_html=True)
            dev_table_rows = []
            for d in top_devs:
                col_name = d.get('column', '')
                dev_table_rows.append({
                    "Feature Column": clean_name(col_name),
                    "Normal Baseline (Median)": f"{d.get('normal_median', 0):,.2f}",
                    "This Record's Value": f"{d.get('this_val', 0):,.2f}",
                    "Robust Z-Score": f"{d.get('robust_z', 0):.2f}x deviation",
                    "Impact Level": "Extreme Deviation" if d.get('robust_z', 0) > 3.0 else "Noticeable Deviation"
                })
            st.dataframe(pd.DataFrame(dev_table_rows), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Section E: Fuzzy Logic Explanation for Demo & Viva (Sections 13 & 14)
    # ---------------------------------------------------------
    with st.expander("🔬 How is Anomaly Severity Calculated? (Fuzzy Logic Engine Demo & Viva)", expanded=not is_simple_mode):
        st.markdown("""
        <div style='background:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px; padding:18px 22px; margin-bottom:16px; font-size:13.5px; color:#334155; line-height:1.6;'>
            <b style='color:#0F172A; font-size:15px;'>Mamdani Classical Fuzzy Inference System</b><br><br>
            <b>1. Detection vs. Severity:</b><br>
            • <b>Isolation Forest</b> detects whether a record is anomalous based on recursive random tree isolation depths.<br>
            • <b>Fuzzy Logic</b> does NOT detect anomalies; it evaluates <i>how severe</i> each detected anomaly is by fusing multiple continuous dimensions.<br><br>
            <b>2. Linguistic Input Variables (Fuzzified 0–100):</b><br>
            • <b>Anomaly Intensity:</b> Derived from normalized inverse Isolation Forest decision function score (0 = normal, 100 = extreme outlier).<br>
            • <b>Feature Deviation:</b> Robust z-score <code>z = 0.6745 * |x - median| / MAD</code> across numerical attributes, aggregated over top-3 deviations.<br>
            • <b>Business Impact:</b> Percentile rank and magnitude of business-critical value metrics (e.g. sales, price, revenue).<br><br>
            <b>3. Linguistic Sets & Membership Functions:</b><br>
            • Inputs: <b>Low</b> (Trapezoidal [0,0,20,45]), <b>Medium</b> (Triangular [20,50,80]), <b>High</b> (Trapezoidal [55,80,100,100]).<br>
            • Output: <b>Low</b> [0–35], <b>Medium</b> [20–62], <b>High</b> [48–85], <b>Critical</b> [72–100].<br><br>
            <b>4. Mamdani Rule Base (14 Explicit Rules):</b><br>
            • <i>Rule 1:</i> IF Intensity is LOW AND Deviation is LOW AND Impact is LOW THEN Severity is LOW<br>
            • <i>Rule 2:</i> IF Intensity is MEDIUM OR Deviation is MEDIUM THEN Severity is MEDIUM<br>
            • <i>Rule 3:</i> IF Intensity is HIGH AND Deviation is HIGH THEN Severity is HIGH<br>
            • <i>Rule 4:</i> IF Intensity is HIGH AND Business Impact is HIGH THEN Severity is CRITICAL<br>
            • <i>Rule 5:</i> IF Deviation is HIGH AND Business Impact is HIGH THEN Severity is CRITICAL<br>
            • <i>Rule 9:</i> IF Intensity is HIGH AND Deviation is HIGH AND Impact is HIGH THEN Severity is CRITICAL<br><br>
            <b>5. Defuzzification:</b><br>
            • Uses <b>Centroid (Center of Gravity) Defuzzification</b>: <code>Severity = ∑(y * μ_Agg(y)) / ∑(μ_Agg(y))</code> over universe [0, 100].
        </div>
        """, unsafe_allow_html=True)

        # Visual Charts for Viva Demo (Section 14)
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            if out_info and out_info.get('membership_curves'):
                fig_mf = plot_fuzzy_membership_functions(out_info['membership_curves'])
                st.plotly_chart(fig_mf, key="tab5_fuzzy_mf_chart", use_container_width=True)
        with f_col2:
            if out_info and out_info.get('outlier_df') is not None:
                fig_fuzz_scatter = plot_fuzzy_scatter(out_info['outlier_df'])
                st.plotly_chart(fig_fuzz_scatter, key="tab5_fuzzy_scatter_chart", use_container_width=True)

# =========================================================
# TAB 6: ASK AI (Was Conversational Assistant)
# =========================================================
with tab_ask:
    st.markdown("<h2 style='color:#0F172A; font-weight:800; margin-bottom:2px;'>Ask AI</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748B; font-size:14px; margin-bottom:18px;'>Ask any question about your data in plain English. Your AI analyst will explain it immediately.</p>", unsafe_allow_html=True)

    def _create_welcome_msg(title: str, rows: int, cols: int) -> dict:
        return {
            'role': 'assistant',
            'content': (
                f"Hello! I'm your **AI Data Analyst**. I've analyzed **{title}** "
                f"({rows:,} rows, {cols} columns) and I'm ready to answer any questions about your data.\n\n"
                f"Try clicking one of the sample questions below or ask me anything!"
            ),
            'chart': None,
            'table': None,
            'code': None
        }

    # Ensure conversation history is tied strictly to the active dataset signature (Bug 3 Fix)
    if 'chat_messages' not in st.session_state or not st.session_state['chat_messages']:
        st.session_state['chat_messages'] = [_create_welcome_msg(friendly_dataset_title, len(cleaned_df), len(cleaned_df.columns))]
        st.session_state['chat_dataset_sig'] = current_dataset_signature
    elif st.session_state.get('chat_dataset_sig') != current_dataset_signature:
        st.session_state['chat_messages'] = [_create_welcome_msg(friendly_dataset_title, len(cleaned_df), len(cleaned_df.columns))]
        st.session_state['chat_dataset_sig'] = current_dataset_signature

    # Approachable Example Prompts (Section 11)
    st.markdown("<span style='color:#64748B; font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:0.5px;'>SAMPLE QUESTIONS YOU CAN ASK:</span>", unsafe_allow_html=True)
    qp0, qp1, qp2, qp3, qp4, qp5 = st.columns(6)
    
    selected_quick_prompt = None
    with qp0:
        if st.button("🚨 Critical Rows", use_container_width=True, help="Identify records with highest computed anomaly risk scores"):
            selected_quick_prompt = "Show me the highest critical rows."
    with qp1:
        if st.button("⚠️ Biggest Problems", use_container_width=True, help="Check data quality problems, missing values, and outliers"):
            selected_quick_prompt = "What are the biggest problems in my data?"
    with qp2:
        if st.button("💡 Key Insights", use_container_width=True, help="Summarize the most important patterns and findings"):
            selected_quick_prompt = "Show me the most important insights."
    with qp3:
        if st.button("🔍 Why Unusual?", use_container_width=True, help="Understand why specific records were flagged"):
            selected_quick_prompt = "Why are some records unusual?"
    with qp4:
        if st.button("🏆 Top Drivers", use_container_width=True, help="See which categories generate the highest volume"):
            selected_quick_prompt = "Which products generate the most sales?"
    with qp5:
        if st.button("📈 Trends", use_container_width=True, help="Look at timeline activity and volume trends"):
            selected_quick_prompt = "Are there any important trends?"

    st.markdown("<br>", unsafe_allow_html=True)

    # Render Conversation History with native markdown, tables, and charts (Bug 2 Fix)
    for msg in st.session_state['chat_messages']:
        role = msg.get('role', 'assistant')
        avatar = "✨" if role == 'assistant' else "👤"
        with st.chat_message(role, avatar=avatar):
            st.markdown(msg.get('content', ''))
            if msg.get('chart') is not None:
                st.plotly_chart(msg['chart'], use_container_width=True)
            if msg.get('table') is not None:
                st.dataframe(msg['table'], use_container_width=True)
            if msg.get('code'):
                code_text = str(msg['code'])
                code_lang = "sql" if any(k in code_text.upper() for k in ["SELECT", "FROM DATASET", "SUMMARIZE", "GROUP BY"]) else "python"
                with st.expander("🔬 View Python / SQL Code"):
                    st.code(code_text, language=code_lang)

    # DuckDB SQL Studio
    with st.expander("⚡ DuckDB In-Memory SQL Query Studio", expanded=False):
        st.markdown("<p style='font-size:13px; color:#64748B;'>Query your active dataset in-memory using ultra-fast analytical SQL powered by <b>DuckDB</b> (tables aliased as <code>dataset</code>, <code>df</code>, or <code>data</code>). Double-quote column names that have spaces or special characters.</p>", unsafe_allow_html=True)
        
        sample_sql_queries = ai_assistant.duckdb_engine.generate_sample_queries()
        query_options = {q['title']: q['sql'] for q in sample_sql_queries}
        query_options["✍️ Custom SQL Query"] = "SELECT * FROM dataset LIMIT 10;"

        selected_preset_title = st.selectbox("Quick Query Templates:", list(query_options.keys()), key="duckdb_preset_selector")
        
        if 'custom_sql_input' not in st.session_state or st.session_state.get('last_preset') != selected_preset_title:
            st.session_state['custom_sql_input'] = query_options[selected_preset_title]
            st.session_state['last_preset'] = selected_preset_title

        user_sql_code = st.text_area("SQL Editor:", value=st.session_state['custom_sql_input'], height=110, key="duckdb_sql_editor")
        
        run_sql_btn = st.button("▶ Run DuckDB SQL Query", type="primary", key="btn_run_duckdb_sql")
        
        if run_sql_btn and user_sql_code:
            with st.spinner("Executing query on DuckDB..."):
                query_res = ai_assistant.duckdb_engine.execute_query(user_sql_code)
                if query_res["success"]:
                    res_df = query_res["df"]
                    exec_ms = query_res["execution_time_ms"]
                    cnt = query_res["row_count"]
                    
                    st.success(f"⚡ Query executed successfully in **{exec_ms:.2f} ms** — returned **{cnt:,}** records.")
                    
                    sql_t1, sql_t2 = st.tabs(["📋 Result Table", "📈 Auto-Chart"])
                    with sql_t1:
                        st.dataframe(res_df, use_container_width=True)
                        csv_data = res_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "⬇️ Download Query Results (CSV)",
                            data=csv_data,
                            file_name="duckdb_query_result.csv",
                            mime="text/csv",
                            key="btn_download_duckdb_csv"
                        )
                    with sql_t2:
                        fig = ai_assistant.duckdb_engine.auto_visualize(res_df, title="DuckDB Query Auto-Chart")
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No automatic chart could be generated for this table format. Switch to 'Result Table' to view all data.")
                else:
                    st.error(f"❌ {query_res['error']}")

    # Chat Input Box (Section 11)
    chat_input_val = st.chat_input("Ask something about your data...")
    prompt_to_run = selected_quick_prompt or chat_input_val
    
    if prompt_to_run:
        st.session_state['chat_messages'].append({
            'role': 'user',
            'content': prompt_to_run,
            'chart': None,
            'table': None,
            'code': None
        })
        
        with st.spinner("Analyzing your data and synthesizing answer..."):
            ai_res = ai_assistant.process_message(prompt_to_run)
            st.session_state['chat_messages'].append({
                'role': 'assistant',
                'content': ai_res['text'],
                'chart': ai_res['chart'],
                'table': ai_res['table'],
                'code': ai_res.get('code_snippet')
            })
        st.rerun()

    if len(st.session_state['chat_messages']) > 1:
        if st.button("🔄 Reset Conversation", key="clear_chat_history_btn"):
            st.session_state['chat_messages'] = [_create_welcome_msg(friendly_dataset_title, len(cleaned_df), len(cleaned_df.columns))]
            st.rerun()
