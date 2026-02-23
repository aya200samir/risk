# -*- coding: utf-8 -*-
"""
===========================================================================
⚖️ عدالة - النظام المتكامل لكشف الرشوة والفساد في الأحكام القضائية
===========================================================================
الإصدار: 2.0 (Advanced Edition)
التقنيات: ML, NLP, Graph Analysis, Explainable AI

المميزات:
- توليد بيانات تدريبية ذكية مع شذوذ مبرمج
- معالجة وتنظيف أي ملف CSV تلقائياً
- فهم اللغة العربية والإنجليزية
- كشف الشذوذ مع تفسير الأسباب
- حساب احتمالية الرشوة لكل قاضٍ
- تقارير مفصلة مع توصيات
===========================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime, timedelta
import warnings
import re
import time
import hashlib
import random
from collections import Counter
from io import BytesIO
import base64

warnings.filterwarnings('ignore')

# ==================== مكتبات التعلم الآلي ====================
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_curve, auc,
                             mean_squared_error, r2_score)
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# XGBoost
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except:
    XGB_AVAILABLE = False

# SHAP للتفسير
try:
    import shap
    SHAP_AVAILABLE = True
except:
    SHAP_AVAILABLE = False

# NLP
try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    nltk.download('vader_lexicon', quiet=True)
    NLTK_AVAILABLE = True
except:
    NLTK_AVAILABLE = False

# ==================== إعدادات الصفحة ====================
st.set_page_config(
    page_title="عدالة - نظام كشف الرشوة القضائية",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CSS محسن ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * { 
        font-family: 'Cairo', 'Inter', sans-serif;
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f9ff 0%, #ffffff 100%);
    }
    
    /* الهيدر الرئيسي */
    .main-header {
        background: linear-gradient(135deg, #1a3b5d 0%, #2c5a8c 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 0 0 40px 40px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(26,59,93,0.2);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
    }
    
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    .main-header h1 {
        font-size: 3.5rem;
        font-weight: 900;
        margin-bottom: 0.5rem;
        position: relative;
        z-index: 1;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        font-size: 1.3rem;
        opacity: 0.95;
        max-width: 800px;
        margin: 0 auto;
        position: relative;
        z-index: 1;
    }
    
    /* كروت متطورة */
    .glass-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 25px;
        padding: 1.8rem;
        box-shadow: 0 8px 30px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
        border: 1px solid rgba(44,90,140,0.1);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        box-shadow: 0 15px 40px rgba(26,59,93,0.1);
        transform: translateY(-3px);
        border-color: #2c5a8c;
    }
    
    .card-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1a3b5d;
        margin-bottom: 1.5rem;
        border-bottom: 3px solid #eef2f6;
        padding-bottom: 0.8rem;
    }
    
    /* مقاييس */
    .metric-container {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f8fbff, #ffffff);
        border-radius: 20px;
        padding: 1.5rem;
        text-align: center;
        flex: 1;
        min-width: 150px;
        border: 1px solid #dde5ed;
        box-shadow: 0 5px 15px rgba(0,0,0,0.02);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 900;
        color: #1a3b5d;
        line-height: 1.2;
    }
    
    .metric-label {
        color: #5f6b7a;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    
    .metric-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 50px;
        font-size: 0.8rem;
        font-weight: 700;
        margin-top: 0.5rem;
    }
    
    .badge-high {
        background: #ff4b4b;
        color: white;
    }
    
    .badge-medium {
        background: #ffa64b;
        color: white;
    }
    
    .badge-low {
        background: #4bb543;
        color: white;
    }
    
    /* أزرار */
    .stButton > button {
        background: linear-gradient(135deg, #1a3b5d, #2c5a8c);
        color: white;
        font-weight: 700;
        border: none;
        border-radius: 15px;
        padding: 0.8rem 2rem;
        width: 100%;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(26,59,93,0.3);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2c5a8c, #1a3b5d);
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(26,59,93,0.4);
    }
    
    /* تنبيهات */
    .alert {
        padding: 1.5rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        border-right: 8px solid;
    }
    
    .alert-danger {
        background: linear-gradient(135deg, #ffe5e5, #ffd1d1);
        border-right-color: #ff4b4b;
        color: #b30000;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, #fff0e0, #ffe6cc);
        border-right-color: #ffa64b;
        color: #b35f00;
    }
    
    .alert-success {
        background: linear-gradient(135deg, #e0f2e0, #cce6cc);
        border-right-color: #4bb543;
        color: #1e6b1e;
    }
    
    .alert-info {
        background: linear-gradient(135deg, #e0f0ff, #cce4ff);
        border-right-color: #2c5a8c;
        color: #1a3b5d;
    }
    
    /* تذييل */
    .footer {
        background: linear-gradient(135deg, #1a3b5d, #2c5a8c);
        color: white;
        padding: 2rem;
        border-radius: 40px 40px 0 0;
        margin-top: 3rem;
        text-align: center;
    }
    
    /* تبويبات */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: rgba(255,255,255,0.8);
        padding: 0.5rem;
        border-radius: 50px;
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 50px;
        padding: 0.8rem 2rem;
        font-weight: 700;
        color: #5f6b7a;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1a3b5d, #2c5a8c) !important;
        color: white !important;
    }
    
    /* Progress bar */
    .progress-container {
        width: 100%;
        height: 12px;
        background: #eef2f6;
        border-radius: 6px;
        margin: 1rem 0;
        overflow: hidden;
    }
    
    .progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #4bb543, #ffa64b, #ff4b4b);
        border-radius: 6px;
        transition: width 0.5s ease;
    }
    
    /* Feature importance */
    .feature-row {
        display: flex;
        align-items: center;
        margin: 0.8rem 0;
        gap: 1rem;
    }
    
    .feature-name {
        min-width: 150px;
        font-weight: 600;
        color: #1a3b5d;
    }
    
    .feature-bar-container {
        flex: 1;
        height: 12px;
        background: #eef2f6;
        border-radius: 6px;
        overflow: hidden;
    }
    
    .feature-bar {
        height: 100%;
        background: linear-gradient(90deg, #2c5a8c, #4a90e2);
        border-radius: 6px;
    }
    
    .feature-value {
        min-width: 60px;
        text-align: right;
        font-weight: 700;
        color: #2c5a8c;
    }
</style>
""", unsafe_allow_html=True)

# ==================== تهيئة حالة الجلسة ====================
def init_session():
    """تهيئة جميع متغيرات الجلسة"""
    defaults = {
        'data_loaded': False,
        'model_trained': False,
        'df_raw': None,
        'df_clean': None,
        'df_anomalies': None,
        'model_pack': None,
        'judge_corruption_scores': {},
        'lawyer_network': None,
        'explainations': {},
        'training_data_generated': False,
        'file_info': {},
        'bribery_cases': []
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session()

# ==================== دوال معالجة النصوص ====================

def detect_language(text):
    """اكتشاف لغة النص (عربي/إنجليزي)"""
    if not text or pd.isna(text):
        return 'unknown'
    
    text = str(text)
    arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')
    english_pattern = re.compile(r'[a-zA-Z]+')
    
    arabic_count = len(arabic_pattern.findall(text))
    english_count = len(english_pattern.findall(text))
    
    if arabic_count > english_count:
        return 'arabic'
    elif english_count > arabic_count:
        return 'english'
    else:
        return 'mixed'

def extract_text_features(text_series):
    """استخراج ميزات من النصوص"""
    features = []
    
    for text in text_series:
        text = str(text) if not pd.isna(text) else ''
        
        # طول النص
        text_len = len(text)
        
        # عدد الكلمات
        words = len(text.split())
        
        # لغة النص
        lang = detect_language(text)
        is_arabic = 1 if lang == 'arabic' else 0
        is_english = 1 if lang == 'english' else 0
        
        # كلمات مشبوهة (عربي)
        arabic_suspicious = ['رشوة', 'عمولة', 'اتفاق', 'مقابل', 'هدية', 'مصلحة', 'حساب خاص']
        suspicious_ar = sum(1 for word in arabic_suspicious if word in text)
        
        # كلمات مشبوهة (إنجليزي)
        english_suspicious = ['bribe', 'commission', 'agreement', 'special', 'gift', 'personal account']
        suspicious_en = sum(1 for word in english_suspicious if word in text.lower())
        
        # علامات الترقيم (قد تشير إلى نص غامض)
        punctuation_count = len(re.findall(r'[!?;:.,]', text))
        
        # نص مكرر
        words_list = text.split()
        unique_ratio = len(set(words_list)) / max(len(words_list), 1)
        
        features.append({
            'text_length': text_len,
            'word_count': words,
            'is_arabic': is_arabic,
            'is_english': is_english,
            'suspicious_words_ar': suspicious_ar,
            'suspicious_words_en': suspicious_en,
            'punctuation_count': punctuation_count,
            'unique_words_ratio': unique_ratio
        })
    
    return pd.DataFrame(features)

def sentiment_analysis(text):
    """تحليل المشاعر في النص"""
    if not NLTK_AVAILABLE or pd.isna(text):
        return 0, 0
    
    try:
        sia = SentimentIntensityAnalyzer()
        scores = sia.polarity_scores(str(text))
        return scores['compound'], scores['pos'] - scores['neg']
    except:
        return 0, 0

# ==================== توليد بيانات تدريبية متقدمة ====================

def generate_advanced_training_data(n_cases=5000, n_judges=15, n_lawyers=30):
    """
    توليد بيانات تدريبية ذكية مع شذوذ مبرمج
    """
    np.random.seed(42)
    random.seed(42)
    
    # أسماء قضاة (عربي/إنجليزي)
    judges_ar = [f'القاضي {name}' for name in ['أحمد', 'محمد', 'فاطمة', 'سارة', 'خالد', 'نورة', 'عمر', 'ليلى', 'ياسر', 'هند']]
    judges_en = [f'Judge {name}' for name in ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis']]
    judges = judges_ar + judges_en
    judges = judges[:n_judges]
    
    # أسماء محامين
    lawyers_ar = [f'محامي {name}' for name in ['علي', 'حسن', 'مريم', 'إبراهيم', 'منى', 'طارق', 'سوسن', 'جمال']]
    lawyers_en = [f'Lawyer {name}' for name in ['Taylor', 'Anderson', 'Thomas', 'Jackson', 'White', 'Harris', 'Martin']]
    lawyers = lawyers_ar + lawyers_en
    lawyers = lawyers[:n_lawyers]
    
    # أنواع القضايا
    case_types_ar = ['جنائي', 'مدني', 'تجاري', 'إداري', 'أسرة', 'عمالي', 'ضريبي', 'عقاري']
    case_types_en = ['Criminal', 'Civil', 'Commercial', 'Administrative', 'Family', 'Labor', 'Tax', 'Real Estate']
    case_types = case_types_ar + case_types_en
    
    # نتائج
    outcomes_ar = ['قبول', 'رفض', 'تأجيل', 'إعادة نظر', 'براءة', 'إدانة']
    outcomes_en = ['Accepted', 'Rejected', 'Postponed', 'Review', 'Acquittal', 'Conviction']
    outcomes = outcomes_ar + outcomes_en
    
    # مناطق
    districts_ar = ['الشمالية', 'الجنوبية', 'الشرقية', 'الغربية', 'الوسطى']
    districts_en = ['Northern', 'Southern', 'Eastern', 'Western', 'Central']
    districts = districts_ar + districts_en
    
    # توليد البيانات
    data = []
    
    # تحديد القضاة الفاسدين (2-3 قضاة)
    corrupt_judges = random.sample(judges, k=random.randint(2, 3))
    corrupt_lawyers = random.sample(lawyers, k=random.randint(3, 5))
    
    for i in range(n_cases):
        judge = random.choice(judges)
        lawyer = random.choice(lawyers)
        case_type = random.choice(case_types)
        district = random.choice(districts)
        
        # خصائص القضية الأساسية
        evidence_strength = np.random.normal(3, 0.8)  # 1-5
        evidence_strength = max(1, min(5, evidence_strength))
        
        case_duration = int(np.random.gamma(5, 10))  # أيام
        case_duration = max(1, min(90, case_duration))
        
        # تحديد ما إذا كانت القضية مشبوهة
        is_corrupt = 0
        
        # 1. شذوذ مرتبط بالقاضي الفاسد
        if judge in corrupt_judges:
            if lawyer in corrupt_lawyers:
                # قاض فاسد + محامٍ فاسد = نسبة فساد عالية
                is_corrupt = np.random.choice([0, 1], p=[0.3, 0.7])
            else:
                is_corrupt = np.random.choice([0, 1], p=[0.8, 0.2])
        
        # 2. شذوذ في قوة الأدلة مقابل النتيجة
        if evidence_strength > 4 and random.choice(outcomes) in ['رفض', 'Rejected']:
            is_corrupt = np.random.choice([0, 1], p=[0.6, 0.4])
        
        # 3. شذوذ في مدة القضية
        if case_duration < 2 and evidence_strength < 2:
            is_corrupt = np.random.choice([0, 1], p=[0.7, 0.3])
        elif case_duration > 60 and evidence_strength > 4:
            is_corrupt = np.random.choice([0, 1], p=[0.5, 0.5])
        
        # 4. شذوذ في تكرار المحامي مع القاضي
        if i > 100 and i % 50 == 0:  # بعض الحالات المصممة للشذوذ
            if random.random() > 0.7:
                is_corrupt = 1
        
        # توليد النص (عربي/إنجليزي حسب القاضي)
        if 'القاضي' in judge or 'محامي' in lawyer:
            language = 'arabic'
        else:
            language = 'english'
        
        if language == 'arabic':
            if is_corrupt:
                text = f"قضية {case_type} - بناء على الأدلة المقدمة والمرافعات، قررت المحكمة {random.choice(outcomes_ar)}. مع الأخذ في الاعتبار الظروف الخاصة والاتفاقات السابقة."
            else:
                text = f"في قضية {case_type} رقم {i+1000}، بعد الاطلاع على الأوراق وسماع المرافعات، حكمت المحكمة بـ {random.choice(outcomes_ar)}."
        else:
            if is_corrupt:
                text = f"Case {case_type} - based on the evidence and pleadings, the court decides {random.choice(outcomes_en)}. Considering special circumstances and prior agreements."
            else:
                text = f"In case #{i+1000}, after reviewing documents and hearing arguments, the court rules {random.choice(outcomes_en)}."
        
        data.append({
            'case_id': f'CASE-{i+1000:05d}',
            'judge': judge,
            'lawyer': lawyer,
            'case_type': case_type,
            'district': district,
            'evidence_strength': round(evidence_strength, 2),
            'duration_days': case_duration,
            'outcome': random.choice(outcomes),
            'text': text,
            'is_corrupt': is_corrupt,
            'language': language
        })
    
    df = pd.DataFrame(data)
    
    # إضافة أعمدة مساعدة
    df['date'] = pd.date_range(end=datetime.now(), periods=len(df), freq='D')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    return df, corrupt_judges, corrupt_lawyers

# ==================== دوال تنظيف ومعالجة البيانات ====================

def auto_clean_dataframe(df):
    """
    تنظيف ومعالجة أي DataFrame تلقائياً
    """
    df_clean = df.copy()
    cleaning_report = {}
    
    # 1. إزالة الأعمدة الفارغة تماماً
    empty_cols = df_clean.columns[df_clean.isnull().all()].tolist()
    if empty_cols:
        df_clean = df_clean.drop(columns=empty_cols)
        cleaning_report['dropped_empty_columns'] = empty_cols
    
    # 2. معالجة الصفوف المكررة
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    cleaning_report['duplicates_removed'] = initial_rows - len(df_clean)
    
    # 3. التعرف على أنواع الأعمدة
    numeric_cols = []
    categorical_cols = []
    text_cols = []
    date_cols = []
    
    for col in df_clean.columns:
        # محاولة تحويل الأعمدة النصية التي تحتوي أرقام
        if df_clean[col].dtype == 'object':
            # التحقق من التاريخ
            try:
                pd.to_datetime(df_clean[col].dropna().iloc[0] if not df_clean[col].dropna().empty else '')
                date_cols.append(col)
                continue
            except:
                pass
            
            # التحقق من الرقم
            try:
                pd.to_numeric(df_clean[col].dropna().iloc[0] if not df_clean[col].dropna().empty else '')
                numeric_cols.append(col)
                continue
            except:
                pass
            
            # التحقق من النص الطويل
            if df_clean[col].astype(str).str.len().mean() > 50:
                text_cols.append(col)
            else:
                categorical_cols.append(col)
        elif df_clean[col].dtype in ['int64', 'float64']:
            numeric_cols.append(col)
    
    cleaning_report['numeric_columns'] = numeric_cols
    cleaning_report['categorical_columns'] = categorical_cols
    cleaning_report['text_columns'] = text_cols
    cleaning_report['date_columns'] = date_cols
    
    # 4. معالجة القيم المفقودة
    missing_before = df_clean.isnull().sum().sum()
    
    for col in numeric_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    for col in categorical_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown')
    
    for col in text_cols:
        df_clean[col] = df_clean[col].fillna('')
    
    for col in date_cols:
        df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else pd.Timestamp.now())
    
    cleaning_report['missing_values_filled'] = missing_before - df_clean.isnull().sum().sum()
    
    # 5. اكتشاف العلاقات بين الأعمدة
    # البحث عن أعمدة القضاة
    judge_cols = [col for col in df_clean.columns if any(word in col.lower() for word in ['قاضي', 'judge', 'القاضي'])]
    if judge_cols:
        cleaning_report['judge_columns'] = judge_cols
    
    # البحث عن أعمدة المحامين
    lawyer_cols = [col for col in df_clean.columns if any(word in col.lower() for word in ['محامي', 'lawyer', 'المحامي'])]
    if lawyer_cols:
        cleaning_report['lawyer_columns'] = lawyer_cols
    
    # البحث عن أعمدة النتائج
    outcome_cols = [col for col in df_clean.columns if any(word in col.lower() for word in ['نتيجة', 'outcome', 'decision', 'قرار'])]
    if outcome_cols:
        cleaning_report['outcome_columns'] = outcome_cols
    
    return df_clean, cleaning_report

def extract_features_from_any_dataframe(df, cleaning_report):
    """
    استخراج ميزات ذكية من أي DataFrame
    """
    df_features = df.copy()
    feature_names = []
    
    # 1. ميزات من الأعمدة الرقمية
    numeric_cols = cleaning_report.get('numeric_columns', [])
    for col in numeric_cols:
        # تطبيع
        df_features[f'{col}_normalized'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1)
        feature_names.append(f'{col}_normalized')
        
        # Z-Score
        df_features[f'{col}_zscore'] = (df[col] - df[col].mean()) / df[col].std()
        feature_names.append(f'{col}_zscore')
    
    # 2. ميزات من الأعمدة النصية
    text_cols = cleaning_report.get('text_columns', [])
    text_features_list = []
    
    for col in text_cols:
        text_features = extract_text_features(df[col])
        for feat in text_features.columns:
            df_features[f'{col}_{feat}'] = text_features[feat]
            feature_names.append(f'{col}_{feat}')
    
    # 3. ميزات من أعمدة القضاة والمحامين
    judge_cols = cleaning_report.get('judge_columns', [])
    lawyer_cols = cleaning_report.get('lawyer_columns', [])
    
    if judge_cols and lawyer_cols:
        # ترميز القضاة والمحامين
        for jcol in judge_cols:
            le = LabelEncoder()
            df_features[f'{jcol}_encoded'] = le.fit_transform(df[jcol].astype(str))
            feature_names.append(f'{jcol}_encoded')
        
        for lcol in lawyer_cols:
            le = LabelEncoder()
            df_features[f'{lcol}_encoded'] = le.fit_transform(df[lcol].astype(str))
            feature_names.append(f'{lcol}_encoded')
    
    # 4. ميزات من أعمدة التاريخ
    date_cols = cleaning_report.get('date_columns', [])
    for col in date_cols:
        df_features[f'{col}_year'] = pd.to_datetime(df[col]).dt.year
        df_features[f'{col}_month'] = pd.to_datetime(df[col]).dt.month
        df_features[f'{col}_dayofweek'] = pd.to_datetime(df[col]).dt.dayofweek
        feature_names.extend([f'{col}_year', f'{col}_month', f'{col}_dayofweek'])
    
    return df_features, feature_names

# ==================== دوال كشف الشذوذ والرشوة ====================

def detect_anomalies_multiple_methods(df, feature_cols, contamination=0.1):
    """
    كشف الشذوذ باستخدام عدة طرق
    """
    X = df[feature_cols].fillna(0)
    
    # توحيد المقاييس
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = {}
    
    # 1. Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    iso_pred = iso_forest.fit_predict(X_scaled)
    results['isolation_forest'] = (iso_pred == -1).astype(int)
    
    # 2. Local Outlier Factor
    lof = LocalOutlierFactor(contamination=contamination)
    lof_pred = lof.fit_predict(X_scaled)
    results['lof'] = (lof_pred == -1).astype(int)
    
    # 3. DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_pred = dbscan.fit_predict(X_scaled)
    results['dbscan'] = (dbscan_pred == -1).astype(int)
    
    # 4. Z-Score based
    z_scores = np.abs((X - X.mean()) / X.std())
    z_score_anomalies = (z_scores > 3).any(axis=1).astype(int)
    results['zscore'] = z_score_anomalies
    
    # دمج النتائج
    anomaly_df = pd.DataFrame(results)
    df['anomaly_score'] = anomaly_df.mean(axis=1)
    df['is_anomaly'] = df['anomaly_score'] > 0.5
    
    return df, anomaly_df

def calculate_corruption_probability(df, anomaly_df, cleaning_report):
    """
    حساب احتمالية الرشوة لكل قاضٍ
    """
    judge_scores = {}
    
    # البحث عن أعمدة القضاة
    judge_cols = cleaning_report.get('judge_columns', [])
    
    if not judge_cols:
        # البحث عن أي عمود نصي قد يمثل القضاة
        for col in cleaning_report.get('categorical_columns', []):
            if any(word in col.lower() for word in ['قاضي', 'judge', 'القاضي']):
                judge_cols = [col]
                break
    
    if not judge_cols:
        return {}
    
    judge_col = judge_cols[0]
    
    # تحليل كل قاضٍ
    for judge in df[judge_col].unique():
        judge_data = df[df[judge_col] == judge]
        judge_anomalies = anomaly_df[df[judge_col] == judge]
        
        # 1. نسبة الشذوذ
        anomaly_rate = judge_anomalies.mean(axis=1).mean() if not judge_anomalies.empty else 0
        
        # 2. تحليل نصوص القضايا
        text_cols = cleaning_report.get('text_columns', [])
        text_suspicion = 0
        
        if text_cols:
            text_col = text_cols[0]
            texts = judge_data[text_col].fillna('')
            
            # كلمات مشبوهة
            arabic_suspicious = ['رشوة', 'عمولة', 'اتفاق', 'مقابل', 'هدية']
            english_suspicious = ['bribe', 'commission', 'agreement', 'special']
            
            for text in texts:
                text = str(text).lower()
                if any(word in text for word in arabic_suspicious + english_suspicious):
                    text_suspicion += 1
            
            text_suspicion = text_suspicion / max(len(texts), 1)
        
        # 3. تحليل مدة القضايا (إذا وجدت)
        duration_cols = [col for col in df.columns if any(word in col.lower() for word in ['duration', 'days', 'مدة'])]
        duration_anomaly = 0
        
        if duration_cols:
            duration_col = duration_cols[0]
            durations = judge_data[duration_col]
            if len(durations) > 1:
                z_scores = np.abs((durations - durations.mean()) / durations.std())
                duration_anomaly = (z_scores > 2).mean()
        
        # 4. تحليل العلاقات مع المحامين
        lawyer_cols = cleaning_report.get('lawyer_columns', [])
        lawyer_anomaly = 0
        
        if lawyer_cols and len(judge_data) > 5:
            lawyer_col = lawyer_cols[0]
            lawyer_counts = judge_data[lawyer_col].value_counts()
            # إذا كان هناك محامٍ يظهر بكثرة غير طبيعية
            if len(lawyer_counts) > 0:
                max_lawyer_ratio = lawyer_counts.max() / len(judge_data)
                if max_lawyer_ratio > 0.3:  # أكثر من 30% مع نفس المحامي
                    lawyer_anomaly = max_lawyer_ratio
        
        # حساب النتيجة النهائية
        corruption_prob = (
            0.4 * anomaly_rate +
            0.3 * text_suspicion +
            0.2 * duration_anomaly +
            0.1 * lawyer_anomaly
        )
        
        judge_scores[judge] = {
            'corruption_probability': min(1, corruption_prob),
            'anomaly_rate': anomaly_rate,
            'text_suspicion': text_suspicion,
            'duration_anomaly': duration_anomaly,
            'lawyer_anomaly': lawyer_anomaly,
            'total_cases': len(judge_data),
            'anomaly_cases': judge_anomalies.mean(axis=1).sum() if not judge_anomalies.empty else 0
        }
    
    return judge_scores

def explain_corruption(judge, score, features, judge_data):
    """
    تفسير أسباب احتمالية الرشوة
    """
    explanation = []
    
    if score['corruption_probability'] > 0.7:
        explanation.append("🚨 **مؤشر خطر مرتفع جداً**")
    elif score['corruption_probability'] > 0.4:
        explanation.append("⚠️ **مؤشر خطر متوسط**")
    else:
        explanation.append("✅ **مؤشر خطر منخفض**")
    
    explanation.append(f"\n**إحصائيات القاضي {judge}:**")
    explanation.append(f"- عدد القضايا: {score['total_cases']}")
    explanation.append(f"- قضايا شاذة: {int(score['anomaly_cases'])} ({score['anomaly_rate']*100:.1f}%)")
    
    # تفسير كل عامل
    factors = []
    if score['anomaly_rate'] > 0.3:
        factors.append("🔴 ارتفاع نسبة القضايا الشاذة")
    elif score['anomaly_rate'] > 0.15:
        factors.append("🟡 نسبة معتدلة من القضايا الشاذة")
    
    if score['text_suspicion'] > 0.2:
        factors.append("🔴 وجود كلمات مشبوهة في نصوص الأحكام")
    
    if score['duration_anomaly'] > 0.3:
        factors.append("🔴 تباين غير طبيعي في مدة القضايا")
    
    if score['lawyer_anomaly'] > 0.3:
        factors.append(f"🔴 تركيز عالٍ مع محامٍ معين ({score['lawyer_anomaly']*100:.1f}% من القضايا)")
    
    if factors:
        explanation.append("\n**العوامل المؤثرة:**")
        explanation.extend(factors)
    else:
        explanation.append("\n✅ لا توجد عوامل خطر واضحة")
    
    # توصيات
    explanation.append("\n**📋 التوصيات:**")
    if score['corruption_probability'] > 0.7:
        explanation.append("- مراجعة عاجلة لجميع قضايا القاضي")
        explanation.append("- تدقيق العلاقات مع المحامين المتكررين")
        explanation.append("- تحليل نصوص الأحكام بشكل موسع")
    elif score['corruption_probability'] > 0.4:
        explanation.append("- مراجعة القضايا الشاذة فقط")
        explanation.append("- متابعة القضايا المستقبلية")
    else:
        explanation.append("- لا توجد توصيات خاصة")
    
    return '\n'.join(explanation)

# ==================== دوال تحليل الشبكات ====================

def build_judge_lawyer_network(df, judge_col, lawyer_col):
    """
    بناء شبكة العلاقات بين القضاة والمحامين
    """
    G = nx.Graph()
    
    # إضافة العقد
    judges = df[judge_col].unique()
    lawyers = df[lawyer_col].unique()
    
    for judge in judges:
        G.add_node(judge, type='judge', size=len(df[df[judge_col] == judge]))
    
    for lawyer in lawyers:
        G.add_node(lawyer, type='lawyer', size=len(df[df[lawyer_col] == lawyer]))
    
    # إضافة الحواف
    for _, row in df.iterrows():
        if G.has_edge(row[judge_col], row[lawyer_col]):
            G[row[judge_col]][row[lawyer_col]]['weight'] += 1
        else:
            G.add_edge(row[judge_col], row[lawyer_col], weight=1)
    
    return G

def plot_network(G, judge_scores):
    """
    رسم شبكة العلاقات
    """
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # ألوان العقد
    node_colors = []
    node_sizes = []
    
    for node in G.nodes():
        if G.nodes[node]['type'] == 'judge':
            # لون حسب درجة الخطورة
            score = judge_scores.get(node, {}).get('corruption_probability', 0)
            if score > 0.7:
                node_colors.append('red')
            elif score > 0.4:
                node_colors.append('orange')
            else:
                node_colors.append('green')
            node_sizes.append(G.nodes[node]['size'] * 20)
        else:
            node_colors.append('blue')
            node_sizes.append(G.nodes[node]['size'] * 10)
    
    # إنشاء figure
    fig = go.Figure()
    
    # إضافة الحواف
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    ))
    
    # إضافة العقد
    node_x = []
    node_y = []
    node_text = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        if G.nodes[node]['type'] == 'judge':
            score = judge_scores.get(node, {}).get('corruption_probability', 0)
            node_text.append(f"{node}<br>نوع: قاضي<br>قضايا: {G.nodes[node]['size']}<br>خطورة: {score*100:.1f}%")
        else:
            node_text.append(f"{node}<br>نوع: محامي<br>قضايا: {G.nodes[node]['size']}")
    
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=2, color='white')
        ),
        text=[node[:15] + '...' if len(node) > 15 else node for node in G.nodes()],
        textposition="top center",
        hoverinfo='text',
        hovertext=node_text
    ))
    
    fig.update_layout(
        title='شبكة العلاقات بين القضاة والمحامين',
        showlegend=False,
        hovermode='closest',
        width=800,
        height=600,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    return fig

# ==================== الصفحة الرئيسية ====================

def main():
    # الهيدر
    st.markdown("""
    <div class="main-header">
        <h1>⚖️ عدالة - نظام كشف الرشوة القضائية</h1>
        <p>تحليل متكامل للأحكام القضائية باستخدام الذكاء الاصطناعي والشبكات</p>
        <div style="margin-top: 1rem;">
            <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 50px; margin: 0 0.5rem;">🤖 ذكاء اصطناعي</span>
            <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 50px; margin: 0 0.5rem;">🔍 كشف شذوذ</span>
            <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 50px; margin: 0 0.5rem;">📊 تحليل شبكات</span>
            <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 50px; margin: 0 0.5rem;">🌐 عربي/إنجليزي</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # الشريط الجانبي
    with st.sidebar:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1a3b5d10, #2c5a8c10); padding: 1.5rem; border-radius: 25px;">
            <h2 style="color: #1a3b5d; text-align: center;">🔧 لوحة التحكم</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # قسم البيانات
        st.markdown("### 📁 مصدر البيانات")
        data_option = st.radio(
            "اختر مصدر البيانات",
            ["🔄 توليد بيانات تجريبية", "📂 رفع ملف CSV"],
            index=0
        )
        
        if data_option == "🔄 توليد بيانات تجريبية":
            n_cases = st.slider("عدد القضايا", 1000, 10000, 5000, step=500)
            
            if st.button("🎲 توليد بيانات ذكية", use_container_width=True):
                with st.spinner("جاري توليد بيانات تدريبية متقدمة..."):
                    df, corrupt_judges, corrupt_lawyers = generate_advanced_training_data(
                        n_cases=n_cases,
                        n_judges=15,
                        n_lawyers=30
                    )
                    
                    st.session_state.df_raw = df
                    st.session_state.corrupt_judges = corrupt_judges
                    st.session_state.corrupt_lawyers = corrupt_lawyers
                    st.session_state.training_data_generated = True
                    
                    # تنظيف البيانات
                    df_clean, report = auto_clean_dataframe(df)
                    st.session_state.df_clean = df_clean
                    st.session_state.cleaning_report = report
                    st.session_state.data_loaded = True
                    
                    st.success(f"✅ تم توليد {len(df)} قضية بنجاح")
                    st.info(f"🔴 قضاة فاسدون مبرمجون: {', '.join(corrupt_judges)}")
        
        else:  # رفع ملف
            uploaded_file = st.file_uploader(
                "اختر ملف CSV",
                type=['csv'],
                help="يدعم العربية والإنجليزية"
            )
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.df_raw = df
                    
                    with st.spinner("جاري تنظيف ومعالجة البيانات..."):
                        df_clean, report = auto_clean_dataframe(df)
                        st.session_state.df_clean = df_clean
                        st.session_state.cleaning_report = report
                        st.session_state.data_loaded = True
                        
                    st.success(f"✅ تم تحميل {len(df)} سجل")
                    
                    # عرض تقرير التنظيف
                    with st.expander("📋 تقرير تنظيف البيانات"):
                        st.json(report)
                        
                except Exception as e:
                    st.error(f"خطأ في قراءة الملف: {str(e)}")
        
        st.markdown("---")
        
        # إعدادات التحليل
        if st.session_state.data_loaded:
            st.markdown("### ⚙️ إعدادات التحليل")
            
            contamination = st.slider(
                "حساسية كشف الشذوذ",
                0.01, 0.3, 0.1, 0.01,
                help="نسبة الحالات المتوقعة كشاذة"
            )
            
            if st.button("🔍 تحليل وكشف الشذوذ", use_container_width=True):
                with st.spinner("جاري تحليل البيانات وكشف الشذوذ..."):
                    df = st.session_state.df_clean
                    report = st.session_state.cleaning_report
                    
                    # استخراج الميزات
                    df_features, feature_names = extract_features_from_any_dataframe(df, report)
                    
                    # اختيار الميزات المناسبة
                    numeric_features = [f for f in feature_names if 'normalized' in f or 'zscore' in f]
                    if not numeric_features:
                        numeric_features = report.get('numeric_columns', [])
                    
                    if numeric_features:
                        # كشف الشذوذ
                        df_anomalies, anomaly_df = detect_anomalies_multiple_methods(
                            df_features, numeric_features, contamination
                        )
                        
                        # حساب احتمالية الرشوة
                        judge_scores = calculate_corruption_probability(
                            df_anomalies, anomaly_df, report
                        )
                        
                        st.session_state.df_anomalies = df_anomalies
                        st.session_state.anomaly_df = anomaly_df
                        st.session_state.judge_scores = judge_scores
                        st.session_state.model_trained = True
                        
                        st.success("✅ تم التحليل بنجاح")
                        
                        # إحصائيات سريعة
                        anomalies_count = df_anomalies['is_anomaly'].sum()
                        st.info(f"🔍 تم اكتشاف {anomalies_count} حالة شاذة")
    
    # المحتوى الرئيسي
    if not st.session_state.data_loaded:
        # شاشة الترحيب
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="glass-card">
                <h3 style="color: #1a3b5d;">🔍 1. تحليل ذكي</h3>
                <p>يكتشف الأنماط غير الطبيعية في الأحكام باستخدام تقنيات متعددة</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="glass-card">
                <h3 style="color: #1a3b5d;">📊 2. شبكة العلاقات</h3>
                <p>يحلل العلاقات بين القضاة والمحامين ويكشف التحيز</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="glass-card">
                <h3 style="color: #1a3b5d;">🧠 3. تفسير النتائج</h3>
                <p>يشرح أسباب الشذوذ ويقيم احتمالية الرشوة</p>
            </div>
            """, unsafe_allow_html=True)
        
        return
    
    # إنشاء التبويبات
    tabs = st.tabs([
        "📊 لوحة المعلومات",
        "🔍 تحليل الشذوذ",
        "👨‍⚖️ تقييم القضاة",
        "🕸️ شبكة العلاقات",
        "📝 تحليل النصوص",
        "📋 البيانات"
    ])
    
    # ========== لوحة المعلومات ==========
    with tabs[0]:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📊 نظرة عامة</div>', unsafe_allow_html=True)
        
        df = st.session_state.df_clean
        report = st.session_state.cleaning_report
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(df):,}</div>
                <div class="metric-label">إجمالي القضايا</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            judge_cols = report.get('judge_columns', [])
            if judge_cols:
                n_judges = df[judge_cols[0]].nunique()
            else:
                n_judges = 'غير محدد'
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{n_judges}</div>
                <div class="metric-label">عدد القضاة</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            if st.session_state.model_trained:
                anomalies = st.session_state.df_anomalies['is_anomaly'].sum()
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{anomalies}</div>
                    <div class="metric-label">حالات شاذة</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-value">-</div>
                    <div class="metric-label">حالات شاذة</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col4:
            if st.session_state.judge_scores:
                high_risk = sum(1 for j in st.session_state.judge_scores.values() 
                              if j['corruption_probability'] > 0.7)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{high_risk}</div>
                    <div class="metric-label">قضاة خطر</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-value">-</div>
                    <div class="metric-label">قضاة خطر</div>
                </div>
                """, unsafe_allow_html=True)
        
        # توزيع البيانات
        col1, col2 = st.columns(2)
        
        with col1:
            if 'case_type' in df.columns:
                fig = px.pie(
                    df['case_type'].value_counts().head(8).reset_index(),
                    values='count',
                    names='case_type',
                    title='توزيع أنواع القضايا'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            judge_cols = report.get('judge_columns', [])
            if judge_cols:
                judge_col = judge_cols[0]
                judge_counts = df[judge_col].value_counts().head(10)
                fig = px.bar(
                    x=judge_counts.values,
                    y=judge_counts.index,
                    orientation='h',
                    title='أكثر القضاة نشاطاً',
                    labels={'x': 'عدد القضايا', 'y': 'القاضي'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ========== تحليل الشذوذ ==========
    with tabs[1]:
        if not st.session_state.model_trained:
            st.warning("⚠️ يرجى تشغيل تحليل الشذوذ أولاً من القائمة الجانبية")
        else:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">🔍 نتائج تحليل الشذوذ</div>', unsafe_allow_html=True)
            
            df_anomalies = st.session_state.df_anomalies
            anomaly_df = st.session_state.anomaly_df
            
            # إحصائيات
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{df_anomalies['is_anomaly'].sum()}</div>
                    <div class="metric-label">حالات شاذة</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{df_anomalies['is_anomaly'].sum()/len(df_anomalies)*100:.1f}%</div>
                    <div class="metric-label">نسبة الشذوذ</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                avg_score = df_anomalies['anomaly_score'].mean()
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{avg_score:.3f}</div>
                    <div class="metric-label">متوسط درجة الشذوذ</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                methods_agreement = anomaly_df.mean(axis=1).mean()
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{methods_agreement*100:.1f}%</div>
                    <div class="metric-label">توافق طرق الكشف</div>
                </div>
                """, unsafe_allow_html=True)
            
            # عرض الحالات الشاذة
            st.markdown("### 🚨 القضايا المشبوهة")
            anomalies_only = df_anomalies[df_anomalies['is_anomaly']]
            
            # اختيار الأعمدة للعرض
            display_cols = [col for col in anomalies_only.columns if col not in 
                          ['anomaly_score', 'is_anomaly'] + list(anomaly_df.columns)]
            display_cols = display_cols[:10]  # حد أقصى 10 أعمدة
            
            st.dataframe(anomalies_only[display_cols].head(20), use_container_width=True)
            
            # توزيع طرق الكشف
            fig = make_subplots(rows=2, cols=2,
                               subplot_titles=['Isolation Forest', 'LOF', 'DBSCAN', 'Z-Score'])
            
            for i, method in enumerate(['isolation_forest', 'lof', 'dbscan', 'zscore']):
                row = i//2 + 1
                col = i%2 + 1
                
                values = anomaly_df[method].value_counts()
                fig.add_trace(
                    go.Pie(labels=['طبيعي', 'شاذ'], values=[values.get(0,0), values.get(1,0)],
                           marker=dict(colors=['#4bb543', '#ff4b4b'])),
                    row=row, col=col
                )
            
            fig.update_layout(height=600, title_text="مقارنة طرق كشف الشذوذ")
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # ========== تقييم القضاة ==========
    with tabs[2]:
        if not st.session_state.judge_scores:
            st.warning("⚠️ لا توجد بيانات كافية لتقييم القضاة")
        else:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">👨‍⚖️ تقييم احتمالية الرشوة للقضاة</div>', unsafe_allow_html=True)
            
            judge_scores = st.session_state.judge_scores
            
            # ترتيب القضاة حسب الخطورة
            sorted_judges = sorted(judge_scores.items(), 
                                  key=lambda x: x[1]['corruption_probability'], 
                                  reverse=True)
            
            # جدول النتائج
            results = []
            for judge, score in sorted_judges:
                results.append({
                    'القاضي': judge,
                    'احتمالية الرشوة': f"{score['corruption_probability']*100:.1f}%",
                    'نسبة الشذوذ': f"{score['anomaly_rate']*100:.1f}%",
                    'كلمات مشبوهة': f"{score['text_suspicion']*100:.1f}%",
                    'شذوذ المدة': f"{score['duration_anomaly']*100:.1f}%",
                    'تكرار محامٍ': f"{score['lawyer_anomaly']*100:.1f}%",
                    'عدد القضايا': score['total_cases']
                })
            
            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True)
            
            # شريط التقدم للقضاة الأكثر خطورة
            st.markdown("### 📊 أكثر القضاة خطورة")
            
            for judge, score in sorted_judges[:5]:
                prob = score['corruption_probability']
                
                if prob > 0.7:
                    badge = '<span class="metric-badge badge-high">خطر مرتفع</span>'
                elif prob > 0.4:
                    badge = '<span class="metric-badge badge-medium">خطر متوسط</span>'
                else:
                    badge = '<span class="metric-badge badge-low">خطر منخفض</span>'
                
                st.markdown(f"""
                <div style="margin: 1rem 0;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span><strong>{judge}</strong> {badge}</span>
                        <span>{prob*100:.1f}%</span>
                    </div>
                    <div class="progress-container">
                        <div class="progress-bar" style="width: {prob*100}%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # تفسير مفصل
            st.markdown("### 🔍 تفسير النتائج")
            
            selected_judge = st.selectbox(
                "اختر قاضياً لعرض التحليل المفصل",
                [j for j, _ in sorted_judges]
            )
            
            if selected_judge:
                score = judge_scores[selected_judge]
                explanation = explain_corruption(
                    selected_judge, 
                    score,
                    st.session_state.cleaning_report,
                    st.session_state.df_anomalies if st.session_state.df_anomalies is not None else st.session_state.df_clean
                )
                
                st.markdown(f"""
                <div class="alert-info">
                    {explanation.replace('\n', '<br>')}
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # ========== شبكة العلاقات ==========
    with tabs[3]:
        if not st.session_state.data_loaded:
            st.warning("⚠️ لا توجد بيانات كافية")
        else:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">🕸️ شبكة العلاقات بين القضاة والمحامين</div>', unsafe_allow_html=True)
            
            df = st.session_state.df_clean
            report = st.session_state.cleaning_report
            
            judge_cols = report.get('judge_columns', [])
            lawyer_cols = report.get('lawyer_columns', [])
            
            if not judge_cols or not lawyer_cols:
                st.warning("⚠️ لا توجد أعمدة كافية للقضاة والمحامين في البيانات")
            else:
                judge_col = judge_cols[0]
                lawyer_col = lawyer_cols[0]
                
                # بناء الشبكة
                G = build_judge_lawyer_network(df, judge_col, lawyer_col)
                
                # رسم الشبكة
                fig = plot_network(G, st.session_state.judge_scores or {})
                st.plotly_chart(fig, use_container_width=True)
                
                # تحليل الشبكة
                st.markdown("### 📊 تحليل الشبكة")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # أكثر القضاة ارتباطاً
                    judge_degrees = [(node, G.degree(node)) for node in G.nodes() 
                                   if G.nodes[node]['type'] == 'judge']
                    judge_degrees.sort(key=lambda x: x[1], reverse=True)
                    
                    st.markdown("**🔗 أكثر القضاة ارتباطاً:**")
                    for judge, degree in judge_degrees[:5]:
                        st.write(f"- {judge}: {degree} علاقة")
                
                with col2:
                    # أكثر المحامين ارتباطاً
                    lawyer_degrees = [(node, G.degree(node)) for node in G.nodes() 
                                    if G.nodes[node]['type'] == 'lawyer']
                    lawyer_degrees.sort(key=lambda x: x[1], reverse=True)
                    
                    st.markdown("**🔗 أكثر المحامين ارتباطاً:**")
                    for lawyer, degree in lawyer_degrees[:5]:
                        st.write(f"- {lawyer}: {degree} علاقة")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # ========== تحليل النصوص ==========
    with tabs[4]:
        if not st.session_state.data_loaded:
            st.warning("⚠️ لا توجد بيانات كافية")
        else:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">📝 تحليل النصوص وفهم السياق</div>', unsafe_allow_html=True)
            
            df = st.session_state.df_clean
            report = st.session_state.cleaning_report
            
            text_cols = report.get('text_columns', [])
            
            if not text_cols:
                st.warning("⚠️ لا توجد أعمدة نصية في البيانات")
            else:
                text_col = st.selectbox("اختر عمود النصوص للتحليل", text_cols)
                
                # تحليل النصوص
                text_features = extract_text_features(df[text_col])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # توزيع اللغات
                    lang_counts = text_features[['is_arabic', 'is_english']].sum()
                    fig = px.pie(
                        values=[lang_counts.get('is_arabic', 0), lang_counts.get('is_english', 0)],
                        names=['عربي', 'إنجليزي'],
                        title='توزيع اللغات في النصوص'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # الكلمات المشبوهة
                    suspicious_cols = [col for col in text_features.columns if 'suspicious' in col]
                    if suspicious_cols:
                        suspicious_total = text_features[suspicious_cols].sum().sum()
                        fig = px.bar(
                            x=text_features[suspicious_cols].sum().index,
                            y=text_features[suspicious_cols].sum().values,
                            title='الكلمات المشبوهة',
                            labels={'x': 'النوع', 'y': 'التكرار'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # عرض عينات من النصوص
                st.markdown("### 📄 عينات من النصوص")
                
                # نص عادي
                normal_samples = df[~df['is_anomaly']][text_col].head(3) if 'is_anomaly' in df.columns else df[text_col].head(3)
                with st.expander("✅ نصوص طبيعية"):
                    for i, text in enumerate(normal_samples):
                        st.write(f"**نص {i+1}:** {text[:200]}..." if len(str(text)) > 200 else text)
                        st.write(f"*اللغة: {detect_language(text)}*")
                        st.write("---")
                
                # نصوص شاذة
                if 'is_anomaly' in df.columns and df['is_anomaly'].any():
                    anomaly_samples = df[df['is_anomaly']][text_col].head(3)
                    with st.expander("🚨 نصوص شاذة (مشبوهة)"):
                        for i, text in enumerate(anomaly_samples):
                            st.write(f"**نص {i+1}:** {text[:200]}..." if len(str(text)) > 200 else text)
                            st.write(f"*اللغة: {detect_language(text)}*")
                            st.write("---")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # ========== عرض البيانات ==========
    with tabs[5]:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📋 عرض البيانات</div>', unsafe_allow_html=True)
        
        df = st.session_state.df_clean
        
        st.markdown(f"**إجمالي السجلات:** {len(df):,}")
        st.markdown(f"**إجمالي الأعمدة:** {len(df.columns)}")
        
        # اختيار الأعمدة للعرض
        all_columns = df.columns.tolist()
        selected_columns = st.multiselect("اختر الأعمدة للعرض", all_columns, default=all_columns[:10])
        
        if selected_columns:
            st.dataframe(df[selected_columns], use_container_width=True)
        
        # إحصائيات سريعة
        with st.expander("📊 إحصائيات سريعة"):
            st.dataframe(df.describe(include='all'), use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # الفوتر
    st.markdown("""
    <div class="footer">
        <h3>⚖️ عدالة - نظام كشف الرشوة القضائية</h3>
        <p>الإصدار 2.0 | يدعم العربية والإنجليزية | تقنيات: ML, NLP, Graph Analysis</p>
        <p style="opacity:0.8; margin-top:1rem;">جميع الحقوق محفوظة © 2026</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
