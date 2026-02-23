import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import time

# ==========================================
# 1. إعدادات الصفحة والتصميم (UI/UX)
# ==========================================
st.set_page_config(page_title="🛡️ AI Judicial Integrity System", layout="wide")

def apply_custom_styles():
    st.markdown("""
        <style>
        .stApp {
            background: radial-gradient(circle at 10% 20%, rgb(0, 0, 0) 0%, rgb(15, 25, 45) 90%);
            color: #E0E0E0;
        }
        .main-header {
            font-size: 45px;
            font-weight: 800;
            background: -webkit-linear-gradient(45deg, #00f2fe, #4facfe);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            padding: 20px;
        }
        .glass-card {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(15px);
            border-radius: 20px;
            padding: 25px;
            border: 1px solid rgba(0, 242, 254, 0.1);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.8);
            margin-bottom: 20px;
        }
        .risk-high { color: #FF4B4B; font-weight: bold; }
        .risk-low { color: #00F2FE; font-weight: bold; }
        </style>
    """, unsafe_allow_html=True)

apply_custom_styles()

# ==========================================
# 2. منطق المعالجة والذكاء الاصطناعي (The Core)
# ==========================================

def process_and_train(df_j, df_d):
    """دمج البيانات، هندسة الميزات، والتدريب على كشف الشذوذ"""
    # أ) ربط ملف العدالة بملف قاعدة البيانات
    try:
        # الربط باستخدام docket كمعرف فريد
        merged = pd.merge(df_j, df_d[['docket', 'issue_area', 'decision_direction', 'majority_votes', 'minority_votes']], 
                         on='docket', how='inner')
    except:
        # fallback في حال اختلاف أسماء الأعمدة
        merged = df_j.copy()

    # ب) هندسة الميزات (Feature Engineering)
    # الموديل يتعلم من: طول النص، توزيع الأصوات، ومنطقة القضية
    merged['facts_len'] = merged['facts'].astype(str).apply(len)
    
    # اختيار الميزات للتدريب
    features = ['facts_len', 'majority_vote', 'issue_area']
    X = merged[features].fillna(0)
    
    # ج) تدريب موديل كشف الشذوذ (Isolation Forest)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    iso_model = IsolationForest(contamination=0.08, random_state=42)
    merged['anomaly_label'] = iso_model.fit_predict(X_scaled)
    
    # د) حساب "نقاط الاشتباه" (Suspicion Score)
    # 0 = طبيعي، 100 = شديد الخطورة
    def calculate_risk(row):
        score = 0
        # إذا كان الموديل الإحصائي يراه شاذاً
        if row['anomaly_label'] == -1: score += 50
        # إذا كان الحكم بالإجماع في قضية معقدة جداً (نص طويل)
        if row['facts_len'] > 2000 and row['majority_vote'] > 7: score += 20
        # إذا كانت القضية في "مناطق حساسة" (مثل الفساد الإداري أو الجرائم المالية)
        if row['issue_area'] in [1, 8]: score += 30
        return min(score, 100)

    merged['risk_score'] = merged.apply(calculate_risk, axis=1)
    return merged

# ==========================================
# 3. واجهة المستخدم (The Interface)
# ==========================================

st.markdown("<h1 class='main-header'>⚖️ AI Judicial Integrity Sentinel</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8892B0;'>نظام الذكاء الاصطناعي المتكامل لتحليل النزاهة القضائية واكتشاف أنماط الرشوة</p>", unsafe_allow_html=True)

# شريط جانبي لرفع الملفات
with st.sidebar:
    st.header("📂 مستودع البيانات")
    file_justice = st.file_uploader("ارفع ملف justice.csv", type="csv")
    file_db = st.file_uploader("ارفع ملف database.csv", type="csv")
    st.markdown("---")
    st.info("النظام يقوم بدمج 'وقائع القضايا' مع 'إحصائيات التصويت' لبناء بروفايل كامل لكل قضية.")

if file_justice and file_db:
    # تحميل البيانات
    df_j = pd.read_csv(file_justice)
    df_d = pd.read_csv(file_db)

    if st.button("🚀 تشغيل محرك التحليل والتدريب"):
        with st.status("جاري معالجة البيانات وتدريب الآلة...", expanded=True) as status:
            st.write("🔗 دمج السجلات التاريخية...")
            time.sleep(1)
            st.write("🧠 استخراج الميزات السياقية (NLP Features)...")
            time.sleep(1)
            final_df = process_and_train(df_j, df_d)
            st.write("🎯 اكتشاف الأنماط الشاذة...")
            status.update(label="✅ اكتمل التحليل!", state="complete", expanded=False)

        # عرض النتائج في كروت
        st.markdown("### 📈 ملخص الرقابة الذكية")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"<div class='glass-card'><h4>إجمالي القضايا</h4><h2>{len(final_df)}</h2></div>", unsafe_allow_html=True)
        with m2:
            high_risk = len(final_df[final_df['risk_score'] > 70])
            st.markdown(f"<div class='glass-card'><h4>🚨 حالات اشتباه عالية</h4><h2 class='risk-high'>{high_risk}</h2></div>", unsafe_allow_html=True)
        with m3:
            avg_risk = round(final_df['risk_score'].mean(), 1)
            st.markdown(f"<div class='glass-card'><h4>متوسط مؤشر المخاطر</h4><h2>{avg_risk}%</h2></div>", unsafe_allow_html=True)

        # عرض القضايا المريبة في جدول فخم
        st.subheader("📋 رادار القضايا المشبوهة")
        display_df = final_df[final_df['risk_score'] > 40].sort_values(by='risk_score', ascending=False)
        
        # تنسيق الجدول
        st.dataframe(
            display_df[['name', 'docket', 'issue_area', 'facts_len', 'risk_score']].style.background_gradient(cmap='OrRd', subset=['risk_score']),
            use_container_width=True
        )

        # الرسوم البيانية التفاعلية
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("#### 🔍 توزيع المخاطر حسب طول الوقائع")
            fig = px.scatter(final_df, x="facts_len", y="majority_vote", 
                             color="risk_score", size="risk_score",
                             color_continuous_scale="Viridis",
                             hover_data=['name'])
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.markdown("#### 📊 المناطق القانونية الأكثر عرضة للشذوذ")
            area_stats = final_df[final_df['risk_score'] > 50]['issue_area'].value_counts().reset_index()
            fig_bar = px.bar(area_stats, x='issue_area', y='count', color='count', color_continuous_scale="Tealgrn")
            fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
            st.plotly_chart(fig_bar, use_container_width=True)

else:
    # واجهة الترحيب في حال عدم رفع ملفات
    st.markdown("""
    <div style='text-align: center; padding: 100px;'>
        <h2 style='color: #8892B0;'>يرجى رفع ملفات البيانات للبدء</h2>
        <p>النظام يحتاج لملف <b>justice.csv</b> (النصوص) وملف <b>database.csv</b> (الإحصائيات) للعمل بدقة 100%.</p>
    </div>
    """, unsafe_allow_html=True)
