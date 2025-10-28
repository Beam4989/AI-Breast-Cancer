# method6_streamlit_app.py
"""
Streamlit Decision Support App (TH/EN) — label-only output
- Language switch (ไทย/English)
- Fixed threshold = 0.50
- Show only "เป็น/ไม่เป็น" (TH) or "Positive/Negative" (EN)
- Breast quadrant options: ALWAYS allow both left/right (per user request)
"""
import streamlit as st
import pandas as pd
import joblib
from utils_common import ART_DIR  # paths anchored to file location

# Load artifacts
pre = joblib.load(ART_DIR / "preprocessor.pkl")
model = joblib.load(ART_DIR / "best_model.pkl")  # LabelOnlyClassifier

UI = {
    "th": {
        # 🔁 เปลี่ยนหัวข้อไทยตามคำขอ
        "title": "เครื่องมือช่วยในการคาดเดาการเป็นมะเร็งเต้านม",
        "caption": "เพื่อการศึกษาเท่านั้น ไม่ใช่การวินิจฉัยทางการแพทย์",
        "section_inputs": "ข้อมูลผู้ป่วย / เคส",
        "age": "ช่วงอายุ", "age_help": "เลือกช่วงอายุ (ปี)",
        "menopause": "ภาวะหมดประจำเดือน", "menopause_help": "สถานะโดยรวมของการมีประจำเดือน",
        "tumor_size": "ขนาดก้อนเนื้อ (มม.)", "tumor_size_help": "ขนาดที่วัดได้ (ช่วงมิลลิเมตร)",
        "inv_nodes": "ต่อมน้ำเหลืองที่พบ (จำนวน)", "inv_nodes_help": "ช่วงจำนวนต่อมน้ำเหลืองที่ตรวจพบ",
        "node_caps": "แคปซูลต่อมน้ำเหลือง", "node_caps_help": "มี/ไม่มี การแตกของแคปซูลต่อมน้ำเหลือง",
        "deg_malig": "ระดับความรุนแรงของพยาธิสภาพ (1-3)", "deg_malig_help": "1 = ต่ำ, 2 = ปานกลาง, 3 = สูง",
        "breast": "ข้างของเต้านม", "breast_help": "ซ้าย หรือ ขวา",
        # 🔁 คำอธิบายเตือนว่าตอนนี้ไม่จำกัดด้านแล้ว
        "breast_quad": "ตำแหน่งก้อนในเต้านม", "breast_quad_help": "เลือกได้ทั้งด้านซ้ายและขวา",
        "irradiat": "เคยได้รับรังสีรักษา", "irradiat_help": "เลือกว่าเคยได้รับหรือไม่",
        "btn_predict": "พยากรณ์",
        "badge_yes": "✅ ผลลัพธ์: เป็นมะเร็งเต้านม",
        "badge_no": "🩷 ผลลัพธ์: ไม่เป็นมะเร็งเต้านม",
        "disclaimer": "คำเตือน: ใช้ผลลัพธ์เพื่อสนับสนุนการตัดสินใจร่วมกับวิจารณญาณทางคลินิก",
        "sidebar_settings": "⚙️ การตั้งค่า",
        "sidebar_lang": "ภาษา",
        "lang_th": "ไทย", "lang_en": "English",
    },
    "en": {
        "title": "Decision Support: Breast Cancer (Preliminary)",
        "caption": "For educational use only — not a medical diagnosis.",
        "section_inputs": "Patient / Case Inputs",
        "age": "Age range", "age_help": "Select age bracket (years)",
        "menopause": "Menopausal status", "menopause_help": "Overall menstrual status",
        "tumor_size": "Tumor size (mm)", "tumor_size_help": "Measured size (millimeter range)",
        "inv_nodes": "Involved lymph nodes (count)", "inv_nodes_help": "Observed node count range",
        "node_caps": "Lymph node capsule", "node_caps_help": "Capsular extension present or not",
        "deg_malig": "Histologic malignancy grade (1-3)", "deg_malig_help": "1 = low, 2 = intermediate, 3 = high",
        "breast": "Breast side", "breast_help": "Left or Right",
        # 🔁 ระบุชัดว่าไม่จำกัดด้าน
        "breast_quad": "Tumor location (quadrant)", "breast_quad_help": "You can choose any left/right quadrant",
        "irradiat": "Received radiotherapy", "irradiat_help": "Select if radiotherapy was given",
        "btn_predict": "Predict",
        "badge_yes": "✅ Result: Positive (breast cancer present)",
        "badge_no": "💙 Result: Negative (breast cancer absent)",
        "disclaimer": "Disclaimer: Use alongside clinical judgment; not a definitive diagnosis.",
        "sidebar_settings": "⚙️ Settings",
        "sidebar_lang": "Language",
        "lang_th": "Thai", "lang_en": "English",
    }
}

# Choice maps (display -> internal)
AGE_TH  = [("10–19 ปี","10-19"),("20–29 ปี","20-29"),("30–39 ปี","30-39"),("40–49 ปี","40-49"),
           ("50–59 ปี","50-59"),("60–69 ปี","60-69"),("70–79 ปี","70-79")]
AGE_EN  = [("10–19 years","10-19"),("20–29 years","20-29"),("30–39 years","30-39"),("40–49 years","40-49"),
           ("50–59 years","50-59"),("60–69 years","60-69"),("70–79 years","70-79")]
MENO_TH = [("อายุต่ำกว่า 40 ปี","lt40"),("อายุตั้งแต่ 40 ปีขึ้นไป","ge40"),("ก่อนหมดประจำเดือน","premeno")]
MENO_EN = [("Under 40","lt40"),("Age ≥ 40","ge40"),("Premenopausal","premeno")]
TS = ["0-4","5-9","10-14","15-19","20-24","25-29","30-34","35-39","40-44","45-49","50-54","55-59"]
TS_TH = [(f"{r} มม.", r) for r in TS]
TS_EN = [(f"{r} mm", r) for r in TS]
INR = ["0-2","3-5","6-8","9-11","12-14","15-17","18-20","21-23","24-26","27-29","30-32","33-35","36-39"]
IN_TH = [(f"{r} ต่อม", r) for r in INR]
IN_EN = [(f"{r} nodes", r) for r in INR]
NC_TH = [("มี","yes"),("ไม่มี","no")]
NC_EN = [("Present","yes"),("Absent","no")]
DM_TH = ["1 (ต่ำ)","2 (ปานกลาง)","3 (สูง)"]
DM_EN = ["1 (low)","2 (intermediate)","3 (high)"]
BR_TH = [("ซ้าย","left"),("ขวา","right")]
BR_EN = [("Left","left"),("Right","right")]

# ✅ รวมตัวเลือกตำแหน่ง “ทั้งซ้ายและขวา” เสมอ (ตามคำขอ)
BQ_ALL_TH = [("ซ้าย-บน","left-up"),("ซ้าย-ล่าง","left-low"),
             ("ขวา-บน","right-up"),("ขวา-ล่าง","right-low"),
             ("กึ่งกลาง","central")]
BQ_ALL_EN = [("Left upper","left-up"),("Left lower","left-low"),
             ("Right upper","right-up"),("Right lower","right-low"),
             ("Central","central")]

IR_TH = [("เคยได้รับ","yes"),("ไม่เคย","no")]
IR_EN = [("Yes","yes"),("No","no")]

def to_value(pairs, display):
    return dict(pairs)[display]

# Sidebar
st.sidebar.header(UI["en"]["sidebar_settings"] + " / " + UI["th"]["sidebar_settings"])
lang = st.sidebar.selectbox(
    UI["en"]["sidebar_lang"] + " / " + UI["th"]["sidebar_lang"],
    options=["th","en"],
    format_func=lambda x: "ไทย" if x=="th" else "English",
    index=0
)
TXT = UI[lang]

st.title(TXT["title"])
st.caption(TXT["caption"])
st.subheader(TXT["section_inputs"])

# Lists
ageL   = AGE_TH if lang=="th" else AGE_EN
menoL  = MENO_TH if lang=="th" else MENO_EN
tsL    = TS_TH if lang=="th" else TS_EN
invL   = IN_TH if lang=="th" else IN_EN
ncL    = NC_TH if lang=="th" else NC_EN
dmL    = DM_TH if lang=="th" else DM_EN
brL    = BR_TH if lang=="th" else BR_EN
irL    = IR_TH if lang=="th" else IR_EN
bqL    = BQ_ALL_TH if lang=="th" else BQ_ALL_EN   # ✅ ใช้รายการรวมเสมอ

# Inputs
age_d  = st.selectbox(TXT["age"], [d for d,_ in ageL], help=TXT["age_help"])
men_d  = st.selectbox(TXT["menopause"], [d for d,_ in menoL], help=TXT["menopause_help"])
ts_d   = st.selectbox(TXT["tumor_size"], [d for d,_ in tsL], help=TXT["tumor_size_help"])
inv_d  = st.selectbox(TXT["inv_nodes"], [d for d,_ in invL], help=TXT["inv_nodes_help"])
nc_d   = st.selectbox(TXT["node_caps"], [d for d,_ in ncL], help=TXT["node_caps_help"])
dm_d   = st.selectbox(TXT["deg_malig"], dmL, index=1, help=TXT["deg_malig_help"])

# ด้านเต้านม (ยังคงมีให้เลือกตามข้อมูลฟีเจอร์)
br_d = st.selectbox(TXT["breast"], [d for d,_ in brL], help=TXT["breast_help"])

# 🔁 ตำแหน่งก้อน: ใช้รายการรวม (ไม่ขึ้นกับด้านแล้ว)
bq_d = st.selectbox(TXT["breast_quad"], [d for d,_ in bqL], help=TXT["breast_quad_help"])

ir_d  = st.selectbox(TXT["irradiat"], [d for d,_ in irL], help=TXT["irradiat_help"])

row = {
    "age":         to_value(ageL, age_d),
    "menopause":   to_value(menoL, men_d),
    "tumor-size":  to_value(tsL, ts_d),
    "inv-nodes":   to_value(invL, inv_d),
    "node-caps":   to_value(ncL, nc_d),
    "deg-malig":   int(dm_d.split()[0]),
    "breast":      to_value(brL, br_d),
    "breast-quad": to_value(bqL, bq_d),   # ✅ map จากรายการรวม
    "irradiat":    to_value(irL, ir_d),
}

THRESHOLD = 0.50  # fixed (kept for clarity)

if st.button(TXT["btn_predict"]):
    X_enc = pre.transform(pd.DataFrame([row]))
    y_pred = int(model.predict(X_enc)[0])  # LabelOnlyClassifier returns label
    if y_pred == 1:
        st.success(TXT["badge_yes"])
    else:
        st.info(TXT["badge_no"])
    st.caption(TXT["disclaimer"])
