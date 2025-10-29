
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="Shanghai East Hospital Endoscopy Hypoglycemia Risk Calculator", layout="wide")

lang = st.sidebar.radio("Language / 语言", ["中文", "English"], index=0)

TXT = {
    "中文": {"title":"消化内镜住院患者低血糖风险计算器（校准版）","caption":"基于多因素Logistic回归的网络护理辅助决策工具（已中心化并经内部校准）",
             "inputs":"输入变量","dx":"原发诊断","dbp":"舒张压分组","glucose":"血糖 (mmol/L)","bun":"尿素氮 (mmol/L)","hb":"血红蛋白 (g/L)",
             "lax":"导泻剂使用","nrs":"NRS-2002 ≥3（营养风险）","compute":"计算风险",
             "pred":"预测概率：{p:.1%}｜风险等级：{tier}","low":"低风险","mid":"中等风险","high":"高风险",
             "advice_low":"常规护理观察；维持正常进食与补液。","advice_mid":"强化营养评估与血糖监测；优化肠道准备方案。","advice_high":"优化血糖控制；个体化肠道准备；密切监测血糖。",
             "viz_title":"模型解释图（可下载PNG）","viz_type":"选择解释图类型","bar":"条形图（重要性）","waterfall":"瀑布图（累积贡献）","force":"力图（正负叠加）",
             "download_png":"下载当前图（PNG）","download_csv":"下载结果（CSV）","footer":"© 2025 同济大学附属东方医院消化内镜中心 · Developed by Yanfen Gu et al.","hint":"请在左侧输入变量后点击“计算风险”。"},
    "English": {"title":"Shanghai East Hospital Endoscopy Hypoglycemia Risk Calculator (Calibrated)","caption":"Web-based decision-support tool with recalibrated intercept and centered continuous predictors.",
             "inputs":"Input Variables","dx":"Primary Diagnosis","dbp":"DBP Category","glucose":"Serum Glucose (mmol/L)","bun":"Blood Urea Nitrogen (mmol/L)","hb":"Hemoglobin (g/L)",
             "lax":"Laxative Use","nrs":"NRS-2002 ≥3 (Nutritional risk)","compute":"Compute Risk",
             "pred":"Predicted probability: {p:.1%} | Risk tier: {tier}","low":"Low risk","mid":"Moderate risk","high":"High risk",
             "advice_low":"Routine care; maintain normal diet and hydration.","advice_mid":"Enhanced nutrition assessment and glucose monitoring.","advice_high":"Optimize glycemic control; individualized bowel prep; close monitoring.",
             "viz_title":"Interpretability plots (downloadable PNG)","viz_type":"Select a plot type","bar":"Bar summary (importance)","waterfall":"Waterfall (cumulative)","force":"Force plot (stacked positive/negative)",
             "download_png":"Download figure (PNG)","download_csv":"Download result (CSV)","footer":"© 2025 Shanghai East Hospital Endoscopy Center · Developed by Yanfen Gu et al.","hint":"Enter variables on the left and click “Compute risk”."}
}

st.title(TXT[lang]["title"])
st.caption(TXT[lang]["caption"])

COEF = {"intercept": -2.000,"dx_colon": -2.296,"dx_gastric": -1.373,"dx_esophageal": -1.191,"dx_other": -2.748,
        "lax_yes": 2.597,"dbp_cat2": -0.007,"dbp_cat3": -0.759,"dbp_cat4": -1.067,
        "bun": 0.090,"glucose": 0.110,"hb": 0.075,"nrs_yes": 1.200}

GLU_MEAN, BUN_MEAN, HB_MEAN = 5.9, 5.5, 130.0

def logistic(x): return 1/(1+np.exp(-x))

st.sidebar.header(TXT[lang]["inputs"])
diagnosis = st.sidebar.selectbox(TXT[lang]["dx"],["Gallbladder/pancreatic disease (reference)","Colonic lesion","Gastric neoplasm","Esophageal lesion","Other diseases"])
dbp_cat = st.sidebar.selectbox(TXT[lang]["dbp"],["< 90 mmHg (reference)","90–100 mmHg","101–110 mmHg","> 110 mmHg"])
glucose = st.sidebar.number_input(TXT[lang]["glucose"],0.0,40.0,GLU_MEAN,0.1)
bun = st.sidebar.number_input(TXT[lang]["bun"],0.0,40.0,BUN_MEAN,0.1)
hb = st.sidebar.number_input(TXT[lang]["hb"],60.0,200.0,HB_MEAN,1.0)
lax = st.sidebar.selectbox(TXT[lang]["lax"],["No / 否","Yes / 是"])
nrs = st.sidebar.selectbox(TXT[lang]["nrs"],["No / 否","Yes / 是"])
go = st.sidebar.button(TXT[lang]["compute"])

def calc_contrib():
    c = {}
    c["Serum_glucose (mmol/L)"] = COEF["glucose"] * (glucose - GLU_MEAN)
    c["Blood_urea_nitrogen (mmol/L)"] = COEF["bun"] * (bun - BUN_MEAN)
    c["Hemoglobin (g/L)"] = COEF["hb"] * (hb - HB_MEAN)
    c["DBP 90–100 mmHg"] = COEF["dbp_cat2"] if dbp_cat=="90–100 mmHg" else 0.0
    c["DBP 101–110 mmHg"] = COEF["dbp_cat3"] if dbp_cat=="101–110 mmHg" else 0.0
    c["DBP > 110 mmHg"] = COEF["dbp_cat4"] if dbp_cat=="> 110 mmHg" else 0.0
    c["Laxative use (Yes)"] = COEF["lax_yes"] if "Yes" in lax else 0.0
    c["NRS-2002 ≥3 (Yes)"] = COEF["nrs_yes"] if "Yes" in nrs else 0.0
    c["Diagnosis: Colonic lesion"] = COEF["dx_colon"] if diagnosis=="Colonic lesion" else 0.0
    c["Diagnosis: Gastric neoplasm"] = COEF["dx_gastric"] if diagnosis=="Gastric neoplasm" else 0.0
    c["Diagnosis: Esophageal lesion"] = COEF["dx_esophageal"] if diagnosis=="Esophageal lesion" else 0.0
    c["Diagnosis: Other diseases"] = COEF["dx_other"] if diagnosis=="Other diseases" else 0.0
    intercept = COEF["intercept"]
    z = intercept + sum(c.values())
    p = logistic(z)
    return z, p, intercept, c

if go:
    z,p,intercept,contrib = calc_contrib()
    if p<0.25: tier=TXT[lang]["low"]; advice=TXT[lang]["advice_low"]; st.success(TXT[lang]["pred"].format(p=p,tier=tier))
    elif p<0.55: tier=TXT[lang]["mid"]; advice=TXT[lang]["advice_mid"]; st.warning(TXT[lang]["pred"].format(p=p,tier=tier))
    else: tier=TXT[lang]["high"]; advice=TXT[lang]["advice_high"]; st.error(TXT[lang]["pred"].format(p=p,tier=tier))
    st.write(advice)

    st.caption("模型版本：Calibrated v1.0 (2025.10)，截距=-2.0，阈值=0.25/0.55，连续变量已中心化")

    st.markdown("---"); st.subheader(TXT[lang]["viz_title"])
    choice = st.selectbox(TXT[lang]["viz_type"],[TXT[lang]["force"],TXT[lang]["waterfall"],TXT[lang]["bar"]])
    df = pd.DataFrame({"Feature":list(contrib.keys()),"Contribution":list(contrib.values())})
    fig,ax = plt.subplots(figsize=(8,4))

    if choice==TXT[lang]["bar"]:
        d = df.assign(abs=np.abs(df["Contribution"])).sort_values("abs")
        ax.barh(d["Feature"], d["abs"]); ax.set_xlabel("|Contribution to log-odds|"); ax.set_ylabel("Feature")
        ax.set_title("Importance summary (|contribution|)")
    elif choice==TXT[lang]["waterfall"]:
        ordered = df.sort_values("Contribution", ascending=False)
        labels = ordered["Feature"].values; vals = ordered["Contribution"].values
        running = intercept; ax.axhline(intercept, color="gray", linestyle="--", linewidth=1)
        for lab,v in zip(labels, vals):
            ax.bar(lab, v, bottom=running if v>=0 else running+v, color="#c0392b" if v>=0 else "#2980b9")
            running += v
        ax.axhline(running, color="black", linestyle="-.", linewidth=1)
        ax.set_ylabel("Log-odds"); ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title("Waterfall (intercept → final log-odds)")
    else:
        ordered = df.sort_values("Contribution", ascending=False)
        pos = ordered[ordered["Contribution"]>0]; neg = ordered[ordered["Contribution"]<=0]
        left = intercept
        for _,r in pos.iterrows():
            ax.barh([0],[r["Contribution"]], left=left, color="#c0392b"); left += r["Contribution"]
        leftn = intercept
        for _,r in neg.iterrows():
            ax.barh([0],[r["Contribution"]], left=leftn+r["Contribution"], color="#2980b9"); leftn += r["Contribution"]
        ax.axvline(0, color="gray", linestyle="--", linewidth=1); ax.set_yticks([]); ax.set_xlabel("Contribution to log-odds")
        ax.set_title("Force-like stacked contributions (red=↑risk, blue=↓risk)")

    st.pyplot(fig)
    buf = BytesIO(); fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    st.download_button(TXT[lang]["download_png"], data=buf.getvalue(), file_name="explanation.png", mime="image/png")

    out = pd.DataFrame({"diagnosis":[diagnosis],"dbp_category":[dbp_cat],"glucose":[glucose],
                        "bun":[bun],"hb":[hb],"laxative":[lax],"nrs2002":[nrs],"logit":[z],"pred_prob":[p]})
    st.download_button(TXT[lang]["download_csv"], data=out.to_csv(index=False).encode("utf-8-sig"),
                       file_name="prediction.csv", mime="text/csv")
else:
    st.info(TXT[lang]["hint"])

st.markdown("---")
st.caption(TXT[lang]["footer"] + " · Version: Calibrated v1.0 (2025.10)")
