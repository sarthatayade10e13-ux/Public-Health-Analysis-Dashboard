import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ---------------- PAGE ----------------
st.set_page_config(page_title="Public Health Dashboard", layout="wide")

# Colorful headings
st.markdown(
    "<h1 style='text-align:center; color:#0f4c81;'>🩺 Public Health Analytics Dashboard</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h3 style='text-align:center; color:#ff6b6b;'>📊 Diabetes predictor- Patient Health Analysis</h3>",
    unsafe_allow_html=True
)

# ---------------- DATASET ----------------
data = {
    "Age": [25, 35, 45, 50, 60, 30, 40, 55, 65, 28, 38, 48, 52, 33, 41],
    "BMI": [22, 28, 31, 29, 35, 24, 27, 33, 36, 23, 30, 32, 34, 26, 29],
    "BP": [80, 85, 90, 88, 95, 82, 86, 91, 96, 81, 89, 92, 94, 84, 87],
    "Sugar": [100, 120, 150, 140, 180, 110, 130, 160, 190, 105, 145, 155, 170, 115, 135],
    "Outcome": [0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0]
}

df = pd.DataFrame(data)

# ---------------- ADD MISSING VALUES ----------------
df.loc[2, "BMI"] = np.nan
df.loc[5, "BP"] = np.nan

# ---------------- KPI CARDS ----------------
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        f"""
        <div style='border:2px solid #3a86ff; padding:20px; border-radius:15px; background-color:#e3f2fd; text-align:center;'>
            <h3>👥 Total Patients</h3>
            <h2>{len(df)}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        f"""
        <div style='border:2px solid #2a9d8f; padding:20px; border-radius:15px; background-color:#e8f5e9; text-align:center;'>
            <h3>🍬 Average Sugar</h3>
            <h2>{round(df["Sugar"].mean(), 2)}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        f"""
        <div style='border:2px solid #e63946; padding:20px; border-radius:15px; background-color:#ffebee; text-align:center;'>
            <h3>⚠️ High Risk</h3>
            <h2>{int(df["Outcome"].sum())}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )
# ---------------- PREVIEW ----------------
st.markdown(
    "<div style='border:2px solid #6a4c93; padding:10px; border-radius:10px; background-color:#f3e8ff;'>"
    "<h3 style='color:#6a4c93;'>📋 Patient Records</h3></div>",
    unsafe_allow_html=True
)
st.dataframe(df)

# ---------------- CLEANING ----------------
st.markdown(
    "<div style='border:2px solid #2a9d8f; padding:10px; border-radius:10px; background-color:#e8f5e9;'>"
    "<h3 style='color:#2a9d8f;'>🧹 Missing Values Handling</h3></div>",
    unsafe_allow_html=True
)
st.write(df.isnull().sum())
df.fillna(df.median(numeric_only=True), inplace=True)
st.success("Missing values filled using median")

# ---------------- STATISTICS ----------------
st.markdown(
    "<div style='border:2px solid #1982c4; padding:10px; border-radius:10px; background-color:#e3f2fd;'>"
    "<h3 style='color:#1982c4;'>📊 Descriptive Statistics</h3></div>",
    unsafe_allow_html=True
)
st.write(df.describe())
# ---------------- MEAN VALUES ----------------
st.markdown(
    "<div style='border:2px solid #4cc9f0; padding:8px; border-radius:10px; background-color:#e0f7ff;'>"
    "<h4 style='color:#0077b6;'>📌 Mean Values</h4></div>",
    unsafe_allow_html=True
)
st.write(df.mean(numeric_only=True))

# ---------------- STANDARD DEVIATION ----------------
st.markdown(
    "<div style='border:2px solid #7209b7; padding:8px; border-radius:10px; background-color:#f3e8ff;'>"
    "<h4 style='color:#7209b7;'>📌 Standard Deviation</h4></div>",
    unsafe_allow_html=True
)
st.write(df.std(numeric_only=True))



# ---------------- NORMALIZATION ----------------
st.markdown(
    "<div style='border:2px solid #ff9f1c; padding:10px; border-radius:10px; background-color:#fff3e0;'>"
    "<h3 style='color:#ff9f1c;'>📏 Data Normalization</h3></div>",
    unsafe_allow_html=True
)
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
st.success("Normalization completed")




# ---------------- HEATMAP ----------------
st.markdown(
    "<div style='border:2px solid #ff006e; padding:10px; border-radius:10px; background-color:#ffe4ec;'>"
    "<h3 style='color:#ff006e;'>🔥 Correlation Heatmap</h3></div>",
    unsafe_allow_html=True
)
corr = df.corr()

fig, ax = plt.subplots(figsize=(8, 5))
img = ax.imshow(corr, cmap="Blues")   # blue theme
ax.set_xticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns)
ax.set_yticks(range(len(corr.columns)))
ax.set_yticklabels(corr.columns)
plt.colorbar(img)
st.pyplot(fig)

# ---------------- HISTOGRAM ----------------
st.markdown(
    "<div style='border:2px solid #008080; padding:10px; border-radius:10px; background-color:#e0f7fa;'>"
    "<h3 style='color:#008080;'>📈 Sugar Histogram</h3></div>",
    unsafe_allow_html=True
)
fig2, ax2 = plt.subplots()
ax2.hist(df["Sugar"], bins=6, color="teal")
ax2.set_xlabel("Sugar")
ax2.set_ylabel("Frequency")
st.pyplot(fig2)
# ---------------- BOXPLOT ----------------
st.markdown(
    "<div style='border:2px solid #8338ec; padding:10px; border-radius:10px; background-color:#f3e8ff;'>"
    "<h3 style='color:#8338ec;'>📦 BMI Box Plot</h3></div>",
    unsafe_allow_html=True
)
fig3, ax3 = plt.subplots()
ax3.boxplot(df["BMI"], patch_artist=True,
            boxprops=dict(facecolor="violet"))
st.pyplot(fig3)

# ---------------- SCATTER ----------------
st.markdown(
    "<div style='border:2px solid #3a86ff; padding:10px; border-radius:10px; background-color:#e3f2fd;'>"
    "<h3 style='color:#3a86ff;'>📉 Age vs Sugar</h3></div>",
    unsafe_allow_html=True
)
fig4, ax4 = plt.subplots()
ax4.scatter(df["Age"], df["Sugar"], color="green")
ax4.set_xlabel("Age")
ax4.set_ylabel("Sugar")
st.pyplot(fig4)

# ---------------- MODEL ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

st.markdown(
    "<div style='border:2px solid #e63946; padding:10px; border-radius:10px; background-color:#ffebee;'>"
    "<h3 style='color:#e63946;'>🤖 Model Accuracy</h3></div>",
    unsafe_allow_html=True
)
st.success(f"Accuracy: {model.score(X_test, y_test)*100:.2f}%")

# ---------------- USER PREDICTION ----------------
st.markdown(
    "<div style='border:2px solid #d00000; padding:10px; border-radius:10px; background-color:#ffe5e5;'>"
    "<h3 style='color:#d00000;'>🧪 Patient Risk Prediction</h3></div>",
    unsafe_allow_html=True
)

age = st.number_input("Enter Age", 1, 100, 30)
bmi = st.number_input("Enter BMI", 10, 50, 25)
bp = st.number_input("Enter BP", 50, 200, 80)
sugar = st.number_input("Enter Sugar", 50, 300, 100)

if st.button("🔍 Predict"):
    sample = scaler.transform([[age, bmi, bp, sugar]])
    pred = model.predict(sample)

    if pred[0] == 1:
        st.error("⚠️ High Diabetes Risk")
    else:
        st.success("✅ Low Diabetes Risk")