from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.metrics import classification_report, confusion_matrix
import shap

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load model dan encoder
model = joblib.load("model_fraud.pkl")
label_encoders = joblib.load("label_encoders.pkl")
expected_columns = model.get_booster().feature_names
dropdown_values = {col: le.classes_.tolist() for col, le in label_encoders.items()}

# SHAP explainer
explainer = shap.TreeExplainer(model)

# Global state
df_global = pd.DataFrame()
prediction_log = []

# Fungsi: Generate visualisasi distribusi
def generate_plot_image():
    if df_global.empty:
        return ""
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(df_global['amount'], kde=True, ax=axs[0], bins=30, color='blue')
    axs[0].set_title('Distribusi Amount')
    sns.countplot(x='isFraud', data=df_global, ax=axs[1], palette='Set2')
    axs[1].set_title('Jumlah Fraud vs Non-Fraud')
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format='png')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return encoded

# Fungsi: Evaluasi model (classification report & confusion matrix)
def get_classification_metrics():
    if df_global.empty:
        return None, None
    y_true = df_global['isFraud']
    X = df_global[expected_columns]
    y_proba = model.predict_proba(X)
    y_pred = (y_proba[:, 1] >= 0.3).astype(int) if y_proba.shape[1] > 1 else [0]*len(X)
    report = classification_report(y_true, y_pred, output_dict=True)
    matrix = confusion_matrix(y_true, y_pred)
    return report, matrix

# Fungsi: Generate SHAP bar chart
def generate_shap_bar(explanation):
    fig, ax = plt.subplots(figsize=(6, 3))
    features, values = zip(*explanation)
    sns.barplot(x=values, y=features, ax=ax, palette='coolwarm')
    ax.set_title("Kontribusi Fitur terhadap Prediksi")
    ax.set_xlabel("SHAP Value")
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format='png')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return encoded

# Fungsi: Generate teks penjelasan SHAP
def generate_shap_text(explanation):
    if not explanation:
        return "Tidak ada penjelasan fitur tersedia."
    lines = []
    for feature, value in explanation:
        arah = "mendorong ke prediksi FRAUD" if value > 0 else "mendorong ke NON-FRAUD"
        lines.append(f"Fitur <b>{feature}</b> memberikan kontribusi sebesar <b>{value:.4f}</b>, {arah}.")
    kesimpulan = "Fitur-fitur di atas adalah yang paling berpengaruh terhadap hasil prediksi model."
    return "<br>".join(lines) + "<br><br>" + kesimpulan

# GET: Halaman utama dashboard
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    if df_global.empty:
        return templates.TemplateResponse("index.html", {"request": request, "no_data": True})

    fraud_count = df_global['isFraud'].value_counts().to_dict()
    image = generate_plot_image()
    report, matrix = get_classification_metrics()

    return templates.TemplateResponse("index.html", {
        "request": request,
        "no_data": False,
        "total": len(df_global),
        "fraud": fraud_count.get(1, 0),
        "nonfraud": fraud_count.get(0, 0),
        "preview": df_global.head(10).to_html(classes="table table-bordered", index=False),
        "plot_image": image,
        "report": report,
        "matrix": matrix
    })

# POST: Upload file CSV
@app.post("/upload", response_class=HTMLResponse)
async def upload_csv(request: Request, file: UploadFile = File(...)):
    global df_global
    try:
        df = pd.read_csv(file.file)
        df = df[df['isFraud'].isin([0, 1])]
        df['amount_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)
        df['diff_balance'] = df['oldbalanceOrg'] - df['newbalanceOrig']

        for col, encoder in label_encoders.items():
            if col in df.columns:
                df[col] = encoder.transform(df[col].astype(str))

        df = df[expected_columns + ['isFraud']]
        df_global = df.copy()
        return await dashboard(request)
    except Exception as e:
        return HTMLResponse(f"<h3>Upload gagal: {e}</h3>")

# GET: Halaman form prediksi manual
@app.get("/predict-page", response_class=HTMLResponse)
async def predict_page(request: Request):
    table_html = ""
    if prediction_log:
        df_log = pd.DataFrame(prediction_log)
        table_html = df_log.tail(10).to_html(classes="table table-striped", index=False)
    return templates.TemplateResponse("predict_page.html", {
        "request": request,
        "dropdowns": dropdown_values,
        "manual_history": table_html
    })

# POST: Proses prediksi manual
@app.post("/predict", response_class=HTMLResponse)
async def predict_manual(
    request: Request,
    amount: float = Form(...),
    oldbalanceOrg: float = Form(...),
    newbalanceOrig: float = Form(...),
    type: str = Form(...),
    card_type: str = Form(...),
    exp_type: str = Form(...),
    gender: str = Form(...),
    city: str = Form(...)
):
    try:
        def encode(col, value):
            return label_encoders[col].transform([value])[0]

        row = {
            "amount": amount,
            "oldbalanceOrg": oldbalanceOrg,
            "newbalanceOrig": newbalanceOrig,
            "amount_ratio": amount / (oldbalanceOrg + 1),
            "diff_balance": oldbalanceOrg - newbalanceOrig,
            "type": encode("type", type),
            "Card Type": encode("Card Type", card_type),
            "Exp Type": encode("Exp Type", exp_type),
            "Gender": encode("Gender", gender),
            "City": encode("City", city)
        }

        input_df = pd.DataFrame([row])
        input_df = input_df[expected_columns]

        proba = model.predict_proba(input_df)[0]
        prob = proba[1] if len(proba) > 1 else 0.0
        is_fraud = int(prob >= 0.3)

        shap_values = explainer.shap_values(input_df)
        explanation = sorted(
            zip(expected_columns, shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]
        shap_img = generate_shap_bar(explanation)
        shap_text = generate_shap_text(explanation)

        prediction_log.append({**row, "prob": round(prob, 4), "label": is_fraud})
        df_log = pd.DataFrame(prediction_log)
        table_html = df_log.tail(10).to_html(classes="table table-striped", index=False)

        return templates.TemplateResponse("predict_page.html", {
            "request": request,
            "dropdowns": dropdown_values,
            "prediction": {
                "prob": f"{prob:.4f}",
                "label": is_fraud,
                "shap_img": shap_img,
                "shap_text": shap_text
            },
            "manual_history": table_html
        })
    except Exception as e:
        return HTMLResponse(f"<h3>Error saat prediksi: {e}</h3>")

# GET: Unduh riwayat prediksi manual
@app.get("/download_log")
async def download_log():
    df_log = pd.DataFrame(prediction_log)
    buf = io.StringIO()
    df_log.to_csv(buf, index=False)
    return HTMLResponse(content=buf.getvalue(), media_type='text/csv')
