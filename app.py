from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import pandas as pd
import joblib
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Loan Default Prediction API")

# عشان Flutter يقدر يبعت طلبات
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# تحميل الـ preprocessor والنموذج
preprocessor = joblib.load("preprocessor.pkl")
input_size = 15   # ←←←←← هتغيّر الرقم ده بس بعد شوية

class NeuralNet(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

model = NeuralNet(input_size)
model.load_state_dict(torch.load("loan_model.pth", map_location=torch.device('cpu')))
model.eval()

# شكل الـ JSON اللي هيجي من التطبيق
class LoanInput(BaseModel):
    Age: int
    Marital_Status: str
    Education_Level: str
    Num_Dependents: int
    Annual_Income: float
    Employment_Type: str
    Employment_Length: int
    Previous_Defaults: int

@app.get("/")
def home():
    return {"message": "السيرفر شغال يا بطل!"}

@app.post("/predict")
def predict(loan: LoanInput):
    try:
        # تحويل البيانات لـ DataFrame
        input_df = pd.DataFrame([{
            'Age': loan.Age,
            'Marital_Status': loan.Marital_Status,
            'Education_Level': loan.Education_Level,
            'Num_Dependents': loan.Num_Dependents,
            'Annual_Income': loan.Annual_Income,
            'Employment_Type': loan.Employment_Type,
            'Employment_Length': loan.Employment_Length,
            'Previous_Defaults': loan.Previous_Defaults
        }])

        # معالجة البيانات بنفس طريقة التدريب
        X_processed = preprocessor.transform(input_df)
        X_tensor = torch.tensor(X_processed, dtype=torch.float32)

        # التنبؤ
        with torch.no_grad():
            prob = model(X_tensor).item()

        prediction = 1 if prob > 0.5 else 0
        risk = "High Risk" if prediction == 1 else "Low Risk"
        prediction_text = "Will Default" if prediction == 1 else "Will Not Default"

        # عوامل مؤثرة بسيطة
        factors = []
        if loan.Annual_Income < 50000: factors.append("Low Annual Income")
        if loan.Previous_Defaults > 0: factors.append("Has Previous Defaults")
        if loan.Num_Dependents > 3: factors.append("Many Dependents")
        if loan.Employment_Length < 2: factors.append("Short Employment Length")

        return {
            "prediction": prediction_text,
            "probability": round(prob, 3),
            "risk_level": risk,
            "top_factors": factors[:3]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))