from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import pandas as pd
import numpy as np
import joblib
import io
from typing import List, Optional
from pydantic import BaseModel

model = joblib.load('random_forest_model.pkl')
features = [
    'aluminium', 'ammonia', 'arsenic', 'barium', 'cadmium', 'chloramine', 
    'chromium', 'copper', 'flouride', 'bacteria', 'viruses', 'lead', 
    'nitrates', 'nitrites', 'mercury', 'perchlorate', 'radium', 
    'selenium', 'silver', 'uranium'
]

app = FastAPI(title="Water Quality Safety Analyzer")

templates = Jinja2Templates(directory="templates")

class WaterSample(BaseModel):
    aluminium: float
    ammonia: float
    arsenic: float
    barium: float
    cadmium: float
    chloramine: float
    chromium: float
    copper: float
    flouride: float
    bacteria: float
    viruses: float
    lead: float
    nitrates: float
    nitrites: float
    mercury: float
    perchlorate: float
    radium: float
    selenium: float
    silver: float
    uranium: float

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict-single")
async def predict_single(sample: WaterSample):
    input_data = pd.DataFrame({
        feature: [getattr(sample, feature)] for feature in features
    })
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]  # Probability of being safe
    
    if hasattr(model, 'feature_importances_'):
        importances = dict(zip(features, model.feature_importances_))
        sorted_importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5])
    else:
        sorted_importances = {}
        
    if prediction == 1:  # Safe
        safety_status = "SAFE"
        safety_class = "text-success"
        recommendation = "This water sample appears safe based on the analyzed parameters."
    else: 
        safety_status = "NOT SAFE"
        safety_class = "text-danger"
        recommendation = "This water sample does not meet safety standards and requires treatment."
    
    return {
        "prediction": int(prediction),
        "safety_status": safety_status,
        "safety_class": safety_class,
        "probability": round(float(probability) * 100, 2),
        "recommendation": recommendation,
        "top_features": sorted_importances
    }

@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):
    content = await file.read()
    
    try:
        if file.filename.endswith('.xlsx') or file.filename.endswith('.xls'):
            df = pd.read_excel(io.BytesIO(content))
        elif file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content))
        else:
            return JSONResponse(
                status_code=400,
                content={"message": "Unsupported file format. Please upload an Excel or CSV file."}
            )
        
        missing_columns = [col for col in features if col not in df.columns]
        if missing_columns:
            return JSONResponse(
                status_code=400,
                content={
                    "message": f"Missing columns in the uploaded file: {', '.join(missing_columns)}",
                    "required_columns": features
                }
            )
        
        input_data = df[features]
        
        predictions = model.predict(input_data)
        probabilities = model.predict_proba(input_data)[:, 1]  # Probability of being safe
        
        df['prediction'] = predictions
        df['probability'] = probabilities
        df['safety_status'] = df['prediction'].apply(lambda x: "SAFE" if x == 1 else "NOT SAFE")
        
        results = []
        for i, row in df.iterrows():
            result = {
                "id": i + 1,
                "prediction": int(row['prediction']),
                "safety_status": row['safety_status'],
                "probability": round(float(row['probability']) * 100, 2)
            }
            for feature in features:
                result[feature] = float(row[feature]) if not pd.isna(row[feature]) else 0.0
                
            results.append(result)
        
        summary = {
            "total_samples": len(df),
            "safe_count": int(sum(predictions)),
            "unsafe_count": int(len(predictions) - sum(predictions)),
            "safe_percentage": round(float(sum(predictions) / len(predictions) * 100), 2)
        }
        
        return {
            "results": results,
            "summary": summary
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error processing file: {str(e)}"}
        )

@app.get("/api/sample-template")
async def get_sample_template():
    """Generate a sample Excel template for users to fill in"""
    df = pd.DataFrame(columns=features)
    df.loc[0] = [0.01, 0.5, 0.001, 0.05, 0.001, 1.0, 0.05, 
                0.5, 1.0, 0.0, 0.0, 0.005, 5.0, 0.1, 
                0.001, 0.001, 1.0, 0.01, 0.01, 0.001]
    output = io.BytesIO()
    df.to_excel(output, index=False)
    output.seek(0)
    
    return Response(
        content=output.getvalue(),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=water_quality_template.xlsx"}
    )

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

