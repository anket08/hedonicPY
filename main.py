from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import json
import os
from pydantic import BaseModel
import logging

app = FastAPI(title="Bike Arbitrage Engine")

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files from 'static' folder
app.mount("/static", StaticFiles(directory="static"), name="static")

csv_file = "Arbitrage_Opportunities_Output.csv"
rejected_file = "rejected_trades.json"

class RejectRequest(BaseModel):
    trade_id: str

def load_rejected():
    if os.path.exists(rejected_file):
        with open(rejected_file, "r") as f:
            return set(json.load(f))
    return set()

def save_rejected(rejected_set):
    with open(rejected_file, "w") as f:
        json.dump(list(rejected_set), f)

@app.get("/")
async def root():
    return FileResponse('static/index.html')

@app.get("/api/opportunities")
async def get_opportunities():
    if not os.path.exists(csv_file):
        return {"error": "CSV data not found. Run the ML pipeline first."}
        
    try:
        df = pd.read_csv(csv_file)
        
        # Create a unique 'trade_id' string for each row to easily track rejects
        df['trade_id'] = df['bike_name'] + "|" + df['buy_city'] + "->" + df['sell_city'] + "|" + df['age_bucket'].astype(str)
        
        rejected = load_rejected()
        
        # Filter out rejected
        if rejected:
            df = df[~df['trade_id'].isin(rejected)]
            
        # Format some numerical columns nicely before sending
        df['net_profit'] = df['net_profit'].round(0).astype(int)
        df['buy_price'] = df['buy_price'].round(0).astype(int)
        df['sell_price'] = df['sell_price'].round(0).astype(int)
        df['risk_adjusted_profit'] = df['risk_adjusted_profit'].round(0).astype(int)
        df['distance_km'] = df['distance_km'].round(0).astype(int)
        
        # Top 100 after filtering
        data = df.head(100).to_dict(orient='records')
        return {"data": data}
        
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/reject")
async def reject_opportunity(req: RejectRequest):
    try:
        rejected = load_rejected()
        rejected.add(req.trade_id)
        save_rejected(rejected)
        return {"status": "success", "message": f"{req.trade_id} has been hidden."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

from bike_arbitrage_engine import BikeArbitrageEngine

class BikeEntry(BaseModel):
    bike_name: str
    age: int
    kms_driven: int
    power: int
    buy_price: int
    buy_city: str
    owner: str = "First Owner"
    first_owner: int = 1

@app.post("/api/evaluate-new")
async def evaluate_new_bike(entry: BikeEntry):
    try:
        bike_dict = entry.dict()
        bike_dict['first_owner'] = 1 if bike_dict['owner'] == "First Owner" else 0
        result = BikeArbitrageEngine.evaluate_new_entry(bike_dict)
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
