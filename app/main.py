from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, Body
from fastapi.security import APIKeyHeader
import os
from dotenv import load_dotenv

# Import des services
from app.services.bim_service import BimService
from app.services.excel_service import ExcelService
from app.services.db_service import DbService  # <--- AJOUT IMPORTANT

load_dotenv()

app = FastAPI(
    title=os.environ.get("PROJECT_NAME", "Baikal Worker"),
    version=os.environ.get("VERSION", "1.0")
)

# Sécurité
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

async def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key != os.environ.get("API_KEY"):
        raise HTTPException(status_code=403, detail="Clé API invalide")
    return api_key

@app.get("/")
def read_root():
    return {"status": "online", "service": "Baikal Worker"}

@app.post("/analyze-data", dependencies=[Depends(get_api_key)])
async def analyze_data(file: UploadFile = File(...)):
    excel_service = ExcelService()
    result = await excel_service.process_file(file)
    return result

# --- C'EST ICI QUE CA CHANGE ---
@app.post("/analyze-bim", dependencies=[Depends(get_api_key)])
async def analyze_bim(
    stream_id: str = Body(..., embed=True), 
    object_id: str = Body(..., embed=True)
):
    # 1. Analyse (Simulation)
    bim_service = BimService(token=None)
    bim_result = await bim_service.get_metadata(stream_id, object_id)
    
    # 2. Sauvegarde en Base de Données (Supabase)
    db_service = DbService()
    
    # On prépare l'objet à sauvegarder
    task_record = {
        "bim_object_id": object_id,
        "bim_object_type": "Wall", # Simulé
        "status": "completed",
        "raw_properties": bim_result.get("fake_data", {}),
        "ai_analysis": {"comment": "Analyse automatique V1"}
    }
    
    # On sauvegarde et on récupère le résultat
    save_result = db_service.save_task(task_record)
    
    # 3. Retourne le tout (Analyse + Confirmation de sauvegarde)
    return {
        "analysis": bim_result,
        "storage": save_result
    }
    