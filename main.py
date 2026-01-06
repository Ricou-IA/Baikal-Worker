"""
==============================================
BAIKAL WORKER - API FastAPI
Analyse BIM (Speckle) + Données (Excel/CSV)
==============================================
"""

import os
import json
import logging
from typing import Optional, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import anthropic
from supabase import create_client, Client

# ==============================================
# CONFIGURATION & LOGGING
# ==============================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("baikal-worker")

# Variables d'environnement
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SPECKLE_TOKEN = os.getenv("SPECKLE_TOKEN")
SPECKLE_SERVER_URL = os.getenv("SPECKLE_SERVER_URL", "https://app.speckle.systems")

# Modèle Claude par défaut
CLAUDE_MODEL = "claude-sonnet-4-5-20250929"

# ==============================================
# INITIALISATION
# ==============================================

app = FastAPI(
    title="Baikal Worker API",
    description="Worker Python pour l'analyse BIM et données financières",
    version="1.0.0"
)

# CORS - Autoriser les appels depuis le frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Client Supabase (initialisé si les credentials sont présents)
supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Connexion Supabase etablie")
    except Exception as e:
        logger.error(f"Erreur connexion Supabase: {e}")

# Client Anthropic
claude_client: Optional[anthropic.Anthropic] = None
if ANTHROPIC_API_KEY:
    try:
        claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        logger.info("Client Anthropic initialise")
    except Exception as e:
        logger.error(f"Erreur initialisation Anthropic: {e}")


# ==============================================
# MODÈLES PYDANTIC (Requêtes/Réponses)
# ==============================================

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    services: dict


class BIMAnalysisRequest(BaseModel):
    project_id: str = Field(..., description="ID du projet Supabase")
    bim_object: dict = Field(..., description="Objet BIM JSON (depuis Speckle)")
    context: Optional[str] = Field(None, description="Contexte RAG optionnel (CCTP)")


class BIMAnalysisResponse(BaseModel):
    success: bool
    bim_object_id: str
    analysis: dict
    task_id: Optional[str] = None


class DataAnalysisRequest(BaseModel):
    project_id: str = Field(..., description="ID du projet Supabase")
    question: str = Field(..., description="Question utilisateur sur les données")
    file_url: Optional[str] = Field(None, description="URL du fichier Excel/CSV")


class DataAnalysisResponse(BaseModel):
    success: bool
    question: str
    answer: str
    code_executed: Optional[str] = None
    data_preview: Optional[dict] = None


# ==============================================
# ENDPOINTS
# ==============================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Endpoint de healthcheck - Verifie l'etat des services
    """
    services = {
        "supabase": "connected" if supabase else "not_configured",
        "anthropic": "connected" if claude_client else "not_configured",
        "speckle": "configured" if SPECKLE_TOKEN else "not_configured",
        "gemini_fallback": "configured" if GOOGLE_API_KEY else "not_configured"
    }
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        services=services
    )


@app.post("/analyze-bim", response_model=BIMAnalysisResponse)
async def analyze_bim(request: BIMAnalysisRequest):
    """
    Analyse un objet BIM et determine :
    - Le lot concerne (Gros Oeuvre, Peinture, etc.)
    - L'action a realiser
    - Les attributs pertinents
    
    Enregistre le resultat dans project_tasks (Supabase)
    """
    
    if not claude_client:
        raise HTTPException(
            status_code=503,
            detail="Service Anthropic non configure (ANTHROPIC_API_KEY manquant)"
        )
    
    try:
        # Extraction des infos de l'objet BIM
        bim_object = request.bim_object
        bim_object_id = bim_object.get("guid") or bim_object.get("id", "unknown")
        bim_object_type = bim_object.get("ifc_type") or bim_object.get("type", "Unknown")
        
        # Construction du prompt d'analyse
        prompt = f"""Tu es un expert BTP. Analyse cet objet BIM et determine :
1. Le LOT concerne (ex: Gros Oeuvre, Peinture, Platrerie, Menuiserie, etc.)
2. L'ACTION principale a realiser sur cet objet
3. Les ATTRIBUTS pertinents pour le suivi de chantier

OBJET BIM (JSON):
```json
{json.dumps(bim_object, indent=2, ensure_ascii=False)}
```

{f"CONTEXTE CCTP: {request.context}" if request.context else ""}

Reponds UNIQUEMENT avec un JSON valide au format:
{{
    "lot": "Nom du lot",
    "action": "Action principale",
    "priority": "haute|moyenne|basse",
    "materials": ["materiau1", "materiau2"],
    "notes": "Remarques eventuelles"
}}
"""
        
        # Appel Claude
        message = claude_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Parse de la reponse
        response_text = message.content[0].text
        
        # Extraction du JSON (gere les cas ou Claude ajoute du texte autour)
        try:
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                analysis = json.loads(response_text[json_start:json_end])
            else:
                analysis = {"raw_response": response_text, "parse_error": True}
        except json.JSONDecodeError:
            analysis = {"raw_response": response_text, "parse_error": True}
        
        # Enregistrement dans Supabase (si connecte)
        task_id = None
        if supabase:
            try:
                insert_data = {
                    "project_id": request.project_id,
                    "bim_object_id": bim_object_id,
                    "bim_object_type": bim_object_type,
                    "status": "pending",
                    "raw_properties": bim_object,
                    "ai_analysis": analysis
                }
                
                result = supabase.table("project_tasks").insert(insert_data).execute()
                
                if result.data:
                    task_id = result.data[0].get("id")
                    logger.info(f"Tache creee: {task_id}")
                    
            except Exception as e:
                logger.error(f"Erreur insertion Supabase: {e}")
        
        return BIMAnalysisResponse(
            success=True,
            bim_object_id=bim_object_id,
            analysis=analysis,
            task_id=task_id
        )
        
    except anthropic.APIError as e:
        logger.error(f"Erreur API Anthropic: {e}")
        raise HTTPException(status_code=502, detail=f"Erreur API Claude: {str(e)}")
    except Exception as e:
        logger.error(f"Erreur analyse BIM: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze-data", response_model=DataAnalysisResponse)
async def analyze_data(request: DataAnalysisRequest):
    """
    Analyse des donnees Excel/CSV avec generation de code Pandas.
    Le LLM genere le code, Python l'execute = pas d'hallucination sur les calculs.
    """
    
    if not claude_client:
        raise HTTPException(
            status_code=503,
            detail="Service Anthropic non configure (ANTHROPIC_API_KEY manquant)"
        )
    
    try:
        # DataFrame de test (a remplacer par le vrai chargement)
        df_test = pd.DataFrame({
            "Lot": ["Gros Oeuvre", "Peinture", "Platrerie", "Menuiserie", "Electricite"],
            "Montant_HT": [450000, 85000, 120000, 95000, 78000],
            "Avancement": [0.75, 0.30, 0.50, 0.45, 0.60]
        })
        
        # Construction du prompt pour generer le code Pandas
        prompt = f"""Tu es un expert en analyse de donnees Python/Pandas.
L'utilisateur pose une question sur un DataFrame.

STRUCTURE DU DATAFRAME (df):
Colonnes: {list(df_test.columns)}
Types: {df_test.dtypes.to_dict()}
Apercu (5 premieres lignes):
{df_test.head().to_string()}

QUESTION UTILISATEUR: {request.question}

Genere UNIQUEMENT le code Python qui repond a cette question.
Le DataFrame s'appelle 'df'.
Le resultat doit etre stocke dans une variable 'result'.
Ne mets PAS de markdown, juste le code brut.

Exemple de format attendu:
result = df['Montant_HT'].sum()
"""
        
        # Appel Claude pour generer le code
        message = claude_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=512,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        generated_code = message.content[0].text.strip()
        
        # Nettoyage du code (supprime les backticks si presents)
        if generated_code.startswith("```"):
            lines = generated_code.split("\n")
            generated_code = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
        
        # Execution securisee du code
        local_vars = {"df": df_test, "pd": pd}
        try:
            exec(generated_code, {"__builtins__": {}}, local_vars)
            result = local_vars.get("result", "Aucun resultat")
            
            # Conversion du resultat en string lisible
            if isinstance(result, pd.DataFrame):
                answer = result.to_string()
            elif isinstance(result, pd.Series):
                answer = result.to_string()
            else:
                answer = str(result)
                
        except Exception as exec_error:
            logger.error(f"Erreur execution code: {exec_error}")
            answer = f"Erreur d'execution: {exec_error}"
        
        return DataAnalysisResponse(
            success=True,
            question=request.question,
            answer=answer,
            code_executed=generated_code,
            data_preview={
                "columns": list(df_test.columns),
                "row_count": len(df_test)
            }
        )
        
    except anthropic.APIError as e:
        logger.error(f"Erreur API Anthropic: {e}")
        raise HTTPException(status_code=502, detail=f"Erreur API Claude: {str(e)}")
    except Exception as e:
        logger.error(f"Erreur analyse donnees: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze-data/upload")
async def analyze_data_with_upload(
    question: str,
    project_id: str,
    file: UploadFile = File(...)
):
    """
    Analyse un fichier Excel/CSV uploade directement.
    """
    
    if not claude_client:
        raise HTTPException(
            status_code=503,
            detail="Service Anthropic non configure"
        )
    
    try:
        # Lecture du fichier
        filename = file.filename.lower()
        contents = await file.read()
        
        if filename.endswith(".csv"):
            df = pd.read_csv(pd.io.common.BytesIO(contents))
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(pd.io.common.BytesIO(contents))
        else:
            raise HTTPException(
                status_code=400,
                detail="Format non supporte. Utilisez CSV, XLS ou XLSX."
            )
        
        # Limite de taille pour eviter les timeouts
        if len(df) > 10000:
            logger.warning(f"DataFrame tronque: {len(df)} -> 10000 lignes")
            df = df.head(10000)
        
        # Construction du prompt
        prompt = f"""Tu es un expert en analyse de donnees Python/Pandas.

STRUCTURE DU DATAFRAME (df):
Colonnes: {list(df.columns)}
Types: {df.dtypes.to_dict()}
Nombre de lignes: {len(df)}
Apercu:
{df.head(10).to_string()}

QUESTION: {question}

Genere UNIQUEMENT le code Python Pandas pour repondre.
Le DataFrame s'appelle 'df'. Stocke le resultat dans 'result'.
Pas de markdown, juste le code.
"""
        
        # Appel Claude
        message = claude_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        )
        
        generated_code = message.content[0].text.strip()
        
        # Nettoyage
        if generated_code.startswith("```"):
            lines = generated_code.split("\n")
            generated_code = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
        
        # Execution
        local_vars = {"df": df, "pd": pd}
        try:
            exec(generated_code, {"__builtins__": {}}, local_vars)
            result = local_vars.get("result", "Aucun resultat")
            
            if isinstance(result, (pd.DataFrame, pd.Series)):
                answer = result.to_string()
            else:
                answer = str(result)
                
        except Exception as exec_error:
            answer = f"Erreur d'execution: {exec_error}"
        
        return DataAnalysisResponse(
            success=True,
            question=question,
            answer=answer,
            code_executed=generated_code,
            data_preview={
                "columns": list(df.columns),
                "row_count": len(df)
            }
        )
        
    except Exception as e:
        logger.error(f"Erreur: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================
# POINT D'ENTREE
# ==============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
# Version 1.1 - Auto deploy test
