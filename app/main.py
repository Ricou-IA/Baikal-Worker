"""
==============================================
BAIKAL WORKER - API FastAPI
Analyse BIM (Speckle) + Données (Excel/CSV)
Version 1.2 - Speckle GraphQL Client intégré
==============================================
"""

import os
import json
import logging
from typing import Optional, List, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import anthropic
import httpx
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
    version="1.2.0"
)

# CORS - Autoriser les appels depuis le frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Client Supabase
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
# CLIENT SPECKLE (API GraphQL)
# ==============================================

class SpeckleClient:
    """Client Speckle maison utilisant l'API GraphQL - sans dépendance externe"""
    
    def __init__(self, server_url: str, token: str):
        self.server_url = server_url.rstrip("/")
        self.graphql_url = f"{self.server_url}/graphql"
        self.token = token
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
    
    async def _query(self, query: str, variables: dict = None) -> dict:
        """Execute une requête GraphQL"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.graphql_url,
                json={"query": query, "variables": variables or {}},
                headers=self.headers,
                timeout=30.0
            )
            response.raise_for_status()
            result = response.json()
            
            if "errors" in result:
                raise Exception(f"GraphQL Error: {result['errors']}")
            
            return result.get("data", {})
    
    async def get_user(self) -> dict:
        """Récupère les infos de l'utilisateur connecté"""
        query = """
        query {
            activeUser {
                id
                name
                email
            }
        }
        """
        data = await self._query(query)
        return data.get("activeUser", {})
    
    async def get_streams(self, limit: int = 25) -> List[dict]:
        """Liste les streams (projets) de l'utilisateur"""
        query = """
        query($limit: Int!) {
            activeUser {
                streams(limit: $limit) {
                    items {
                        id
                        name
                        description
                        createdAt
                        updatedAt
                    }
                }
            }
        }
        """
        data = await self._query(query, {"limit": limit})
        return data.get("activeUser", {}).get("streams", {}).get("items", [])
    
    async def get_stream(self, stream_id: str) -> dict:
        """Récupère les détails d'un stream"""
        query = """
        query($id: String!) {
            stream(id: $id) {
                id
                name
                description
                createdAt
                updatedAt
                branches {
                    items {
                        id
                        name
                        commits(limit: 5) {
                            items {
                                id
                                message
                                createdAt
                                referencedObject
                            }
                        }
                    }
                }
            }
        }
        """
        data = await self._query(query, {"id": stream_id})
        return data.get("stream", {})
    
    async def get_commit(self, stream_id: str, commit_id: str) -> dict:
        """Récupère un commit spécifique"""
        query = """
        query($stream_id: String!, $commit_id: String!) {
            stream(id: $stream_id) {
                commit(id: $commit_id) {
                    id
                    message
                    createdAt
                    referencedObject
                }
            }
        }
        """
        data = await self._query(query, {"stream_id": stream_id, "commit_id": commit_id})
        return data.get("stream", {}).get("commit", {})
    
    async def get_object(self, stream_id: str, object_id: str) -> dict:
        """Récupère un objet Speckle (données BIM)"""
        url = f"{self.server_url}/objects/{stream_id}/{object_id}"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                headers=self.headers,
                timeout=60.0
            )
            response.raise_for_status()
            return response.json()
    
    async def get_object_children(self, stream_id: str, object_id: str, limit: int = 100, depth: int = 2) -> List[dict]:
        """Récupère les enfants d'un objet (éléments BIM)"""
        query = """
        query($stream_id: String!, $object_id: String!, $limit: Int!, $depth: Int!) {
            stream(id: $stream_id) {
                object(id: $object_id) {
                    id
                    speckleType
                    totalChildrenCount
                    children(limit: $limit, depth: $depth) {
                        objects {
                            id
                            speckleType
                            data
                        }
                    }
                }
            }
        }
        """
        data = await self._query(query, {
            "stream_id": stream_id,
            "object_id": object_id,
            "limit": limit,
            "depth": depth
        })
        
        obj = data.get("stream", {}).get("object", {})
        children = obj.get("children", {}).get("objects", [])
        return children


# Instance du client Speckle (si configuré)
speckle_client: Optional[SpeckleClient] = None
if SPECKLE_TOKEN:
    speckle_client = SpeckleClient(SPECKLE_SERVER_URL, SPECKLE_TOKEN)
    logger.info("Client Speckle initialise")


# ==============================================
# MODÈLES PYDANTIC
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


class SpeckleStreamResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    createdAt: Optional[str] = None
    updatedAt: Optional[str] = None


# ==============================================
# ENDPOINTS - HEALTH
# ==============================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Endpoint de healthcheck"""
    services = {
        "supabase": "connected" if supabase else "not_configured",
        "anthropic": "connected" if claude_client else "not_configured",
        "speckle": "connected" if speckle_client else "not_configured",
        "gemini_fallback": "configured" if GOOGLE_API_KEY else "not_configured"
    }
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        services=services
    )


# ==============================================
# ENDPOINTS - SPECKLE
# ==============================================

@app.get("/speckle/user")
async def get_speckle_user():
    """Récupère les infos de l'utilisateur Speckle connecté"""
    if not speckle_client:
        raise HTTPException(status_code=503, detail="Speckle non configuré (SPECKLE_TOKEN manquant)")
    
    try:
        user = await speckle_client.get_user()
        return {"success": True, "user": user}
    except Exception as e:
        logger.error(f"Erreur Speckle: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/speckle/streams", response_model=List[SpeckleStreamResponse])
async def list_speckle_streams(limit: int = Query(25, ge=1, le=100)):
    """Liste les streams (projets) Speckle de l'utilisateur"""
    if not speckle_client:
        raise HTTPException(status_code=503, detail="Speckle non configuré (SPECKLE_TOKEN manquant)")
    
    try:
        streams = await speckle_client.get_streams(limit)
        return streams
    except Exception as e:
        logger.error(f"Erreur Speckle: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/speckle/streams/{stream_id}")
async def get_speckle_stream(stream_id: str):
    """Récupère les détails d'un stream avec ses branches et commits"""
    if not speckle_client:
        raise HTTPException(status_code=503, detail="Speckle non configuré")
    
    try:
        stream = await speckle_client.get_stream(stream_id)
        return {"success": True, "stream": stream}
    except Exception as e:
        logger.error(f"Erreur Speckle: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/speckle/streams/{stream_id}/objects/{object_id}")
async def get_speckle_object(stream_id: str, object_id: str):
    """Récupère un objet BIM spécifique"""
    if not speckle_client:
        raise HTTPException(status_code=503, detail="Speckle non configuré")
    
    try:
        obj = await speckle_client.get_object(stream_id, object_id)
        return {"success": True, "object": obj}
    except Exception as e:
        logger.error(f"Erreur Speckle: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/speckle/streams/{stream_id}/objects/{object_id}/children")
async def get_speckle_object_children(
    stream_id: str,
    object_id: str,
    limit: int = Query(100, ge=1, le=1000),
    depth: int = Query(2, ge=1, le=5)
):
    """Récupère les éléments enfants d'un objet (éléments BIM du modèle)"""
    if not speckle_client:
        raise HTTPException(status_code=503, detail="Speckle non configuré")
    
    try:
        children = await speckle_client.get_object_children(stream_id, object_id, limit, depth)
        return {
            "success": True,
            "count": len(children),
            "objects": children
        }
    except Exception as e:
        logger.error(f"Erreur Speckle: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/speckle/analyze-stream/{stream_id}")
async def analyze_speckle_stream(
    stream_id: str,
    project_id: str = Query(..., description="ID du projet Supabase"),
    branch_name: str = Query("main", description="Nom de la branche"),
    limit: int = Query(50, ge=1, le=500, description="Nombre max d'objets à analyser")
):
    """
    Analyse un stream Speckle complet :
    1. Récupère le dernier commit de la branche
    2. Extrait les objets BIM
    3. Analyse chaque objet avec Claude
    4. Enregistre les tâches dans Supabase
    """
    if not speckle_client:
        raise HTTPException(status_code=503, detail="Speckle non configuré")
    if not claude_client:
        raise HTTPException(status_code=503, detail="Anthropic non configuré")
    
    try:
        # 1. Récupérer le stream et son dernier commit
        stream = await speckle_client.get_stream(stream_id)
        
        # Trouver la branche
        branches = stream.get("branches", {}).get("items", [])
        target_branch = next((b for b in branches if b["name"] == branch_name), None)
        
        if not target_branch:
            raise HTTPException(status_code=404, detail=f"Branche '{branch_name}' non trouvée")
        
        commits = target_branch.get("commits", {}).get("items", [])
        if not commits:
            raise HTTPException(status_code=404, detail="Aucun commit trouvé")
        
        latest_commit = commits[0]
        root_object_id = latest_commit.get("referencedObject")
        
        # 2. Récupérer les objets enfants
        children = await speckle_client.get_object_children(stream_id, root_object_id, limit, depth=2)
        
        # 3. Analyser chaque objet et créer les tâches
        tasks_created = []
        errors = []
        
        for child in children:
            try:
                # Analyse avec Claude
                analysis = await _analyze_single_bim_object(child)
                
                # Enregistrer dans Supabase
                if supabase:
                    insert_data = {
                        "project_id": project_id,
                        "bim_object_id": child.get("id", "unknown"),
                        "bim_object_type": child.get("speckleType", "Unknown"),
                        "status": "pending",
                        "raw_properties": child.get("data", {}),
                        "ai_analysis": analysis
                    }
                    
                    result = supabase.table("project_tasks").insert(insert_data).execute()
                    if result.data:
                        tasks_created.append({
                            "id": result.data[0].get("id"),
                            "bim_object_id": child.get("id"),
                            "lot": analysis.get("lot", "Non déterminé")
                        })
                        
            except Exception as e:
                logger.error(f"Erreur analyse objet {child.get('id')}: {e}")
                errors.append({"object_id": child.get("id"), "error": str(e)})
                continue
        
        return {
            "success": True,
            "stream_id": stream_id,
            "stream_name": stream.get("name"),
            "commit_id": latest_commit.get("id"),
            "objects_found": len(children),
            "tasks_created": len(tasks_created),
            "tasks": tasks_created,
            "errors": errors if errors else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur analyse stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _analyze_single_bim_object(bim_object: dict) -> dict:
    """Analyse un objet BIM avec Claude"""
    
    # Préparer les données (limiter la taille)
    obj_data = bim_object.get("data", {})
    if isinstance(obj_data, dict):
        obj_data_str = json.dumps(obj_data, indent=2, ensure_ascii=False)[:3000]
    else:
        obj_data_str = str(obj_data)[:3000]
    
    prompt = f"""Tu es un expert BTP/Construction. Analyse cet objet BIM extrait d'une maquette numérique.

OBJET BIM:
- Type Speckle: {bim_object.get('speckleType', 'Unknown')}
- ID: {bim_object.get('id', 'Unknown')}
- Données: {obj_data_str}

Détermine:
1. Le LOT concerné (Gros Oeuvre, Charpente, Couverture, Menuiseries Ext., Menuiseries Int., Plâtrerie, Peinture, Électricité, Plomberie, CVC, etc.)
2. L'ACTION principale à réaliser
3. La PRIORITÉ (haute, moyenne, basse)
4. Les MATÉRIAUX identifiés

Réponds UNIQUEMENT avec un JSON valide:
{{
    "lot": "Nom du lot",
    "action": "Action principale à réaliser",
    "priority": "haute|moyenne|basse",
    "materials": ["matériau1", "matériau2"],
    "notes": "Remarques éventuelles"
}}"""
    
    message = claude_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    
    response_text = message.content[0].text
    
    try:
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        if json_start != -1 and json_end > json_start:
            return json.loads(response_text[json_start:json_end])
    except json.JSONDecodeError:
        pass
    
    return {"raw_response": response_text, "parse_error": True}


# ==============================================
# ENDPOINTS - ANALYSE BIM MANUEL
# ==============================================

@app.post("/analyze-bim", response_model=BIMAnalysisResponse)
async def analyze_bim(request: BIMAnalysisRequest):
    """Analyse un objet BIM fourni manuellement et enregistre dans Supabase"""
    
    if not claude_client:
        raise HTTPException(status_code=503, detail="Service Anthropic non configuré")
    
    try:
        bim_object = request.bim_object
        bim_object_id = bim_object.get("guid") or bim_object.get("id", "unknown")
        bim_object_type = bim_object.get("ifc_type") or bim_object.get("speckleType") or bim_object.get("type", "Unknown")
        
        prompt = f"""Tu es un expert BTP/Construction. Analyse cet objet BIM.

OBJET BIM (JSON):
```json
{json.dumps(bim_object, indent=2, ensure_ascii=False)[:4000]}
```

{f"CONTEXTE CCTP: {request.context}" if request.context else ""}

Détermine:
1. Le LOT concerné (Gros Oeuvre, Charpente, Couverture, Menuiseries, Plâtrerie, Peinture, Électricité, Plomberie, CVC, etc.)
2. L'ACTION principale à réaliser
3. La PRIORITÉ (haute, moyenne, basse)
4. Les MATÉRIAUX identifiés

Réponds UNIQUEMENT avec un JSON valide:
{{
    "lot": "Nom du lot",
    "action": "Action principale",
    "priority": "haute|moyenne|basse",
    "materials": ["matériau1", "matériau2"],
    "notes": "Remarques éventuelles"
}}"""
        
        message = claude_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = message.content[0].text
        
        try:
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                analysis = json.loads(response_text[json_start:json_end])
            else:
                analysis = {"raw_response": response_text, "parse_error": True}
        except json.JSONDecodeError:
            analysis = {"raw_response": response_text, "parse_error": True}
        
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
                    logger.info(f"Tâche créée: {task_id}")
                    
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


# ==============================================
# ENDPOINTS - ANALYSE DONNÉES EXCEL/CSV
# ==============================================

@app.post("/analyze-data", response_model=DataAnalysisResponse)
async def analyze_data(request: DataAnalysisRequest):
    """Analyse des données Excel/CSV avec génération de code Pandas"""
    
    if not claude_client:
        raise HTTPException(status_code=503, detail="Service Anthropic non configuré")
    
    try:
        # DataFrame de démonstration (à remplacer par chargement réel)
        df_test = pd.DataFrame({
            "Lot": ["Gros Oeuvre", "Peinture", "Plâtrerie", "Menuiserie", "Électricité"],
            "Montant_HT": [450000, 85000, 120000, 95000, 78000],
            "Avancement": [0.75, 0.30, 0.50, 0.45, 0.60]
        })
        
        prompt = f"""Tu es un expert en analyse de données Python/Pandas.

STRUCTURE DU DATAFRAME (df):
Colonnes: {list(df_test.columns)}
Types: {df_test.dtypes.to_dict()}
Aperçu:
{df_test.head().to_string()}

QUESTION: {request.question}

Génère UNIQUEMENT le code Python Pandas pour répondre.
Le DataFrame s'appelle 'df'. Stocke le résultat dans 'result'.
Pas de markdown, juste le code Python brut.
"""
        
        message = claude_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        )
        
        generated_code = message.content[0].text.strip()
        
        # Nettoyage du code
        if generated_code.startswith("```"):
            lines = generated_code.split("\n")
            generated_code = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
        
        # Exécution sécurisée
        local_vars = {"df": df_test, "pd": pd}
        try:
            exec(generated_code, {"__builtins__": {}}, local_vars)
            result = local_vars.get("result", "Aucun résultat")
            
            if isinstance(result, (pd.DataFrame, pd.Series)):
                answer = result.to_string()
            else:
                answer = str(result)
                
        except Exception as exec_error:
            logger.error(f"Erreur exécution code: {exec_error}")
            answer = f"Erreur d'exécution: {exec_error}"
        
        return DataAnalysisResponse(
            success=True,
            question=request.question,
            answer=answer,
            code_executed=generated_code,
            data_preview={"columns": list(df_test.columns), "row_count": len(df_test)}
        )
        
    except anthropic.APIError as e:
        logger.error(f"Erreur API Anthropic: {e}")
        raise HTTPException(status_code=502, detail=f"Erreur API Claude: {str(e)}")
    except Exception as e:
        logger.error(f"Erreur analyse données: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze-data/upload")
async def analyze_data_with_upload(
    question: str,
    project_id: str,
    file: UploadFile = File(...)
):
    """Analyse un fichier Excel/CSV uploadé directement"""
    
    if not claude_client:
        raise HTTPException(status_code=503, detail="Service Anthropic non configuré")
    
    try:
        filename = file.filename.lower()
        contents = await file.read()
        
        # Lecture selon le format
        if filename.endswith(".csv"):
            df = pd.read_csv(pd.io.common.BytesIO(contents))
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(pd.io.common.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Format non supporté. Utilisez CSV, XLS ou XLSX.")
        
        # Limite de taille
        if len(df) > 10000:
            logger.warning(f"DataFrame tronqué: {len(df)} -> 10000 lignes")
            df = df.head(10000)
        
        prompt = f"""Tu es un expert en analyse de données Python/Pandas.

STRUCTURE DU DATAFRAME (df):
Colonnes: {list(df.columns)}
Types: {df.dtypes.to_dict()}
Nombre de lignes: {len(df)}
Aperçu (10 premières lignes):
{df.head(10).to_string()}

QUESTION: {question}

Génère UNIQUEMENT le code Python Pandas pour répondre.
Le DataFrame s'appelle 'df'. Stocke le résultat dans 'result'.
Pas de markdown, juste le code Python brut.
"""
        
        message = claude_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        )
        
        generated_code = message.content[0].text.strip()
        
        # Nettoyage
        if generated_code.startswith("```"):
            lines = generated_code.split("\n")
            generated_code = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
        
        # Exécution
        local_vars = {"df": df, "pd": pd}
        try:
            exec(generated_code, {"__builtins__": {}}, local_vars)
            result = local_vars.get("result", "Aucun résultat")
            
            if isinstance(result, (pd.DataFrame, pd.Series)):
                answer = result.to_string()
            else:
                answer = str(result)
                
        except Exception as exec_error:
            answer = f"Erreur d'exécution: {exec_error}"
        
        return DataAnalysisResponse(
            success=True,
            question=question,
            answer=answer,
            code_executed=generated_code,
            data_preview={"columns": list(df.columns), "row_count": len(df)}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================
# POINT D'ENTREE
# ==============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    