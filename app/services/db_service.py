import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

class DbService:
    def __init__(self):
        url: str = os.environ.get("SUPABASE_URL")
        key: str = os.environ.get("SUPABASE_KEY")
        
        if not url or not key:
            print("⚠️ ATTENTION: Identifiants Supabase manquants dans .env")
            self.client = None
        else:
            self.client: Client = create_client(url, key)

    def save_task(self, task_data: dict):
        if not self.client:
            return {"error": "Database not connected"}
            
        try:
            # --- CORRECTION MAJEURE ICI ---
            # On ne déballe plus le résultat en (data, count).
            # On récupère l'objet response complet.
            response = self.client.schema("arpet").table("project_tasks").insert(task_data).execute()
            
            # On vérifie si on a des données insérées (response.data)
            inserted_data = response.data if hasattr(response, 'data') else "Donnée envoyée"
            
            return {"status": "saved_in_arpet", "db_response": inserted_data}

        except Exception as e:
            print(f"❌ Erreur DB: {e}")
            # C'est cette erreur qui t'empêchait de voir les données
            return {"status": "error", "details": str(e)}
            