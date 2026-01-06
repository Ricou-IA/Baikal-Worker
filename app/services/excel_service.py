import pandas as pd
from fastapi import UploadFile
import io

class ExcelService:
    async def process_file(self, file: UploadFile):
        try:
            # On lit le contenu du fichier
            contents = await file.read()
            
            # On utilise Pandas pour lire l'Excel
            df = pd.read_excel(io.BytesIO(contents))
            
            # Analyse basique pour le MVP
            result = {
                "filename": file.filename,
                "total_rows": len(df),
                "columns": list(df.columns),
                "preview_data": df.head(3).to_dict(orient="records") # Aperçu des 3 premières lignes
            }
            
            return {"status": "success", "data": result}
            
        except Exception as e:
            print(f"❌ Erreur Excel: {e}")
            return {"status": "error", "message": str(e)}
            