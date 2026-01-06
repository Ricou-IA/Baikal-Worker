# app/services/bim_service.py
from specklepy.api.client import SpeckleClient
from specklepy.transports.server import ServerTransport
from specklepy.api import operations

class BimService:
    def __init__(self, token: str = None):
        self.token = token
        # Si on n'a pas de token, on ne peut pas se connecter (pour l'instant c'est ok)
        self.client = SpeckleClient(host="speckle.xyz")
        if token:
            self.client.authenticate_with_token(token)

    async def get_metadata(self, stream_id: str, object_id: str):
        """
        Simule la récupération des données d'un objet 3D.
        Dans la version finale, cela téléchargera la géométrie réelle.
        """
        try:
            # Pour le MVP, on vérifie juste que la connexion au serveur Speckle fonctionne
            # Ici, on triche un peu pour le test si pas de token :
            if not self.token:
                return {
                    "status": "simulation", 
                    "message": "Pas de token Speckle configuré. Mode démo activé.",
                    "object_id": object_id,
                    "stream_id": stream_id,
                    "fake_data": {"volume": 12.5, "material": "Concrete", "level": "R+1"}
                }

            # Code réel (sera activé quand tu auras ta clé)
            # transport = ServerTransport(client=self.client, stream_id=stream_id)
            # obj = operations.receive(object_id, transport)
            # return obj
            
            return {"status": "connected", "msg": "Token valide, prêt à recevoir"}

        except Exception as e:
            return {"error": str(e)}
            