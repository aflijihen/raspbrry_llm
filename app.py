import os
import logging
import time
from dotenv import load_dotenv
from recommendation_engine import RecommendationEngine
from mqtt_handler import DataHandler  # Assurez-vous que la classe DataHandler est correctement importée

load_dotenv()

if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG)

    # Initialiser l'engine de recommandations
    rec_engine = RecommendationEngine(
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        pushbullet_api_key=os.getenv('PUSHBULLET_API_KEY'),
    )
    
    # Initialiser le DataHandler pour collecter les données des capteurs localement
    data_handler = DataHandler()

    while True:
        # Collecter les données des capteurs
        sensor_data_json = data_handler.collect_sensor_data()
        
        if sensor_data_json:
            # Traiter les données avec le moteur de recommandations
            recommendation = rec_engine.generate_recommendation(sensor_data_json)
            
            if recommendation:
                # Envoyer une notification avec les recommandations générées
                rec_engine.notify("Recommandation Capteurs", recommendation)
        
        # Pause entre les cycles de collecte
        time.sleep(10)


