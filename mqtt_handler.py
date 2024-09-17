import time
import RPi.GPIO as GPIO
import json

class DataHandler:
    def __init__(self):
        # Configuration des broches GPIO pour les capteurs
        self.temp_sensor_pin = 18  # Exemple de broche pour le capteur de température
        self.ph_sensor_pin = 23    # Exemple de broche pour le capteur de pH
        self.ec_sensor_pin = 24    # Exemple de broche pour le capteur EC
        self.light_sensor_pin = 25 # Exemple de broche pour le capteur de luminosité
        
        # Initialiser GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.temp_sensor_pin, GPIO.IN)
        GPIO.setup(self.ph_sensor_pin, GPIO.IN)
        GPIO.setup(self.ec_sensor_pin, GPIO.IN)
        GPIO.setup(self.light_sensor_pin, GPIO.IN)

    def collect_sensor_data(self):
        """
        Cette fonction collecte les données de plusieurs capteurs connectés via GPIO,
        les formate en JSON, puis retourne le JSON sous forme de chaîne de caractères.
        """
        try:
            # Collecte des données depuis les capteurs
            temp_value = GPIO.input(self.temp_sensor_pin)
            ph_value = GPIO.input(self.ph_sensor_pin)
            ec_value = GPIO.input(self.ec_sensor_pin)
            light_value = GPIO.input(self.light_sensor_pin)
            
            # Créer un dictionnaire des valeurs collectées
            sensor_data = {
                'temperature': temp_value,
                'pH': ph_value,
                'EC': ec_value,
                'brightness': light_value
            }

            # Convertir le dictionnaire en JSON
            sensor_data_json = json.dumps(sensor_data)
            
            print("Données collectées et converties en JSON : ", sensor_data_json)
            return sensor_data_json
        
        except Exception as e:
            print(f"Erreur lors de la collecte des données : {e}")
            return None

