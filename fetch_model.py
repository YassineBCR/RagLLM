import requests
from bs4 import BeautifulSoup
import json

# URL de la page à scraper
url = "https://ollama.com/library"

# Envoyer une requête GET à la page
response = requests.get(url)

# Vérifier si la requête a réussi
if response.status_code == 200:
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extraire le contenu (cette partie doit être adaptée à la structure HTML de la page)
    data = []
    models = soup.find_all('div', class_='model-card')  # Exemple de classe, à adapter

    for model in models:
        name = model.find('h2').text if model.find('h2') else "Nom non trouvé"
        description = model.find('p').text if model.find('p') else "Description non trouvée"
        data.append({'name': name, 'description': description})

    # Sauvegarder les données dans un fichier JSON
    with open('ollama_models.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print("Données scrappées et sauvegardées dans ollama_models.json")
else:
    print("Échec de la récupération de la page. Code d'état :", response.status_code)
