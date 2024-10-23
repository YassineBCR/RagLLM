import whisper

# Fonction pour transcrire un fichier audio en texte
def transcribe_audio_to_text(audio_file_path, output_txt_path):
    # Charger le modèle Whisper
    model = whisper.load_model("large")  # "medium", "large", ou "small"
  # Vous pouvez choisir "tiny", "small", "medium", "large"
    
    # Transcrire l'audio
    result = model.transcribe(audio_file_path)
    
    # Extraire le texte transcrit
    transcribed_text = result['text']
    
    # Sauvegarder le texte dans un fichier .txt
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write(transcribed_text)
    
    print(f"Transcription terminée. Le texte est enregistré dans {output_txt_path}")

# Exemple d'utilisation
if __name__ == "__main__":
    audio_file_path = "testaudio.mp3"  # Remplacez par le chemin de votre fichier audio
    output_txt_path = "fichier_output.txt"  # Chemin de sortie pour le fichier texte
    
    # Appel de la fonction pour transcrire et sauvegarder
    transcribe_audio_to_text(audio_file_path, output_txt_path)
