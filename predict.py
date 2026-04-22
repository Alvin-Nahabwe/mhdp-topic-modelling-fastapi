import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

def main():
    print("Loading specialized NLP Architecture...")
    # 1. Fetch our state-of-the-art African Dialect Model
    embedding_model = SentenceTransformer("Davlan/afro-xlmr-base")

    # 2. Load the lightweight BERTopic architecture from the directory
    topic_model = BERTopic.load("./bertopic_supervised/model", embedding_model=embedding_model)

    # 3. Load the symptom dictionary we mapped earlier
    custom_labels = pd.read_csv("./bertopic_supervised/topic_info.csv").set_index('Topic')['Symptom_Label'].to_dict()

    print("\n--- INFERENCE TEST START ---")
    
    # Passing mixed Luganda and English test sentences
    new_transcripts = [
        "Sifuna tulo, whenever I try to close my eyes I overthink.",
        "Buli lunaku awulira amaloboozi agamusindika, she hears voices."
    ]

    # Run prediction
    predicted_topics, probabilities = topic_model.transform(new_transcripts)

    for text, topic_id, score in zip(new_transcripts, predicted_topics, probabilities):
        symptom_name = custom_labels.get(topic_id, "Unknown Symptom")
        keywords = topic_model.get_topic(topic_id)
        # Parse out just the string keywords from the representation tuples
        clean_keywords = [word for word, weight in keywords]
        
        print(f"\nPatient Text: '{text}'")
        print(f"Predicted Symptom: {symptom_name} >> (Confidence: {score:.4f})")
        print(f"Trigger Keywords Matrix: {clean_keywords}")

    print("\n--- INFERENCE SUCCESS ---")

if __name__ == "__main__":
    main()
