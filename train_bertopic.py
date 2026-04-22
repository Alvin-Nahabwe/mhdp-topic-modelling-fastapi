import pandas as pd
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import os

def main():
    print("Loading dataset...")
    df = pd.read_csv("clinical_document.csv")
    
    docs = df['segment_transcript'].fillna("").astype(str).tolist()
    targets = df['target'].tolist()
    
    print(f"Loaded {len(docs)} documents.")

    print("\nPre-computing embeddings to speed up pipeline...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(docs, show_progress_bar=True)

    print("\nConfiguring structural BERTopic improvements...")
    # 1. Parameter Tuning: Custom UMAP for local structure
    umap_model = UMAP(n_neighbors=5, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
    
    # 2. Tips & Tricks: Stop Words Vectorizer
    vectorizer_model = CountVectorizer(stop_words="english")
    
    # 3. Tips & Tricks: Diversified MMR Representation
    representation_model = MaximalMarginalRelevance(diversity=0.3)

    # Initialize BERTopic model with newly researched blocks
    topic_model = BERTopic(
        min_topic_size=5, 
        umap_model=umap_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        verbose=True
    )

    # Fit the model: passing 'y' enables semi-supervised modeling
    print("Fitting model (this may take a minute depending on resources)...")
    topics, probs = topic_model.fit_transform(docs, embeddings=embeddings, y=targets)

    # Establish output directory structure
    base_dir = "./bertopic_semi_supervised"
    os.makedirs(base_dir, exist_ok=True)

    # Output Topic Info
    print("\nSaving Topic Info to CSV...")
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(os.path.join(base_dir, "topic_info.csv"), index=False)

    print("\nEvaluating Model Metrics...")
    # 1. Topic Diversity calculation
    topics_dict = topic_model.get_topics()
    valid_topics = [t for t in topics_dict if t != -1]
    
    top_n = 10
    unique_words = set()
    total_words = 0
    
    for topic_id in valid_topics:
        # retrieve top_n words for the given topic
        words = [w for w, _ in topics_dict[topic_id][:top_n]]
        unique_words.update(words)
        total_words += len(words)
        
    diversity_score = len(unique_words) / total_words if total_words > 0 else 0

    # 2. Topic Coherence calculation Using Gensim
    from gensim.corpora.dictionary import Dictionary
    from gensim.models.coherencemodel import CoherenceModel
    
    print("Computing gensim Coherence metrics...")
    tokenized_docs = [str(doc).lower().split() for doc in docs]
    dictionary = Dictionary(tokenized_docs)
    topic_words = [[w for w, _ in topics_dict[topic_id]] for topic_id in valid_topics]
    
    cm = CoherenceModel(topics=topic_words, texts=tokenized_docs, dictionary=dictionary, coherence='c_v')
    coherence_score = cm.get_coherence()
    
    metrics_out = f"--- BERTopic Evaluation Metrics ---\n" \
                  f"Topic Diversity (PUW): {diversity_score:.4f}\n" \
                  f"Topic Coherence (C_v): {coherence_score:.4f}\n"
    
    print(metrics_out)
    with open(os.path.join(base_dir, "metrics.txt"), "w") as f:
        f.write(metrics_out)

    # Visualizations Output
    print("\nSaving Interactive HTML Visualizations...")
    viz_dir = os.path.join(base_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    try:
        fig_topics = topic_model.visualize_topics()
        fig_topics.write_html(os.path.join(viz_dir, "topics_distance.html"))
    except Exception as e:
        print(f"Failed to generate visualize_topics: {e}")
        
    try:
        fig_barchart = topic_model.visualize_barchart()
        fig_barchart.write_html(os.path.join(viz_dir, "topics_barchart.html"))
    except Exception as e:
        print(f"Failed to generate visualize_barchart: {e}")

    try:
        fig_hierarchy = topic_model.visualize_hierarchy()
        fig_hierarchy.write_html(os.path.join(viz_dir, "topics_hierarchy.html"))
    except Exception as e:
        print(f"Failed to generate visualize_hierarchy: {e}")

    # Save the model
    model_path = os.path.join(base_dir, "model")
    print(f"\nSaving model to {model_path}...")
    topic_model.save(model_path)
    
    print("Pipeline executed successfully!")

if __name__ == "__main__":
    main()
