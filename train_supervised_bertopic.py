import pandas as pd
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sentence_transformers import SentenceTransformer
from bertopic.backend import BaseEmbedder
from bertopic.cluster import BaseCluster
from bertopic.dimensionality import BaseDimensionalityReduction
from umap import UMAP
import os

def main():
    print("Loading dataset...")
    df = pd.read_csv("clinical_document.csv")
    
    docs = df['segment_transcript'].fillna("").astype(str).tolist()
    targets = df['target'].tolist()
    
    print(f"Loaded {len(docs)} documents.")

    print("\nPre-computing embeddings using Multilingual Cross-Lingual Pipeline...")
    # Upgraded explicitly to Davlan/afro-xlmr-base for State-of-the-Art African Dialect parsing
    embedding_model = SentenceTransformer("Davlan/afro-xlmr-base")
    embeddings = embedding_model.encode(docs, show_progress_bar=True)

    print("\nConfiguring structural BERTopic improvements for Supervised Training...")
    
    # 1. Empty models to bypass dimensionality reduction and clustering
    empty_dimensionality_model = BaseDimensionalityReduction()
    empty_cluster_model = BaseCluster()

    # 2. Vocabulary Filtration Expansion
    # Curated Luganda stop words and conversational filler markers
    luganda_stops = [
        "kale", "mbu", "anti", "ate", "mpoozi", "bambi", "naye", "era", "nga", "oba",
        "ssebo", "nnyabo", "wangi", "yee", "nedda", "ddala", "nnyo", "nze", "ggwe", 
        "gwe", "ye", "ffe", "mmwe", "mwe", "bo", "wange", "kino", "ekyo", "bino", 
        "ebyo", "ne", "mu", "ku", "wa", "nti", "bwe", "eri", "okuva", "paka", "ko",
        "ki", "kiki", "ani", "ddi", "lwaki", "otya", "mutya", "ndi", "oli", "ali", 
        "tuli", "bali", "nina", "alina"
    ]

    document_specific_stops = [
    "like", "just", "hmm", "eeh", "eh", "ah", "don", "t", "s", 
    "kati", "awo", "waliwo", "na", "kuba", "umm", "mmh", "mpozi"
    ]

    # Merge with your existing lists
    custom_stop_words = list(ENGLISH_STOP_WORDS) + luganda_stops + document_specific_stops
    vectorizer_model = CountVectorizer(stop_words=custom_stop_words)
    
    # We must explicitly limit MMR to 5 words, otherwise it ignores the BERTopic limit and pads to 30!
    representation_model = MaximalMarginalRelevance(diversity=0.3, top_n_words=3)

    # Initialize BERTopic model with strictly 5 words to prevent trailing noise extraction
    topic_model = BERTopic(
        top_n_words=3,
        umap_model=empty_dimensionality_model,
        hdbscan_model=empty_cluster_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        verbose=True
    )

    # Fit the model: passing 'y' acts as explicit distinct clusters!
    print("Fitting Supervised model (this skips stochastic mapping and uses predefined targets)...")
    topics, probs = topic_model.fit_transform(docs, embeddings=embeddings, y=targets)

    # Establish output directory structure
    base_dir = "./bertopic_supervised"
    os.makedirs(base_dir, exist_ok=True)

    # Output Topic Info
    print("\nSaving Topic Info to CSV...")
    topic_info = topic_model.get_topic_info()
    
    # Force BERTopic to definitively prune its internal 30-word padding bug before writing representations 
    if 'Representation' in topic_info.columns:
        topic_info['Representation'] = topic_info['Representation'].apply(lambda x: [w for w in x if w != ''][:3])

    # Inject the actual textual Symptom Labels mapping to the Topic IDs!
    # Fix: BERTopic re-indexes topics by size dynamically, we map using the returned 'topics' sequence directly!
    # Exclude topic -1 (outlier bucket) — it contains mixed unclassified documents
    true_label_map = {t: l for t, l in zip(topics, df['symptom_label'].tolist()) if t != -1}
    topic_info.insert(2, 'Symptom_Label', topic_info['Topic'].map(true_label_map))
        
    topic_info.to_csv(os.path.join(base_dir, "topic_info.csv"), index=False)

    print("\nEvaluating Supervised Model Metrics...")
    # 1. Topic Diversity calculation
    topics_dict = topic_model.get_topics()
    valid_topics = [t for t in topics_dict if t != -1]
    
    top_n = 3
    unique_words = set()
    total_words = 0
    
    for topic_id in valid_topics:
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
    
    metrics_out = f"--- BERTopic SUPERVISED Evaluation Metrics ---\n" \
                  f"Topic Diversity (PUW): {diversity_score:.4f}\n" \
                  f"Topic Coherence (C_v): {coherence_score:.4f}\n"
    
    print(metrics_out)
    with open(os.path.join(base_dir, "metrics.txt"), "w") as f:
        f.write(metrics_out)

    # Visualizations Output
    print("\nGenerating Interactive Best-Practice Visualizations...")
    viz_dir = os.path.join(base_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    print("Running temporary UMAP strictly to render a Document scatter plot...")
    try:
        reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine', random_state=42).fit_transform(embeddings)
        fig_docs = topic_model.visualize_documents(docs, reduced_embeddings=reduced_embeddings, custom_labels=True)
        fig_docs.write_html(os.path.join(viz_dir, "topics_documents_scatter.html"))
    except Exception as e:
        print(f"Failed to generate visualize_documents: {e}")
    
    try:
        fig_topics = topic_model.visualize_topics()
        fig_topics.write_html(os.path.join(viz_dir, "topics_distance.html"))
    except Exception as e:
        print(f"Skipping visualize_topics (Expected to fail without integrated UMAP distances): {e}")
        
    try:
        fig_barchart = topic_model.visualize_barchart()
        fig_barchart.write_html(os.path.join(viz_dir, "topics_barchart.html"))
    except Exception as e:
        print(f"Failed to generate visualize_barchart: {e}")

    try:
        print("Rendering Topics Per Class visualization...")
        classes = df['symptom_label'].tolist()
        # Since we use a fully supervised model, this generates purely correlated distributions
        topics_per_class = topic_model.topics_per_class(docs, classes=classes)
        fig_classes = topic_model.visualize_topics_per_class(topics_per_class, top_n_topics=10)
        fig_classes.write_html(os.path.join(viz_dir, "topics_per_class.html"))
    except Exception as e:
        print(f"Failed to generate visualize_topics_per_class: {e}")

    try:
        fig_hierarchy = topic_model.visualize_hierarchy()
        fig_hierarchy.write_html(os.path.join(viz_dir, "topics_hierarchy.html"))
    except Exception as e:
        print(f"Skipping visualize_hierarchy (May fail without integrated HDBSCAN tree): {e}")

    # Save the model efficiently
    model_path = os.path.join(base_dir, "model")
    print(f"\nSaving highly optimized lightweight safetensors model to {model_path}...")
    topic_model.save(model_path, serialization="safetensors", save_embedding_model=False)
    
    print("Supervised Pipeline executed successfully!")

if __name__ == "__main__":
    main()
