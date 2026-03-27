import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE

# Upewniamy się, że folder na wykresy istnieje
os.makedirs("lab2plots", exist_ok=True)

def plot_wordclouds(texts: list, labels: list, class_names: list):
    """Generuje chmurę słów dla całego korpusu i dla każdej klasy."""
    # 1. Chmura dla całego korpusu
    all_text = " ".join(texts)
    wc_corpus = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc_corpus, interpolation='bilinear')
    plt.axis('off')
    plt.savefig("lab2plots/wordcloud_corpus.png")
    plt.close()

    # 2. Chmura dla poszczególnych klas
    for i, class_name in enumerate(class_names):
        class_texts = [texts[j] for j in range(len(texts)) if labels[j] == i]
        if class_texts:
            class_text_joined = " ".join(class_texts)
            wc_class = WordCloud(width=800, height=400, background_color='white').generate(class_text_joined)
            plt.figure(figsize=(10, 5))
            plt.imshow(wc_class, interpolation='bilinear')
            plt.title(f"Klasa: {class_name}")
            plt.axis('off')
            # Czyszczenie nazwy klasy z niedozwolonych znaków w nazwie pliku
            safe_class_name = class_name.replace("/", "_").replace("\\", "_")
            plt.savefig(f"lab2plots/wordcloud_class_{safe_class_name}.png")
            plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, embedding_name: str, model_name: str):
    """Generuje i zapisuje macierz pomyłek."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix: {embedding_name.upper()} + {model_name.upper()}')
    plt.ylabel('Rzeczywista klasa')
    plt.xlabel('Przewidziana klasa')
    plt.tight_layout()
    plt.savefig(f"lab2plots/confusion_{embedding_name}_{model_name}.png")
    plt.close()

def plot_embeddings(X, y, dataset: str, modelname: str, representation: str):
    """Generuje wizualizacje PCA, t-SNE i TruncatedSVD dla danych."""
    # Ograniczamy liczbę próbek do wizualizacji, żeby t-SNE nie trwało wieków
    limit = min(1000, X.shape[0])
    X_viz, y_viz = X[:limit], y[:limit]

    methods = {
        'pca': PCA(n_components=2),
        'tsne': TSNE(n_components=2, random_state=42),
        'svd': TruncatedSVD(n_components=2)
    }

    for method_name, reducer in methods.items():
        try:
            X_reduced = reducer.fit_transform(X_viz)
            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_viz, cmap='viridis', alpha=0.6)
            plt.colorbar(scatter, label='Klasy')
            plt.title(f'Wizualizacja {method_name.upper()} ({representation})')
            filepath = f"lab2plots/{dataset}_{modelname}_{representation}_{method_name}_embedding.png"
            plt.savefig(filepath)
            plt.close()
        except Exception as e:
            print(f"Pominięto wizualizację {method_name} (często SVD gryzie się z ujemnymi wartościami): {e}")

def save_feature_importance(model, vectorizer, class_names, dataset_name: str):
    """Zapisuje Feature Importance (top 10) dla modelu, jeśli to możliwe."""
    if hasattr(model, 'feature_importances_') and hasattr(vectorizer, 'get_feature_names_out'):
        importances = model.feature_importances_
        features = vectorizer.get_feature_names_out()
        indices = np.argsort(importances)[::-1][:10]
        
        with open(f"lab2plots/{dataset_name}_feature_importance.txt", "w", encoding="utf-8") as f:
            f.write("Top 10 cech (Feature Importance):\n")
            for i in indices:
                f.write(f"{features[i]}: {importances[i]:.4f}\n")

def save_results(embedding: str, model: str, accuracy: float, macro_f1: float, seed: int):
    """Zapisuje metryki do pliku CSV."""
    file_exists = os.path.isfile("lab2results.csv")
    with open("lab2results.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["embedding", "model", "accuracy", "macro_f1", "seed"]) #
        writer.writerow([embedding, model, round(accuracy, 4), round(macro_f1, 4), seed])

def save_similar_words_and_plot(model, words: list):
    """Zapisuje podobne słowa z Word2Vec/GloVe i generuje ich wykresy."""
    found_words = []
    vectors = []
    
    with open("lab2_similar_words.txt", "w", encoding="utf-8") as f:
        for word in words:
            if word in model:
                similar = model.most_similar(word, topn=5)
                f.write(f"Podobne do '{word}': {similar}\n")
                
                # Zbieramy dane do wykresu
                found_words.append(word)
                vectors.append(model[word])
                for sim_w, _ in similar:
                    found_words.append(sim_w)
                    vectors.append(model[sim_w])
            else:
                f.write(f"Słowo '{word}' nie istnieje w słowniku modelu.\n")
                
    if len(vectors) > 5:
        X = np.array(vectors)
        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        plt.figure(figsize=(10, 8))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c='red')
        for i, word in enumerate(found_words):
            plt.annotate(word, (X_pca[i, 0], X_pca[i, 1]))
        plt.title("Wizualizacja słów - PCA")
        plt.savefig("lab2plots/word_embedding_pca.png") #
        plt.close()
        
        # t-SNE (wymaga więcej próbek (tzw. perplexity), więc używamy mniejszej wartości perplexity dla małej ilości słów)
        perplexity = min(5, len(X) - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        X_tsne = tsne.fit_transform(X)
        plt.figure(figsize=(10, 8))
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c='blue')
        for i, word in enumerate(found_words):
            plt.annotate(word, (X_tsne[i, 0], X_tsne[i, 1]))
        plt.title("Wizualizacja słów - t-SNE")
        plt.savefig("lab2plots/word_embedding_tsne.png") #
        plt.close()