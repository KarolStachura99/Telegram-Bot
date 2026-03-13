import os
from datetime import datetime
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import time 
from datetime import datetime


# Czy folder "plots" w ogóle istnieje
os.makedirs("plots", exist_ok=True)

def generate_filename() -> str:
    """Generuje wymaganą przez prowadzącego nazwę pliku, zapobiegając nadpisywaniu."""
    while True:
        now = datetime.now()
        filepath = f"plots/Sentence_{now.strftime('%Y-%m-%d_%H-%M-%S')}.png"
        
        # Jeśli plik z tej samej sekundy jeszcze nie istnieje, możemy go użyć
        if not os.path.exists(filepath):
            return filepath
            
        # Jeśli istnieje, komputer jest za szybki. Czekamy 1 sekundę i próbujemy ponownie.
        time.sleep(1)

def plot_histogram(tokens: list) -> str:
    """Tworzy histogram długości tokenów i zwraca ścieżkę do wygenerowanego pliku."""
    filepath = generate_filename()
    
    # Obliczamy długość każdego słowa
    lengths = [len(token) for token in tokens]
    
    plt.figure(figsize=(8, 6))
    # Rysujemy słupki
    plt.hist(lengths, bins=range(1, max(lengths)+2) if lengths else [1], edgecolor='black', alpha=0.7)
    plt.title('Histogram długości tokenów')
    plt.xlabel('Długość tokenu (liczba znaków)')
    plt.ylabel('Częstotliwość')
    
    # Zapisujemy do pliku i zamykamy wykres z pamięci
    plt.savefig(filepath)
    plt.close()
    
    return filepath

def plot_wordcloud(tokens: list) -> str:
    """Tworzy chmurę najczęstszych słów i zwraca ścieżkę do pliku."""
    filepath = generate_filename()
    text = " ".join(tokens)
    
    if not text.strip():
        text = "Brak_słów"
        
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    
    plt.savefig(filepath)
    plt.close()
    
    return filepath

def plot_bar_chart(tokens: list, top_n: int = 10) -> str:
    """Tworzy wykres słupkowy najczęstszych tokenów."""
    filepath = generate_filename()
    
    if not tokens:
        plt.figure()
        plt.savefig(filepath)
        plt.close()
        return filepath
        
    counter = Counter(tokens)
    common = counter.most_common(top_n)
    words = [w for w, count in common]
    counts = [count for w, count in common]
    
    plt.figure(figsize=(10, 6))
    plt.bar(words, counts, color='skyblue', edgecolor='black')
    plt.title(f'Top {top_n} najczęstszych słów')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig(filepath)
    plt.close()
    
    return filepath