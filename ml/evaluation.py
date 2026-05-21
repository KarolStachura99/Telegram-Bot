import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import config

def evaluate_and_plot_matrix(y_true: list, y_pred: list, model_name: str, dataset_name: str) -> dict:
    """
    Oblicza statystyki (Accuracy, Precision, Recall, F1) i rysuje macierz pomyłek.
    Zwraca słownik z wynikami.
    """
    # 1. Obliczanie metryk (używamy average='weighted', aby poradzić sobie z klasyfikacją wieloklasową i binarną)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # 2. Generowanie Macierzy Pomyłek (Confusion Matrix)
    cm = confusion_matrix(y_true, y_pred)
    
    # 3. Rysowanie wykresu za pomocą biblioteki Seaborn
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Macierz pomyłek: {model_name.upper()} ({dataset_name})')
    plt.ylabel('Prawdziwa etykieta')
    plt.xlabel('Przewidziana etykieta')
    
    # Zapisywanie obrazka na dysku
    os.makedirs(config.PLOTS_LAB3_DIR, exist_ok=True)
    matrix_path = os.path.join(config.PLOTS_LAB3_DIR, f"confusion_{model_name}_{dataset_name}.png")
    plt.savefig(matrix_path, bbox_inches='tight')
    plt.close()
    
    # Zwracamy wszystkie wyliczone dane
    return {
        "Model": model_name.upper(),
        "Dataset": dataset_name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "Matrix_Path": matrix_path
    }

def save_results_to_csv(results_list: list, filename: str = "lab3results.csv"):
    """Zapisuje listę wyników do pliku CSV"""
    filepath = os.path.join(os.getcwd(), filename)
    file_exists = os.path.isfile(filepath)
    
    with open(filepath, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Zapisz nagłówki tylko jeśli plik jest nowy
        if not file_exists:
            writer.writerow(["Model", "Dataset", "Accuracy", "Precision", "Recall", "F1-Score"])
            
        for res in results_list:
            writer.writerow([res["Model"], res["Dataset"], res["Accuracy"], res["Precision"], res["Recall"], res["F1-Score"]])