import os
import pickle
import numpy as np
import config

def predict_sequential(method: str, text: str, dataset: str = "custom") -> tuple:
    """Wczytuje wytrenowany model i dokonuje predykcji dla pojedynczego tekstu."""
    
    model_path = os.path.join(config.MODELS_DIR, f"{method}_{dataset}.h5")
    tok_path = os.path.join(config.MODELS_DIR, f"{method}_{dataset}_tokenizer.pkl")
    le_path = os.path.join(config.MODELS_DIR, f"{method}_{dataset}_label_encoder.pkl")
    
    # Sprawdzenie, czy model w ogóle został wcześniej wytrenowany
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Brak modelu! Najpierw wytrenuj: /train {method} {dataset}")
        
    # Lazy loading TensorFlow (tylko wtedy, gdy funkcja jest wywoływana)
    import tensorflow as tf
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    
    # Wczytywanie artefaktów
    model = tf.keras.models.load_model(model_path)
    with open(tok_path, 'rb') as f:
        tokenizer = pickle.load(f)
    with open(le_path, 'rb') as f:
        le = pickle.load(f)
        
    # Preprocessing tekstu: tokenizacja i padding
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=config.MAX_LEN)
    
    # Wykonanie predykcji
    prediction = model.predict(padded, verbose=0)
    
    # Dekodowanie wyniku w zależności od liczby klas
    if prediction.shape[1] == 1:
        # Klasyfikacja binarna (np. IMDB, Amazon)
        score = prediction[0][0]
        label_idx = int(score > 0.5)
        confidence = score if label_idx == 1 else 1.0 - score
    else:
        # Klasyfikacja wieloklasowa (np. Twój custom)
        label_idx = np.argmax(prediction[0])
        confidence = prediction[0][label_idx]
        
    label = le.inverse_transform([label_idx])[0]
        
    # Tłumaczenie cyfrowych etykiet z IMDB/Amazon na tekst 
    if str(label) == "0":
        label = "negatywny"
    elif str(label) == "1":
        label = "pozytywny"
    
    # Czyszczenie pamięci po predykcji
    tf.keras.backend.clear_session()
    
    return str(label), float(confidence)