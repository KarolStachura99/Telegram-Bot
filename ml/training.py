import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import config
from ml.datasets import load_dataset

def train_sequential_model(model_type: str, dataset_name: str) -> tuple:
    """Trenuje model sieci neuronowej i zapisuje go do pliku .h5."""
    
    # ========================================================
    # NAJPIERW POBIERAMY DANE (Hugging Face / PyArrow)
    # ========================================================
    texts, labels, class_names = load_dataset(dataset_name)
    if len(texts) == 0:
        raise ValueError(f"Zbiór danych {dataset_name} jest pusty!")

    # ========================================================
    # DOPIERO TERAZ ŁADUJEMY TENSORFLOW
    # ========================================================
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, GRU, Dense, Dropout
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras import backend as K

    model_type = model_type.lower()
    
    # 2. Zamiana etykiet tekstowych na liczby
    le = LabelEncoder()
    y = le.fit_transform(labels)
    num_classes = len(np.unique(y))
    
    # Upewniamy się, że folder istnieje
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    
    # Zapis Kodera Etykiet
    le_path = os.path.join(config.MODELS_DIR, f"{model_type}_{dataset_name}_label_encoder.pkl")
    with open(le_path, 'wb') as f:
        pickle.dump(le, f)

    # 3. Tokenizacja
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    
    # Zapis Tokenizera
    tok_path = os.path.join(config.MODELS_DIR, f"{model_type}_{dataset_name}_tokenizer.pkl")
    with open(tok_path, 'wb') as f:
        pickle.dump(tokenizer, f)

    # 4. Wyrównanie zdań do MAX_LEN (Padding)
    X = pad_sequences(sequences, maxlen=config.MAX_LEN)
    vocab_size = len(tokenizer.word_index) + 1

    # 5. Budowa Architektury Sieci
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=config.EMBEDDING_DIM, input_length=config.MAX_LEN))
    
    if model_type == "simplernn":
        model.add(SimpleRNN(64))
    elif model_type == "lstm":
        model.add(LSTM(64))
    elif model_type == "gru":
        model.add(GRU(64))
    else:
        raise ValueError("Wybierz poprawny model: simplernn, lstm lub gru.")

    model.add(Dropout(0.5))

    # Warstwa wyjściowa
    if num_classes == 2:
        model.add(Dense(1, activation='sigmoid'))
        loss_fn = 'binary_crossentropy'
    else:
        model.add(Dense(num_classes, activation='softmax'))
        loss_fn = 'sparse_categorical_crossentropy'

    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

    # 6. Smażenie (Trening sieci)
    history = model.fit(X, y, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, validation_split=0.2, verbose=0)

    # 7. Zapis Mózgu do pliku .h5
    model_path = os.path.join(config.MODELS_DIR, f"{model_type}_{dataset_name}.h5")
    model.save(model_path)

    # 8. Rysowanie wykresu nauki
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Trening Accuracy')
    plt.plot(history.history['val_accuracy'], label='Walidacja Accuracy')
    plt.title(f'Historia uczenia: {model_type.upper()} na {dataset_name}')
    plt.xlabel('Epoka')
    plt.ylabel('Skuteczność')
    plt.legend()
    
    # Upewniamy się, że folder na wykresy istnieje
    os.makedirs(config.PLOTS_LAB3_DIR, exist_ok=True)
    plot_path = os.path.join(config.PLOTS_LAB3_DIR, f"train_history_{model_type}_{dataset_name}.png")
    plt.savefig(plot_path)
    plt.close()

    # ========================================================
    # CZYSZCZENIE PAMIĘCI
    # ========================================================
    final_accuracy = history.history['accuracy'][-1]
    K.clear_session()

    return model_path, plot_path, final_accuracy