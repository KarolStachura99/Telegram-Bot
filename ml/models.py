from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def train_model(method: str, X_train: np.ndarray, y_train: np.ndarray, gridsearch: bool, seed: int):
    """
    Inicjalizuje i trenuje wybrany model klasyfikacji.
    Jeśli gridsearch=True, uruchamia strojenie hiperparametrów za pomocą GridSearchCV.
    Zwraca wytrenowany model (lub najlepszy estymator z Grid Search).
    """
    method = method.lower()
    model = None
    param_grid = {}

    """
    MultinomialNB nie przyjmuje ujemnych wartości wejściowych,
    co jest problemem przy Word2Vec i GloVe. W locie skalujemy dane do zakresu [0, 1].
    """
    if method == "nb":
        if np.any(X_train < 0):
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            
        model = MultinomialNB()
        param_grid = {'alpha': [0.1, 0.5, 1.0]} #
        
    elif method == "rf":
        model = RandomForestClassifier(random_state=seed)
        param_grid = {
            'n_estimators': [100, 300],
            'max_depth': [None, 10, 20]
        } #
        
    elif method == "logreg":
        model = LogisticRegression(random_state=seed, max_iter=1000)
        param_grid = {'C': [0.1, 1, 10]} #
        
    elif method == "mlp":
        model = MLPClassifier(random_state=seed, max_iter=500)
        param_grid = {
            'hidden_layer_sizes': [(128,), (256, 128)]
        } #
        
    else:
        raise ValueError(f"Nieznana metoda klasyfikacji: {method}")

    # Uruchomienie Grid Search lub standardowego treningu
    if gridsearch:
        print(f"Rozpoczynam Grid Search dla modelu {method.upper()}...")
        # cv=3 dla przyspieszenia obliczeń, n_jobs=-1 wykorzystuje wszystkie rdzenie procesora
        search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, scoring='accuracy')
        search.fit(X_train, y_train)
        print(f"Najlepsze parametry dla {method}: {search.best_params_}")
        return search.best_estimator_
    else:
        print(f"Trenowanie bazowego modelu {method.upper()} (seed={seed})...")
        model.fit(X_train, y_train)
        return model