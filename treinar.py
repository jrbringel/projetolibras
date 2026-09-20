import os

import pandas as pd
from joblib import dump
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def main():
    csv_path = "dados_libras_2maos.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            "Nao encontrei 'dados_libras_2maos.csv'. Rode o coletor e aperte 's' algumas vezes para gerar dados."
        )

    df = pd.read_csv(csv_path, header=None)
    if df.shape[1] < 2:
        raise RuntimeError("CSV invalido: esperado features + label na ultima coluna.")

    # O coletor salva o CSV com header na primeira vez. Remove essa linha se existir.
    try:
        last_col = str(df.shape[1] - 1)
        if str(df.iloc[0, 0]) == "0" and str(df.iloc[0, -1]) == last_col:
            df = df.iloc[1:].reset_index(drop=True)
    except Exception:
        pass

    X = df.iloc[:, :-1].astype("float32")
    y = df.iloc[:, -1].astype(str)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced")),
        ]
    )

    # Se tiver poucas amostras por classe, nao da para fazer split com stratify.
    counts = y.value_counts()
    can_split = len(df) >= 5 and y.nunique() >= 2 and counts.min() >= 2

    if can_split:
        test_size = 0.2 if len(df) >= 20 else 0.3
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Amostras: {len(df)} | Classes: {y.nunique()} | Acuracia: {acc:.3f}")
        print(classification_report(y_test, y_pred))
    else:
        model.fit(X, y)
        print(
            f"Amostras: {len(df)} | Classes: {y.nunique()} | "
            "Poucos dados para teste; modelo treinado com tudo."
        )

    os.makedirs("models", exist_ok=True)
    out_path = os.path.join("models", "classificador_libras_2maos.pkl")
    dump(model, out_path)
    print(f"Modelo salvo em: {out_path}")


if __name__ == "__main__":
    main()

