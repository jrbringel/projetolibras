import argparse
import os

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from libras_features import RAW_FEATURES, HandLandmarkNormalizer, mirror_raw


def _normalize_label(raw: str) -> str:
    s = str(raw).strip()
    if s.startswith("videos_"):
        s = s[len("videos_") :]
    s = s.replace("_", " ")
    return " ".join(p for p in s.split(" ") if p).lower()


def _drop_index_header(df: pd.DataFrame) -> pd.DataFrame:
    """O CSV e gravado com cabecalho 0,1,2,...,126. O pandas le isso como numeros."""
    row = pd.to_numeric(df.iloc[0], errors="coerce")
    expected = list(range(df.shape[1]))
    if row.notna().all() and [int(v) for v in row.tolist()] == expected:
        return df.iloc[1:].reset_index(drop=True)
    return df


def load_dataset(csv_path: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Nao encontrei '{csv_path}'. Rode o coletor (tecla 's') ou o importar_videos.py para gerar dados."
        )

    df = pd.read_csv(csv_path, header=None)
    if df.shape[1] != RAW_FEATURES + 1:
        raise RuntimeError(
            f"CSV invalido: esperado {RAW_FEATURES} features + label, encontrado {df.shape[1]} colunas."
        )

    df = _drop_index_header(df)
    y = df.iloc[:, -1].map(_normalize_label)
    keep = ~(y.eq("") | y.str.fullmatch(r"\d+"))
    df = df.loc[keep].reset_index(drop=True)
    X = df.iloc[:, :-1].astype("float32").to_numpy()
    y = df.iloc[:, -1].map(_normalize_label).to_numpy()
    return X, y


def build_model() -> Pipeline:
    svc = SVC(kernel="rbf", C=4.0, gamma="scale", class_weight="balanced")
    return Pipeline(
        steps=[
            ("landmarks", HandLandmarkNormalizer()),
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(svc, method="sigmoid", cv=5, ensemble=False)),
        ]
    )


def augment(X: np.ndarray, y: np.ndarray, jitter: float = 0.01, copies: int = 1):
    """Espelho horizontal + pequeno ruido, so no conjunto de treino."""
    rng = np.random.default_rng(42)
    parts_x = [X, mirror_raw(X)]
    parts_y = [y, y]
    for _ in range(copies):
        for base in (X, mirror_raw(X)):
            noise = rng.normal(0.0, jitter, size=base.shape).astype(np.float32)
            noise[np.abs(base) < 1e-9] = 0.0  # mao ausente continua ausente
            parts_x.append(base + noise)
            parts_y.append(y)
    return np.concatenate(parts_x), np.concatenate(parts_y)


def main():
    ap = argparse.ArgumentParser(description="Treina o classificador de sinais (2 maos).")
    ap.add_argument("--csv", default="dados_libras_2maos.csv", help="CSV com 126 features + label.")
    ap.add_argument("--out", default=os.path.join("models", "classificador_libras_2maos.pkl"))
    ap.add_argument("--no_augment", action="store_true", help="Treina sem espelhamento e sem ruido.")
    args = ap.parse_args()

    X, y = load_dataset(args.csv)
    counts = pd.Series(y).value_counts()
    keep = counts[counts >= 10].index
    if len(keep) < counts.size:
        mask = pd.Series(y).isin(keep).to_numpy()
        dropped = counts[counts < 10]
        print("Ignorando classes com poucas amostras:")
        for name, n in dropped.items():
            print(f"  {name}: {n}")
        X, y = X[mask], y[mask]
    classes = pd.Series(y).value_counts()
    print("Amostras por sinal:")
    for name, n in classes.items():
        print(f"  {name}: {n}")

    model = build_model()
    can_split = len(y) >= 5 and classes.size >= 2 and classes.min() >= 2

    if can_split:
        test_size = 0.2 if len(y) >= 20 else 0.3
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        if not args.no_augment:
            X_train, y_train = augment(X_train, y_train)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"\nAmostras: {len(y)} | Classes: {classes.size} | Acuracia no teste: {acc:.3f}")
        print(classification_report(y_test, y_pred))
        # Modelo final usa todos os dados, ja que o teste acima mediu a qualidade.
        X_all, y_all = (X, y) if args.no_augment else augment(X, y)
        model = build_model().fit(X_all, y_all)
    else:
        X_all, y_all = (X, y) if args.no_augment else augment(X, y)
        model.fit(X_all, y_all)
        print(
            f"\nAmostras: {len(y)} | Classes: {classes.size} | "
            "Poucos dados para teste; modelo treinado com tudo."
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    dump(model, args.out)
    print(f"Sinais reconhecidos: {', '.join(model.classes_)}")
    print(f"Modelo salvo em: {args.out}")


if __name__ == "__main__":
    main()

