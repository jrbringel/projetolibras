import os
import time
import urllib.request

import cv2
import mediapipe as mp
import pandas as pd

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

CSV_PATH = "dados_libras_2maos.csv"
MODEL_TASK_PATH = os.path.join("models", "hand_landmarker.task")
NUM_HANDS = 2
LANDMARKS_PER_HAND = 21
FEATURES_PER_HAND = LANDMARKS_PER_HAND * 3  # x,y,z
FEATURES_TOTAL = FEATURES_PER_HAND * NUM_HANDS  # 126


def _ensure_hand_model(model_path: str) -> str:
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if os.path.exists(model_path):
        return model_path

    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    print("Baixando modelo do MediaPipe (primeira vez)...")
    urllib.request.urlretrieve(url, model_path)
    return model_path


def _open_camera():
    backends = []
    if hasattr(cv2, "CAP_DSHOW"):
        backends.append(("DSHOW", cv2.CAP_DSHOW))
    if hasattr(cv2, "CAP_MSMF"):
        backends.append(("MSMF", cv2.CAP_MSMF))
    backends.append(("AUTO", cv2.CAP_ANY))

    indices = [0, 1, 2, 3]
    for idx in indices:
        for backend_name, backend in backends:
            cap = cv2.VideoCapture(idx, backend)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            ok, frame = cap.read()
            if ok and frame is not None:
                print(f"Camera aberta: index={idx} backend={backend_name}")
                return cap
            cap.release()
    return None


def _draw_hand_landmarks(image_bgr, landmarks):
    h, w = image_bgr.shape[:2]
    for lm in landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        cv2.circle(image_bgr, (x, y), 3, (0, 255, 0), -1)


def _features_for_hand(hand_landmarks):
    feats = []
    for lm in hand_landmarks:
        feats.extend([lm.x, lm.y, lm.z])
    return feats


def _extract_two_hands_features(results):
    zeros = [0.0] * FEATURES_PER_HAND
    out_left = None
    out_right = None

    if not results.hand_landmarks:
        return zeros + zeros

    handed = getattr(results, "handedness", None)

    for i, hand in enumerate(results.hand_landmarks[:NUM_HANDS]):
        feats = _features_for_hand(hand)
        label = None
        if handed and i < len(handed) and handed[i] and len(handed[i]) > 0:
            label = getattr(handed[i][0], "category_name", None)

        if label == "Left" and out_left is None:
            out_left = feats
        elif label == "Right" and out_right is None:
            out_right = feats
        else:
            if out_right is None:
                out_right = feats
            elif out_left is None:
                out_left = feats

    if out_right is None:
        out_right = zeros
    if out_left is None:
        out_left = zeros

    # Ordem fixa: Right depois Left (126 valores)
    return out_right + out_left


def main():
    task_path = _ensure_hand_model(MODEL_TASK_PATH)
    base_options = mp_python.BaseOptions(model_asset_path=task_path)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=NUM_HANDS,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)
    print("MediaPipe carregado (Tasks HandLandmarker)!")

    cap = _open_camera()
    print("DICA: Se a câmera não abrir, verifique o Iriun no celular e no PC.")
    if cap is None:
        raise RuntimeError("Nao consegui abrir a camera.")

    label = input("Qual sinal vamos coletar? (ex: bom_dia): ").strip()
    dados_coletados = []

    cv2.namedWindow("COLETOR LIBRAS", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("COLETOR LIBRAS", 960, 540)

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            print("Nao consegui ler frame da camera.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)
        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        if results.hand_landmarks:
            for hand in results.hand_landmarks[:NUM_HANDS]:
                _draw_hand_landmarks(frame, hand)

        cv2.imshow("COLETOR LIBRAS", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            feats = _extract_two_hands_features(results)
            linha = feats + [label]
            dados_coletados.append(linha)
            print(f"Registrado! Total: {len(dados_coletados)}")
        elif key == ord("q"):
            break

    if dados_coletados:
        df = pd.DataFrame(dados_coletados)
        df.to_csv(
            CSV_PATH,
            mode="a",
            header=not os.path.exists(CSV_PATH),
            index=False,
        )
        print(f"Arquivo '{CSV_PATH}' atualizado!")

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()