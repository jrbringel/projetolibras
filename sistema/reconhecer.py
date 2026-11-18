import os
import time
import urllib.request

import cv2
import mediapipe as mp
from joblib import load

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = os.path.join("models", "classificador_libras_2maos.pkl")
HAND_TASK_PATH = os.path.join("models", "hand_landmarker.task")
NUM_HANDS = 2
LANDMARKS_PER_HAND = 21
FEATURES_PER_HAND = LANDMARKS_PER_HAND * 3


def _pretty_label(raw: str) -> str:
    s = str(raw).strip()
    if s.startswith("videos_"):
        s = s[len("videos_") :]
    s = s.replace("_", " ").strip()
    s = " ".join(p for p in s.split(" ") if p)
    return s.capitalize() if s else "?"


def _clamp01(v: float) -> float:
    try:
        v = float(v)
    except Exception:
        return 0.0
    if v < 0:
        return 0.0
    if v > 1:
        return 1.0
    return v


def _conf_color(conf: float):
    # Vermelho -> Amarelo -> Verde
    c = _clamp01(conf)
    if c < 0.5:
        # red -> yellow
        t = c / 0.5
        return (0, int(255 * t), 255)
    t = (c - 0.5) / 0.5
    return (0, 255, int(255 * (1 - t)))


def _draw_hud(frame, label: str, conf: float, hands: int, fps: float):
    x, y = 14, 14
    pad = 14
    font = cv2.FONT_HERSHEY_SIMPLEX

    title_scale = 1.25
    title_thick = 2
    info_scale = 0.75
    info_thick = 2

    conf = _clamp01(conf)
    title = label
    info = f"maos: {hands}   fps: {fps:.1f}   q = sair"

    (tw, th), _ = cv2.getTextSize(title, font, title_scale, title_thick)
    (iw, ih), _ = cv2.getTextSize(info, font, info_scale, info_thick)

    bar_w = max(260, tw)
    w = max(tw, iw, bar_w) + pad * 2
    h = th + ih + 44 + pad * 2

    # Sombra
    shadow = frame.copy()
    cv2.rectangle(shadow, (x + 4, y + 6), (x + w + 4, y + h + 6), (0, 0, 0), -1)
    cv2.addWeighted(shadow, 0.25, frame, 0.75, 0, frame)

    # Fundo com blur leve (só na área do HUD)
    x2, y2 = x + w, y + h
    roi = frame[y:y2, x:x2]
    if roi.size:
        blurred = cv2.GaussianBlur(roi, (9, 9), 0)
        cv2.addWeighted(blurred, 0.65, roi, 0.35, 0, roi)

    # Painel
    cv2.rectangle(frame, (x, y), (x2, y2), (20, 20, 20), -1)
    cv2.rectangle(frame, (x, y), (x2, y2), (80, 80, 80), 2)

    # Texto
    cv2.putText(
        frame,
        title,
        (x + pad, y + pad + th),
        font,
        title_scale,
        (0, 255, 255),
        title_thick,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        info,
        (x + pad, y + pad + th + ih + 10),
        font,
        info_scale,
        (230, 230, 230),
        info_thick,
        cv2.LINE_AA,
    )

    # Barra de confiança
    bar_x = x + pad
    bar_y = y + pad + th + ih + 24
    bar_h = 16
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    fill_w = int(bar_w * conf)
    if fill_w > 0:
        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + fill_w, bar_y + bar_h),
            _conf_color(conf),
            -1,
        )
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (110, 110, 110), 1)
    cv2.putText(
        frame,
        f"confianca: {conf:.2f}",
        (bar_x, bar_y + bar_h + 18),
        font,
        0.65,
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )


def _ensure_hand_model(model_path: str) -> str:
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if os.path.exists(model_path):
        return model_path

    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    print("Baixando modelo do MediaPipe (primeira vez)...")
    urllib.request.urlretrieve(url, model_path)
    return model_path


def _draw_hand_landmarks(image_bgr, landmarks):
    h, w = image_bgr.shape[:2]
    for lm in landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        cv2.circle(image_bgr, (x, y), 3, (0, 255, 0), -1)


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

    return out_right + out_left


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Nao achei 'models/classificador_libras_2maos.pkl'. Rode primeiro: python treinar.py"
        )
    model = load(MODEL_PATH)

    hand_task_path = _ensure_hand_model(HAND_TASK_PATH)
    base_options = mp_python.BaseOptions(model_asset_path=hand_task_path)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=NUM_HANDS,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    cap = _open_camera()
    if cap is None:
        raise RuntimeError(
            "Nao consegui abrir a camera. Feche apps que usam a camera e confirme permissoes no Windows."
        )

    cv2.namedWindow("RECONHECEDOR LIBRAS", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RECONHECEDOR LIBRAS", 960, 540)

    last_label = None
    last_conf = 0.0
    last_t = time.time()
    fps = 0.0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            print("Nao consegui ler frame da camera.")
            break
        now = time.time()
        dt = now - last_t
        last_t = now
        if dt > 0:
            # suaviza FPS para ficar estável
            inst = 1.0 / dt
            fps = inst if fps == 0.0 else (0.9 * fps + 0.1 * inst)

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)
        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        label_text = "Sem mao"
        conf = 0.0
        hands_count = len(results.hand_landmarks) if results.hand_landmarks else 0
        if results.hand_landmarks:
            for hand in results.hand_landmarks[:NUM_HANDS]:
                _draw_hand_landmarks(frame, hand)

            features = _extract_two_hands_features(results)
            pred = model.predict([features])[0]
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba([features])[0]
                conf = float(probs.max())

            last_label, last_conf = str(pred), conf
            pretty = _pretty_label(last_label)
            label_text = pretty
        else:
            last_conf = 0.0

        _draw_hud(frame, label_text, last_conf, hands_count, fps)

        cv2.imshow("RECONHECEDOR LIBRAS", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()

