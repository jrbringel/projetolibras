import os
import sys
import time
import urllib.request
from pathlib import Path

_venv_python = Path(__file__).resolve().parent / ".venv" / "Scripts" / "python.exe"
if _venv_python.exists() and Path(sys.executable).resolve() != _venv_python.resolve():
    os.execv(str(_venv_python), [str(_venv_python), *sys.argv])
import threading
from collections import Counter, deque

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from joblib import load

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = os.path.join("models", "classificador_libras_2maos.pkl")
HAND_TASK_PATH = os.path.join("models", "hand_landmarker.task")
NUM_HANDS = 2
LANDMARKS_PER_HAND = 21
FEATURES_PER_HAND = LANDMARKS_PER_HAND * 3
LOCK_THRESHOLD = 0.72
LOCK_FRAMES = 5
RELEASE_FRAMES = 6  # frames seguidos sem mao para soltar o sinal travado
# Limiares por sinal: "precisa de ajuda" permanece exigente; "te amo" (1 mao) e mais tolerante.
LOCK_THRESHOLD_BY_LABEL = {
    "precisa de ajuda": 0.70,
    "te amo": 0.55,
    "bom dia": 0.70,
    "boa noite": 0.70,
}
LOCK_FRAMES_BY_LABEL = {
    "precisa de ajuda": 3,
    "te amo": 3,
}


def _ensure_hand_model(model_path: str) -> str:
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    if os.path.exists(model_path):
        return model_path
    url = (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    )
    urllib.request.urlretrieve(url, model_path)
    return model_path


def _pretty_label(raw: str) -> str:
    s = str(raw).strip()
    if s.startswith("videos_"):
        s = s[len("videos_") :]
    s = s.replace("_", " ").strip()
    s = " ".join(p for p in s.split(" ") if p)
    return s if s else "?"


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
        elif out_right is None:
            out_right = feats
        elif out_left is None:
            out_left = feats
    return (out_right or zeros) + (out_left or zeros)


class RecognizerState:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Modelo nao encontrado: {MODEL_PATH}. Rode: python treinar.py")
        self.model = load(MODEL_PATH)
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=_ensure_hand_model(HAND_TASK_PATH)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=NUM_HANDS,
            min_hand_detection_confidence=0.62,
            min_hand_presence_confidence=0.62,
            min_tracking_confidence=0.45,
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self.ts = int(time.time() * 1000)
        self.history = deque(maxlen=12)
        self.locked_label = ""
        self.lock_streak = 0
        self.empty_streak = 0
        self.lock = threading.Lock()
        self.classes = [_pretty_label(c).lower() for c in getattr(self.model, "classes_", [])]

    def reset(self):
        self.history.clear()
        self.locked_label = ""
        self.lock_streak = 0
        self.empty_streak = 0


STATE = None
_STATE_LOCK = threading.Lock()


def _get_state():
    global STATE
    with _STATE_LOCK:
        if STATE is None:
            STATE = RecognizerState()
    return STATE


def _vote_label(history):
    filtered = [x for x in history if x and x != "sem mao"]
    if not filtered:
        return ""
    return Counter(filtered).most_common(1)[0][0]


def _is_help_sign(label: str) -> bool:
    s = (label or "").lower()
    return "precisa" in s and "ajuda" in s


def _lock_threshold(label: str) -> float:
    return float(LOCK_THRESHOLD_BY_LABEL.get((label or "").lower(), LOCK_THRESHOLD))


def _lock_frames(label: str) -> int:
    return int(LOCK_FRAMES_BY_LABEL.get((label or "").lower(), LOCK_FRAMES))


def recognize_frame(frame_rgb: np.ndarray) -> dict:
    st = _get_state()
    frame_rgb = cv2.flip(np.ascontiguousarray(frame_rgb), 1)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    # O HandLandmarker em modo VIDEO exige timestamps crescentes e nao e thread-safe.
    with st.lock:
        now_ms = int(time.time() * 1000)
        st.ts = max(st.ts + 1, now_ms)
        results = st.landmarker.detect_for_video(mp_image, st.ts)
        hands = len(results.hand_landmarks) if results.hand_landmarks else 0
        conf = 0.0
        label = ""
        if hands:
            st.empty_streak = 0
            features = _extract_two_hands_features(results)
            if hasattr(st.model, "predict_proba"):
                proba = st.model.predict_proba([features])[0]
                idx = int(np.argmax(proba))
                conf = float(proba[idx])
                label = _pretty_label(st.model.classes_[idx]).lower()
            else:
                label = _pretty_label(st.model.predict([features])[0]).lower()
            st.history.append(label if conf >= _lock_threshold(label) else "")
        else:
            st.history.append("")
            st.empty_streak += 1
            if st.empty_streak >= RELEASE_FRAMES and (st.locked_label or st.history):
                # A pessoa baixou as maos: solta o sinal para o proximo poder entrar.
                st.reset()
        stable = _vote_label(st.history)
        needed = _lock_threshold(label) if label else LOCK_THRESHOLD
        if stable and conf >= needed and label == stable:
            st.lock_streak += 1
            if st.lock_streak >= _lock_frames(stable):
                st.locked_label = stable
                st.lock_streak = 0
        else:
            st.lock_streak = 0
        # Sinal critico substitui outro sinal ja travado.
        if (
            st.locked_label
            and not _is_help_sign(st.locked_label)
            and _is_help_sign(label)
            and _is_help_sign(stable)
            and conf >= _lock_threshold(label)
        ):
            st.locked_label = stable
        shown = st.locked_label or ""
        waiting = (not shown) or hands == 0
        candidate = stable if (not shown and stable) else ""
        priority = _is_help_sign(shown) or _is_help_sign(candidate)
    return {
        "frase": "" if waiting else shown,
        "candidate": candidate,
        "confidence": round(conf, 3),
        "hands": hands,
        "waiting": waiting,
        "priority": bool(priority),
        "status": "Aguardando sinal" if waiting else "Sinal reconhecido",
    }


flask_app = Flask(__name__)
flask_app.config["TEMPLATES_AUTO_RELOAD"] = True


@flask_app.get("/")
def home():
    return render_template("index.html")


@flask_app.get("/api/health")
def api_health():
    st = _get_state()
    return jsonify({"ok": True, "sinais": st.classes, "modelo": MODEL_PATH})


@flask_app.post("/api/recognize/reset")
def api_recognize_reset():
    st = _get_state()
    with st.lock:
        st.reset()
    return jsonify({"ok": True})


@flask_app.post("/api/recognize")
def api_recognize():
    upload = request.files.get("frame")
    if upload is None:
        return jsonify({"error": "Envie um frame."}), 400
    data = np.frombuffer(upload.read(), dtype=np.uint8)
    if data.size == 0 or data.size > 6_000_000:
        return jsonify({"error": "Imagem invalida."}), 400
    image_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return jsonify({"error": "Imagem invalida."}), 400
    h, w = image_bgr.shape[:2]
    if w > 640:
        image_bgr = cv2.resize(image_bgr, (640, int(h * 640 / w)), interpolation=cv2.INTER_AREA)
    frame_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return jsonify(recognize_frame(frame_rgb))


if __name__ == "__main__":
    _get_state()
    flask_app.run(host="127.0.0.1", port=7860, debug=False, threaded=True)
