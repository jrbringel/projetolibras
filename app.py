import time
import os
import urllib.request
from collections import deque, Counter

import cv2
import gradio as gr
import mediapipe as mp
import numpy as np
from joblib import load

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


MODEL_PATH = os.path.join("models", "classificador_libras_2maos.pkl")
HAND_TASK_PATH = os.path.join("models", "hand_landmarker.task")

NUM_HANDS = 2
LANDMARKS_PER_HAND = 21
FEATURES_PER_HAND = LANDMARKS_PER_HAND * 3


def _ensure_hand_model(model_path: str) -> str:
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if os.path.exists(model_path):
        return model_path
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    urllib.request.urlretrieve(url, model_path)
    return model_path


def _pretty_label(raw: str) -> str:
    s = str(raw).strip()
    if s.startswith("videos_"):
        s = s[len("videos_") :]
    s = s.replace("_", " ").strip()
    s = " ".join(p for p in s.split(" ") if p)
    return s.title() if s else "?"


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


def _draw_hand_landmarks_bgr(image_bgr, landmarks):
    h, w = image_bgr.shape[:2]
    for lm in landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        cv2.circle(image_bgr, (x, y), 3, (0, 255, 0), -1)


class RecognizerState:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Modelo nao encontrado: {MODEL_PATH}. Rode: python treinar.py")
        self.model = load(MODEL_PATH)

        task_path = _ensure_hand_model(HAND_TASK_PATH)
        base_options = mp_python.BaseOptions(model_asset_path=task_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=NUM_HANDS,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(options)

        self.ts = int(time.time() * 1000)
        self.history = deque(maxlen=12)
        self.locked_label = "Sem mao"
        self.lock_streak = 0

    def close(self):
        try:
            self.landmarker.close()
        except Exception:
            pass


STATE = None


def _get_state():
    global STATE
    if STATE is None:
        STATE = RecognizerState()
    return STATE


def _vote_label(history):
    if not history:
        return None
    c = Counter(history)
    return c.most_common(1)[0][0]


def _render_pills(items):
    pills = []
    for it in items:
        safe = str(it).replace("<", "&lt;").replace(">", "&gt;")
        cls = "pill dim" if safe.lower() == "sem mao" else "pill"
        pills.append(f"<span class='{cls}'>{safe}</span>")
    return "<div class='pills'>" + "".join(pills) + "</div>"


def predict(frame_rgb: np.ndarray, enable_lock: bool, lock_threshold: float, lock_frames: int):
    st = _get_state()
    if frame_rgb is None:
        return None, "Sem imagem", 0.0, 0, "", ""

    # Gradio fornece RGB (H,W,3)
    frame_rgb = np.asarray(frame_rgb)
    if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
        return frame_rgb, "Frame invalido"

    # espelha para ficar natural
    frame_rgb = cv2.flip(frame_rgb, 1)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    st.ts += 33  # monotonic crescente (aprox 30fps)
    results = st.landmarker.detect_for_video(mp_image, st.ts)

    hands_count = len(results.hand_landmarks) if results.hand_landmarks else 0
    label = "Sem mao"
    conf = 0.0

    if results.hand_landmarks:
        for hand in results.hand_landmarks[:NUM_HANDS]:
            _draw_hand_landmarks_bgr(frame_bgr, hand)

        features = _extract_two_hands_features(results)
        pred_label = st.model.predict([features])[0]
        label = _pretty_label(pred_label)

        if hasattr(st.model, "predict_proba"):
            probs = st.model.predict_proba([features])[0]
            conf = float(np.max(probs))

        st.history.append(label)
    else:
        st.history.append("Sem mao")

    stable = _vote_label([x for x in st.history if x != "Sem mao"]) or "Sem mao"
    text = f"Sinal: {stable}"

    # Lock label: só atualiza quando tiver confiança alta por N frames seguidos
    lock_threshold = float(lock_threshold or 0.0)
    lock_frames = int(lock_frames or 1)
    if lock_frames < 1:
        lock_frames = 1

    if enable_lock and stable != "Sem mao":
        if conf >= lock_threshold:
            st.lock_streak += 1
        else:
            st.lock_streak = 0

        if st.lock_streak >= lock_frames:
            st.locked_label = stable
            st.lock_streak = 0
    else:
        # sem lock: acompanha o estável
        st.locked_label = stable
        st.lock_streak = 0

    # HUD simples (o resto fica na UI do Gradio)
    cv2.rectangle(frame_bgr, (10, 10), (10 + 520, 10 + 50), (0, 0, 0), -1)
    cv2.addWeighted(frame_bgr, 0.85, frame_bgr, 0.15, 0, frame_bgr)
    cv2.putText(frame_bgr, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    out_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    shown = st.locked_label if enable_lock else stable
    pretty = shown if shown == "Sem mao" else _pretty_label(shown)
    history = list(st.history)[-8:][::-1]
    history_html = _render_pills(history)
    conf_bar = f"<div class='bar'><div class='barFill' style='width:{int(conf*100)}%'></div></div>"
    return out_rgb, pretty, conf, hands_count, history_html, conf_bar


CSS = """
body {
  background: radial-gradient(1200px 700px at 20% 0%, rgba(0,180,255,.18), transparent 60%),
              radial-gradient(900px 500px at 80% 20%, rgba(120,80,255,.12), transparent 60%),
              linear-gradient(180deg, #050816 0%, #050816 55%, #03040d 100%) !important;
}
.container { max-width: 1200px !important; }
.hero h2 { margin-bottom: 0.25rem; }
.hero p { margin-top: 0.35rem; opacity: .85; }
.cardish {
  border: 1px solid rgba(120, 200, 255, 0.22) !important;
  background: linear-gradient(180deg, rgba(10,18,40,.72), rgba(6,10,25,.72)) !important;
  box-shadow: 0 0 0 1px rgba(0, 255, 255, 0.06), 0 18px 60px rgba(0,0,0,.55) !important;
  border-radius: 18px !important;
}
.labelBig { font-size: 1.35rem; font-weight: 700; color: #e8fbff; letter-spacing: .2px; }
.subtle { opacity: .85; }
.pills { display:flex; flex-wrap:wrap; gap:8px; }
.pill {
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid rgba(0, 200, 255, .30);
  background: rgba(0, 180, 255, .10);
  color: #d9f6ff;
  font-weight: 600;
  font-size: .9rem;
}
.pill.dim {
  border-color: rgba(160,160,160,.22);
  background: rgba(255,255,255,.06);
  color: rgba(255,255,255,.75);
}
.bar {
  height: 14px;
  border-radius: 999px;
  background: rgba(255,255,255,.08);
  border: 1px solid rgba(120,200,255,.22);
  overflow: hidden;
}
.barFill {
  height: 100%;
  background: linear-gradient(90deg, #00b4ff 0%, #18e3ff 45%, #7c5cff 100%);
  box-shadow: 0 0 22px rgba(24,227,255,.32);
  transition: width 140ms linear;
}
"""

theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="cyan",
    neutral_hue="slate",
)

with gr.Blocks(title="Reconhecedor Libras", theme=theme, css=CSS) as demo:
    gr.HTML(
        "<div class='hero'>"
        "<h2>Reconhecedor LIBRAS</h2>"
        "<p>Azul futurista • Webcam no navegador • Estável por votação de frames</p>"
        "</div>"
    )

    with gr.Row():
        with gr.Column(scale=2):
            cam = gr.Image(sources=["webcam"], type="numpy", label="Webcam", streaming=True, elem_classes=["cardish"])
            out = gr.Image(type="numpy", label="Saída (landmarks)", elem_classes=["cardish"])

        with gr.Column(scale=1):
            gr.HTML("<div class='cardish' style='padding:16px'>", visible=True)
            label_box = gr.Textbox(label="Sinal reconhecido", interactive=False)
            conf_slider = gr.Slider(0, 1, value=0, step=0.01, label="Confiança", interactive=False)
            hands_box = gr.Number(label="Mãos detectadas", interactive=False)
            conf_bar = gr.HTML(label="")
            history_html = gr.HTML(label="")
            gr.Markdown("### Estabilização (lock)")
            enable_lock = gr.Checkbox(value=True, label="Travar label (evita ficar piscando)")
            lock_threshold = gr.Slider(0, 1, value=0.75, step=0.01, label="Confiança mínima p/ trocar")
            lock_frames = gr.Slider(1, 30, value=8, step=1, label="Frames seguidos p/ trocar")
            gr.HTML("</div>", visible=True)

    cam.stream(
        fn=predict,
        inputs=[cam, enable_lock, lock_threshold, lock_frames],
        outputs=[out, label_box, conf_slider, hands_box, history_html, conf_bar],
    )

if __name__ == "__main__":
    demo.launch()

