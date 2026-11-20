import argparse
import os
import random
import time
import urllib.request

import cv2
import mediapipe as mp
import pandas as pd

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

NUM_HANDS = 2
LANDMARKS_PER_HAND = 21
FEATURES_PER_HAND = LANDMARKS_PER_HAND * 3


def _ensure_hand_model(model_path: str) -> str:
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if os.path.exists(model_path):
        return model_path

    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    print("Baixando modelo do MediaPipe (primeira vez)...")
    urllib.request.urlretrieve(url, model_path)
    return model_path


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

    # Ordem fixa: Right depois Left
    return out_right + out_left


def _iter_video_files(root_dir: str):
    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in exts:
                yield os.path.join(dirpath, fn)


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Importa videos e gera amostras (landmarks) em CSV no formato de 2 maos.\n\n"
            "Estrutura recomendada:\n"
            "  videos/bom_dia/1.mp4\n"
            "  videos/boa_noite/2.mp4\n"
        )
    )
    ap.add_argument("--videos_dir", default="videos", help="Pasta raiz com subpastas por label.")
    ap.add_argument("--out_csv", default="dados_libras_2maos.csv", help="CSV de saida (append).")
    ap.add_argument("--every_n_frames", type=int, default=3, help="Coletar 1 amostra a cada N frames.")
    ap.add_argument("--min_hands", type=int, default=1, help="Minimo de maos detectadas para salvar amostra.")
    ap.add_argument("--max_per_video", type=int, default=500, help="Limite de amostras por video.")
    ap.add_argument(
        "--repeat_until_max",
        action="store_true",
        help="Se um video gerar poucas amostras, faz novas passadas ate tentar chegar no max_per_video.",
    )
    ap.add_argument(
        "--max_passes_per_video",
        type=int,
        default=5,
        help="Numero maximo de passadas por video quando --repeat_until_max estiver ativo.",
    )
    ap.add_argument("--show", action="store_true", help="Mostra preview enquanto importa (mais lento).")
    args = ap.parse_args()

    if not os.path.isdir(args.videos_dir):
        raise FileNotFoundError(
            f"Nao encontrei a pasta '{args.videos_dir}'. Crie, por exemplo: videos/bom_dia/..."
        )

    task_path = _ensure_hand_model(os.path.join("models", "hand_landmarker.task"))
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

    rows = []
    total = 0
    global_ts = int(time.time() * 1000) - 1
    videos = list(_iter_video_files(args.videos_dir))
    if not videos:
        raise FileNotFoundError(f"Nenhum video encontrado em '{args.videos_dir}'.")

    for path in videos:
        label = os.path.basename(os.path.dirname(path)).strip()
        if not label:
            continue

        per_video = 0
        print(f"Importando: {path} | label={label}")

        passes = 0
        while True:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                print(f"Falhou ao abrir: {path}")
                break

            frame_idx = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if args.repeat_until_max and total_frames > 0:
                # Comeca em pontos diferentes para variar amostras por passada
                start_frame = int((passes / max(args.max_passes_per_video, 1)) * total_frames)
                jitter = int(total_frames * 0.05)
                start_frame = max(0, min(total_frames - 1, start_frame + random.randint(-jitter, jitter)))
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break

                frame_idx += 1
                if args.every_n_frames > 1 and (frame_idx % args.every_n_frames) != 0:
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                # timestamp precisa ser monotonicamente crescente em TODAS as chamadas do landmarker
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                if fps <= 0:
                    fps = 30.0
                step_ms = max(1, int(1000.0 / fps))
                global_ts += step_ms
                timestamp_ms = global_ts
                results = landmarker.detect_for_video(mp_image, timestamp_ms)

                hands_count = len(results.hand_landmarks) if results.hand_landmarks else 0
                if hands_count < args.min_hands:
                    if args.show:
                        cv2.imshow("IMPORTAR", frame)
                        if (cv2.waitKey(1) & 0xFF) == ord("q"):
                            break
                    continue

                feats = _extract_two_hands_features(results)
                rows.append(feats + [label])
                per_video += 1
                total += 1

                if args.show:
                    cv2.putText(
                        frame,
                        f"{label} | +{per_video}",
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow("IMPORTAR", frame)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        break

                if per_video >= args.max_per_video:
                    break

            cap.release()
            if args.show:
                cv2.destroyWindow("IMPORTAR")

            if not args.repeat_until_max:
                break

            passes += 1
            if per_video >= args.max_per_video:
                break
            if passes >= max(1, args.max_passes_per_video):
                break

        print(f"  amostras desse video: {per_video}")

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(
            args.out_csv,
            mode="a",
            header=not os.path.exists(args.out_csv),
            index=False,
        )
        print(f"OK: {len(rows)} novas amostras em '{args.out_csv}' (total nessa importacao: {total})")
    else:
        print("Nenhuma amostra foi gerada (talvez maos nao detectadas).")

    landmarker.close()


if __name__ == "__main__":
    main()

