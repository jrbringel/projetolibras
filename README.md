# Projeto Libras

Reconhecedor de sinais de Libras pela câmera (MediaPipe + classificador).

## Ambiente

Use **Python 3.13** (não o 3.14).

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Como usar

1. Coletar: `python coletor.py` ou `python importar_videos.py`
2. Treinar: `python treinar.py`
3. Interface: `python app.py` (ou `python reconhecer.py`)
