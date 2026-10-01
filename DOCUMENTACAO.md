# IVY LIBRAS — Documentação do software

**Totem de atendimento policial acessível em Libras**  
Protótipo acadêmico. Não é documento policial oficial e não possui validade de registro.

| Item | Valor |
|---|---|
| Nome | IVY LIBRAS |
| Órgão nesta versão | Polícia Civil |
| Endereço local | http://127.0.0.1:7860 |
| Linguagem | Python 3.13 |
| Interface | Flask + HTML/CSS/JavaScript (Bootstrap 5 e Bootstrap Icons) |
| Reconhecimento | MediaPipe HandLandmarker + classificador SVM (scikit-learn) |
| Data desta documentação | 30/09/2026 |

---

## 1. O que o sistema faz

O IVY LIBRAS organiza um atendimento da Polícia Civil para pessoas que se comunicam em Libras ou por texto.

1. A pessoa escolhe **atendimento por câmera** ou **atendimento assistido**.
2. A Ivy conduz uma conversa guiada (participação, segurança, fato, local, identificação).
3. No modo câmera, sinais treinados entram na conversa e no relatório.
4. O policial acompanha em tempo real no painel.
5. O boletim pode ser impresso, gerado em PDF ou compartilhado.

A Ivy **não** usa inteligência artificial generativa. As respostas são roteiro contextual: ela ecoa o que foi dito, classifica o motivo (roubo, furto, ameaça etc.) e segue o formulário. O registro oficial continua com o profissional.

---

## 2. Como abrir o terminal e o ambiente

Todos os comandos abaixo são digitados no **PowerShell**, na pasta do projeto.

No Cursor: **Terminal → New Terminal** (`Ctrl+'`).

```powershell
cd C:\Users\junior\Documents\projetoLibras
.\.venv\Scripts\Activate.ps1
```

O prompt deve mostrar `(.venv)` no início. Sem isso, o Python do sistema (3.14) não tem OpenCV/MediaPipe e o programa falha.

Se o PowerShell bloquear o `Activate.ps1`:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Alternativa sem ativar o venv:

```powershell
.\.venv\Scripts\python.exe app.py
```

O `app.py` também se corrige sozinho: se for chamado pelo Python do sistema, ele reabre no `.venv`.

---

## 3. Instalação (primeira vez)

Use **Python 3.13**. O 3.14 não é suportado neste projeto.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Dependências principais (`requirements.txt`): MediaPipe, OpenCV, pandas, NumPy, scikit-learn, joblib, Flask.

---

## 4. Comandos do reconhecimento (Libras)

| Comando | Função | Teclas / opções |
|---|---|---|
| `python coletor.py` | Coleta amostras pela webcam e grava em `dados_libras_2maos.csv` | Pede o nome do sinal. **S** grava uma amostra. **Q** salva e sai |
| `python importar_videos.py` | Lê vídeos em `videos/<nome_do_sinal>/` e acrescenta amostras no CSV | ver opções abaixo |
| `python treinar.py` | Treina o classificador e salva `models/classificador_libras_2maos.pkl` | mostra acurácia no terminal |
| `python reconhecer.py` | Janela de teste: mostra o sinal reconhecido na webcam | **Q** sai |
| `python app.py` | Sobe o totem | **Ctrl+C** encerra |

### 4.1 Sinal novo (fluxo completo)

1. Crie a pasta `videos/nome_do_sinal/` e coloque o(s) `.mp4` **ou** rode `python coletor.py`.
2. `python importar_videos.py` (se usou vídeo).
3. `python treinar.py`
4. Encerre o totem (Ctrl+C) e rode de novo `python app.py`.
5. Recarregue http://127.0.0.1:7860

O nome da pasta vira o rótulo do sinal (`te_amo` → **te amo**).

### 4.2 `importar_videos.py`

```powershell
python importar_videos.py --help
python importar_videos.py
python importar_videos.py --videos_dir videos\te_amo
python importar_videos.py --every_n_frames 1 --max_per_video 800 --repeat_until_max --max_passes_per_video 10
python importar_videos.py --show
```

| Opção | Padrão | Significado |
|---|---|---|
| `--videos_dir` | `videos` | Pasta raiz; cada subpasta é um sinal |
| `--out_csv` | `dados_libras_2maos.csv` | CSV de saída (acrescenta, não apaga) |
| `--every_n_frames` | 3 | 1 amostra a cada N quadros |
| `--min_hands` | 1 | Mínimo de mãos detectadas para gravar |
| `--max_per_video` | 500 | Limite de amostras por vídeo |
| `--repeat_until_max` | desligado | Várias passadas até tentar chegar no limite |
| `--max_passes_per_video` | 5 | Passadas quando `--repeat_until_max` está ativo |
| `--show` | desligado | Preview na tela (mais lento) |

### 4.3 `treinar.py`

```powershell
python treinar.py
python treinar.py --csv dados_libras_2maos.csv
python treinar.py --out models\classificador_libras_2maos.pkl
python treinar.py --no_augment
```

`--no_augment` desliga o espelhamento e o ruído usados no treino.

Classes com menos de 10 amostras são ignoradas (evita ruído no modelo).

### 4.4 Roteiro para demonstração

```powershell
cd C:\Users\junior\Documents\projetoLibras
.\.venv\Scripts\Activate.ps1
python app.py
```

Abrir **http://127.0.0.1:7860** → Atendimento por câmera → Começar → Ligar câmera.

---

## 5. Sinais treinados nesta versão

| Sinal (na tela) | Pasta de vídeo |
|---|---|
| bom dia | `videos/bom_dia` |
| boa noite | `videos/boa_noite` |
| precisa de ajuda | `videos/precisa_de_ajuda` |
| te amo | `videos/te_amo` |

O que cada sinal faz no totem:

- **bom dia / boa noite** — cumprimento na conversa e no relatório; a pergunta atual não muda.
- **precisa de ajuda** — marca “precisa de ajuda = Sim” no painel e no PDF.
- **te amo** — registrado na conversa e no relatório; a pergunta atual não muda.

Nome, fato, local e demais campos do boletim **não** vêm da câmera. Continuam pelos botões e pelo texto. O widget VLibras traduz as perguntas da Ivy.

---

## 6. Funcionalidades do totem

### 6.1 Telas

| Tela | Função |
|---|---|
| Início | Escolher tipo de atendimento |
| Atendimento | Conversa com a Ivy, câmera (se Libras), dados ao vivo |
| Painel do policial | Mesma conversa + resumo; o policial pode enviar uma pergunta |
| Relatório | Boletim montado, impressão, PDF e compartilhamento |
| Configuração | Órgão do totem (somente administrador) |

### 6.2 Tipos de atendimento

- **Por câmera** — webcam do PC, USB ou câmera frontal do celular (`facingMode: user`, com fallback). Sinais treinados + botões + texto.
- **Assistido** — conversa e formulário, sem câmera.

O órgão **não** é escolhido na página inicial. Nesta versão é **Polícia Civil**, definido em Configuração. Outros órgãos aparecem bloqueados.

### 6.3 Conversa guiada (Ivy)

Ordem típica: participação → segurança → relato → motivo → perguntas da categoria → local → quando → pessoas → testemunhas → provas → identificação (nome, CPF, nascimento, naturalidade, estado civil, profissão, telefone, endereço) → confirmação.

A Ivy classifica o motivo a partir do texto (roubo, furto, violência doméstica, ameaça, injúria, agressão, fraude/PIX, dano, perda/extravio ou outro) e faz perguntas específicas da categoria.

Frases úteis na conversa:

| O que escrever | Efeito |
|---|---|
| `não entendi` / `repete` | Repete a pergunta atual |
| `voltar` | Volta à pergunta anterior |
| `pular` / `não sei` / `prefiro não informar` | Deixa campo opcional em branco |
| `corrigir local para Rua 10` | Altera um dado |
| `cancelar` (na correção) | Mantém o resumo |
| `Ver relatório` / `Novo atendimento` | Depois de confirmar |

Validações: CPF (11 dígitos e dígito verificador), telefone com DDD, data de nascimento (`12 de maio de 1990`, `05/03/90`), nome com sobrenome. “Ontem”, “hoje” e “anteontem” viram data no relatório.

Limites de campo (não quebram o layout): nome 80, relato 700, chat 500, CPF 14, telefone 15, data 10, endereço 140.

### 6.4 Câmera

- Ligar / desligar câmera.
- Gravar sinalização (vídeo da sessão, para revisão).
- Cada sinal confirmado entra na conversa **e** na seção **Sinais reconhecidos pela câmera** (horário e confiança).
- O sinal só fecha depois de quadros estáveis (confiança ≥ 0,72). Quando as mãos saem da imagem, o sinal solta para o próximo poder entrar.

### 6.5 Painel do policial

Acompanha a conversa sem substituir o atendimento humano. Pergunta digitada pelo policial aparece na tela da pessoa e pode ser traduzida pelo VLibras.

### 6.6 Relatório e PDF

O relatório reúne identificação, fato, envolvidos cadastrados, narrativa e sinais da câmera.

Botões:

- **Gerar PDF** (também no painel de dados e na caixa da câmera)
- **Compartilhar PDF**
- Imprimir
- E-mail / WhatsApp (abrem o aplicativo; o PDF precisa ser anexado)
- **Novo atendimento** (limpa a tela; pede confirmação se o atendimento não foi concluído)

O PDF é gerado no navegador (biblioteca jsPDF). Não existe endpoint de relatório no servidor.

O atendimento **não** é gravado automaticamente no disco. Recarregar a página apaga a sessão, salvo o PDF já baixado. Apenas o órgão (`ivy_org`) fica no `localStorage`.

### 6.7 Pessoas envolvidas

Cadastro adicional ilimitado (vítima, testemunha, solicitante, autor/suspeito), com editar e remover. Entra no relatório e no PDF.

---

## 7. API REST (métodos HTTP)

Contrato usado pelo totem. Estilo **REST** sobre HTTP: cada caminho é um recurso; o **método** indica a operação.  
Base: `http://127.0.0.1:7860`  
Respostas da API: `application/json; charset=utf-8`  
Autenticação da API: **não há**. Não existe CRUD de ocorrência no servidor (o boletim fica no navegador).

Não são expostos **PUT**, **PATCH** nem **DELETE**.

| Método | Recurso | Corpo | Sucesso | Falha |
|---|---|---|---|---|
| **GET** | `/` | — | HTML do totem | — |
| **GET** | `/api/health` | — | 200 JSON (`ok`, `sinais`, `modelo`) | — |
| **POST** | `/api/recognize` | `multipart/form-data`, campo `frame` (JPEG) | 200 JSON do reconhecimento | **400** JSON `error` |
| **POST** | `/api/recognize/reset` | — | 200 `{ "ok": true }` | — |

**GET** em `/api/health` é leitura idempotente (o `INICIAR.bat` espera HTTP 200 para abrir o navegador).  
**POST** em `/api/recognize` envia um quadro; o estado interno do HandLandmarker (modo VIDEO) avança — não é idempotente.

### 7.1 Exemplos

```http
GET /api/health HTTP/1.1
Host: 127.0.0.1:7860
```

```json
{
  "ok": true,
  "sinais": ["boa noite", "bom dia", "precisa de ajuda", "te amo"],
  "modelo": "models\\classificador_libras_2maos.pkl"
}
```

```http
POST /api/recognize HTTP/1.1
Host: 127.0.0.1:7860
Content-Type: multipart/form-data
```

Campo: `frame` = arquivo JPEG. Resposta:

```json
{
  "frase": "precisa de ajuda",
  "candidate": "",
  "confidence": 0.94,
  "hands": 2,
  "waiting": false,
  "priority": true,
  "status": "Sinal reconhecido"
}
```

`POST /api/recognize/reset` zera a trava do sinal (chamado ao ligar a câmera).

O PDF do relatório **não** tem endpoint REST: é gerado no cliente (jsPDF).

---

## 8. Arquitetura (resumo técnico)

```
Webcam / vídeo
    → MediaPipe HandLandmarker (até 2 mãos, 21 pontos, xyz)
    → 126 valores brutos (mão direita + mão esquerda)
    → libras_features.py (posição e escala relativas ao punho; etapa dentro do .pkl)
    → SVM + probabilidade calibrada
    → frase na conversa e no relatório
```

Arquivos principais:

| Arquivo | Papel |
|---|---|
| `app.py` | Servidor Flask e API de reconhecimento |
| `templates/index.html` | Toda a interface e a máquina de estados do boletim |
| `coletor.py` | Coleta pela webcam |
| `importar_videos.py` | Coleta a partir de vídeos |
| `treinar.py` | Treino do classificador |
| `reconhecer.py` | Teste visual na webcam |
| `libras_features.py` | Normalização das landmarks (usada no modelo salvo) |
| `dados_libras_2maos.csv` | Amostras (features + rótulo) |
| `models/classificador_libras_2maos.pkl` | Modelo treinado |
| `models/hand_landmarker.task` | Modelo MediaPipe (baixado na primeira execução) |
| `videos/<sinal>/` | Vídeos de treino |

`dados_libras.csv` e `models/classificador_libras.pkl` (se existirem) são da versão antiga de uma mão e **não** são usados pelo totem atual.

---

## 9. Limitações conhecidas

- Protótipo acadêmico; o boletim gerado não substitui o registro oficial.
- Vocabulário da câmera limitado aos sinais treinados.
- Sem IA generativa: a conversa segue um roteiro.
- Sem persistência de atendimento no servidor.
- API local, sem login e sem HTTPS.
- sklearn pode avisar diferença de versão ao carregar o `.pkl`; o reconhecimento continua funcionando.
- O totem precisa ser reiniciado depois de `python treinar.py` para carregar o modelo novo.

---

## 10. Aviso legal

Este software é um **protótipo acadêmico**. Organiza a comunicação e a coleta de informações. Não gera boletim de ocorrência oficial, não substitui o atendimento humano e não possui validade de registro policial.
