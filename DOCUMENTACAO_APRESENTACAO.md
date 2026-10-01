# IVY LIBRAS — Documentação para apresentação

**Totem de atendimento da Polícia Civil acessível em Libras**  
Protótipo acadêmico. Não é documento policial oficial e não possui validade de registro.

Este texto cobre **somente o que entra na demonstração do site**. Coleta de amostras, importação de vídeos e retreino do classificador ficam de fora (podem ser mostrados em vídeo gravado).

| Item | Valor |
|---|---|
| Nome | IVY LIBRAS |
| Órgão | Polícia Civil |
| Como iniciar na banca | Clique em `INICIAR.bat` (neste PC) **ou** `INICIAR_ONLINE.bat` (link para outro notebook) |
| Endereço local | http://127.0.0.1:7860 |
| Estilo da API | REST sobre HTTP (JSON + `multipart/form-data`) |
| Servidor | Flask 3, Python 3.13 |
| Data | 30/09/2026 |

---

## 1. Como ligar o sistema na apresentação

Faça isto **antes de chegar na banca**, neste computador, com internet se for usar o notebook emprestado.

### 1.1 Demonstração neste PC (projetor / HDMI)

1. Abra a pasta do projeto.
2. Clique duas vezes em **`INICIAR.bat`**.
3. Espere o Chrome ou o Edge abrir em `http://127.0.0.1:7860`.
4. **Não feche a janela preta** enquanto apresentar.
5. Para encerrar: feche a janela ou `Ctrl+C`.

O `.bat` sobe o servidor Flask e só abre o navegador depois que `GET /api/health` responde **200**.

### 1.2 Demonstração no notebook emprestado (câmera de lá)

1. Clique em **`INICIAR_ONLINE.bat`**.
2. Confira neste PC: `http://127.0.0.1:7860`.
3. Copie o link `https://….trycloudflare.com` (Desktop → `IVY_LIBRAS_LINK.txt`).
4. No outro notebook: Chrome ou Edge, **https**, permitir a câmera.
5. Deixe este PC ligado, tampa aberta, janela do `.bat` aberta.

A câmera usada é a **do aparelho que abriu o link**, não a deste PC.

---

## 2. O que será mostrado no site

1. **Início** — atendimento por câmera (Libras) ou assistido (texto).
2. **Conversa com a Ivy** — perguntas guiadas; dados ditos no chat preenchem o formulário.
3. **Câmera** (modo Libras) — sinais já treinados: *bom dia*, *boa noite*, *precisa de ajuda*, *te amo*.
4. **Área policial** — senha de verificação do protótipo: **123**. A pessoa atendida não entra sem senha.
5. **Relatório** — no topo: **Gerar PDF**, **Imprimir**, **Novo atendimento** (zera o histórico da tela).

**Finalizar atendimento** grava o horário de encerramento e abre o relatório. **Novo atendimento** apaga conversa e dados da tela (PDF já baixado permanece no computador).

A Ivy **não** usa modelo generativo (não é ChatGPT). As falas são roteiro contextual.

---

## 3. API REST — métodos de comunicação

O professor pediu os **métodos HTTP** usados. O totem é um serviço REST **local** (e, se o túnel estiver ligado, o mesmo contrato atrás de HTTPS).

Não há CRUD de ocorrência no servidor: o boletim vive no **navegador**. A API existe para **saúde do serviço** e **reconhecimento de um quadro**.

**Base:** `http://127.0.0.1:7860`  
**Formato das respostas:** `application/json`  
**Autenticação da API:** nenhuma (protótipo). A senha **123** é só da tela “Área policial”, no JavaScript.

### 3.1 Tabela de recursos

| Método HTTP | Recurso | Corpo da requisição | Resposta de sucesso | Erros |
|---|---|---|---|---|
| **GET** | `/` | — | HTML do totem (`text/html`) | — |
| **GET** | `/api/health` | — | JSON: modelo carregado e lista de sinais | — |
| **POST** | `/api/recognize` | `multipart/form-data`, campo **`frame`** (JPEG) | JSON do reconhecimento | **400** se faltar arquivo ou a imagem for inválida |
| **POST** | `/api/recognize/reset` | — | JSON `{ "ok": true }` | — |

Não são usados **PUT**, **PATCH** nem **DELETE**: não há recurso persistido de atendimento para alterar ou apagar no servidor.

Por que **GET** em `/api/health`: leitura idempotente, sem efeito colateral, adequada a monitoramento (o `.bat` usa isso para saber se o site já pode abrir).

Por que **POST** em `/api/recognize`: envia um quadro novo; o servidor atualiza o estado temporal do MediaPipe (modo VIDEO) e pode devolver um sinal travado. Não é idempotente.

### 3.2 `GET /api/health`

Verifica se o classificador e o HandLandmarker subiram.

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

### 3.3 `POST /api/recognize`

O JavaScript da câmera envia um JPEG a cada ~280 ms (`FormData`, campo `frame`).

```http
POST /api/recognize HTTP/1.1
Host: 127.0.0.1:7860
Content-Type: multipart/form-data; boundary=...
```

Resposta típica com sinal confirmado:

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

| Campo | Significado |
|---|---|
| `frase` | Sinal estável (vazio se ainda estiver esperando) |
| `candidate` | Hipótese enquanto não trava |
| `confidence` | Probabilidade calibrada (0–1) |
| `hands` | Quantidade de mãos no quadro |
| `waiting` | `true` se não há sinal para entrar na conversa |
| `priority` | `true` no sinal crítico *precisa de ajuda* |

HTTP **400**: `{ "error": "Envie um frame." }` ou `{ "error": "Imagem invalida." }`.

### 3.4 `POST /api/recognize/reset`

Disparado ao **ligar a câmera**, para limpar a trava do sinal anterior.

```json
{ "ok": true }
```

### 3.5 Fluxo de comunicação (o que a banca precisa entender)

```
Navegador (câmera getUserMedia)
    → POST /api/recognize  (JPEG, multipart)
    → Flask (app.py)
    → MediaPipe HandLandmarker  (até 2 mãos, 21 pontos, xyz)
    → vetor de 126 valores + classificador SVM (arquivo .pkl)
    → JSON { frase, confidence, priority, ... }
    → interface preenche conversa e relatório
```

O boletim, o PDF e o painel policial **não** passam por REST: são estado e arquivos no cliente (JavaScript / jsPDF).

---

## 4. Termos técnicos usados na fala

| Termo | Em uma frase, na apresentação |
|---|---|
| **REST** | Estilo de API em que cada URL é um recurso e o verbo HTTP diz a operação (GET lê, POST envia um quadro). |
| **HTTP** | Protocolo de comunicação entre o navegador e o Flask. |
| **GET / POST** | Métodos: GET consulta; POST envia o frame da câmera. |
| **JSON** | Formato texto das respostas da API (`frase`, `confidence`, …). |
| **multipart/form-data** | Como o JPEG vai no POST (`frame`). |
| **Endpoint / rota** | Caminho da API, por exemplo `/api/recognize`. |
| **Flask** | Servidor web em Python que entrega o HTML e a API. |
| **MediaPipe** | Detecta as mãos e os pontos (landmarks) no vídeo. |
| **SVM** | Classificador que, a partir dos pontos, escolhe o sinal (*bom dia*, *precisa de ajuda*, …). |
| **Calibração de probabilidade** | A `confidence` não é um chute cru: vem calibrada para decidir quando travar o sinal. |
| **Cliente / servidor** | Cliente = navegador; servidor = `app.py` neste PC. |
| **getUserMedia** | API do navegador que liga a webcam no aparelho de quem abriu o site. |
| **Protótipo acadêmico** | Organiza a comunicação; não substitui boletim oficial. |

---

## 5. Sinais que a câmera reconhece nesta versão

Já estão no modelo carregado (não será treinado ao vivo):

- **bom dia** / **boa noite** — cumprimento; não apagam o boletim.
- **preciso de ajuda** — prioridade: alerta no painel policial.
- **te amo** — registrado na conversa e no relatório.

Nome, CPF, local e demais campos entram pelo **chat**, pelos **botões** ou pelo **formulário** ao lado.

---

## 6. Aviso

Software de **protótipo acadêmico**. Organiza atendimento acessível em Libras. Não gera registro policial oficial e não possui validade jurídica de boletim de ocorrência.
