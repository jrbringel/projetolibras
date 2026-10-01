# IVY LIBRAS

Totem de atendimento da Polícia Civil acessível em Libras. A câmera reconhece sinais treinados (MediaPipe + classificador scikit-learn) e a conversa guiada organiza o boletim para o policial.

Protótipo acadêmico. Não gera boletim oficial e não substitui o atendimento humano.

## Ambiente

Use **Python 3.13** (o 3.14 não tem OpenCV/MediaPipe neste projeto).

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Com o ambiente ativado, os comandos abaixo usam o Python certo. O `app.py` também se corrige sozinho: se for chamado pelo Python do sistema, ele reabre no `.venv`.

O modelo treinado (`models/classificador_libras_2maos.pkl`) já vem no repositório. Não é obrigatório treinar de novo para abrir o totem.

## Como usar

**Opção A — atalho (Windows):** clique duas vezes em `INICIAR.bat`. O navegador abre em http://127.0.0.1:7860 quando `GET /api/health` responder 200. Não feche a janela preta.

**Opção B — terminal:**

```powershell
.\.venv\Scripts\Activate.ps1
python app.py
```

Abra http://127.0.0.1:7860 → **Atendimento por câmera** ou **Atendimento assistido**.

**Outro computador / celular (mesmo Wi-Fi não é necessário):** neste PC, clique em `INICIAR_ONLINE.bat`. Use o link `https://….trycloudflare.com`. A câmera é a do aparelho que abriu o link.

No totem:

1. Escolha o tipo de atendimento.
2. No modo câmera, permita a webcam e faça um sinal treinado (*bom dia*, *boa noite*, *precisa de ajuda*, *te amo*).
3. Responda à conversa da Ivy (texto, botões ou sinais).
4. Área policial: senha de verificação do protótipo **123**.
5. **Finalizar atendimento** abre o relatório. No topo: **Gerar PDF**, **Imprimir**, **Novo atendimento**.

## Métodos da API

Estilo **REST** sobre HTTP. Base: `http://127.0.0.1:7860`. Respostas em JSON (`application/json`), exceto `GET /` (HTML). Sem autenticação na API. Não há `PUT`, `PATCH` nem `DELETE`: o boletim fica no navegador, não no servidor.

| Método | Recurso | Corpo | Função |
|---|---|---|---|
| `GET` | `/` | — | Página do totem |
| `GET` | `/api/health` | — | Serviço no ar, lista de sinais e caminho do modelo |
| `POST` | `/api/recognize` | `multipart/form-data`, campo **`frame`** (JPEG) | Reconhece o quadro e devolve o sinal |
| `POST` | `/api/recognize/reset` | — | Zera o estado do reconhecedor (ao ligar a câmera) |

Exemplos:

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

Campo `frame` = JPEG. Resposta típica:

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

O PDF do relatório **não** é REST: é gerado no cliente (jsPDF). Contrato completo: [DOCUMENTACAO.md](DOCUMENTACAO.md) (seção 7).

## Funções da ferramenta

| Comando | Função | Teclas / opções |
|---|---|---|
| `INICIAR.bat` | Sobe o totem e abre o navegador | Não feche a janela |
| `INICIAR_ONLINE.bat` | Totem local + túnel HTTPS para outro aparelho | Link em `IVY_LIBRAS_LINK.txt` na Área de Trabalho |
| `python app.py` | Servidor Flask em http://127.0.0.1:7860 | Ctrl+C encerra |
| `python coletor.py` | Grava amostras de um sinal na webcam (`dados_libras_2maos.csv`) | **S** amostra, **Q** salva e sai |
| `python importar_videos.py` | Gera amostras a partir de `videos/<nome_do_sinal>/*.mp4` | `--help`, `--max_per_video`, `--every_n_frames` |
| `python treinar.py` | Treina e salva `models/classificador_libras_2maos.pkl` | `--no_augment` desliga espelho e ruído |
| `python reconhecer.py` | Teste visual na webcam (sem o site) | **Q** sai |

Sinal novo (fora da demonstração do site): coletor ou pasta em `videos/`, depois `treinar.py`, depois **reinicie** o `app.py`. Classes com menos de 10 amostras são ignoradas.

## Como o reconhecimento funciona

- Cada quadro passa pelo MediaPipe HandLandmarker (até 2 mãos, 21 pontos por mão, 126 valores).
- `libras_features.py` deixa as coordenadas relativas ao punho e ao tamanho da mão (perto/longe, centro/canto). Essa etapa fica dentro do modelo salvo.
- O treino espelha as amostras, para a câmera do totem (imagem em espelho) coincidir com os vídeos de referência.
- O servidor só aceita um sinal depois de vários quadros estáveis e solta quando as mãos saem da imagem.

Sinais nesta versão: **bom dia**, **boa noite**, **precisa de ajuda**, **te amo**.

## Totem

- Página inicial: **Atendimento por câmera** ou **Atendimento assistido**. O órgão (Polícia Civil) é da configuração administrativa.
- A Ivy não usa IA generativa. Ela classifica o motivo (roubo, furto, ameaça…) e segue o roteiro.
- Na conversa: **voltar**, **pular** / **prefiro não informar**, **não entendi**, **corrigir local para …**.
- Relatório: PDF, impressão, e-mail, WhatsApp e **Novo atendimento** (limpa a tela).

## Documentação

- [DOCUMENTACAO.md](DOCUMENTACAO.md) — manual completo (instalação, treino, API REST).
- [DOCUMENTACAO_APRESENTACAO.md](DOCUMENTACAO_APRESENTACAO.md) — só o que entra na demonstração do site.
