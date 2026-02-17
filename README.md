# 💢 É ele ali? (Vasco Detector)

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![YOLO](https://img.shields.io/badge/YOLO-Ultralytics-green)
![SAHI](https://img.shields.io/badge/SAHI-Sliced%20Inference-orange)
![Status](https://img.shields.io/badge/Status-Em%20Desenvolvimento-yellow)

Detector em tempo real da **Cruz de Malta** (Vasco da Gama) que monitora a tela do computador, utilizando Inteligência Artificial para identificar até os menores escudos em vídeos ou jogos.

O projeto combina a velocidade do **YOLO (Ultralytics)** com a precisão do **SAHI (Sliced Inference)**, permitindo detectar pequenos objetos que passariam despercebidos em um redimensionamento padrão.

## ✨ Funcionalidades

- 🚀 **Inferência Fatiada (SAHI):** Divide a tela em blocos (slices) para detectar escudos pequenos com "zoom" digital.
- 📸 **Captura Otimizada:** Usa `mss` para capturar a tela com baixa latência.
- 🔊 **Alerta Sonoro:** Toca o hino quando a cruz é detectada.
- 💾 **Log Visual:** Salva automaticamente os frames com detecções na pasta `found/`.
- 🧠 **Treinamento Personalizado:** Scripts prontos para baixar datasets do Roboflow e treinar novos modelos.

## 📸 Demonstração

## 🧱 Estrutura do Projeto

```plaintext
├── main.py            # 🎮 Script principal (Detecção em tempo real + SAHI)
├── treinar.py         # 🏋️ Script de treinamento (Download Roboflow + YOLO)
├── requirements.txt   # 📦 Lista de dependências
├── .env               # 🔑 Chaves de API (Não comitar!)
├── hino.mp3           # 🎵 Arquivo de áudio do hino
├── runs/              # 📈 Resultados dos treinos (Gráficos e Pesos .pt)
└── found/             # 🖼️ Imagens salvas das detecções (Limpa a cada execução)
```

## ⚙️ Instalação

1. Pré-requisitos
   - Python 3.10 ou superior.
   - Recomendado: Placa de vídeo NVIDIA com drivers atualizados (para CUDA).
   - Conta no Roboflow para gerenciar o dataset.

2. Configuração do Ambiente
   Clone o repositório e instale as dependências:

```Bash
# Crie um ambiente virtual
python -m venv venv

# Ative o ambiente (Windows)
.\venv\Scripts\activate

# Instale as dependências
pip install -r requirements.txt
```

> Nota sobre GPU: Se você possui placa NVIDIA, certifique-se de instalar a versão do PyTorch compatível com CUDA para ter performance em tempo real.

3. Variáveis de Ambiente
   Crie um arquivo .env na raiz do projeto e adicione sua chave do Roboflow:

```Ini, TOML
API_KEY=sua_chave_roboflow_aqui
```

## 🚀 Como Usar

1. Treinando o Modelo
   Se você ainda não tem um modelo .pt ou quer melhorar a detecção:
   1. Configure o dataset no Roboflow (Recomendado: Sem Grayscale, com Mosaic ativado).

   2. Execute o script de treino:

   ```Bash
   python treinar.py
   ```

   3. O melhor modelo será salvo em runs/detect/train/weights/best.pt.

2. Rodando a Detecção
   1. Verifique se o caminho do modelo em main.py aponta para o seu arquivo .pt:

   ```Python
   model_path='runs/detect/train/weights/best.pt'
   ```

   2. Execute o detector:

   ```Bash
   python main.py
   ```

## 🐳 Rodando com Docker

Se você não quiser instalar Python ou CUDA no seu sistema, pode usar o Docker para isolar o ambiente. Isso é **altamente recomendado para a etapa de treinamento**.

### Pré-requisitos

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado.
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (para usar a GPU no Docker).

### Como usar

1. **Construir a imagem:**

   ```bash
   docker compose build
   ```

2. **Rodar o Treinamento:**
   Isso iniciará o treinamento usando sua GPU e salvará os resultados na pasta runs/ localmente.

   ```Bash
   docker compose up
   ```

> Nota: Rodar o script de detecção (main.py) via Docker requer configurações avançadas de encaminhamento de X11 (vídeo) e PulseAudio (som), o que pode ser instável no Windows. Para detecção em tempo real, prefira a instalação nativa.

## 🔧 Troubleshooting (Dicas)

- **Erro de multiprocessing no Windows:**
  Se o treinamento travar ou der erro de spawn, garanta que no treinar.py o workers esteja definido como 0:

```Python
workers=0
```

- **A IA não detecta objetos pequenos:**
  O projeto usa SAHI. Ajuste os parâmetros slice_height e slice_width no main.py. Valores menores (ex: 320 ou 480) dão mais "zoom", mas consomem mais processamento.

- **Falsos Positivos (Detectando o nada):**
  Adicione imagens vazias (Negative Samples) no seu dataset do Roboflow (prints da tela sem a cruz) e treine novamente. Isso ensina a IA a ignorar fundos comuns.

## 🛠️ Tecnologias

- Ultralytics YOLO
- SAHI (Slicing Aided Hyper Inference)
- MSS (Screen Capture)
- OpenCV
- Roboflow

## 📄 Licença

Este projeto está sob a licença [MIT](LICENSE).
