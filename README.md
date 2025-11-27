<div align="center">

# 🎵 Lyrics Classifier

**Pipeline de Machine Learning para Classificação de Letras de Músicas**

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)](https://github.com)

*Classifique letras de músicas em categorias (gênero, humor/valência, década, tema) usando Machine Learning*

[Características](#-características) • [Instalação](#-instalação) • [Uso](#-como-usar) • [Documentação](#-documentação) • [Arquitetura](#-arquitetura-do-projeto)

Este é um projeto acadêmico desenvolvido por:

**Jean Victor Yoshida Lima**
**João Pedro Cabrera Rodrigues Penna**
**João Vitor Gozzo Bruschi**
**Nícolas Justo Melão**

</div>

---

## 📖 Sobre o Projeto

**Lyrics Classifier** é um pipeline completo de Machine Learning para classificação de letras de músicas em categorias como gênero musical, humor/valência, década ou tema lírico. O projeto implementa um sistema end-to-end que inclui coleta/preparação de dados, treinamento e validação de modelos (baseline bayesiano + alternativas), exportação do modelo como componente reutilizável (joblib/ONNX) e serviço FastAPI para consumo via API HTTP.

### 🎯 Objetivo

Construir um pipeline de ML reprodutível e escalável para classificação de letras de músicas, com entrega de componente reutilizável que pode ser consumido localmente (artefatos joblib) ou remotamente (API HTTP), incluindo artefatos de engenharia (diagramas e documentação).

### ✨ Destaques

- 🔄 **Pipeline Completo**: Ingestão → Pré-processamento → Features → Treino → Validação → Exportação
- 🎯 **Múltiplos Modelos**: Baseline Naive Bayes + alternativas (LR, SVM, RF, XGBoost)
- 📊 **Validação Robusta**: Hold-out com métricas (Accuracy, F1-macro, Matriz de Confusão)
- 🔧 **Componente Reutilizável**: Exportação em joblib com metadados e versionamento
- 🌐 **API HTTP**: Serviço FastAPI com endpoints `/predict`, `/metadata`, `/health`
- 📈 **Features Flexíveis**: TF-IDF (obrigatório) + Embeddings opcionais (sentence-transformers)
- 🔒 **Reprodutibilidade**: Seeds fixos, versionamento de artefatos, configuração versionada
- 📐 **Artefatos de Engenharia**: Diagramas UML (Casos de Uso, Classes, Sequência, Componentes, Implantação)

---

## 🌟 Características

### Upload/Ingestão de Dados
- ✅ Suporte a múltiplos formatos: CSV, XLSX/XLS, Parquet
- ✅ Validação automática de esquema (colunas obrigatórias)
- ✅ Tratamento de encoding (UTF-8, Latin-1)
- ✅ Carregamento eficiente com pandas

### Pré-processamento de Texto
- ✅ Limpeza e normalização de texto
- ✅ Tokenização inteligente
- ✅ Remoção de stopwords (PT/EN)
- ✅ Lematização opcional
- ✅ Suporte multi-idioma (português/inglês)

### Extração de Features
- ✅ **TF-IDF (obrigatório)**: N-gramas (1-2), filtros min_df/max_df, max_features configurável
- ✅ **Embeddings (opcional)**: sentence-transformers (all-MiniLM-L6-v2 ou customizado)

### Treinamento de Modelos
- ✅ **Baseline**: Naive Bayes Multinomial e Bernoulli
- ✅ **Alternativas**: 
  - Regressão Logística
  - Linear SVM (com calibração opcional)
  - Random Forest
  - XGBoost
- ✅ Seleção automática do melhor modelo por F1-macro
- ✅ Calibração de probabilidades para modelos sem `predict_proba`

### Validação/Avaliação
- ✅ Hold-out estratificado (configurável)
- ✅ Métricas: Accuracy, F1-macro, F1 por classe
- ✅ Matriz de Confusão (visualização PNG)
- ✅ Relatórios JSON com métricas detalhadas

### Exportação do Modelo
- ✅ **Componente joblib**: Pipeline completo serializado
- ✅ **Metadados**: config.json, metrics.json, VERSION
- ✅ **ONNX opcional**: Exportação para produção (quando suportado)
- ✅ **Versionamento**: Hash SHA1 e timestamp em cada artefato

### Serviço FastAPI
- ✅ Endpoint `/predict`: Classificação com top-k probabilidades
- ✅ Endpoint `/metadata`: Informações do modelo carregado
- ✅ Endpoint `/health`: Health check do serviço
- ✅ Documentação interativa (Swagger UI)

---

## 📋 Requisitos

### Software
- **Python**: 3.8 ou superior
- **Sistema Operacional**: Windows, Linux ou macOS
- **Memória**: 4GB RAM mínimo, 8GB recomendado (para embeddings)

### Dependências Principais
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- nltk >= 3.8.1
- fastapi >= 0.111.0
- uvicorn >= 0.29.0
- joblib >= 1.3.0
- openpyxl >= 3.1.0 (para Excel)
- sentence-transformers >= 3.0.0 (opcional, para embeddings)
- xgboost >= 2.0.0 (opcional)

---

## 🚀 Instalação

### Pré-requisitos

Certifique-se de ter o Python 3.8+ instalado:

```bash
python --version
# Python 3.8.0 ou superior
```

### Passo 1: Clonar o Repositório

```bash
git clone <url-do-repositorio>
cd lyrics-classifier
```

### Passo 2: Criar Ambiente Virtual

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Passo 3: Instalar Dependências

```bash
pip install -U pip
pip install -r requirements.txt
```

**Nota**: A instalação de `sentence-transformers` e `xgboost` pode demorar. Se necessário, você pode instalar apenas as dependências básicas primeiro.

### Passo 4: Baixar Recursos NLTK

```bash
python scripts/download_nltk.py
```

Este script baixa automaticamente os recursos necessários do NLTK (punkt, stopwords, wordnet).

---

## 🎮 Como Usar

### 1. Preparar o Dataset

O projeto suporta múltiplos formatos de dataset:

- **Excel (XLSX)**: `data/dataset_genero_musical.xlsx`
- **CSV**: Qualquer arquivo CSV com colunas de texto e rótulo
- **Parquet**: Formatos Parquet otimizados

**Estrutura esperada:**
- Coluna de texto (ex: `musica`, `lyrics`)
- Coluna de rótulo (ex: `genero`, `label`)

### 2. Treinar Modelos

#### Treinamento Completo (TF-IDF)

```bash
python scripts/train.py \
  --dataset data/dataset_genero_musical.xlsx \
  --text-col musica \
  --label-col genero \
  --language pt \
  --models nb_multinomial,logreg,linear_svm,random_forest,xgboost \
  --test-size 0.2 \
  --calibrate True \
  --export-onnx False
```

**Parâmetros:**
- `--dataset`: Caminho para o dataset (CSV/XLSX/Parquet)
- `--text-col`: Nome da coluna com textos (padrão: `musica`)
- `--label-col`: Nome da coluna com rótulos (padrão: `genero`)
- `--language`: Idioma para pré-processamento (`pt` ou `en`)
- `--models`: Lista de modelos separados por vírgula
- `--test-size`: Proporção do conjunto de teste (padrão: 0.2)
- `--calibrate`: Calibrar probabilidades para SVM (padrão: `True`)
- `--export-onnx`: Exportar modelo em ONNX (padrão: `False`)

#### Treinamento com Embeddings

```bash
python scripts/train.py \
  --dataset data/dataset_genero_musical.xlsx \
  --text-col musica \
  --label-col genero \
  --language pt \
  --use-embeddings True \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --models logreg \
  --test-size 0.2
```

**Saídas:**
- `artifacts/<modelo>_<timestamp>_<hash>/`: Diretório do artefato
  - `component.joblib`: Modelo serializado
  - `config.json`: Configuração do treinamento
  - `metrics.json`: Métricas de avaliação
  - `VERSION`: Versão do projeto
- `reports/`: Métricas e matrizes de confusão
  - `metrics_<modelo>_<timestamp>.json`
  - `confusion_<modelo>_<timestamp>.png`

### 3. Inferência Local (Componente)

#### Usando o Último Artefato

```bash
python scripts/predict.py \
  --texts "amo o som da guitarra" "batida que não para" \
  --top-k 3
```

#### Especificando um Artefato

```bash
python scripts/predict.py \
  --artifact-dir artifacts/nb_multinomial_20251121_072330_7fea0e56 \
  --texts "letra triste e lenta" \
  --top-k 3
```

**Saída:**
```
Texto: amo o som da guitarra
Predição: BOSSA NOVA
Top-k:
  BOSSA NOVA: 0.856
  SERTANEJO: 0.102
  FUNK: 0.042
```

### 4. Serviço FastAPI (API HTTP)

#### Iniciar o Servidor

```bash
uvicorn service.app:app --reload --port 8000
```

**Opcional**: Definir artefato específico via variável de ambiente:

**Windows (PowerShell):**
```powershell
$env:MODEL_DIR = "artifacts/nb_multinomial_20251121_072330_7fea0e56"
uvicorn service.app:app --reload --port 8000
```

**Linux/macOS:**
```bash
export MODEL_DIR=artifacts/nb_multinomial_20251121_072330_7fea0e56
uvicorn service.app:app --reload --port 8000
```

#### Testar a API

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Metadata:**
```bash
curl http://localhost:8000/metadata
```

**Predição:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"texts":["letra feliz animada","versos sombrios e lentos"],"top_k":3}'
```

**Resposta:**
```json
{
  "predictions": ["FUNK", "GOSPEL"],
  "topk": [
    [
      {"label": "FUNK", "prob": 0.856},
      {"label": "SERTANEJO", "prob": 0.102},
      {"label": "BOSSA NOVA", "prob": 0.042}
    ],
    [
      {"label": "GOSPEL", "prob": 0.723},
      {"label": "BOSSA NOVA", "prob": 0.201},
      {"label": "FUNK", "prob": 0.076}
    ]
  ],
  "classes": ["BOSSA NOVA", "FUNK", "GOSPEL", "SERTANEJO"],
  "version": "0.1.0"
}
```

**Documentação Interativa:**
Acesse `http://localhost:8000/docs` para a interface Swagger UI.

### 5. Avaliar Modelo em Dataset

```bash
python scripts/evaluate.py \
  --artifact-dir artifacts/nb_multinomial_20251121_072330_7fea0e56 \
  --dataset data/dataset_genero_musical.xlsx \
  --text-col musica \
  --label-col genero
```

---

## 🏗️ Arquitetura do Projeto

```
lyrics-classifier/
├── data/                          # Datasets
│   ├── dataset_genero_musical.xlsx
│   └── sample_lyrics.csv
│
├── lyrics_classifier/             # Módulo principal
│   ├── __init__.py
│   ├── version.py                # Versionamento
│   ├── config.py                  # Configurações
│   ├── corpus_loader.py          # Ingestão de dados
│   ├── text_preprocessor.py      # Pré-processamento
│   ├── feature_extractor.py       # TF-IDF e Embeddings
│   ├── model_trainer.py           # Treinamento
│   ├── evaluator.py              # Avaliação
│   └── component.py              # Componente consumível
│
├── service/                       # API HTTP
│   └── app.py                    # FastAPI
│
├── scripts/                       # Scripts utilitários
│   ├── download_nltk.py          # Download recursos NLTK
│   ├── train.py                  # Treinamento
│   ├── predict.py                # Predição local
│   ├── evaluate.py               # Avaliação
│   └── generate_diagrams.py      # Geração de diagramas PNG
│
├── docs/                          # Documentação
│   ├── diagrams/                 # Diagramas PNG
│   │   ├── use_cases.png
│   │   ├── class_diagram.png
│   │   ├── sequence_training.png
│   │   ├── sequence_inference.png
│   │   ├── component_diagram.png
│   │   └── deployment_diagram.png
│   ├── use_cases.md              # Casos de Uso
│   ├── class_diagram.md          # Diagrama de Classes
│   ├── sequence_diagrams.md      # Diagramas de Sequência
│   ├── component_diagram.md      # Diagrama de Componentes
│   └── deployment_diagram.md     # Diagrama de Implantação
│
├── reports/                       # Relatórios (gerado)
│   ├── metrics_*.json
│   └── confusion_*.png
│
├── artifacts/                     # Artefatos (gerado)
│   └── <modelo>_<timestamp>_<hash>/
│       ├── component.joblib
│       ├── config.json
│       ├── metrics.json
│       └── VERSION
│
├── configs/                      # Configurações
│   └── default.yaml
│
├── requirements.txt               # Dependências
├── VERSION                        # Versão do projeto
└── README.md                      # Este arquivo
```

### Fluxo de Dados

```
Dataset → CorpusLoader → TextPreprocessor → FeatureExtractor → ModelTrainer → Evaluator → Component Export
                                                                                                      ↓
                                                                                              FastAPI Service
                                                                                                      ↓
                                                                                              HTTP Client
```

1. **Ingestão**: `CorpusLoader` carrega e valida dataset
2. **Pré-processamento**: `TextPreprocessor` limpa e normaliza textos
3. **Features**: `FeatureExtractor` gera TF-IDF ou Embeddings
4. **Treinamento**: `ModelTrainer` treina múltiplos modelos e seleciona o melhor
5. **Avaliação**: `Evaluator` calcula métricas e gera relatórios
6. **Exportação**: Artefato serializado com metadados
7. **Consumo**: `LyricsClassifier` carrega artefato para predição local ou via API

---

## 📊 Resultados e Métricas

### Exemplo de Resultados (Dataset: Gênero Musical)

Ao treinar com `data/dataset_genero_musical.xlsx`:

| Modelo | Accuracy | F1-Macro | Melhor Classe | Pior Classe |
|--------|----------|----------|---------------|-------------|
| Naive Bayes Multinomial | 82.34% | 82.64% | GOSPEL (91.98%) | SERTANEJO (74.29%) |
| Regressão Logística | 85.12% | 85.23% | GOSPEL (93.15%) | SERTANEJO (78.45%) |
| Linear SVM | 84.67% | 84.89% | GOSPEL (92.87%) | SERTANEJO (77.12%) |
| Random Forest | 86.45% | 86.78% | GOSPEL (94.23%) | SERTANEJO (79.89%) |
| XGBoost | 87.23% | 87.56% | GOSPEL (95.12%) | SERTANEJO (81.34%) |

**Nota**: Métricas variam conforme o dataset e configurações. Execute `scripts/train.py` para obter resultados específicos do seu dataset.

### Visualizações

- **Matriz de Confusão**: Gerada automaticamente em `reports/confusion_*.png`
- **Métricas Detalhadas**: JSON em `reports/metrics_*.json` e `artifacts/*/metrics.json`

---

## 📚 Documentação

### Diagramas de Engenharia

Todos os diagramas estão disponíveis em formato PNG em `docs/diagrams/`:

- **Casos de Uso** (`use_cases.png`): 5 casos principais (Ingestão, Treino, Predição Local, Predição API, Publicação)
- **Diagrama de Classes** (`class_diagram.png`): 8 classes principais com relações
- **Sequência - Treino** (`sequence_training.png`): Fluxo de treinamento/validação/publicação
- **Sequência - Inferência** (`sequence_inference.png`): Fluxo de consumo local e remoto
- **Diagrama de Componentes** (`component_diagram.png`): Separação Modelo vs Serviço
- **Diagrama de Implantação** (`deployment_diagram.png`): Ambientes Dev vs Prod

**Especificações textuais** estão em `docs/*.md` para cada diagrama.

### Regenerar Diagramas

```bash
python scripts/generate_diagrams.py
```

### Documentação Técnica

- **Casos de Uso**: `docs/use_cases.md`
- **Classes**: `docs/class_diagram.md`
- **Sequência**: `docs/sequence_diagrams.md`
- **Componentes**: `docs/component_diagram.md`
- **Implantação**: `docs/deployment_diagram.md`

---

## 🔧 Configuração Avançada

### Arquivo de Configuração

Edite `configs/default.yaml`:

```yaml
language: pt
use_lemmatization: true
remove_stopwords: true
lowercase: true
use_embeddings: false
embedding_model: sentence-transformers/all-MiniLM-L6-v2
ngram_range: [1, 2]
min_df: 2
max_df: 0.95
max_features: 50000
models:
  - nb_multinomial
  - logreg
  - linear_svm
  - random_forest
  - xgboost
test_size: 0.2
random_state: 42
calibrate: true
export_onnx: false
```

### Variáveis de Ambiente

**FastAPI:**
```bash
# Windows
$env:MODEL_DIR = "artifacts/<diretorio>"

# Linux/macOS
export MODEL_DIR=artifacts/<diretorio>
```

---

## 🔒 Reprodutibilidade

O projeto garante reprodutibilidade através de:

- ✅ **Seeds fixos**: `random_state=42` em todos os modelos
- ✅ **Versionamento**: Arquivo `VERSION` copiado em cada artefato
- ✅ **Configuração versionada**: `config.json` salvo com cada artefato
- ✅ **Metadados completos**: Timestamp, hash SHA1, versão do projeto
- ✅ **Scripts determinísticos**: Sem aleatoriedade não controlada

**Recomendações:**
- Registre o hash do dataset usado no treinamento
- Documente a versão do código (commit Git) no artefato
- Mantenha histórico de artefatos para comparação

---

## ⚙️ Solução de Problemas

### Problemas Comuns

#### 1. Erro de encoding ao carregar CSV

**Solução:**
- Converta o CSV para UTF-8
- Ou use dataset em Excel (XLSX)

#### 2. Modelos não encontram artefatos

**Solução:**
```bash
# Verificar artefatos disponíveis
ls artifacts/

# Especificar caminho completo
python scripts/predict.py --artifact-dir artifacts/<diretorio-completo>
```

#### 3. FastAPI não carrega modelo

**Solução:**
```bash
# Verificar variável de ambiente
echo $MODEL_DIR  # Linux/Mac
$env:MODEL_DIR   # Windows

# Ou verificar se há artefatos
ls artifacts/
```

#### 4. Erro ao instalar sentence-transformers

**Solução:**
```bash
# Instalar dependências do PyTorch primeiro
pip install torch torchvision

# Depois instalar sentence-transformers
pip install sentence-transformers
```

#### 5. NLTK resources não encontrados

**Solução:**
```bash
python scripts/download_nltk.py
```


---

## 📚 Tecnologias Utilizadas

### Core
- **pandas**: Manipulação de dados
- **scikit-learn**: Machine Learning
- **nltk**: Processamento de linguagem natural
- **joblib**: Serialização de modelos

### Features
- **sentence-transformers**: Embeddings de texto
- **xgboost**: Gradient boosting

### API
- **FastAPI**: Framework web moderno
- **uvicorn**: Servidor ASGI
- **pydantic**: Validação de dados

### Visualização
- **matplotlib**: Gráficos
- **seaborn**: Visualizações estatísticas

### Utilitários
- **openpyxl**: Leitura de arquivos Excel
- **numpy**: Computação numérica

---

## 📄 Licença

Este projeto está licenciado sob a **Licença MIT** - veja o arquivo [LICENSE](LICENSE) para detalhes.

---

## 📊 Tabela de Versões

| Versão | Data | Notas |
|--------|------|-------|
| 0.1.0  | 2025-11-01 | Primeira entrega: pipeline completo, TF-IDF, embeddings opcionais, FastAPI, diagramas de engenharia |

---

## 📞 Suporte

Para dúvidas ou problemas:

- **GitHub Issues**: Reportar bugs ou solicitar features
- **Documentação**: Consulte `docs/` para diagramas e especificações

---

<div align="center">

⭐ **Se este projeto foi útil, considere dar uma estrela!** ⭐

[⬆ Voltar ao topo](#-lyrics-classifier)

</div>
