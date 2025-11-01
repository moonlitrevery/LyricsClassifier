# Diagrama de Classes e Justificativa

```mermaid
classDiagram
    class CorpusLoader{
      +load_dataset(path, text_col, label_col)
    }
    class TextPreprocessor{
      -language: str
      -lowercase: bool
      -remove_stopwords: bool
      -use_lemmatization: bool
      +transform(texts): List~str~
    }
    class FeatureExtractor{
    }
    class TfidfFeatures{
      +fit_transform(texts)
      +transform(vec, texts)
    }
    class EmbeddingFeatures{
      +fit_transform(texts)
      +transform(texts)
    }
    class ModelTrainer{
      +train(X_texts, y_labels): TrainedModel
    }
    class Evaluator{
      +evaluate_predictions(y_true, y_pred)
      +save_confusion_matrix(cm, labels, path)
    }
    class LyricsClassifier{
      +load(artifact_dir)
      +predict(texts): List~str~
      +predict_proba(texts): List~List~float~~
      +topk(texts, k): List~List~{label,prob}~
    }
    class MetricsReport{
      +to_json()
    }

    CorpusLoader --> TextPreprocessor : produz textos
    TextPreprocessor --> TfidfFeatures : insumo
    TextPreprocessor --> EmbeddingFeatures : opcional
    ModelTrainer --> TfidfFeatures
    ModelTrainer --> EmbeddingFeatures
    ModelTrainer --> LyricsClassifier : exporta artefato
    Evaluator --> MetricsReport
```

Justificativa (fronteiras de responsabilidade):
- CorpusLoader cuida apenas de I/O e validação de esquema do dataset, isolando acesso a dados.
- TextPreprocessor encapsula decisões linguísticas (case, stopwords, lematização) mantendo FeatureExtractor agnóstico.
- FeatureExtractor separa TF-IDF de Embeddings para permitir escolha/combinação sem afetar o restante.
- ModelTrainer orquestra treino/validação e seleção do melhor estimador; não conhece detalhes de I/O da API.
- Evaluator centraliza métricas e visualizações, produzindo artefatos reprodutíveis.
- LyricsClassifier é o componente consumível (local/API), com contrato estável de `predict`, `predict_proba` e `topk`.
- MetricsReport materializa resultados em JSON, facilitando auditoria e publicação.