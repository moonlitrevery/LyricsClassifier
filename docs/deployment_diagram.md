# Diagrama de Implantação

```mermaid
flowchart LR
  subgraph DEV["Ambiente de Desenvolvimento"]
    DEVAPP["Notebook/CLI"]
    DEVAPI["FastAPI local"]
    DEVART[("artifacts/ local")]
  end

  subgraph PROD["Ambiente de Produção"]
    SVC["Serviço FastAPI em VM/Container Host"]
    ARTPROD[("Artefatos modelo versão N")]
    MON["Logs/Monitoramento"]
  end

  DEVAPP -->|"treino"| DEVART
  DEVAPI -->|"carrega"| DEVART

  DEVART ==>|"publicar"| ARTPROD

  SVC -->|"monta/configura"| ARTPROD
  SVC -->|"logs/metrics"| MON

  CLIENTE([Clientes]) -->|"HTTP"| SVC
```

- Consumo local roda no mesmo host do cliente (carrega `artifacts/` diretamente).
- Serviço HTTP roda em host dedicado; em produção, apontado para artefatos publicados e versionados.
- Diferenças dev vs prod: hot-reload, logs verbosos, credenciais e observabilidade.
