# 🛵 Customer Segmentation: RFM Analysis & K-Means Clustering

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Manipulation-150458)
![Status](https://img.shields.io/badge/Status-Concluído-success)

## 📌 Visão Geral do Projeto
Este projeto aplica técnicas de Machine Learning não supervisionado para resolver um problema clássico de negócios: **entender o comportamento do usuário e segmentar a base de clientes**. 

Utilizando dados de transações (inspirados no modelo de negócio do Rappi), foi construído um pipeline ponta a ponta que extrai dados, calcula as métricas **RFM (Recency, Frequency, Monetary)** e aplica o algoritmo **K-Means** para encontrar clusters naturais de consumidores.

> **⚠️ Nota de Confidencialidade:** Como este projeto foi desenvolvido resolvendo um problema real, todas as credenciais, queries originais sensíveis e dados proprietários foram removidos ou anonimizados em conformidade com as diretrizes de proteção de dados (LGPD).

## 🎯 Impacto de Negócio
A segmentação permite que os times de Marketing e CRM tomem decisões baseadas em dados, como:
* **Retenção:** Identificar clientes "Campeões" que estão perdendo engajamento (aumento de Recência).
* **Upsell/Cross-sell:** Focar nos clientes com alta Frequência, mas baixo Ticket Médio (Monetary).
* **Otimização de ROI:** Reduzir o custo de aquisição (CAC) e focar em campanhas personalizadas para cada cluster.

## 🏗️ Estrutura do Repositório

O projeto foi refatorado utilizando boas práticas de engenharia de software (modularização de scripts):

```text
📦 RFM-Clustering
 ┣ 📂 models/             # Modelos K-Means pré-treinados exportados (.joblib) para K de 3 a 7
 ┣ 📂 query/              # Arquivos SQL utilizados para extração inicial dos dados
 ┣ 📂 src/                # Código fonte modularizado
 ┃ ┣ 📜 data_loader.py    # Ingestão e limpeza primária dos dados
 ┃ ┣ 📜 features.py       # Engenharia de features (Cálculo do R, F e M)
 ┃ ┣ 📜 k_num_experiment.py # Experimentos para definição do 'K' ideal (Elbow Method / Silhouette)
 ┃ ┣ 📜 predict.py        # Script para inferência de novos dados nos modelos treinados
 ┃ ┣ 📜 visualize_profiles.py # Geração de gráficos e perfis dos clusters
 ┃ ┗ 📜 finalize_segments.py  # Consolidação dos clusters finais e regras de negócio
 ┣ 📜 main.py             # Orquestrador do pipeline (Executa de ponta a ponta)
 ┣ 📜 pipeline.log        # Arquivo de log das execuções
 ┗ 📜 README.md           # Documentação do projeto
