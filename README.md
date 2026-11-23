# Brain Tumor Classification & Segmentation — LIPAI Onboarding (FACOM/UFU)

Projeto de extensão desenvolvido no âmbito do **LIPAI Onboarding** (**Laboratório de Inteligência Artificial e Pesquisa Aplicada — FACOM/UFU**), com o objetivo de estruturar um pipeline reprodutível de Visão Computacional aplicada a imagens de Ressonância Magnética (MRI) cerebral para publicação científica.

O estudo investiga o impacto da segmentação semântica prévia da região tumoral (**ROI**) no desempenho de modelos leves de classificação multiclasse (*glioma*, *meningioma*, *notumor* e *pituitary*).

---

## 📋 Visão Geral do Pipeline

O projeto está estruturado em etapas progressivas e reprodutíveis:

1. **Etapa 1 — Classificação Baseline (Imagens Completas):** Treinamento e avaliação da arquitetura leve `ConvNeXt-Atto` utilizando as imagens completas de MRI ($224 \times 224$) como linha de base quantitativa (**Concluída**).
2. **Etapa 2 — Segmentação Semântica da Região Tumoral:** Treinamento do modelo `SegFormer` (`nvidia/mit-b0`) para geração automática de máscaras binárias de tumores cerebrais (**Em andamento**).
3. **Etapa 3 — Classificação Guiada por Segmentação & Análise Comparativa:** Avaliação do ganho de desempenho e explicabilidade ao classificar apenas a região de interesse segmentada, consolidando os resultados para o artigo científico (**Planejada**).

---

## 🧪 Etapa 1: Classificação Baseline (`ConvNeXt-Atto`)

### Metodologia e Reprodutibilidade
* **Arquitetura:** `ConvNeXt-Atto` pré-treinada no *ImageNet* (via `timm`), adaptada para classificação multiclasse em 4 classes clínicas (`glioma`, `meningioma`, `notumor`, `pituitary`).
* **Padronização e Particionamento (`splits.csv`):** Conversão explícita para RGB ($224 \times 224$), normalização *ImageNet* e divisão estratificada sem vazamento de dados (assinatura SHA-256 verificada):
  * **Treino:** 4.855 imagens (85% do conjunto `Training/`)
  * **Validação:** 857 imagens (15% do conjunto `Training/`)
  * **Teste Independente:** 1.311 imagens (conjunto `Testing/` original)
* **Protocolo de Treinamento:** 25 épocas com otimizador `AdamW` (`lr = 1e-4`, `weight_decay = 1e-2`), escalonador `CosineAnnealingLR` e precisão mista (`AMP`) em GPU T4. O *checkpoint* final foi selecionado pelo maior **Macro F1-Score** na validação (atingido na **Época 20**).

### Resultados Obtidos (P1)

| Partição | Amostras | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | Erros Totais |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Validação** | 857 | **99,30%** | 99,28% | 99,31% | **99,29%** | 6 / 857 |
| **Teste** | 1.311 | **99,31%** | 99,26% | 99,30% | **99,28%** | 9 / 1.311 |

* **Desempenho por Classe (Conjunto de Teste):** Todas as classes superaram 99% de *recall*: `pituitary` (**99,7%** — 299/300), `notumor` (**99,5%** — 403/405), `glioma` (**99,0%** — 297/300) e `meningioma` (**99,0%** — 303/306).
* **Análise de Erros:** Os 9 erros residuais no conjunto de teste concentraram-se principalmente entre `meningioma` e `pituitary` (4 casos somados nos dois sentidos) e entre `glioma` e `meningioma` (2 casos), comportamento consistente com a proximidade anatômica e sobreposição morfológica nas imagens completas sem recorte.

### Visualizações e Gráficos de Avaliação

#### Curvas de Aprendizado (Loss e Accuracy)
![Curvas de Aprendizado P1](results/figures/curvas_aprendizado_p1.png)

#### Matrizes de Confusão (Validação e Teste Independente)
![Matrizes de Confusão P1](results/figures/matrizes_confusao_p1.png)

---

## 📂 Estrutura do Repositório

```text
Brain-Tumor-Dataset-Seg-Class/
├── README.md                                  # Documentação principal do projeto
├── .gitignore                                 # Ignorar caches, checkpoints pesados (.pth) e dados locais
├── notebooks/
│   ├── P1_Classificacao_ConvNeXt_Atto.ipynb   # Pipeline completo e executado da Etapa 1 (Baseline)
│   └── P2_Segmentacao_SegFormer.ipynb         # Pipeline estruturado da Etapa 2 (SegFormer)
├── data_splits/
│   └── splits.csv                             # Particionamento estratificado fixo (reprodutibilidade)
└── results/
    ├── P1_classificacao.xlsx                  # Métricas completas, histórico de épocas e matrizes
    └── figures/
        ├── curvas_aprendizado_p1.png          # Evolução de Loss e Accuracy
        └── matrizes_confusao_p1.png           # Matrizes de confusão (Validação e Teste)
```

---

## 🛠️ Tecnologias Utilizadas

* **Linguagem & Deep Learning:** Python 3, PyTorch, `timm`, Hugging Face `transformers`
* **Análise & Visualização:** NumPy, Pandas, Scikit-Learn, Matplotlib, OpenPyXL
