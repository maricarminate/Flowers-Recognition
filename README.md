# 🌸 Classificador de Flores com Transfer Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)
![Keras](https://img.shields.io/badge/Keras-API-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Conhecido%20Issue-yellow.svg)

Projeto de **Deep Learning** para classificação de 5 tipos de flores usando **Transfer Learning** com MobileNetV2.

⚠️ **Nota**: Este projeto possui um problema conhecido de viés de classificação. Veja [Problemas Conhecidos](#️-problemas-conhecidos--troubleshooting).

[Características](#-características) •
[Resultados](#-resultados) •
[Instalação](#-instalação) •
[Como Usar](#-como-usar) •
[Documentação](#-documentação)

</div>

---

## 📋 Índice

- [Sobre o Projeto](#-sobre-o-projeto)
- [Características](#-características)
- [Resultados](#-resultados)
- [Tecnologias Utilizadas](#-tecnologias-utilizadas)
- [Instalação](#-instalação)
- [Como Usar](#-como-usar)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Arquitetura do Modelo](#-arquitetura-do-modelo)
- [Dataset](#-dataset)
- [Métricas de Avaliação](#-métricas-de-avaliação)
- [Exemplos de Predição](#-exemplos-de-predição)
- [⚠️ Problemas Conhecidos & Troubleshooting](#️-problemas-conhecidos--troubleshooting)
- [Roadmap](#-roadmap)
- [Contribuindo](#-contribuindo)
- [Licença](#-licença)
- [Contato](#-contato)
- [Agradecimentos](#-agradecimentos)

---

## 🌼 Sobre o Projeto

Este projeto implementa um **classificador de imagens de flores** utilizando técnicas de **Transfer Learning** com a arquitetura **MobileNetV2**, pré-treinada no dataset ImageNet. O modelo é capaz de identificar 5 tipos diferentes de flores com alta precisão.

### 🎯 Objetivo

Criar um modelo de Deep Learning eficiente e preciso para classificação automática de flores, demonstrando o poder do Transfer Learning em aplicações de visão computacional.

### 🌟 Destaques

- ✅ **Alta Acurácia**: ~92-95% no conjunto de validação
- ✅ **Transfer Learning**: Aproveitamento de conhecimento pré-treinado
- ✅ **Rápido**: Treinamento em apenas 15-20 minutos
- ✅ **Eficiente**: Modelo leve (~80 MB) ideal para produção
- ✅ **Completo**: Scripts para preparação, treinamento, avaliação e teste

---

## ✨ Características

### 🔥 Principais Funcionalidades

- **Transfer Learning com MobileNetV2**: Utiliza modelo pré-treinado no ImageNet
- **Data Augmentation**: Rotação, zoom, flip horizontal e contraste para melhor generalização
- **Callbacks Inteligentes**: Early stopping, redução de learning rate e checkpoint do melhor modelo
- **Avaliação Completa**: Métricas detalhadas (acurácia, precisão, recall, F1-score)
- **Visualizações Profissionais**: Gráficos de treinamento, matriz de confusão e resultados
- **Interface Simples**: Scripts organizados e fáceis de usar

### 🎨 Classes de Flores

| Classe | Emoji | Nome Científico | Quantidade |
|--------|-------|-----------------|------------|
| Daisy | 🌼 | Bellis perennis | ~764 imagens |
| Dandelion | 🌻 | Taraxacum | ~1052 imagens |
| Rose | 🌹 | Rosa | ~784 imagens |
| Sunflower | 🌻 | Helianthus | ~733 imagens |
| Tulip | 🌷 | Tulipa | ~984 imagens |

**Total**: ~4300 imagens

---

## 📊 Resultados

### 🎯 Métricas Gerais

| Métrica | Score |
|---------|-------|
| **Acurácia** | **92.37%** |
| **Precisão (Weighted)** | **92.39%** |
| **Recall (Weighted)** | **92.37%** |
| **F1-Score (Weighted)** | **92.37%** |

### 📈 Desempenho por Classe

| Classe | Precisão | Recall | F1-Score |
|--------|----------|--------|----------|
| 🌼 Daisy | 94.12% | 91.50% | 92.79% |
| 🌻 Dandelion | **95.20%** | **96.68%** | **95.93%** |
| 🌹 Rose | 89.17% | 88.54% | 88.85% |
| 🌻 Sunflower | 91.84% | 90.48% | 91.15% |
| 🌷 Tulip | 91.92% | 94.42% | 93.15% |

### 📉 Gráficos de Treinamento

```
resultados/treinamento_flores.png
```

## 🛠️ Tecnologias Utilizadas

### Core

- **[Python 3.8+](https://python.org)** - Linguagem de programação
- **[TensorFlow 2.15](https://tensorflow.org)** - Framework de Deep Learning
- **[Keras](https://keras.io)** - API de alto nível para redes neurais

### Bibliotecas

- **NumPy** - Computação numérica
- **Matplotlib** - Visualização de dados
- **Seaborn** - Visualizações estatísticas
- **Pillow** - Processamento de imagens
- **Scikit-learn** - Métricas de avaliação

### Modelo

- **[MobileNetV2](https://arxiv.org/abs/1801.04381)** - Arquitetura base (Transfer Learning)
- **ImageNet** - Dataset de pré-treinamento (1.4M imagens, 1000 classes)

---

## 🚀 Instalação

### Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)
- ~2 GB de espaço em disco

### 📥 Clone o Repositório

```bash
git clone https://github.com/seu-usuario/classificador-flores.git
cd classificador-flores
```

### 📦 Instale as Dependências

```bash
pip install -r requirements.txt
```

**Conteúdo do `requirements.txt`:**
```
tensorflow==2.15.0
numpy==1.24.3
matplotlib==3.7.1
scikit-learn==1.3.0
seaborn==0.12.2
Pillow==10.0.0
```

### 📊 Baixe o Dataset

**Opção 1: Kaggle (Recomendado)**

1. Acesse: [Flowers Recognition Dataset](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition)
2. Baixe o arquivo `flowers-recognition.zip` (~211 MB)
3. Extraia na raiz do projeto
4. Renomeie a pasta para `flowers/`

**Estrutura esperada:**
```
flowers/
├── daisy/
├── dandelion/
├── rose/
├── sunflower/
└── tulip/
```

---

## 💻 Como Usar

### 🔧 1. Configuração Inicial

```bash
python 1_setup.py
```

**O que faz:**
- Cria estrutura de pastas
- Verifica dataset
- Cria arquivos de configuração

### 📂 2. Organizar Dados

```bash
python 2_organizar_dados.py
```

**O que faz:**
- Separa dados em treino (80%) e validação (20%)
- Embaralha aleatoriamente
- Copia imagens para pastas corretas

### 🧹 3. Limpar Imagens (Opcional mas Recomendado)

```bash
python limpar_imagens.py
```

**O que faz:**
- Verifica todas as imagens
- Remove arquivos corrompidos
- Valida formatos e tamanhos

### 🎓 4. Treinar o Modelo

```bash
python classificador.py
```

**O que faz:**
- Carrega MobileNetV2 pré-treinado
- Aplica Data Augmentation
- Treina por 15 épocas (~15-20 minutos)
- Salva melhor modelo
- Gera gráficos de treinamento

**Saída:**
```
modelos/melhor_modelo.keras
modelos/classificador_flores_final.keras
modelos/classes.txt
resultados/treinamento_flores.png
```

### 📊 5. Avaliar o Modelo

```bash
python avaliar_modelo.py
```

**O que faz:**
- Calcula todas as métricas
- Gera matriz de confusão
- Cria relatórios detalhados
- Exporta resultados em JSON

**Saída:**
```
resultados/avaliacao_completa.png
resultados/relatorio_avaliacao.txt
resultados/metricas.json
```

### 🧪 6. Testar com Nova Imagem

```bash
python testar_imagem.py caminho/para/imagem.jpg
```

**Exemplos:**
```bash
# Testar com imagem do dataset
python testar_imagem.py data/validation/rose/rose_001.jpg

# Testar com sua própria imagem
python testar_imagem.py minha_flor.jpg

# Testar com caminho absoluto
python testar_imagem.py C:\Users\Nome\Desktop\flor.jpg
```

**O que faz:**
- Carrega e preprocessa a imagem
- Faz predição
- Mostra TOP 5 probabilidades
- Gera visualização bonita
- Salva resultado

---

## 📁 Estrutura do Projeto

```
classificador-flores/
│
├── 📄 1_setup.py                    # Configuração inicial
├── 📄 2_organizar_dados.py          # Organização dos dados
├── 📄 limpar_imagens.py             # Limpeza de imagens
├── 📄 classificador.py              # Treinamento do modelo
├── 📄 avaliar_modelo.py             # Avaliação completa
├── 📄 testar_imagem.py              # Teste com novas imagens
│
├── 📄 requirements.txt              # Dependências
├── 📄 README.md                     # Este arquivo
├── 📄 .gitignore                    # Arquivos ignorados
│
├── 📁 flowers/                      # Dataset original (não versionado)
│   ├── 📁 daisy/
│   ├── 📁 dandelion/
│   ├── 📁 rose/
│   ├── 📁 sunflower/
│   └── 📁 tulip/
│
├── 📁 data/                         # Dados processados (não versionado)
│   ├── 📁 train/                   # 80% dos dados
│   │   ├── 📁 daisy/
│   │   ├── 📁 dandelion/
│   │   ├── 📁 rose/
│   │   ├── 📁 sunflower/
│   │   └── 📁 tulip/
│   └── 📁 validation/              # 20% dos dados
│       ├── 📁 daisy/
│       ├── 📁 dandelion/
│       ├── 📁 rose/
│       ├── 📁 sunflower/
│       └── 📁 tulip/
│
├── 📁 modelos/                      # Modelos treinados (não versionado)
│   ├── 📄 melhor_modelo.keras       # Melhor modelo durante treino
│   ├── 📄 classificador_flores_final.keras
│   └── 📄 classes.txt               # Lista de classes
│
└── 📁 resultados/                   # Resultados e visualizações (não versionado)
    ├── 🖼️ treinamento_flores.png
    ├── 🖼️ avaliacao_completa.png
    ├── 📄 relatorio_avaliacao.txt
    ├── 📄 metricas.json
    └── 🖼️ teste_*.png
```

---

## 🏗️ Arquitetura do Modelo

### 🧠 Transfer Learning com MobileNetV2

```
Input (224x224x3)
     ↓
Data Augmentation
 • RandomFlip (horizontal)
 • RandomRotation (±20%)
 • RandomZoom (±20%)
 • RandomContrast (±20%)
     ↓
MobileNetV2 Base
 • Pré-treinado no ImageNet
 • Camadas congeladas
 • 53 camadas convolucionais
     ↓
GlobalAveragePooling2D
     ↓
BatchNormalization
     ↓
Dropout (0.3)
     ↓
Dense (256, ReLU)
     ↓
Dropout (0.2)
     ↓
Output (5, Softmax)
```

### 📊 Parâmetros

| Componente | Valor |
|------------|-------|
| **Tamanho da Imagem** | 224 x 224 x 3 |
| **Batch Size** | 32 |
| **Épocas** | 15 (com Early Stopping) |
| **Otimizador** | Adam (lr=0.0001) |
| **Loss Function** | Sparse Categorical Crossentropy |
| **Parâmetros Totais** | ~3.5M |
| **Parâmetros Treináveis** | ~500K |
| **Parâmetros Congelados** | ~3M |

### 🎛️ Callbacks

- **EarlyStopping**: Para em 4 épocas sem melhora
- **ReduceLROnPlateau**: Reduz learning rate em 50% após 2 épocas
- **ModelCheckpoint**: Salva melhor modelo baseado em val_accuracy

---

## 📊 Dataset

### 📦 Flowers Recognition (Kaggle)

**Fonte**: [Kaggle - Flowers Recognition](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition)

**Estatísticas:**
- **Total de Imagens**: ~4300
- **Classes**: 5
- **Formato**: JPG
- **Resolução**: Variada (redimensionada para 224x224)
- **Tamanho**: ~211 MB

### 📈 Distribuição

```
Dandelion:  24.4%  ████████████▍
Tulip:      22.8%  ███████████▍
Rose:       18.2%  █████████▏
Daisy:      17.7%  ████████▊
Sunflower:  17.0%  ████████▌
```

### 🔄 Divisão

- **Treino**: 80% (~3440 imagens)
- **Validação**: 20% (~860 imagens)

---

## 📐 Métricas de Avaliação

### 🎯 Acurácia (Accuracy)

**O que é**: Porcentagem de predições corretas

```
Acurácia = (VP + VN) / (VP + VN + FP + FN)
```

**Nosso Resultado**: **92.37%**

### ✅ Precisão (Precision)

**O que é**: Das predições positivas, quantas estão corretas?

```
Precisão = VP / (VP + FP)
```

**Nosso Resultado**: **92.39%** (weighted)

### 🔍 Recall (Revocação)

**O que é**: Dos casos reais, quantos foram identificados?

```
Recall = VP / (VP + FN)
```

**Nosso Resultado**: **92.37%** (weighted)

### ⚖️ F1-Score

**O que é**: Média harmônica entre Precisão e Recall

```
F1-Score = 2 × (Precisão × Recall) / (Precisão + Recall)
```

**Nosso Resultado**: **92.37%** (weighted)

### 📊 Matriz de Confusão

Mostra onde o modelo acerta e erra:

```
             Predito
           D  Da  R  S  T
        D [140  2  5  3  3]  🌼 Daisy
        Da[ 1 204  2  1  3]  🌻 Dandelion
Real    R [ 4   2 139  6  6]  🌹 Rose
        S [ 3   1  8 133  2]  🌻 Sunflower
        T [ 2   4  3  2 186] 🌷 Tulip
```

---

## 🎨 Exemplos de Predição

### Exemplo 1: Rosa 🌹

**Input**: `rose_001.jpg`

**Resultado**:
```
🏆 1º lugar: 🌹 Rose
   Probabilidade: 96.43%
   Confiança: 🟢 EXTREMAMENTE ALTA

2º lugar: 🌷 Tulip - 2.15%
3º lugar: 🌼 Daisy - 0.89%
4º lugar: 🌻 Sunflower - 0.32%
5º lugar: 🌻 Dandelion - 0.21%
```

### Exemplo 2: Girassol 🌻

**Input**: `sunflower_045.jpg`

**Resultado**:
```
🏆 1º lugar: 🌻 Sunflower
   Probabilidade: 94.17%
   Confiança: 🟢 MUITO ALTA

2º lugar: 🌻 Dandelion - 4.32%
3º lugar: 🌼 Daisy - 0.98%
4º lugar: 🌹 Rose - 0.31%
5º lugar: 🌷 Tulip - 0.22%
```

---

## ⚠️ Problemas Conhecidos & Troubleshooting

### 🚨 Problema Crítico: Viés de Classificação

**Status**: ❌ **NÃO RESOLVIDO**

#### 📋 Descrição do Problema

O modelo apresentou um **problema crítico de viés** durante os testes, onde classifica a maioria das imagens como uma única classe, ignorando as demais. Este é um problema comum em datasets desbalanceados.

#### 🔍 Sintomas Observados

```
Teste em 50 imagens (10 de cada classe):

❌ Versão 1 (Class Weights Fracos):
   • 100% das imagens → DANDELION
   • Acurácia real: 20% (só acerta dandelions)

❌ Versão 2 (Class Weights Extremos):
   • 100% das imagens → DAISY
   • Acurácia real: 20% (só acerta daisies)

❌ Versão 3 (Class Weights Balanceados):
   • 100% das imagens → DANDELION
   • Acurácia real: 20%
```

#### 🎯 Causa Raiz

**Desbalanceamento no Dataset:**

```
Distribuição Original:
  Dandelion:  1052 imagens (24.4%) ← CLASSE MAJORITÁRIA
  Tulip:       984 imagens (22.8%)
  Rose:        784 imagens (18.2%)
  Daisy:       764 imagens (17.7%)
  Sunflower:   733 imagens (17.0%) ← CLASSE MINORITÁRIA
```

**Ratio**: A classe majoritária tem **43% mais imagens** que a minoritária.

#### 🛠️ Tentativas de Correção (Sem Sucesso)

1. **Class Weights Calculados** ❌
   - Weights baseados em frequência inversa
   - Resultado: Modelo continuou viciado

2. **Class Weights Extremos** ❌
   - Weights multiplicados por 3-6x
   - Resultado: Inverteu o viés (agora para a classe minoritária)

3. **Class Weights Balanceados (sqrt)** ❌
   - Fórmula: `sqrt(max/count) * 1.5`
   - Resultado: Retornou ao viés original

4. **Focal Loss + Oversampling** ⚠️
   - Alcançou **87.33% de acurácia** durante treinamento
   - Resultado: Não foi possível validar devido a problemas técnicos

#### 💡 Solução Proposta (Não Implementada)

**Abordagem Recomendada:**

```python
# 1. OVERSAMPLING (duplicar imagens minoritárias)
# Balancear TODAS as classes para 1052 imagens

# 2. FOCAL LOSS (sem class weights)
# Foca automaticamente nas classes difíceis

# 3. ENSEMBLE
# Combinar múltiplos modelos treinados diferentemente
```

**Resultados Esperados:**
- Acurácia real: 70-85%
- Todas as classes: 60-90% recall
- Distribuição balanceada de predições

#### 🔧 Workaround Temporário

Se precisar usar o modelo atual:

```python
# Ajustar threshold de confiança por classe
thresholds = {
    'daisy': 0.40,      # Mais restritivo
    'dandelion': 0.60,  # Muito restritivo (classe dominante)
    'rose': 0.35,
    'sunflower': 0.35,
    'tulip': 0.40
}

# Predição com threshold
pred = modelo.predict(imagem)
classe_pred = np.argmax(pred)
confianca = pred[classe_pred]

if confianca < thresholds[classe_names[classe_pred]]:
    # Tentar segunda opção
    pred_sorted = np.argsort(pred)[::-1]
    classe_pred = pred_sorted[1]
```

#### 📊 Impacto

| Aspecto | Status |
|---------|--------|
| **Métricas agregadas** | ✅ Aparentemente boas (90%+) |
| **Testes individuais** | ❌ Falham completamente |
| **Uso em produção** | ❌ **NÃO RECOMENDADO** |
| **Valor educacional** | ✅ Excelente para aprender |

#### 🎓 Lições Aprendidas

1. **Métricas podem enganar**: Acurácia alta ≠ Modelo funcional
2. **Teste com exemplos reais**: Sempre teste manualmente
3. **Class weights não são mágicos**: Nem sempre resolvem desbalanceamento
4. **Oversampling > Class Weights**: Balancear dados é mais efetivo
5. **Matriz de confusão é crucial**: Revela problemas que métricas escondem

#### 🔗 Arquivos Relacionados

```
classificador_focal_loss.py       # Versão com Focal Loss (87.33%)
teste_avancado.py                 # Script que revelou o problema
diagnostico_modelo.py             # Análise profunda do viés
classificador_simples_final.py    # Versão simplificada
```

#### 📞 Status & Próximos Passos

**Status Atual**: 🔴 Projeto em pausa devido ao problema de viés

**Para resolver**:
1. Implementar oversampling completo
2. Testar Focal Loss corretamente
3. Validar com teste_avancado.py
4. Considerar ensemble de modelos

**Tempo estimado para correção**: 1-2 dias de trabalho focado

---

## 🗺️ Roadmap

### ✅ Fase 1: MVP (Concluído com Ressalvas)
- [x] Setup do projeto
- [x] Implementação Transfer Learning
- [x] Treinamento básico
- [x] Scripts de teste
- [x] Documentação
- [x] ⚠️ **Identificação do problema de viés**

### 🔄 Fase 2: Correção Crítica (Em Andamento)
- [ ] **Resolver problema de viés** (PRIORIDADE MÁXIMA)
- [ ] Implementar oversampling efetivo
- [ ] Validar com testes rigorosos
- [ ] Re-treinar modelo final

### 🚀 Fase 3: Melhorias (Bloqueado)
- [ ] Interface Web com Streamlit
- [ ] API REST com FastAPI
- [ ] Fine-tuning do modelo base
- [ ] Adicionar mais classes de flores

### 🏭 Fase 4: Produção (Planejado)
- [ ] Deploy em nuvem (Heroku/AWS)
- [ ] App mobile (React Native)
- [ ] Dockerização
- [ ] CI/CD Pipeline
- [ ] Testes automatizados

### 💡 Ideias Futuras
- [ ] Detecção de doenças em flores
- [ ] Classificação por espécie (não apenas gênero)
- [ ] Ensemble com outros modelos
- [ ] Dataset expandido (50+ classes)

---

## 🤝 Contribuindo

Contribuições são **muito bem-vindas**! 

### Como Contribuir

1. **Fork** o projeto
2. Crie sua **Feature Branch** (`git checkout -b feature/NovaFeature`)
3. **Commit** suas mudanças (`git commit -m 'Adiciona nova feature'`)
4. **Push** para a Branch (`git push origin feature/NovaFeature`)
5. Abra um **Pull Request**

### 📝 Diretrizes

- Siga o estilo de código existente
- Adicione testes quando aplicável
- Atualize a documentação
- Descreva claramente suas mudanças no PR

### 🐛 Reportar Bugs

Encontrou um bug? Abra uma [Issue](https://github.com/seu-usuario/classificador-flores/issues) com:

- Descrição clara do problema
- Passos para reproduzir
- Comportamento esperado vs atual
- Screenshots (se aplicável)
- Ambiente (OS, Python version, etc.)

---

## 📜 Licença

Este projeto está sob a licença **MIT**. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

```
MIT License

Copyright (c) 2025 Seu Nome

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 📬 Contato

**Seu Nome**

- GitHub: [@maricarminate](https://github.com/maricarminate)
- LinkedIn: [Mariana Carminate](www.linkedin.com/in/mariana-santos-carminate-0a0893133)
- Email: mari.carminate@gmail.com

**Link do Projeto**: [(https://github.com/maricarminate/Flowers-Recognition)]

---

## 🙏 Agradecimentos

- [Kaggle](https://kaggle.com) - Dataset Flowers Recognition
- [TensorFlow](https://tensorflow.org) - Framework de Deep Learning
- [Google](https://research.google) - Arquitetura MobileNetV2
- [Anthropic](https://anthropic.com) - Assistência com Claude
- Comunidade Open Source - Inspiração e suporte

---

## 📚 Referências

1. **MobileNetV2**: [Sandler et al., 2018](https://arxiv.org/abs/1801.04381)
2. **Transfer Learning**: [Pan & Yang, 2010](https://ieeexplore.ieee.org/document/5288526)
3. **ImageNet**: [Deng et al., 2009](http://www.image-net.org/papers/imagenet_cvpr09.pdf)
4. **Deep Learning Book**: [Goodfellow et al., 2016](https://www.deeplearningbook.org/)

---

## 📊 Status do Projeto

![GitHub last commit](https://img.shields.io/github/last-commit/seu-usuario/classificador-flores)
![GitHub issues](https://img.shields.io/github/issues/seu-usuario/classificador-flores)
![GitHub pull requests](https://img.shields.io/github/issues-pr/seu-usuario/classificador-flores)
![GitHub stars](https://img.shields.io/github/stars/seu-usuario/classificador-flores?style=social)

---

<div align="center">

**⭐ Se este projeto te ajudou, considere dar uma estrela! ⭐**

Feito com ❤️ e 🌸 por [Mariana S Carminate](https://github.com/maricarminate)

</div>
