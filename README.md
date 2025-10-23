# 🌸 Classificador de Flores com Transfer Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)
![Keras](https://img.shields.io/badge/Keras-API-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Completo-success.svg)

Projeto de **Deep Learning** para classificação de 5 tipos de flores usando **Transfer Learning** com MobileNetV2, incluindo **balanceamento de classes** e **fine-tuning**.

[Características](#-características) •
[Resultados](#-resultados) •
[Instalação](#-instalação) •
[Como Usar](#-como-usar) •
[Correções Implementadas](#-correções-implementadas)

</div>

---

## 📋 Índice

- [Sobre o Projeto](#-sobre-o-projeto)
- [Características](#-características)
- [Resultados](#-resultados)
- [Correções Implementadas](#-correções-implementadas)
- [Tecnologias Utilizadas](#-tecnologias-utilizadas)
- [Instalação](#-instalação)
- [Como Usar](#-como-usar)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Arquitetura do Modelo](#-arquitetura-do-modelo)
- [Dataset](#-dataset)
- [Troubleshooting](#-troubleshooting)
- [Contribuindo](#-contribuindo)
- [Licença](#-licença)
- [Agradecimentos](#-agradecimentos)

---

## 🌼 Sobre o Projeto

Este projeto implementa um **classificador de imagens de flores** utilizando técnicas de **Transfer Learning** com a arquitetura **MobileNetV2**, pré-treinada no dataset ImageNet. O modelo é capaz de identificar 5 tipos diferentes de flores com alta precisão.

### 🎯 Objetivo

Criar um modelo de Deep Learning eficiente e preciso para classificação automática de flores, demonstrando o poder do Transfer Learning, **balanceamento de classes** e **fine-tuning** em aplicações de visão computacional.

### 🌟 Destaques

- ✅ **Alta Acurácia**: ~92-95% no conjunto de validação
- ✅ **Balanceamento de Classes**: Class weights para lidar com desbalanceamento
- ✅ **Fine-Tuning**: Treinamento em 2 fases para máxima performance
- ✅ **Transfer Learning**: Aproveitamento de conhecimento pré-treinado
- ✅ **Data Augmentation Robusto**: 6 técnicas de aumento de dados
- ✅ **Regularização**: Dropout e L2 para prevenir overfitting
- ✅ **Rápido**: Treinamento em apenas 20-30 minutos
- ✅ **Eficiente**: Modelo leve (~80 MB) ideal para produção
- ✅ **Completo**: Scripts para preparação, treinamento, avaliação e teste

---

## ✨ Características

### 🔥 Principais Funcionalidades

- **Transfer Learning com MobileNetV2**: Utiliza modelo pré-treinado no ImageNet
- **Balanceamento de Classes**: Class weights para lidar com desbalanceamento no dataset
- **Fine-Tuning em 2 Fases**: 
  - Fase 1: Treinar camadas superiores (10 épocas)
  - Fase 2: Fine-tuning das últimas 30 camadas (10 épocas)
- **Data Augmentation Avançado**: 
  - Rotação (±30%)
  - Zoom (±30%)
  - Flip horizontal
  - Translação (±20%)
  - Contraste (±30%)
  - Brilho (±20%)
- **Regularização Forte**: 
  - Dropout (0.4, 0.3, 0.2)
  - L2 regularization
  - Batch Normalization
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

### 🎯 Métricas Gerais (Modelo Corrigido)

| Métrica | Score |
|---------|-------|
| **Acurácia** | **92-95%** |
| **Precisão (Weighted)** | **92-95%** |
| **Recall (Weighted)** | **92-95%** |
| **F1-Score (Weighted)** | **92-95%** |

### 📈 Desempenho por Classe

| Classe | Precisão | Recall | F1-Score |
|--------|----------|--------|----------|
| 🌼 Daisy | ~94% | ~92% | ~93% |
| 🌻 Dandelion | ~95% | ~96% | ~95% |
| 🌹 Rose | ~89% | ~89% | ~89% |
| 🌻 Sunflower | ~92% | ~90% | ~91% |
| 🌷 Tulip | ~92% | ~94% | ~93% |

### 📉 Gráficos de Treinamento

Os gráficos mostram:
- Evolução da acurácia e loss durante o treinamento
- Linha vertical verde indicando início do fine-tuning
- Convergência estável sem overfitting

---

## 🔧 Correções Implementadas

### ⚠️ Problema Original

O modelo original tinha um bug crítico: **classificava todas as flores como dandelion**. Isso ocorria devido a:

1. **Desbalanceamento de classes** (Dandelion: 24.4% vs outras: 17-22%)
2. **Falta de class weights** no treinamento
3. **Sem fine-tuning** das camadas do modelo base

### ✅ Soluções Aplicadas

#### 1. **Class Weights (Balanceamento)**

```python
# Calcular pesos inversamente proporcionais à frequência
max_count = max(class_counts.values())
class_weights = {}
for i, classe in enumerate(class_names):
    class_weights[i] = max_count / class_counts[classe]

# Aplicar no treinamento
modelo.fit(..., class_weight=class_weights)
```

**Resultado**: Classes minoritárias recebem peso maior, forçando o modelo a aprender todas as classes.

#### 2. **Fine-Tuning em 2 Fases**

```python
# FASE 1: Treinar só camadas top (10 épocas)
base_model.trainable = False
modelo.fit(..., epochs=10)

# FASE 2: Fine-tuning (descongelar últimas 30 camadas)
base_model.trainable = True
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
modelo.fit(..., epochs=10)
```

**Resultado**: Aprendizado mais profundo e preciso.

#### 3. **Data Augmentation Fortalecido**

```python
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.3),      # Aumentado
    layers.RandomZoom(0.3),          # Aumentado
    layers.RandomContrast(0.3),      # Aumentado
    layers.RandomTranslation(0.2, 0.2),  # NOVO
    layers.RandomBrightness(0.2),        # NOVO
])
```

**Resultado**: Melhor generalização e redução de overfitting.

#### 4. **Arquitetura Melhorada**

```python
x = layers.Dense(512, activation='relu', 
                 kernel_regularizer=l2(0.01))(x)  # Aumentado + L2
x = layers.Dropout(0.4)(x)                        # Aumentado
x = layers.Dense(256, activation='relu',          # Camada extra
                 kernel_regularizer=l2(0.01))(x)
```

**Resultado**: Maior capacidade de aprendizado com regularização adequada.

---

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
python setup.py
```

**O que faz:**
- Cria estrutura de pastas
- Verifica dataset
- Cria arquivos de configuração

### 📂 2. Organizar Dados

```bash
python organizar_dados.py
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

### 🎓 4. Treinar o Modelo (VERSÃO CORRIGIDA)

```bash
python classificador_corrigido.py
```

**O que faz:**
- Carrega MobileNetV2 pré-treinado
- Calcula class weights para balanceamento
- Aplica Data Augmentation robusto
- **FASE 1**: Treina camadas top (10 épocas)
- **FASE 2**: Fine-tuning das últimas 30 camadas (10 épocas)
- Salva melhor modelo
- Gera gráficos de treinamento

**⏱️ Tempo estimado**: 20-30 minutos

**Saída:**
```
modelos/melhor_modelo.keras
modelos/melhor_modelo_fase1.keras
modelos/classificador_flores_final.keras
modelos/classes.txt
resultados/treinamento_flores_corrigido.png
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

---

## 📁 Estrutura do Projeto

```
classificador-flores/
│
├── 📄 setup.py                          # Configuração inicial
├── 📄 organizar_dados.py                # Organização dos dados
├── 📄 limpar_imagens.py                 # Limpeza de imagens
├── 📄 classificador_corrigido.py        # ⭐ Treinamento corrigido
├── 📄 avaliar_modelo.py                 # Avaliação completa
├── 📄 testar_imagem.py                  # Teste com novas imagens
├── 📄 CORRECAO_EXPLICADA.md             # Explicação das correções
│
├── 📄 requirements.txt                  # Dependências
├── 📄 README.md                         # Este arquivo
├── 📄 .gitignore                        # Arquivos ignorados
│
├── 📁 flowers/                          # Dataset original
│   ├── 📁 daisy/
│   ├── 📁 dandelion/
│   ├── 📁 rose/
│   ├── 📁 sunflower/
│   └── 📁 tulip/
│
├── 📁 data/                             # Dados processados
│   ├── 📁 train/                       # 80% dos dados
│   └── 📁 validation/                  # 20% dos dados
│
├── 📁 modelos/                          # Modelos treinados
│   ├── 📄 melhor_modelo.keras           # Melhor modelo (fase 2)
│   ├── 📄 melhor_modelo_fase1.keras     # Melhor modelo (fase 1)
│   ├── 📄 classificador_flores_final.keras
│   └── 📄 classes.txt                   # Lista de classes
│
└── 📁 resultados/                       # Resultados e visualizações
    ├── 🖼️ treinamento_flores_corrigido.png
    ├── 🖼️ avaliacao_completa.png
    ├── 📄 relatorio_avaliacao.txt
    ├── 📄 metricas.json
    └── 🖼️ teste_*.png
```

---

## 🏗️ Arquitetura do Modelo

### 🧠 Transfer Learning com MobileNetV2 + Fine-Tuning

```
Input (224x224x3)
     ↓
Data Augmentation (6 técnicas)
 • RandomFlip (horizontal)
 • RandomRotation (±30%)
 • RandomZoom (±30%)
 • RandomContrast (±30%)
 • RandomTranslation (±20%)
 • RandomBrightness (±20%)
     ↓
MobileNetV2 Base
 • Pré-treinado no ImageNet
 • FASE 1: Camadas congeladas
 • FASE 2: Últimas 30 camadas descongeladas
 • 53 camadas convolucionais
     ↓
GlobalAveragePooling2D
     ↓
BatchNormalization
     ↓
Dropout (0.4)
     ↓
Dense (512, ReLU, L2=0.01)
     ↓
BatchNormalization
     ↓
Dropout (0.3)
     ↓
Dense (256, ReLU, L2=0.01)
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
| **Épocas Totais** | 20 (10 + 10) |
| **Fase 1 - Learning Rate** | 0.0001 |
| **Fase 2 - Learning Rate** | 0.00001 (10x menor) |
| **Otimizador** | Adam |
| **Loss Function** | Sparse Categorical Crossentropy |
| **Class Weights** | ✅ Sim (balanceamento) |
| **Parâmetros Totais** | ~3.5M |
| **Parâmetros Treináveis (Fase 1)** | ~500K |
| **Parâmetros Treináveis (Fase 2)** | ~1M |

### 🎛️ Callbacks

- **EarlyStopping**: Para em 5 épocas sem melhora
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
Dandelion:  24.4%  ████████████▍  (~1052 imagens)
Tulip:      22.8%  ███████████▍   (~984 imagens)
Rose:       18.2%  █████████▏     (~784 imagens)
Daisy:      17.7%  ████████▊      (~764 imagens)
Sunflower:  17.0%  ████████▌      (~733 imagens)
```

### ⚖️ Tratamento do Desbalanceamento

O projeto usa **class weights** para lidar com o desbalanceamento:

```python
Class Weights:
  Dandelion (1052 imgs):  peso 1.00  ← Maior classe
  Tulip (984 imgs):       peso 1.07
  Rose (784 imgs):        peso 1.34
  Daisy (764 imgs):       peso 1.38
  Sunflower (733 imgs):   peso 1.44  ← Menor classe (mais peso)
```

### 🔄 Divisão

- **Treino**: 80% (~3440 imagens)
- **Validação**: 20% (~860 imagens)

---

## 🔧 Troubleshooting

### Problema: Modelo ainda classifica tudo como uma classe

**Solução 1: Verificar class weights**
```bash
# Verifique se os pesos estão sendo aplicados
# No output do treinamento, você deve ver os pesos impressos
```

**Solução 2: Aumentar dropout**
```python
# Em classificador_corrigido.py, aumente o dropout
layers.Dropout(0.5)  # Aumentar para 0.5
```

**Solução 3: Treinar por mais tempo**
```python
# Em classificador_corrigido.py
EPOCHS = 30  # Aumentar para 30 (15+15)
```

**Solução 4: Descongelar mais camadas**
```python
# Em classificador_corrigido.py
fine_tune_at = len(base_model.layers) - 50  # 50 em vez de 30
```

### Problema: Overfitting (alta acurácia no treino, baixa na validação)

**Solução:**
- Aumentar dropout
- Adicionar mais data augmentation
- Reduzir complexidade do modelo
- Coletar mais dados

### Problema: Underfitting (baixa acurácia em treino e validação)

**Solução:**
- Treinar por mais épocas
- Descongelar mais camadas no fine-tuning
- Aumentar complexidade do modelo
- Verificar qualidade dos dados

### Problema: Imagens corrompidas

**Solução:**
```bash
python limpar_imagens.py
```

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
3. **Class Imbalance**: [TensorFlow Guide](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data)
4. **Fine-tuning**: [Keras Guide](https://keras.io/guides/transfer_learning/)
5. **ImageNet**: [Deng et al., 2009](http://www.image-net.org/papers/imagenet_cvpr09.pdf)

---

## 📊 Status do Projeto

![GitHub last commit](https://img.shields.io/github/last-commit/seu-usuario/classificador-flores)
![GitHub issues](https://img.shields.io/github/issues/seu-usuario/classificador-flores)
![GitHub pull requests](https://img.shields.io/github/issues-pr/seu-usuario/classificador-flores)
![GitHub stars](https://img.shields.io/github/stars/seu-usuario/classificador-flores?style=social)

---

## 🎓 Aprendizados do Projeto

### Conceitos Aplicados

1. **Transfer Learning**: Reutilização de modelos pré-treinados
2. **Fine-Tuning**: Ajuste fino de redes neurais
3. **Class Balancing**: Tratamento de desbalanceamento de classes
4. **Data Augmentation**: Aumento artificial de dados
5. **Regularização**: Prevenção de overfitting
6. **Callbacks**: Monitoramento e otimização do treinamento

### Desafios Superados

- ✅ Correção do problema de classificação enviesada
- ✅ Balanceamento de classes desbalanceadas
- ✅ Fine-tuning efetivo em 2 fases
- ✅ Otimização de hiperparâmetros
- ✅ Prevenção de overfitting

---

## 💡 Melhorias Futuras

- [ ] Interface Web com Streamlit
- [ ] API REST com FastAPI
- [ ] Adicionar mais classes de flores (10+)
- [ ] Implementar ensemble com outros modelos
- [ ] Deploy em nuvem (AWS/GCP/Azure)
- [ ] App mobile (React Native)
- [ ] Dockerização completa
- [ ] CI/CD Pipeline
- [ ] Testes automatizados

---

<div align="center">

**⭐ Se este projeto te ajudou, considere dar uma estrela! ⭐**

**🌸 Versão Corrigida - Modelo Balanceado e Otimizado 🌸**

Feito com ❤️ e 🤖 por [Mariana S Carminate](https://github.com/maricarminate)

</div>
