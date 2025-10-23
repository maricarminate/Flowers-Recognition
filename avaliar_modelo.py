"""
SCRIPT DE AVALIAÇÃO COMPLETA DO MODELO
Gera relatório detalhado com todas as métricas de desempenho
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import json
from datetime import datetime

print("=" * 70)
print("📊 AVALIAÇÃO COMPLETA DO MODELO DE FLORES")
print("=" * 70)

# Configurações
IMG_SIZE = 224
BATCH_SIZE = 32
DIRETORIO_VALIDACAO = 'data/validation'

# Emojis para flores
EMOJI_MAP = {
    'daisy': '🌼',
    'dandelion': '🌻',
    'rose': '🌹',
    'sunflower': '🌻',
    'tulip': '🌷'
}

print("\n[1/6] Carregando modelo treinado...")

# Carregar modelo
try:
    if os.path.exists('modelos/melhor_modelo.keras'):
        modelo = keras.models.load_model('modelos/melhor_modelo.keras')
        modelo_usado = 'melhor_modelo.keras'
    else:
        modelo = keras.models.load_model('modelos/classificador_flores_final.keras')
        modelo_usado = 'classificador_flores_final.keras'
    
    print(f"✅ Modelo carregado: {modelo_usado}")
except Exception as e:
    print(f"❌ Erro ao carregar modelo: {e}")
    print("💡 Execute primeiro: python classificador.py")
    exit(1)

# Carregar classes
try:
    with open('modelos/classes.txt', 'r', encoding='utf-8') as f:
        class_names = [linha.strip() for linha in f.readlines()]
    print(f"✅ Classes carregadas: {len(class_names)} classes")
except:
    print("⚠️  Arquivo classes.txt não encontrado, usando ordem padrão")
    class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

print("\n[2/6] Carregando dados de validação...")

# Carregar dataset de validação
val_ds = keras.preprocessing.image_dataset_from_directory(
    DIRETORIO_VALIDACAO,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False  # Importante para manter ordem
)

print(f"✅ Dados de validação carregados")

# Contar imagens por classe
print("\n📊 Distribuição do dataset de validação:")
for classe in class_names:
    caminho = f'{DIRETORIO_VALIDACAO}/{classe}'
    if os.path.exists(caminho):
        num = len([f for f in os.listdir(caminho) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        emoji = EMOJI_MAP.get(classe, '🌸')
        print(f"   {emoji} {classe:12} {num:4} imagens")

print("\n[3/6] Fazendo predições no conjunto de validação...")

# Fazer predições
y_true = []
y_pred = []

num_batches = len(val_ds)
for i, (images, labels) in enumerate(val_ds):
    if (i + 1) % 5 == 0:
        print(f"   Progresso: {i + 1}/{num_batches} batches processados...")
    
    predictions = modelo.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(predictions, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print(f"✅ {len(y_true)} predições realizadas")

print("\n[4/6] Calculando métricas de desempenho...")

# Calcular métricas
accuracy = accuracy_score(y_true, y_pred)
precision_macro = precision_score(y_true, y_pred, average='macro')
precision_weighted = precision_score(y_true, y_pred, average='weighted')
recall_macro = recall_score(y_true, y_pred, average='macro')
recall_weighted = recall_score(y_true, y_pred, average='weighted')
f1_macro = f1_score(y_true, y_pred, average='macro')
f1_weighted = f1_score(y_true, y_pred, average='weighted')

print("✅ Métricas calculadas")

print("\n[5/6] Gerando visualizações...")

# Criar figura com 3 subplots
fig = plt.figure(figsize=(20, 6))

# ==================== SUBPLOT 1: MATRIZ DE CONFUSÃO ====================
ax1 = plt.subplot(1, 3, 1)

# Calcular matriz de confusão
cm = confusion_matrix(y_true, y_pred)

# Plotar com seaborn
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues',
    xticklabels=[f"{EMOJI_MAP.get(c, '🌸')} {c.capitalize()}" for c in class_names],
    yticklabels=[f"{EMOJI_MAP.get(c, '🌸')} {c.capitalize()}" for c in class_names],
    cbar_kws={'label': 'Número de Predições'},
    ax=ax1
)

ax1.set_title('📊 Matriz de Confusão', fontsize=16, fontweight='bold', pad=20)
ax1.set_xlabel('Predição', fontsize=13, fontweight='bold')
ax1.set_ylabel('Verdadeiro', fontsize=13, fontweight='bold')
plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

# ==================== SUBPLOT 2: MÉTRICAS POR CLASSE ====================
ax2 = plt.subplot(1, 3, 2)

# Calcular métricas por classe
report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

classes_display = [f"{EMOJI_MAP.get(c, '🌸')} {c.capitalize()}" for c in class_names]
precision_per_class = [report[c]['precision'] * 100 for c in class_names]
recall_per_class = [report[c]['recall'] * 100 for c in class_names]
f1_per_class = [report[c]['f1-score'] * 100 for c in class_names]

x = np.arange(len(classes_display))
width = 0.25

bars1 = ax2.bar(x - width, precision_per_class, width, label='Precisão', color='#3498db', alpha=0.8)
bars2 = ax2.bar(x, recall_per_class, width, label='Recall', color='#2ecc71', alpha=0.8)
bars3 = ax2.bar(x + width, f1_per_class, width, label='F1-Score', color='#e74c3c', alpha=0.8)

ax2.set_xlabel('Classe', fontsize=13, fontweight='bold')
ax2.set_ylabel('Score (%)', fontsize=13, fontweight='bold')
ax2.set_title('📈 Métricas por Classe', fontsize=16, fontweight='bold', pad=20)
ax2.set_xticks(x)
ax2.set_xticklabels(classes_display, rotation=45, ha='right')
ax2.legend(fontsize=11)
ax2.set_ylim([0, 105])
ax2.grid(axis='y', alpha=0.3, linestyle='--')

# Adicionar valores nas barras
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

# ==================== SUBPLOT 3: MÉTRICAS GERAIS ====================
ax3 = plt.subplot(1, 3, 3)

# Dados das métricas gerais
metricas_nomes = ['Acurácia', 'Precisão\n(Macro)', 'Precisão\n(Weighted)', 
                  'Recall\n(Macro)', 'Recall\n(Weighted)', 'F1-Score\n(Macro)', 'F1-Score\n(Weighted)']
metricas_valores = [
    accuracy * 100,
    precision_macro * 100,
    precision_weighted * 100,
    recall_macro * 100,
    recall_weighted * 100,
    f1_macro * 100,
    f1_weighted * 100
]

colors = ['#9b59b6', '#3498db', '#5dade2', '#2ecc71', '#58d68d', '#e74c3c', '#ec7063']
bars = ax3.barh(metricas_nomes, metricas_valores, color=colors, alpha=0.8, edgecolor='black')

ax3.set_xlabel('Score (%)', fontsize=13, fontweight='bold')
ax3.set_title('🎯 Métricas Gerais do Modelo', fontsize=16, fontweight='bold', pad=20)
ax3.set_xlim([0, 105])
ax3.grid(axis='x', alpha=0.3, linestyle='--')

# Adicionar valores
for i, (bar, valor) in enumerate(zip(bars, metricas_valores)):
    ax3.text(valor + 1, i, f'{valor:.2f}%', 
            va='center', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('resultados/avaliacao_completa.png', dpi=300, bbox_inches='tight')
print("✅ Visualizações salvas: resultados/avaliacao_completa.png")

print("\n[6/6] Gerando relatório detalhado...")

# Gerar relatório em texto
relatorio_texto = f"""
{'=' * 70}
📊 RELATÓRIO DE AVALIAÇÃO DO MODELO
{'=' * 70}

Data/Hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
Modelo: {modelo_usado}

{'=' * 70}
🎯 MÉTRICAS GERAIS
{'=' * 70}

Acurácia Geral:           {accuracy * 100:.2f}%

Precisão (Macro Avg):     {precision_macro * 100:.2f}%
Precisão (Weighted Avg):  {precision_weighted * 100:.2f}%

Recall (Macro Avg):       {recall_macro * 100:.2f}%
Recall (Weighted Avg):    {recall_weighted * 100:.2f}%

F1-Score (Macro Avg):     {f1_macro * 100:.2f}%
F1-Score (Weighted Avg):  {f1_weighted * 100:.2f}%

{'=' * 70}
📈 MÉTRICAS POR CLASSE
{'=' * 70}

"""

for classe in class_names:
    emoji = EMOJI_MAP.get(classe, '🌸')
    relatorio_texto += f"\n{emoji} {classe.upper()}\n"
    relatorio_texto += f"   Precisão:  {report[classe]['precision'] * 100:6.2f}%\n"
    relatorio_texto += f"   Recall:    {report[classe]['recall'] * 100:6.2f}%\n"
    relatorio_texto += f"   F1-Score:  {report[classe]['f1-score'] * 100:6.2f}%\n"
    relatorio_texto += f"   Suporte:   {report[classe]['support']:6} imagens\n"

relatorio_texto += f"\n{'=' * 70}\n"
relatorio_texto += f"📊 TOTAL DE IMAGENS AVALIADAS: {len(y_true)}\n"
relatorio_texto += f"{'=' * 70}\n"

# Salvar relatório
with open('resultados/relatorio_avaliacao.txt', 'w', encoding='utf-8') as f:
    f.write(relatorio_texto)

print("✅ Relatório salvo: resultados/relatorio_avaliacao.txt")

# Salvar métricas em JSON
metricas_json = {
    'data_avaliacao': datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
    'modelo': modelo_usado,
    'metricas_gerais': {
        'acuracia': float(accuracy),
        'precisao_macro': float(precision_macro),
        'precisao_weighted': float(precision_weighted),
        'recall_macro': float(recall_macro),
        'recall_weighted': float(recall_weighted),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted)
    },
    'metricas_por_classe': {
        classe: {
            'precisao': float(report[classe]['precision']),
            'recall': float(report[classe]['recall']),
            'f1_score': float(report[classe]['f1-score']),
            'suporte': int(report[classe]['support'])
        }
        for classe in class_names
    },
    'total_imagens': int(len(y_true))
}

with open('resultados/metricas.json', 'w', encoding='utf-8') as f:
    json.dump(metricas_json, f, indent=4, ensure_ascii=False)

print("✅ Métricas JSON salvas: resultados/metricas.json")

# Exibir relatório na tela
print("\n" + "=" * 70)
print("🎉 AVALIAÇÃO CONCLUÍDA!")
print("=" * 70)

print(relatorio_texto)

# Análise de desempenho
print("=" * 70)
print("💡 ANÁLISE DE DESEMPENHO")
print("=" * 70)

if accuracy >= 0.95:
    print("\n   🌟 EXCELENTE! Modelo de altíssima qualidade!")
elif accuracy >= 0.90:
    print("\n   👍 MUITO BOM! Modelo pronto para produção!")
elif accuracy >= 0.85:
    print("\n   ✅ BOM! Desempenho satisfatório!")
elif accuracy >= 0.75:
    print("\n   ⚠️  RAZOÁVEL. Considere mais treinamento ou dados.")
else:
    print("\n   ❌ BAIXO. Modelo precisa de melhorias significativas.")

# Identificar melhor e pior classe
melhor_classe = max(class_names, key=lambda c: report[c]['f1-score'])
pior_classe = min(class_names, key=lambda c: report[c]['f1-score'])

emoji_melhor = EMOJI_MAP.get(melhor_classe, '🌸')
emoji_pior = EMOJI_MAP.get(pior_classe, '🌸')

print(f"\n   🏆 Melhor classe: {emoji_melhor} {melhor_classe.capitalize()} "
      f"(F1: {report[melhor_classe]['f1-score']*100:.2f}%)")
print(f"   📉 Pior classe: {emoji_pior} {pior_classe.capitalize()} "
      f"(F1: {report[pior_classe]['f1-score']*100:.2f}%)")

print("\n" + "=" * 70)
print("📁 ARQUIVOS GERADOS:")
print("=" * 70)
print("   ✅ resultados/avaliacao_completa.png")
print("   ✅ resultados/relatorio_avaliacao.txt")
print("   ✅ resultados/metricas.json")
print("=" * 70)