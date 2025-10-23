print("🌸 CLASSIFICADOR DE FLORES - TRANSFER LEARNING")
print("=" * 70)

# Imports
print("\n[1/6] Importando bibliotecas...")
import sys
import os

# FIX: Resolver problema de encoding no Windows
if sys.platform == 'win32':
    import locale
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    
# FIX: Forçar UTF-8 no Python
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Continua o resto do código normalmente...

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt
import numpy as np

print("✅ Bibliotecas importadas!")

# Configurações
IMG_SIZE = 224  # MobileNetV2 usa 224x224
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.0001

DIRETORIO_TREINO = 'data/train'
DIRETORIO_VALIDACAO = 'data/validation'

print("\n[2/6] Preparando dados...")
print(f"   Tamanho das imagens: {IMG_SIZE}x{IMG_SIZE}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Épocas: {EPOCHS}")

# Criar datasets usando a API do TensorFlow
train_ds = keras.preprocessing.image_dataset_from_directory(
    DIRETORIO_TREINO,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    seed=123
)

val_ds = keras.preprocessing.image_dataset_from_directory(
    DIRETORIO_VALIDACAO,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    seed=123
)

# Obter nomes das classes
class_names = train_ds.class_names
num_classes = len(class_names)

print(f"✅ Dados carregados!")
print(f"   Classes: {class_names}")
print(f"   Número de classes: {num_classes}")

# Otimização de performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Data Augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
])

# Preprocessing para MobileNetV2
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

print("\n[3/6] Criando modelo com Transfer Learning...")

# Carregar MobileNetV2 pré-treinado
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Congelar camadas do modelo base
base_model.trainable = False

print(f"   Base: MobileNetV2")
print(f"   Camadas congeladas: {len(base_model.layers)}")

# Construir modelo completo
inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

modelo = keras.Model(inputs, outputs)

print("✅ Modelo criado!")

# Compilar
modelo.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n📊 Resumo do modelo:")
modelo.summary()

# Callbacks
print("\n[4/6] Configurando callbacks...")

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        'modelos/melhor_modelo.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

print("✅ Callbacks configurados!")

# Treinar
print("\n[5/6] INICIANDO TREINAMENTO...")
print("=" * 70)
print("⏰ Tempo estimado: 15-20 minutos")
print("💡 Acompanhe o progresso abaixo:")
print("=" * 70 + "\n")

historico = modelo.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

print("\n" + "=" * 70)
print("✅ TREINAMENTO CONCLUÍDO!")
print("=" * 70)

# Salvar modelo final
print("\n[6/6] Salvando modelo e resultados...")

modelo.save('modelos/classificador_flores_final.keras')
print("✅ Modelo salvo: modelos/classificador_flores_final.keras")

# Salvar classes
with open('modelos/classes.txt', 'w', encoding='utf-8') as f:
    for classe in class_names:
        f.write(f"{classe}\n")
print("✅ Classes salvas: modelos/classes.txt")

# Gerar gráficos
print("\n📊 Gerando gráficos...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Acurácia
epochs_range = range(1, len(historico.history['accuracy']) + 1)
ax1.plot(epochs_range, historico.history['accuracy'], 
         'b-o', label='Treino', linewidth=2.5, markersize=8)
ax1.plot(epochs_range, historico.history['val_accuracy'], 
         'r-o', label='Validação', linewidth=2.5, markersize=8)
ax1.set_title('🎯 Acurácia do Modelo', fontsize=18, fontweight='bold', pad=20)
ax1.set_xlabel('Época', fontsize=14)
ax1.set_ylabel('Acurácia', fontsize=14)
ax1.legend(fontsize=13, loc='lower right')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_ylim([0, 1])

# Loss
ax2.plot(epochs_range, historico.history['loss'], 
         'b-o', label='Treino', linewidth=2.5, markersize=8)
ax2.plot(epochs_range, historico.history['val_loss'], 
         'r-o', label='Validação', linewidth=2.5, markersize=8)
ax2.set_title('📉 Loss do Modelo', fontsize=18, fontweight='bold', pad=20)
ax2.set_xlabel('Época', fontsize=14)
ax2.set_ylabel('Loss', fontsize=14)
ax2.legend(fontsize=13, loc='upper right')
ax2.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('resultados/treinamento_flores.png', dpi=300, bbox_inches='tight')
print("✅ Gráficos salvos: resultados/treinamento_flores.png")

# Resultado final
print("\n" + "=" * 70)
print("🎉 PROJETO CONCLUÍDO COM SUCESSO!")
print("=" * 70)

acuracia_final = historico.history['val_accuracy'][-1]
loss_final = historico.history['val_loss'][-1]
melhor_acuracia = max(historico.history['val_accuracy'])

print(f"\n📊 RESULTADOS FINAIS:")
print(f"   🎯 Acurácia Final: {acuracia_final:.4f} ({acuracia_final*100:.2f}%)")
print(f"   🌟 Melhor Acurácia: {melhor_acuracia:.4f} ({melhor_acuracia*100:.2f}%)")
print(f"   📉 Loss Final: {loss_final:.4f}")

if acuracia_final > 0.95:
    print("   🌟 EXCELENTE! Transfer Learning funcionou perfeitamente!")
elif acuracia_final > 0.90:
    print("   👍 MUITO BOM! Resultado acima do esperado!")
elif acuracia_final > 0.85:
    print("   ✅ BOM! Resultado satisfatório!")
else:
    print("   ⚠️  Razoável. Considere treinar por mais épocas.")

print(f"\n📁 ARQUIVOS GERADOS:")
print(f"   ✅ modelos/classificador_flores_final.keras")
print(f"   ✅ modelos/melhor_modelo.keras")
print(f"   ✅ modelos/classes.txt")
print(f"   ✅ resultados/treinamento_flores.png")

print(f"\n📊 CLASSES TREINADAS:")
emoji_map = {
    'daisy': '🌼',
    'dandelion': '🌻',
    'rose': '🌹',
    'sunflower': '🌻',
    'tulip': '🌷'
}
for i, classe in enumerate(class_names, 1):
    emoji = emoji_map.get(classe, '🌸')
    print(f"   {i}. {emoji} {classe.capitalize()}")

print(f"\n📝 PRÓXIMO PASSO:")
print(f"   python 5_testar_imagem.py caminho/para/imagem.jpg")

print("\n" + "=" * 70)
print("🌸 Obrigado por usar o Classificador de Flores! 🌸")
print("=" * 70)