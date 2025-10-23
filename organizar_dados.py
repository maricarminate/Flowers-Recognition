"""
Script para reorganizar dados quando TUDO está em data/train
Move 20% para data/validation
"""

import os
import shutil
import random

print("=" * 70)
print("🔧 REORGANIZANDO: SEPARAR TREINO E VALIDAÇÃO")
print("=" * 70)

PORCENTAGEM_VALIDACAO = 0.2  # 20% para validação
CLASSES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

print("\n[1/3] Verificando situação atual...")

# Verificar data/train
if not os.path.exists('data/train'):
    print("❌ Pasta data/train não encontrada!")
    exit(1)

# Contar imagens em train
print("\n📊 Imagens em data/train:")
total_train = 0
for classe in CLASSES:
    caminho = f'data/train/{classe}'
    if os.path.exists(caminho):
        num = len([f for f in os.listdir(caminho) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        total_train += num
        print(f"   🌸 {classe:12} {num:4} imagens")

print(f"\n   📊 TOTAL: {total_train} imagens")

# Contar em validation
print("\n📊 Imagens em data/validation:")
total_val = 0
for classe in CLASSES:
    caminho = f'data/validation/{classe}'
    if os.path.exists(caminho):
        num = len([f for f in os.listdir(caminho) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        total_val += num
        print(f"   🌸 {classe:12} {num:4} imagens")

print(f"\n   📊 TOTAL: {total_val} imagens")

if total_val > 0:
    print("\n✅ Já existe dados em validação!")
    print("⚠️  Se quiser reorganizar, delete data/validation/* primeiro")
    exit(0)

print("\n[2/3] Movendo 20% das imagens para validação...")

total_movidos = 0

for classe in CLASSES:
    origem = f'data/train/{classe}'
    destino = f'data/validation/{classe}'
    
    if not os.path.exists(origem):
        print(f"   ⚠️  {origem} não existe")
        continue
    
    # Criar pasta de destino se não existir
    os.makedirs(destino, exist_ok=True)
    
    # Listar todas as imagens
    todas_imagens = [f for f in os.listdir(origem) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Embaralhar
    random.seed(42)
    random.shuffle(todas_imagens)
    
    # Calcular quantas mover
    num_total = len(todas_imagens)
    num_mover = int(num_total * PORCENTAGEM_VALIDACAO)
    
    # Selecionar imagens para mover
    imagens_mover = todas_imagens[:num_mover]
    
    print(f"\n   🌸 {classe:12}")
    print(f"      Total: {num_total} → Movendo: {num_mover} para validação")
    
    # Mover imagens
    movidas = 0
    for img in imagens_mover:
        try:
            shutil.move(
                os.path.join(origem, img),
                os.path.join(destino, img)
            )
            movidas += 1
        except Exception as e:
            print(f"      ⚠️  Erro ao mover {img}: {e}")
    
    total_movidos += movidas
    print(f"      ✅ Movidas: {movidas} imagens")

print("\n[3/3] Verificando resultado final...")

print("\n" + "=" * 70)
print("📊 CONTAGEM FINAL")
print("=" * 70)

print("\n📁 TREINO:")
total_train_final = 0
for classe in CLASSES:
    caminho = f'data/train/{classe}'
    if os.path.exists(caminho):
        num = len([f for f in os.listdir(caminho) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        total_train_final += num
        print(f"   🌸 {classe:12} {num:4} imagens")
print(f"\n   📊 TOTAL: {total_train_final} imagens")

print("\n📁 VALIDAÇÃO:")
total_val_final = 0
for classe in CLASSES:
    caminho = f'data/validation/{classe}'
    if os.path.exists(caminho):
        num = len([f for f in os.listdir(caminho) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        total_val_final += num
        print(f"   🌸 {classe:12} {num:4} imagens")
print(f"\n   📊 TOTAL: {total_val_final} imagens")

print("\n" + "=" * 70)
print("✅ REORGANIZAÇÃO CONCLUÍDA!")
print("=" * 70)
print(f"\n   📊 Divisão final: {total_train_final} treino / {total_val_final} validação")
print(f"   📊 Porcentagem: {(total_train_final/(total_train_final+total_val_final)*100):.1f}% / {(total_val_final/(total_train_final+total_val_final)*100):.1f}%")
print("\n📝 PRÓXIMO PASSO:")
print("   python classificador.py")
print("=" * 70)