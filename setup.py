import os

print("=" * 70)
print("🌸 SETUP: CLASSIFICADOR DE FLORES")
print("=" * 70)

print("\n[1/3] Criando estrutura de pastas...")

# Pastas principais
pastas = [
    'data',
    'data/train',
    'data/train/daisy',
    'data/train/dandelion',
    'data/train/rose',
    'data/train/sunflower',
    'data/train/tulip',
    'data/validation',
    'data/validation/daisy',
    'data/validation/dandelion',
    'data/validation/rose',
    'data/validation/sunflower',
    'data/validation/tulip',
    'modelos',
    'resultados'
]

for pasta in pastas:
    os.makedirs(pasta, exist_ok=True)
    print(f"   ✅ {pasta}")

print("\n✅ Estrutura criada!")

# Verificar se existe pasta flowers/
print("\n[2/3] Verificando dataset...")

if os.path.exists('flowers'):
    print("✅ Pasta 'flowers/' encontrada!")
    
    # Contar imagens
    classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
    total = 0
    
    print("\n📊 Contagem por classe:")
    for classe in classes:
        caminho = f'flowers/{classe}'
        if os.path.exists(caminho):
            num = len([f for f in os.listdir(caminho) if f.endswith(('.jpg', '.jpeg', '.png'))])
            total += num
            print(f"   🌸 {classe:12} {num:4} imagens")
        else:
            print(f"   ❌ {classe:12} pasta não encontrada!")
    
    print(f"\n   📊 TOTAL: {total} imagens")
else:
    print("⚠️  Pasta 'flowers/' NÃO encontrada!")
    print("\n💡 INSTRUÇÕES:")
    print("   1. Baixe o dataset:")
    print("      https://www.kaggle.com/datasets/alxmamaev/flowers-recognition")
    print("   2. Extraia o arquivo flowers-recognition.zip")
    print("   3. Copie a pasta 'flowers' para este diretório")
    print("   4. Execute este script novamente")

print("\n[3/3] Criando arquivos de configuração...")

# Criar requirements.txt se não existir
if not os.path.exists('requirements.txt'):
    with open('requirements.txt', 'w') as f:
        f.write("""tensorflow==2.15.0
numpy==1.24.3
matplotlib==3.7.1
scikit-learn==1.3.0
seaborn==0.12.2
Pillow==10.0.0
""")
    print("   ✅ requirements.txt criado")
else:
    print("   ℹ️  requirements.txt já existe")

# Criar .gitignore se não existir
if not os.path.exists('.gitignore'):
    with open('.gitignore', 'w') as f:
        f.write("""# Python
__pycache__/
*.py[cod]
venv/
env/

# Dados
flowers/
data/
*.jpg
*.jpeg
*.png

# Modelos
modelos/
*.keras
*.h5

# Resultados
resultados/

# IDE
.vscode/
.idea/
*.swp
.DS_Store
""")
    print("   ✅ .gitignore criado")
else:
    print("   ℹ️  .gitignore já existe")

print("\n" + "=" * 70)
print("✅ SETUP CONCLUÍDO!")
print("=" * 70)

if os.path.exists('flowers'):
    print("\n📝 PRÓXIMO PASSO:")
    print("   python 2_organizar_dados.py")
else:
    print("\n⚠️  ANTES DE CONTINUAR:")
    print("   Baixe e extraia o dataset na pasta 'flowers/'")

print("=" * 70)