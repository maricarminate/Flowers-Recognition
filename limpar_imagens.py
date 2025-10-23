"""
SCRIPT: Limpar Imagens Corrompidas
Verifica e remove imagens inválidas de data/train e data/validation
"""

import os
from PIL import Image

print("=" * 70)
print("🧹 LIMPEZA DE IMAGENS CORROMPIDAS")
print("=" * 70)

# Classes de flores
CLASSES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Diretórios para verificar
diretorios = []
for tipo in ['train', 'validation']:
    for classe in CLASSES:
        caminho = f'data/{tipo}/{classe}'
        if os.path.exists(caminho):
            diretorios.append(caminho)

def verificar_imagem(caminho):
    """
    Verifica se a imagem está OK
    Retorna (True, "OK") ou (False, "motivo do erro")
    """
    try:
        # Abrir imagem
        img = Image.open(caminho)
        
        # Verificar se consegue ler
        img.verify()
        
        # Reabrir (verify fecha o arquivo)
        img = Image.open(caminho)
        img.load()
        
        # Verificar tamanho mínimo (pelo menos 10x10 pixels)
        if img.size[0] < 10 or img.size[1] < 10:
            return False, "Imagem muito pequena"
        
        # Verificar se tem pelo menos 1 canal
        if hasattr(img, 'mode'):
            if img.mode not in ['RGB', 'RGBA', 'L', 'P']:
                return False, f"Modo inválido: {img.mode}"
        
        return True, "OK"
    
    except FileNotFoundError:
        return False, "Arquivo não encontrado"
    except Image.UnidentifiedImageError:
        return False, "Formato de imagem não reconhecido"
    except OSError as e:
        return False, f"Erro ao ler arquivo: {str(e)}"
    except Exception as e:
        return False, f"Erro desconhecido: {str(e)}"

print("\n🔍 Iniciando verificação...\n")

total_verificadas = 0
total_removidas = 0
total_ok = 0
problemas_por_tipo = {}

for diretorio in diretorios:
    print(f"📁 {diretorio}")
    
    # Listar arquivos de imagem
    arquivos = [f for f in os.listdir(diretorio) 
               if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]
    
    if len(arquivos) == 0:
        print(f"   ⚠️  Pasta vazia!")
        print()
        continue
    
    print(f"   Total de arquivos: {len(arquivos)}")
    
    removidas_aqui = 0
    
    # Verificar cada imagem
    for i, arquivo in enumerate(arquivos):
        caminho_completo = os.path.join(diretorio, arquivo)
        total_verificadas += 1
        
        # Mostrar progresso a cada 100 imagens
        if (i + 1) % 100 == 0:
            print(f"   Progresso: {i + 1}/{len(arquivos)} verificadas...")
        
        # Verificar imagem
        ok, mensagem = verificar_imagem(caminho_completo)
        
        if not ok:
            print(f"   ❌ PROBLEMA: {arquivo}")
            print(f"      Motivo: {mensagem}")
            
            # Contar tipo de problema
            if mensagem not in problemas_por_tipo:
                problemas_por_tipo[mensagem] = 0
            problemas_por_tipo[mensagem] += 1
            
            # Tentar remover
            try:
                os.remove(caminho_completo)
                print(f"      🗑️  Removida!")
                total_removidas += 1
                removidas_aqui += 1
            except Exception as e:
                print(f"      ⚠️  Erro ao remover: {e}")
        else:
            total_ok += 1
    
    # Resumo do diretório
    if removidas_aqui == 0:
        print(f"   ✅ Todas as {len(arquivos)} imagens estão OK!")
    else:
        print(f"   ⚠️  {removidas_aqui} imagens removidas")
        print(f"   ✅ {len(arquivos) - removidas_aqui} imagens OK")
    
    print()

# Resumo geral
print("=" * 70)
print("📊 RESUMO DA LIMPEZA")
print("=" * 70)

print(f"\n   🔍 Total verificadas: {total_verificadas}")
print(f"   ✅ Imagens OK: {total_ok}")
print(f"   ❌ Imagens removidas: {total_removidas}")

if total_removidas > 0:
    porcentagem = (total_removidas / total_verificadas) * 100
    print(f"\n   ⚠️  {total_removidas} imagens corrompidas foram removidas!")
    print(f"   📊 Porcentagem: {porcentagem:.2f}%")
    
    # Mostrar tipos de problemas
    if problemas_por_tipo:
        print(f"\n   📋 PROBLEMAS ENCONTRADOS:")
        for problema, count in sorted(problemas_por_tipo.items(), key=lambda x: x[1], reverse=True):
            print(f"      • {problema}: {count} imagem(ns)")
else:
    print(f"\n   🎉 Nenhuma imagem corrompida encontrada!")
    print(f"   ✨ Todas as {total_verificadas} imagens estão perfeitas!")

# Contagem final por classe
print("\n" + "=" * 70)
print("📁 CONTAGEM FINAL POR CLASSE")
print("=" * 70)

print("\n📁 TREINO:")
total_treino = 0
for classe in CLASSES:
    caminho = f'data/train/{classe}'
    if os.path.exists(caminho):
        num = len([f for f in os.listdir(caminho) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        total_treino += num
        emoji = {'daisy': '🌼', 'dandelion': '🌻', 'rose': '🌹', 
                'sunflower': '🌻', 'tulip': '🌷'}.get(classe, '🌸')
        print(f"   {emoji} {classe:12} {num:4} imagens")
print(f"\n   📊 TOTAL: {total_treino} imagens")

print("\n📁 VALIDAÇÃO:")
total_val = 0
for classe in CLASSES:
    caminho = f'data/validation/{classe}'
    if os.path.exists(caminho):
        num = len([f for f in os.listdir(caminho) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        total_val += num
        emoji = {'daisy': '🌼', 'dandelion': '🌻', 'rose': '🌹', 
                'sunflower': '🌻', 'tulip': '🌷'}.get(classe, '🌸')
        print(f"   {emoji} {classe:12} {num:4} imagens")
print(f"\n   📊 TOTAL: {total_val} imagens")

print("\n" + "=" * 70)
print("✅ LIMPEZA CONCLUÍDA!")
print("=" * 70)

if total_treino + total_val < 100:
    print("\n⚠️  ATENÇÃO: Poucas imagens disponíveis!")
    print("   Recomendado: pelo menos 100 imagens por classe")
elif total_val == 0:
    print("\n⚠️  ATENÇÃO: Sem imagens de validação!")
    print("   Execute: python reorganizar_treino_validacao.py")
else:
    print("\n✅ Dataset pronto para treinamento!")
    print("\n📝 PRÓXIMO PASSO:")
    print("   python classificador.py")

print("=" * 70)