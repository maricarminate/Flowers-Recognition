"""
SCRIPT: Testar Modelo com Novas Imagens
Uso: python testar_imagem.py caminho/para/imagem.jpg
"""

import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from PIL import Image

# Mapear emojis para flores
EMOJI_MAP = {
    'daisy': '🌼',
    'dandelion': '🌻',
    'rose': '🌹',
    'sunflower': '🌻',
    'tulip': '🌷'
}

def carregar_modelo():
    """Carrega o modelo treinado e as classes"""
    try:
        # Tentar carregar o melhor modelo
        if tf.io.gfile.exists('modelos/melhor_modelo.keras'):
            modelo = keras.models.load_model('modelos/melhor_modelo.keras')
            print("   📦 Modelo: melhor_modelo.keras")
        elif tf.io.gfile.exists('modelos/classificador_flores_final.keras'):
            modelo = keras.models.load_model('modelos/classificador_flores_final.keras')
            print("   📦 Modelo: classificador_flores_final.keras")
        else:
            print("   ❌ Nenhum modelo encontrado!")
            return None, None
        
        # Carregar classes
        with open('modelos/classes.txt', 'r', encoding='utf-8') as f:
            classes = [linha.strip() for linha in f.readlines()]
        
        return modelo, classes
    
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        print("💡 Execute primeiro: python classificador.py")
        return None, None

def preprocessar_imagem(caminho, img_size=224):
    """Carrega e preprocessa a imagem"""
    # Carregar imagem
    img = Image.open(caminho)
    img_original = img.copy()
    
    # Converter RGBA para RGB se necessário
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Redimensionar
    img = img.resize((img_size, img_size))
    img_array = np.array(img)
    
    # Adicionar dimensão do batch
    img_array = np.expand_dims(img_array, axis=0)
    
    # Preprocessamento do MobileNetV2
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    
    return img_original, img_array

def prever_flor(modelo, img_array, classes):
    """Faz a predição e retorna resultados ordenados"""
    predicoes = modelo.predict(img_array, verbose=0)[0]
    
    # Ordenar por probabilidade (maior para menor)
    indices_ordenados = np.argsort(predicoes)[::-1]
    
    resultados = []
    for idx in indices_ordenados:
        resultados.append({
            'classe': classes[idx],
            'probabilidade': predicoes[idx] * 100
        })
    
    return resultados

def visualizar_resultado(img, resultados, caminho_original):
    """Cria visualização bonita do resultado"""
    
    fig = plt.figure(figsize=(16, 7))
    
    # Layout: imagem + gráfico de barras horizontais
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2], wspace=0.3)
    
    # LADO ESQUERDO: Imagem original
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(img)
    ax1.axis('off')
    
    # Título com resultado principal
    top_result = resultados[0]
    emoji = EMOJI_MAP.get(top_result['classe'], '🌸')
    
    cor_titulo = '#2ecc71' if top_result['probabilidade'] > 80 else '#f39c12'
    
    ax1.set_title(
        f"{emoji} {top_result['classe'].upper()}\n{top_result['probabilidade']:.1f}% de Confiança",
        fontsize=20,
        fontweight='bold',
        pad=25,
        color=cor_titulo
    )
    
    # LADO DIREITO: Gráfico de probabilidades
    ax2 = fig.add_subplot(gs[1])
    
    # Preparar dados
    classes_display = []
    probs = []
    cores = []
    
    for i, r in enumerate(resultados):
        emoji = EMOJI_MAP.get(r['classe'], '🌸')
        classes_display.append(f"{emoji} {r['classe'].capitalize()}")
        probs.append(r['probabilidade'])
        
        # Cor: verde para o primeiro, azul para os outros
        if i == 0:
            cores.append('#2ecc71')  # Verde
        else:
            cores.append('#3498db')  # Azul
    
    # Criar barras horizontais
    y_pos = np.arange(len(classes_display))
    bars = ax2.barh(y_pos, probs, color=cores, alpha=0.85, edgecolor='black', linewidth=1.5)
    
    # Adicionar valores nas barras
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        width = bar.get_width()
        
        # Cor do texto
        text_color = 'white' if prob > 50 else 'black'
        
        ax2.text(
            width / 2, 
            bar.get_y() + bar.get_height() / 2,
            f'{prob:.1f}%',
            ha='center',
            va='center',
            fontsize=13,
            fontweight='bold',
            color=text_color
        )
    
    # Configurar eixos
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(classes_display, fontsize=13)
    ax2.set_xlabel('Probabilidade (%)', fontsize=14, fontweight='bold')
    ax2.set_title('📊 Distribuição de Probabilidades', fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlim(0, 100)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.invert_yaxis()  # Maior probabilidade no topo
    
    # Linha vertical em 50%
    ax2.axvline(x=50, color='red', linestyle='--', alpha=0.3, linewidth=1)
    
    plt.tight_layout()
    
    # Salvar
    nome_classe = resultados[0]['classe']
    timestamp = __import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')
    nome_saida = f"resultados/teste_{nome_classe}_{timestamp}.png"
    plt.savefig(nome_saida, dpi=200, bbox_inches='tight')
    print(f"💾 Resultado salvo: {nome_saida}")
    
    plt.show()

def main():
    """Função principal"""
    
    print("=" * 70)
    print("🌸 TESTAR MODELO - CLASSIFICADOR DE FLORES")
    print("=" * 70)
    
    # Verificar argumento
    if len(sys.argv) < 2:
        print("\n❌ ERRO: Forneça o caminho da imagem!")
        print("\n📖 USO CORRETO:")
        print("   python testar_imagem.py caminho/para/imagem.jpg")
        print("\n📝 EXEMPLOS:")
        print("   python testar_imagem.py minha_flor.jpg")
        print("   python testar_imagem.py C:\\Users\\Nome\\Desktop\\rosa.jpg")
        print("   python testar_imagem.py flowers/rose/image_001.jpg")
        print("   python testar_imagem.py data/validation/tulip/tulip_100.jpg")
        print("\n💡 DICA: Você pode testar com imagens do próprio dataset!")
        print("=" * 70)
        return
    
    caminho_imagem = sys.argv[1]
    
    # Verificar se arquivo existe
    import os
    if not os.path.exists(caminho_imagem):
        print(f"\n❌ ERRO: Arquivo não encontrado!")
        print(f"   Caminho: {caminho_imagem}")
        return
    
    # Carregar modelo
    print("\n📦 Carregando modelo...")
    modelo, classes = carregar_modelo()
    
    if modelo is None:
        return
    
    print(f"✅ Modelo carregado!")
    print(f"   Classes disponíveis: {len(classes)}")
    classes_str = ', '.join([EMOJI_MAP.get(c, '🌸') + ' ' + c.capitalize() for c in classes])
    print(f"   {classes_str}")
    
    # Carregar e preprocessar imagem
    print(f"\n🖼️  Carregando imagem: {caminho_imagem}")
    try:
        img, img_array = preprocessar_imagem(caminho_imagem)
        print("✅ Imagem carregada e preprocessada!")
        print(f"   Tamanho original: {img.size}")
    except Exception as e:
        print(f"❌ Erro ao carregar imagem: {e}")
        return
    
    # Fazer predição
    print("\n🧠 Fazendo predição...")
    resultados = prever_flor(modelo, img_array, classes)
    
    # Mostrar resultados no terminal
    print("\n" + "=" * 70)
    print("📊 RESULTADOS DA PREDIÇÃO")
    print("=" * 70)
    
    for i, resultado in enumerate(resultados, 1):
        emoji = EMOJI_MAP.get(resultado['classe'], '🌸')
        prob = resultado['probabilidade']
        classe = resultado['classe'].capitalize()
        
        if i == 1:
            print(f"\n   🏆 {i}º lugar: {emoji} {classe}")
            print(f"      Probabilidade: {prob:.2f}%")
            
            # Nível de confiança
            if prob > 95:
                print(f"      Confiança: 🟢 EXTREMAMENTE ALTA")
            elif prob > 85:
                print(f"      Confiança: 🟢 MUITO ALTA")
            elif prob > 70:
                print(f"      Confiança: 🟡 ALTA")
            elif prob > 50:
                print(f"      Confiança: 🟠 MODERADA")
            else:
                print(f"      Confiança: 🔴 BAIXA (modelo incerto)")
        else:
            print(f"\n   {i}º lugar: {emoji} {classe} - {prob:.2f}%")
    
    print("\n" + "=" * 70)
    
    # Visualizar
    print("\n📊 Gerando visualização...")
    visualizar_resultado(img, resultados, caminho_imagem)
    
    print("\n" + "=" * 70)
    print("✅ TESTE CONCLUÍDO!")
    print("=" * 70)
    print("\n💡 DICA: Teste com outras imagens para ver o desempenho!")
    print("=" * 70)

if __name__ == "__main__":
    main()