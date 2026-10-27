import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings('ignore')

def funcao_richards(t, K, A, B, M, V):
    arg = -B * (t - M)
    arg = np.clip(arg, -100, 100)
    denominador = (1.0 + np.exp(arg)) ** (1.0 / max(V, 1e-6))
    return A + (K - A) / denominador

def ajustar_janela_cega(y_treino):
    """Ajusta a curva recebendo APENAS a porcentagem liberada (Treino)"""
    tamanho_treino = len(y_treino)
    t_treino = np.arange(tamanho_treino) 
    
    K_init, A_init = np.max(y_treino), np.min(y_treino)
    chutes = [K_init, A_init, 0.1, tamanho_treino / 2.0, 1.0]
    limites_inf = [np.min(y_treino), -np.inf, 0.001, -np.inf, 0.001]
    limites_sup = [np.inf, np.inf, 10.0, np.inf, 50.0]
    
    try:
        popt, _ = curve_fit(
            funcao_richards, t_treino, y_treino, 
            p0=chutes, bounds=(limites_inf, limites_sup), maxfev=10000
        )
        return popt
    except RuntimeError:
        return None

if __name__ == "__main__":
    print("=" * 60)
    print("TESTE DE PREDIÇÃO CEGA (TRAIN/TEST SPLIT)")
    print("=" * 60)

    # Entradas do usuário
    entrada_janela = input("Digite o tamanho total da janela (ex: 100): ")
    tamanho_janela = int(entrada_janela) if entrada_janela.strip() else 100

    entrada_porcent = input("Digite a porcentagem visível para o modelo (ex: 80): ")
    porcentagem = float(entrada_porcent) if entrada_porcent.strip() else 80.0
    
    # Validação do arquivo
    arquivo = Path("sinteticos/dados_sinteticos-2.csv")

    try:
        df = pd.read_csv(arquivo)
        df.columns = [col.strip().lower() for col in df.columns]
        valores_totais = df['close'].values if 'close' in df.columns else df['valores'].values
    except KeyError as e:
        print(f"❌ Erro de coluna: {e}")
        sys.exit()

    # Define quantos pontos representam o Treino e o Teste
    tamanho_treino = int(tamanho_janela * (porcentagem / 100.0))
    tamanho_teste = tamanho_janela - tamanho_treino
    passo = tamanho_janela # Pula janela por janela

    print("\n" + "=" * 60)
    print(f"Janela Total: {tamanho_janela} | Treino: {tamanho_treino} ({porcentagem}%) | Teste Cego: {tamanho_teste} ({(100-porcentagem):.1f}%)")
    print("=" * 60)

    inicio = 0
    num_janela = 1

    while (inicio + tamanho_janela) <= len(valores_totais):
        fim = inicio + tamanho_janela
        
        # Pega a janela completa (100%), mas o modelo só vai ver a parte do treino
        y_janela_completa = valores_totais[inicio:fim]
        y_treino = y_janela_completa[:tamanho_treino]
        y_teste_oculto = y_janela_completa[tamanho_treino:]
        
        t_completo = np.arange(tamanho_janela)
        t_treino = t_completo[:tamanho_treino]
        t_teste = t_completo[tamanho_treino:]
        
        # O modelo é ajustado APENAS com os 80% (y_treino)
        parametros = ajustar_janela_cega(y_treino)
        
        plt.figure(figsize=(10, 5))
        
        # Plotando os dados reais
        plt.plot(t_treino, y_treino, color='gray', linewidth=2.5, label=f'Dados Reais (Treino - {porcentagem}%)')
        plt.plot(t_teste, y_teste_oculto, color='steelblue', linewidth=2.5, label=f'Dados Reais Ocultos (Teste)')
        
        # Linha delimitadora visual
        plt.axvline(x=t_treino[-1], color='black', linestyle=':', label='Fronteira do Conhecimento (Corte)')
        
        if parametros is not None:
            # Projetando a linha de Richards para toda a janela usando os parâmetros encontrados nos 80%
            y_fit_treino = funcao_richards(t_treino, *parametros)
            y_pred_teste = funcao_richards(t_teste, *parametros)
            
            r2_treino = r2_score(y_treino, y_fit_treino)
            r2_teste = r2_score(y_teste_oculto, y_pred_teste) # Avalia o acerto no escuro
            
            plt.plot(t_treino, y_fit_treino, color='red', linewidth=3, label='Richards (Ajuste)')
            plt.plot(t_teste, y_pred_teste, color='red', linewidth=2.5, linestyle='--', label='Richards (Projeção Cega)')

            print(f"\n✅ ONDA {num_janela} | Treino: R² {r2_treino:.4f} | Previsão Oculta: R² {r2_teste:.4f}")
            titulo = f"Onda {num_janela}: R² Ajuste = {r2_treino:.2f} | R² Projeção Cega = {r2_teste:.2f}"
        else:
            print(f"\n❌ ONDA {num_janela} | FALHA NO AJUSTE DO TREINO")
            titulo = f"Onda {num_janela}: Falha no Ajuste"
            
        plt.title(titulo, fontweight='bold')
        plt.xlabel("Dias na Janela (Resetado para 0)")
        plt.ylabel("Valores")
        plt.legend(loc="best")
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show() 
        
        inicio += passo
        num_janela += 1