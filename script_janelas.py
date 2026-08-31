import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. FUNÇÃO DE RICHARDS
# ==============================================================================
def funcao_richards(t, K, A, B, M, V):
    """Equação da Curva de Richards"""
    arg = -B * (t - M)
    arg = np.clip(arg, -100, 100)
    denominador = (1.0 + np.exp(arg)) ** (1.0 / max(V, 1e-6))
    return A + (K - A) / denominador

# ==============================================================================
# 2. FUNÇÃO DE AJUSTE POR JANELA
# ==============================================================================
def ajustar_janela(y_janela):
    """Realiza o chute inicial dinâmico e o ajuste matemático para a janela"""
    tamanho = len(y_janela)
    t_janela = np.arange(tamanho) # RESETA O TEMPO (0 até tamanho-1)
    
    K_init = np.max(y_janela)
    A_init = np.min(y_janela)
    B_init = 0.1
    M_init = tamanho / 2.0
    V_init = 1.0
    
    chutes = [K_init, A_init, B_init, M_init, V_init]
    limites_inf = [np.min(y_janela), -np.inf, 0.001, -np.inf, 0.001]
    limites_sup = [np.inf, np.inf, 10.0, np.inf, 50.0]
    
    try:
        popt, _ = curve_fit(
            funcao_richards, t_janela, y_janela, 
            p0=chutes, bounds=(limites_inf, limites_sup), maxfev=10000
        )
        return t_janela, popt
    except RuntimeError:
        return t_janela, None

# ==============================================================================
# 3. EXECUÇÃO PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    tamanho_janela = 60
    if len(sys.argv) > 1:
        try:
            tamanho_janela = int(sys.argv[1])
        except ValueError:
            print("Valor inválido para a janela. Usando padrão (60).")

    passo = tamanho_janela 

    print("=" * 50)
    print(f"INICIANDO ANÁLISE - JANELA: {tamanho_janela} DIAS | PASSO: {passo} DIAS")
    print("=" * 50)

    nome_arquivo = "dados_sinteticos-2.csv"
    try:
        df = pd.read_csv(nome_arquivo)
        valores_totais = df['valores'].values
        tempo_total = df['tempo'].values
    except FileNotFoundError:
        print(f"Erro: Arquivo '{nome_arquivo}' não encontrado.")
        sys.exit()

    # ==============================================================================
    # 4. LOOP DAS JANELAS DESLIZANTES COM LOGS DE PARÂMETROS
    # ==============================================================================
    inicio = 0
    num_janela = 1
    todos_ajustes = []

    while (inicio + tamanho_janela) <= len(valores_totais):
        fim = inicio + tamanho_janela
        y_atual = valores_totais[inicio:fim]
        
        t_zerado, parametros = ajustar_janela(y_atual)
        
        plt.figure(figsize=(8, 5))
        plt.bar(t_zerado, y_atual, color='lightgray', edgecolor='gray', label='Valores Reais (Barras)')
        
        if parametros is not None:
            K_opt, A_opt, B_opt, M_opt, V_opt = parametros
            y_fit = funcao_richards(t_zerado, *parametros)
            
            # --- IMPRIME OS PARÂMETROS DA ONDA NO TERMINAL ---
            print(f"\n✅ ONDA {num_janela} | Índices: [{inicio} a {fim-1}] ajustada com sucesso!")
            print("-" * 40)
            print(f"  K (Teto)       = {K_opt:.5f}")
            print(f"  A (Chão)       = {A_opt:.5f}")
            print(f"  B (Velocidade) = {B_opt:.5f}")
            print(f"  M (Inflexão)   = {M_opt:.5f}")
            print(f"  V (Assimetria) = {V_opt:.5f}")
            print("-" * 40)
            # -------------------------------------------------
            
            plt.plot(t_zerado, y_fit, color='red', linewidth=3, label='Ajuste de Richards')
            todos_ajustes.append((inicio, fim, y_fit))
            titulo = f"Janela {num_janela}: Índices [{inicio} a {fim-1}] - Ajuste OK"
        else:
            print(f"\n❌ ONDA {num_janela} | Índices: [{inicio} a {fim-1}] - FALHA NO AJUSTE")
            titulo = f"Janela {num_janela}: Índices [{inicio} a {fim-1}] - Falha no Ajuste"
            
        plt.title(titulo, fontweight='bold')
        plt.xlabel("Tempo (Resetado para 0)")
        plt.ylabel("Valores")
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()
        
        inicio += passo
        num_janela += 1

    # ==============================================================================
    # 5. PLOTAGEM DO CONJUNTO DE DADOS COMPLETO AO FINAL
    # ==============================================================================
    print("\nExibindo panorama completo de todas as ondas...")
    
    plt.figure(figsize=(14, 6))
    plt.bar(tempo_total, valores_totais, color='lightgray', edgecolor='gray', label='Dados Históricos')
    
    for (idx_inicio, idx_fim, y_fit_janela) in todos_ajustes:
        t_original = tempo_total[idx_inicio:idx_fim]
        plt.plot(t_original, y_fit_janela, linewidth=2.5, label=f'Onda {idx_inicio} a {idx_fim-1}')
        plt.axvline(x=t_original[0], color='black', linestyle=':', alpha=0.4)

    plt.title(f"Panorama Geral: Múltiplas Ondas Detectadas ({tamanho_janela} dias cada)", fontsize=14, fontweight='bold')
    plt.xlabel("Tempo Total Contínuo")
    plt.ylabel("Valores")
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()