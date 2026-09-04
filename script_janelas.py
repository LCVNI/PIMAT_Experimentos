import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import warnings
import downloader_forex as fx

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
    """Realiza o chute inicial dinâmico e o ajuste matemático para a janela (treino)"""
    tamanho = len(y_janela)
    t_janela = np.arange(tamanho) 
    
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
# 3. EXECUÇÃO PRINCIPAL (MODO INTERATIVO)
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("CONFIGURAÇÃO DA ANÁLISE E PREDIÇÃO DE ONDAS (RICHARDS)")
    print("=" * 60)

    # Entradas de configuração da janela e predição
    entrada_janela = input("Digite o tamanho da janela de ajuste (ex: 60): ")
    try:
        tamanho_janela = int(entrada_janela) if entrada_janela.strip() else 60
    except ValueError:
        print("Valor inválido. Usando o padrão de 60.")
        tamanho_janela = 60

    entrada_pred = input("Digite o horizonte de predição à frente (ex: 15): ")
    try:
        horizonte_predicao = int(entrada_pred) if entrada_pred.strip() else 15
    except ValueError:
        print("Valor inválido. Usando o padrão de 15.")
        horizonte_predicao = 15

    passo = tamanho_janela 

    print("\n--- Dados do Arquivo ---")
    moeda1 = input("Primeira moeda (ex: EUR): ").strip().upper()
    moeda2 = input("Segunda moeda (ex: USD): ").strip().upper()
    timeframe = input("Timeframe (ex: weekly, daily, 15m): ").strip().lower()
    data_inicio = input("Data inicio (YYYYMMDD): ").strip()
    data_fim = input("Data fim (YYYYMMDD): ").strip()

    # Nome dinâmico
    nome_arquivo = f"{moeda1}_{moeda2}_{timeframe}_{data_inicio}_{data_fim}.csv"
    arquivo = Path(nome_arquivo)

    # Verifica download
    if not arquivo.is_file():
        print(f"O arquivo {nome_arquivo} não existe! Iniciando download...")
        fx.baixar_forex(moeda1, moeda2, timeframe, data_inicio, data_fim)
  
    print("\n" + "=" * 60)
    print(f"ANÁLISE - JANELA: {tamanho_janela} | PREDIÇÃO: {horizonte_predicao} | TIMEFRAME: {timeframe}")
    print("=" * 60)

    try:
        df = pd.read_csv(nome_arquivo)
        df.columns = [col.strip().lower() for col in df.columns]
        valores_totais = df['close'].values if 'close' in df.columns else df['valores'].values
        tempo_total = np.arange(len(valores_totais)) 
    except FileNotFoundError:
        print(f"\n❌ Erro: Arquivo '{nome_arquivo}' não encontrado.")
        sys.exit()
    except KeyError as e:
        print(f"\n❌ Erro: Coluna de preço não encontrada. Detalhe: {e}")
        sys.exit()

    # ==============================================================================
    # 4. LOOP DAS JANELAS DESLIZANTES COM PREDIÇÃO
    # ==============================================================================
    inicio = 0
    num_janela = 1
    todos_ajustes = []
    todas_predicoes = []

    while (inicio + tamanho_janela) <= len(valores_totais):
        fim = inicio + tamanho_janela
        fim_futuro = min(fim + horizonte_predicao, len(valores_totais))
        
        y_atual = valores_totais[inicio:fim]
        y_futuro_real = valores_totais[fim:fim_futuro]
        
        t_zerado, parametros = ajustar_janela(y_atual)
        t_futuro = np.arange(tamanho_janela, tamanho_janela + len(y_futuro_real))
        
        plt.figure(figsize=(10, 5))
        
        # Plota os DADOS REAIS como linha (cinza sólido para treino, azul para futuro)
        plt.plot(t_zerado, y_atual, color='gray', linewidth=2, label='Dados Reais (Ajuste)')
        if len(y_futuro_real) > 0:
            plt.plot(t_futuro, y_futuro_real, color='steelblue', linewidth=2, alpha=0.8, label='Dados Reais (Futuro)')
        
        if parametros is not None:
            K_opt, A_opt, B_opt, M_opt, V_opt = parametros
            y_fit = funcao_richards(t_zerado, *parametros)
            
            # Cálculo do R² para avaliar o ajuste da janela
            r2 = r2_score(y_atual, y_fit)
            
            plt.plot(t_zerado, y_fit, color='red', linewidth=3, label='Richards (Ajuste)')
            todos_ajustes.append((inicio, fim, y_fit))
            
            if len(y_futuro_real) > 0:
                y_pred = funcao_richards(t_futuro, *parametros)
                plt.plot(t_futuro, y_pred, color='red', linewidth=2.5, linestyle='--', label='Richards (Predição)')
                todas_predicoes.append((fim, fim_futuro, y_pred))

            print(f"\n✅ ONDA {num_janela} | Índices: [{inicio} a {fim-1}] ajustada com sucesso!")
            print("-" * 40)
            print(f"  R² (Qualidade) = {r2:.4f} ({(r2*100):.2f}%)")
            print(f"  K (Teto)       = {K_opt:.5f}")
            print(f"  A (Chão)       = {A_opt:.5f}")
            print(f"  B (Velocidade) = {B_opt:.5f}")
            print(f"  M (Inflexão)   = {M_opt:.5f}")
            print(f"  V (Assimetria) = {V_opt:.5f}")
            print("-" * 40)
            
            titulo = f"Onda {num_janela}: Ajuste [{inicio}:{fim-1}] | Predição [{fim}:{fim_futuro-1}] | R² = {r2:.2f}"
        else:
            print(f"\n❌ ONDA {num_janela} | Índices: [{inicio} a {fim-1}] - FALHA NO AJUSTE")
            titulo = f"Onda {num_janela}: Falha no Ajuste"
            
        plt.title(titulo, fontweight='bold')
        plt.xlabel(f"Tempo em períodos ({timeframe}) - Resetado para 0")
        plt.ylabel("Valores")
        plt.legend(loc="best")
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show() 
        
        inicio += passo
        num_janela += 1

    # ==============================================================================
    # 5. PLOTAGEM DO CONJUNTO DE DADOS COMPLETO AO FINAL
    # ==============================================================================
    print("\nExibindo panorama completo de todas as ondas e predições...")
    
    plt.figure(figsize=(14, 6))
    
    # Linha contínua cinza para representar a série temporal original inteira
    plt.plot(tempo_total, valores_totais, color='gray', linewidth=2, alpha=0.6, label='Dados Históricos Reais')
    
    for i, (idx_inicio, idx_fim, y_fit_janela) in enumerate(todos_ajustes):
        t_original = tempo_total[idx_inicio:idx_fim]
        plt.plot(t_original, y_fit_janela, linewidth=2.5, label=f'Onda {i+1} (Ajuste)')
        plt.axvline(x=t_original[0], color='black', linestyle=':', alpha=0.4)

    for idx_inicio_pred, idx_fim_pred, y_pred_janela in todas_predicoes:
        t_pred = tempo_total[idx_inicio_pred:idx_fim_pred]
        plt.plot(t_pred, y_pred_janela, color='black', linewidth=2, linestyle='--', alpha=0.8)
    
    if todas_predicoes:
        plt.plot([], [], color='black', linewidth=2, linestyle='--', label='Predições Além da Janela')

    plt.title(f"Panorama Geral ({timeframe}): Janelas de {tamanho_janela} | Predição de {horizonte_predicao}", fontsize=14, fontweight='bold')
    plt.xlabel(f"Histórico Total (Timeframe: {timeframe})")
    plt.ylabel("Valores")
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()