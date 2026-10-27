import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
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
# 2. EXECUÇÃO PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("ANÁLISE DE PROJEÇÃO CEGA (EXECUÇÃO ÚNICA)")
    print("=" * 60)

    # 1. Solicita as entradas ao usuário
    try:
        entrada_tamanho = input("Digite o tamanho total dos dados a considerar (ex: 100): ")
        tamanho_total = int(entrada_tamanho)
        
        entrada_porcentagem = input("Digite a porcentagem dos dados para treino (ex: 80): ")
        porcentagem = float(entrada_porcentagem)
    except ValueError:
        print("❌ Erro: Por favor, digite apenas números válidos.")
        sys.exit()

    # 2. Carrega o arquivo fixo
    caminho_arquivo = Path("sinteticos/dados_sinteticos-1.csv")
    
    if not caminho_arquivo.is_file():
        print(f"❌ Erro: Arquivo '{caminho_arquivo}' não encontrado.")
        print("Certifique-se de que a pasta 'sinteticos' existe no mesmo local do script.")
        sys.exit()

    try:
        df = pd.read_csv(caminho_arquivo)
        # Padroniza nomes de colunas
        df.columns = [col.strip().lower() for col in df.columns]
        valores = df['close'].values if 'close' in df.columns else df['valores'].values
    except KeyError as e:
        print(f"❌ Erro: Coluna de dados não encontrada no arquivo. Detalhe: {e}")
        sys.exit()

    # Verifica se o arquivo tem dados suficientes
    if len(valores) < tamanho_total:
        print(f"⚠️ Aviso: O arquivo tem apenas {len(valores)} linhas. Usando o tamanho máximo disponível.")
        tamanho_total = len(valores)

    # 3. Faz o Split (Divisão) de Treino e Teste
    valores_recorte = valores[:tamanho_total]
    tamanho_treino = int(tamanho_total * (porcentagem / 100.0))
    
    if tamanho_treino < 5:
        print("❌ Erro: O tamanho do treino é muito pequeno para ajustar a curva (mínimo 5 pontos).")
        sys.exit()

    y_treino = valores_recorte[:tamanho_treino]
    y_teste = valores_recorte[tamanho_treino:]
    
    t_completo = np.arange(tamanho_total)
    t_treino = t_completo[:tamanho_treino]
    t_teste = t_completo[tamanho_treino:]

    print("\n" + "-" * 60)
    print(f"Resumo do Split:")
    print(f"  Tamanho Total   : {tamanho_total} pontos")
    print(f"  Treino (Visível): {len(y_treino)} pontos ({porcentagem}%)")
    print(f"  Teste (Oculto)  : {len(y_teste)} pontos ({(100-porcentagem):.1f}%)")
    print("-" * 60)

    # 4. Ajuste Matemático (Curve Fit) usando apenas o Treino
    print("\nCalculando parâmetros... Aguarde.")
    
    K_init, A_init = np.max(y_treino), np.min(y_treino)
    chutes = [K_init, A_init, 0.1, len(y_treino) / 2.0, 1.0]
    limites_inf = [np.min(y_treino), -np.inf, 0.001, -np.inf, 0.001]
    limites_sup = [np.inf, np.inf, 10.0, np.inf, 50.0]
    
    try:
        parametros, _ = curve_fit(
            funcao_richards, t_treino, y_treino, 
            p0=chutes, bounds=(limites_inf, limites_sup), maxfev=10000
        )
        
        K_opt, A_opt, B_opt, M_opt, V_opt = parametros
        
        # Gera as projeções (linhas vermelhas)
        y_fit_treino = funcao_richards(t_treino, *parametros)
        y_pred_teste = funcao_richards(t_teste, *parametros) if len(t_teste) > 0 else []
        
        # Calcula as notas
        r2_treino = r2_score(y_treino, y_fit_treino)
        
        print("\n✅ AJUSTE CONCLUÍDO COM SUCESSO!")
        print("=" * 40)
        print("PARÂMETROS ENCONTRADOS:")
        print(f"  K (Teto)       = {K_opt:.5f}")
        print(f"  A (Chão)       = {A_opt:.5f}")
        print(f"  B (Velocidade) = {B_opt:.5f}")
        print(f"  M (Inflexão)   = {M_opt:.5f}")
        print(f"  V (Assimetria) = {V_opt:.5f}")
        print("-" * 40)
        print("PERFORMANCE DO MODELO:")
        print(f"  R² no Treino (Base do Ajuste): {r2_treino:.4f}")
        
        if len(y_teste) > 0:
            r2_teste = r2_score(y_teste, y_pred_teste)
            print(f"  R² no Teste (Projeção Cega)  : {r2_teste:.4f}")
            titulo = f"Modelo Richards | R² Treino: {r2_treino:.2f} | R² Teste: {r2_teste:.2f}"
        else:
            titulo = f"Modelo Richards | R² Treino: {r2_treino:.2f} | (Sem Teste)"
        print("=" * 40)

        # 5. Plotagem
        plt.figure(figsize=(10, 6))
        
        # Dados Reais
        plt.plot(t_treino, y_treino, color='gray', linewidth=2.5, label='Dados Reais (Treino)')
        if len(y_teste) > 0:
            plt.plot(t_teste, y_teste, color='steelblue', linewidth=2.5, label='Dados Reais Ocultos (Teste)')
            
        # Fronteira
        if len(y_teste) > 0:
            plt.axvline(x=t_treino[-1], color='black', linestyle=':', label='Fronteira (Início da Projeção)')
        
        # Curva de Richards
        plt.plot(t_treino, y_fit_treino, color='red', linewidth=3, label='Richards (Ajuste)')
        if len(y_teste) > 0:
            plt.plot(t_teste, y_pred_teste, color='red', linewidth=2.5, linestyle='--', label='Richards (Projeção)')

        plt.title(titulo, fontweight='bold', fontsize=12)
        plt.xlabel("Tempo (Índice)", fontsize=11)
        plt.ylabel("Valores", fontsize=11)
        plt.legend(loc="best")
        plt.grid(axis='both', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    except RuntimeError:
        print("\n❌ FALHA NO AJUSTE: O algoritmo não conseguiu encontrar uma convergência.")
        print("Isso geralmente acontece quando a porção de treino não apresenta um padrão de curva logística claro.")