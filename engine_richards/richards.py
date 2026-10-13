import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import warnings

# Ignora avisos de overflow internos durante as tentativas do curve_fit
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. DEFINIÇÃO DA FUNÇÃO DE RICHARDS
# ==============================================================================
def funcao_richards(t, K, A, B, M, V):
    """
    Equação da Curva de Richards.
    K: Capacidade de carga (Teto)
    A: Nível inicial (Chão)
    B: Velocidade de crescimento
    M: Ponto de inflexão (Tempo de maior aceleração)
    V: Assimetria da curva
    """
    # Limita o argumento do exponencial para evitar erro de 'overflow' no Python
    arg = -B * (t - M)
    arg = np.clip(arg, -100, 100)
    
    # max(V, 1e-6) evita divisão por zero caso o algoritmo tente testar V = 0
    denominador = (1.0 + np.exp(arg)) ** (1.0 / max(V, 1e-6))
    return A + (K - A) / denominador

# ==============================================================================
# 2. CARREGAMENTO DOS DADOS
# ==============================================================================
nome_arquivo = "dados_sinteticos-1.csv"

try:
    df = pd.read_csv(nome_arquivo)
    
    # Pega os valores das colunas 'tempo' e 'valores'
    t = df['tempo'].values
    y = df['valores'].values
    
except FileNotFoundError:
    print(f"Erro: Arquivo '{nome_arquivo}' não encontrado.")
    print("Certifique-se de que o arquivo está na mesma pasta do script.")
    exit()
except KeyError as e:
    print(f"Erro: Coluna não encontrada. Verifique se o CSV possui as colunas 'tempo' e 'valores'. Detalhe: {e}")
    exit()

# ==============================================================================
# 3. CONFIGURAÇÃO E AJUSTE DO MODELO (CURVE FIT)
# ==============================================================================
# "Chutes" iniciais baseados nos próprios dados para ajudar o curve_fit
K_init = np.max(y)          # O teto provavelmente é perto do valor máximo real
A_init = np.min(y)          # O chão provavelmente é perto do valor mínimo real
B_init = 0.1                # Chute padrão de velocidade
M_init = np.median(t)       # O meio do tempo é um bom chute para a inflexão
V_init = 1.0                # Começa assumindo uma curva simétrica (Logística)

chutes_iniciais = [K_init, A_init, B_init, M_init, V_init]

# Limites lógicos para os parâmetros (K, A, B, M, V)
# Ex: V não pode ser negativo; B tem que ser positivo.
limite_inferior = [np.min(y), -np.inf, 0.001, -np.inf, 0.001]
limite_superior = [np.inf, np.inf, 10.0, np.inf, 50.0]

try:
    print("Calculando o ajuste da curva. Aguarde...\n")
    # Otimização por mínimos quadrados
    parametros_otimizados, matriz_covariancia = curve_fit(
        funcao_richards, 
        t, 
        y, 
        p0=chutes_iniciais, 
        bounds=(limite_inferior, limite_superior),
        maxfev=10000 # Dá até 10 mil tentativas para o algoritmo encontrar a resposta
    )
    
    K_opt, A_opt, B_opt, M_opt, V_opt = parametros_otimizados
    
    # ==============================================================================
    # 4. CÁLCULO DO R²
    # ==============================================================================
    y_predito = funcao_richards(t, *parametros_otimizados)
    r2 = r2_score(y, y_predito)
    
    # Exibe os resultados no terminal
    print("=" * 40)
    print("RESULTADOS DO AJUSTE DE RICHARDS")
    print("=" * 40)
    print(f"R² (Qualidade do Ajuste): {r2:.4f} ({(r2*100):.2f}%)")
    print("-" * 40)
    print("Parâmetros Encontrados:")
    print(f"  K (Teto)       = {K_opt:.5f}")
    print(f"  A (Chão)       = {A_opt:.5f}")
    print(f"  B (Velocidade) = {B_opt:.5f}")
    print(f"  M (Inflexão)   = {M_opt:.5f}")
    print(f"  V (Assimetria) = {V_opt:.5f}")
    print("=" * 40)

    # ==============================================================================
    # 5. PLOTAGEM DO GRÁFICO
    # ==============================================================================
    plt.figure(figsize=(10, 6))
    
    # Plota os dados reais como pontos dispersos
    plt.bar(t, y, color='blue', label='Dados Originais (Reais)', alpha=0.6, edgecolor='k')
    
    # Plota a linha da função de Richards ajustada
    # Criamos uma linha suave gerando mais pontos de tempo entre o min e max
    t_suave = np.linspace(min(t), max(t), 500)
    y_suave = funcao_richards(t_suave, *parametros_otimizados)
    plt.plot(t_suave, y_suave, color='red', linewidth=2.5, label='Curva de Richards Ajustada')
    
    # Estética do gráfico
    plt.title(f"Ajuste da Função de Richards (R² = {r2:.4f})", fontsize=14, fontweight='bold')
    plt.xlabel("Tempo", fontsize=12)
    plt.ylabel("Valores", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"\nFalha ao ajustar a curva de Richards.")
    print(f"Detalhe do erro: {e}")
    print("Dica: Os dados podem estar muito ruidosos ou não formam um crescimento em 'S' claro.")