import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ==============================================================================
# 1. CARREGAMENTO DOS DADOS
# ==============================================================================
def carregar_dados_csv(caminho_arquivo: str) -> pd.DataFrame:
    """Carrega o arquivo CSV, padroniza as colunas e ordena por data."""
    if not os.path.exists(caminho_arquivo):
        raise FileNotFoundError(f"Erro: O arquivo '{caminho_arquivo}' não foi encontrado.")

    df = pd.read_csv(caminho_arquivo)
    
    # Padroniza nomes para minúsculo
    df.columns = [col.strip().lower() for col in df.columns]

    colunas_necessarias = ["timestamp", "open", "high", "low", "close"]
    for col in colunas_necessarias:
        if col not in df.columns:
            raise ValueError(f"Erro: Coluna '{col}' não encontrada no arquivo CSV.")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df

# ==============================================================================
# 2. ALGORITMO DE DETECÇÃO DE TENDÊNCIA
# ==============================================================================
def detectar_inicio_movimento(df: pd.DataFrame, n: int = 3, coluna: str = "close", estrito: bool = False) -> pd.DataFrame:
    """Identifica topos e fundos analisando 'n' períodos para trás e para frente."""
    sinais = []
    precos = df[coluna].values
    datas = df["timestamp"].dt.strftime("%Y-%m-%d").values

    for i in range(n, len(precos) - n):
        passado = precos[i - n : i + 1]
        futuro = precos[i : i + n + 1]

        if estrito:
            passado_caindo = all(passado[j] > passado[j + 1] for j in range(len(passado) - 1))
            passado_subindo = all(passado[j] < passado[j + 1] for j in range(len(passado) - 1))
            futuro_subindo = all(futuro[j] < futuro[j + 1] for j in range(len(futuro) - 1))
            futuro_caindo = all(futuro[j] > futuro[j + 1] for j in range(len(futuro) - 1))

            if passado_caindo and futuro_subindo:
                sinais.append({"data": datas[i], "preco": precos[i], "tipo": "Início de Alta (Fundo)"})
            elif passado_subindo and futuro_caindo:
                sinais.append({"data": datas[i], "preco": precos[i], "tipo": "Início de Baixa (Topo)"})
        else:
            variacao_passada = passado[-1] - passado[0]
            variacao_futura = futuro[-1] - futuro[0]

            quedas_passado = sum(passado[j] > passado[j + 1] for j in range(len(passado) - 1))
            subidas_futuro = sum(futuro[j] < futuro[j + 1] for j in range(len(futuro) - 1))
            subidas_passado = sum(passado[j] < passado[j + 1] for j in range(len(passado) - 1))
            quedas_futuro = sum(futuro[j] > futuro[j + 1] for j in range(len(futuro) - 1))

            if quedas_passado >= (n - 1) and subidas_futuro >= (n - 1) and variacao_passada < 0 and variacao_futura > 0:
                sinais.append({"data": datas[i], "preco": precos[i], "tipo": "Início de Alta (Giro)"})
            elif subidas_passado >= (n - 1) and quedas_futuro >= (n - 1) and variacao_passada > 0 and variacao_futura < 0:
                sinais.append({"data": datas[i], "preco": precos[i], "tipo": "Início de Baixa (Giro)"})

    return pd.DataFrame(sinais)

# ==============================================================================
# 3. PLOTAGEM DO GRÁFICO (SEM BURACOS E COM TRACEJADOS DESTACADOS)
# ==============================================================================
def plotar_grafico_tendencias(df: pd.DataFrame, sinais: pd.DataFrame):
    """Gera o gráfico de preços e marca os pontos de mudança com linhas tracejadas grossas."""
    if df.empty:
        print("Erro: DataFrame está vazio.")
        return

    plt.figure(figsize=(14, 7))
    
    # Índice numérico sequencial para evitar buracos de sábado e domingo
    x_indices = np.arange(len(df))
    
    # Linha principal de preços
    plt.plot(x_indices, df["close"], label="Preço de Fechamento", color="#2b2b2b", linewidth=1.5, zorder=3)

    if not sinais.empty:
        sinais["data"] = pd.to_datetime(sinais["data"])
        
        altas = sinais[sinais["tipo"].str.contains("Alta")]
        baixas = sinais[sinais["tipo"].str.contains("Baixa")]

        # Marcações de ALTA (Verde)
        primeira_alta = True
        for _, row in altas.iterrows():
            idx = df.index[df['timestamp'] == row['data']].tolist()[0]
            preco_minimo = df.loc[idx, 'low']
            
            plt.axvline(x=idx, color="green", linestyle="--", linewidth=2.5, zorder=1,
                        label="Início de Alta" if primeira_alta else "")
            plt.scatter(idx, preco_minimo - 0.003, color="green", s=120, marker="^", zorder=5)
            primeira_alta = False

        # Marcações de BAIXA (Vermelho)
        primeira_baixa = True
        for _, row in baixas.iterrows():
            idx = df.index[df['timestamp'] == row['data']].tolist()[0]
            preco_maximo = df.loc[idx, 'high']
            
            plt.axvline(x=idx, color="red", linestyle="--", linewidth=2.5, zorder=1,
                        label="Início de Baixa" if primeira_baixa else "")
            plt.scatter(idx, preco_maximo + 0.003, color="red", s=120, marker="v", zorder=5)
            primeira_baixa = False

    # Formata o eixo X para mostrar as datas em vez dos números dos índices
    def formatar_data(x, pos):
        if int(x) >= 0 and int(x) < len(df):
            return df['timestamp'].iloc[int(x)].strftime('%d/%m/%y')
        return ""
    
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(formatar_data))
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(15))

    # Estética do gráfico
    plt.title("Detecção de Movimentos e Mudança de Tendência", fontsize=15, fontweight="bold")
    plt.ylabel("Preço", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.4, zorder=0)
    plt.legend(loc='best', fontsize=11)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ==============================================================================
# 4. EXECUÇÃO PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    # NOME DO SEU ARQUIVO AQUI
    arquivo = "EUR_USD_daily_20240101_20250101.csv"
    
    try:
        # 1. Carrega os dados
        print(f"Lendo dados do arquivo: {arquivo}")
        df_dados = carregar_dados_csv(arquivo)
        
        # 2. Roda a detecção de movimento (usando estrito=False para capturar mais movimentos)
        print("Calculando mudanças de tendência...")
        df_sinais = detectar_inicio_movimento(df_dados, n=3, coluna="close", estrito=False)
        
        if not df_sinais.empty:
            print("\nSinais Encontrados:")
            print(df_sinais.to_string(index=False))
        else:
            print("\nNenhum sinal encontrado com os parâmetros atuais.")
            
        # 3. Plota o gráfico na tela
        print("\nGerando gráfico...")
        plotar_grafico_tendencias(df_dados, df_sinais)
        
    except Exception as e:
        print(f"\nOcorreu um erro: {e}")