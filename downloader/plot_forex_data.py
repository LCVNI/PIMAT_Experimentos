import pandas as pd
import mplfinance as mpf
from pathlib import Path

def plot_forex_chart(file_path: Path, timeframe_pandas: str, timeframe_label: str, num_candles: int):
    print(f"\n[*] Carregando dados validados de {file_path.name}...")
    df = pd.read_parquet(file_path)
    
    df.set_index('timestamp', inplace=True)
    
    print(f"[*] Agrupando {len(df)} ticks em velas de {timeframe_label}...")
    
    ohlc_dict = {
        'bid': ['first', 'max', 'min', 'last'], 
        'volume': 'sum'                         
    }
    
    df_resampled = df.resample(timeframe_pandas).agg(ohlc_dict)
    df_resampled.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df_resampled.dropna(inplace=True)
    
    df_plot = df_resampled.iloc[-num_candles:]
    
    if df_plot.empty:
        print(f"[!] Aviso: O arquivo não possui dados suficientes para gerar {num_candles} velas no timeframe {timeframe_label}.")
        return

    # Extração dinâmica do par de moedas a partir do nome do arquivo (ex: USDGBP_202001... -> USD/GBP)
    nome_arquivo = file_path.name
    par_moeda = nome_arquivo[:6] if len(nome_arquivo) >= 6 else "PAR"
    titulo_par = f"{par_moeda[:3]}/{par_moeda[3:]}" if len(par_moeda) == 6 else par_moeda

    print(f"[*] Renderizando o motor gráfico ({len(df_plot)} velas)... [Feche a janela do gráfico para continuar o terminal]")
    
    cores_mercado = mpf.make_marketcolors(
        up='#aaddff',     
        down='#000080',   
        edge='inherit',
        wick='inherit',
        volume='in'
    )
    estilo_visual = mpf.make_mpf_style(base_mpf_style='nightclouds', marketcolors=cores_mercado)
    
    mpf.plot(
        df_plot,
        type='candle',
        style=estilo_visual,
        volume=True,
        title=f"\n{titulo_par} - Últimas {len(df_plot)} Velas ({timeframe_label})",
        ylabel='Preço de Compra (Bid)',
        ylabel_lower='Volume Total'
    )
    print("[+] Gráfico encerrado com sucesso.")

def menu_selecao_arquivo(diretorio: Path):
    if not diretorio.exists():
        print(f"[!] O diretório '{diretorio}' não existe.")
        return None
        
    # Busca estritamente os arquivos que já passaram pela limpeza (_clean)
    arquivos = list(diretorio.glob("*_clean.parquet"))
    
    if not arquivos:
        print(f"[!] Nenhum arquivo validado (_clean) encontrado na pasta '{diretorio}'.")
        print("    -> Rode o script de validação/limpeza primeiro.")
        return None
        
    print(f"\n[*] Arquivos validados disponíveis:")
    for i, arquivo in enumerate(arquivos):
        print(f"  [{i}] {arquivo.name}")
        
    try:
        escolha = int(input("\nDigite o número do arquivo que deseja plotar: "))
        if 0 <= escolha < len(arquivos):
            return arquivos[escolha]
        else:
            print("[!] Opção inválida.")
    except ValueError:
        print("[!] Entrada inválida. Digite um número.")
    return None

if __name__ == "__main__":
    print("="*60)
    print(" Motor Gráfico Quantitativo (Candlesticks) ")
    print("="*60)
    
    dir_padrao = Path("data/forex")
    arquivo_alvo = menu_selecao_arquivo(dir_padrao)
    
    if arquivo_alvo:
        mapa_timeframes = {
            '1M': '1min', '5M': '5min', '15M': '15min', '30M': '30min', 
            '1H': '1h', '4H': '4h', '1D': '1D', '1W': '1W', '1MW': 'ME'
        }
        
        print("\n[Timeframes disponíveis]: " + ", ".join(mapa_timeframes.keys()))
        
        escolha_tf = ""
        while escolha_tf not in mapa_timeframes:
            escolha_tf = input("[?] Digite o timeframe desejado: ").strip().upper()
            if escolha_tf not in mapa_timeframes:
                print("[!] Opção inválida.")
                
        tf_pandas = mapa_timeframes[escolha_tf]
        
        try:
            qtd = int(input("[?] Quantidade de velas para renderizar na tela (Padrão: 150): ").strip())
        except ValueError:
            qtd = 150
            print("    -> Usando valor padrão: 150")
            
        plot_forex_chart(
            file_path=arquivo_alvo, 
            timeframe_pandas=tf_pandas, 
            timeframe_label=escolha_tf,
            num_candles=qtd
        )