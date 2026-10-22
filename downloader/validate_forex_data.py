import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

def validate_tick_data(file_path: Path):
    print(f"\n[*] Iniciando auditoria de alta performance no arquivo: {file_path.name}")
    
    try:
        parquet_file = pq.ParquetFile(file_path)
        colunas_disponiveis = parquet_file.schema.names
        
        colunas_para_ler = ['timestamp']
        if 'bid' in colunas_disponiveis: colunas_para_ler.append('bid')
        elif 'bidPrice' in colunas_disponiveis: colunas_para_ler.append('bidPrice')
        
        if 'ask' in colunas_disponiveis: colunas_para_ler.append('ask')
        elif 'askPrice' in colunas_disponiveis: colunas_para_ler.append('askPrice')
        
        for vol_col in ['bid_volume', 'ask_volume', 'bidVolume', 'askVolume', 'volume']:
            if vol_col in colunas_disponiveis: colunas_para_ler.append(vol_col)
            
        df = pd.read_parquet(file_path, engine='pyarrow', columns=colunas_para_ler)
        total_linhas_iniciais = len(df)
        
        renames = {'bidPrice': 'bid', 'askPrice': 'ask'}
        df = df.rename(columns={k: v for k, v in renames.items() if k in df.columns})
            
        if 'bid_volume' in df.columns and 'ask_volume' in df.columns:
            df['volume'] = df['bid_volume'] + df['ask_volume']
        elif 'bidVolume' in df.columns and 'askVolume' in df.columns:
            df['volume'] = df['bidVolume'] + df['askVolume']
            
        if 'volume' not in df.columns:
            df['volume'] = 0.0
            
        colunas_exigidas = ['timestamp', 'ask', 'bid', 'volume']
        
        for col in colunas_exigidas:
            if col not in df.columns:
                print(f"[!] Erro fatal: Coluna essencial '{col}' ausente.")
                return False
                
        df = df[colunas_exigidas]
        
        if df.isnull().any().any():
            linhas_antes = len(df)
            df = df.dropna()
            print(f"  [!] Anomalia: {linhas_antes - len(df)} linhas com valores nulos removidas.")

        # =========================================================================
        # BLOCO DE RESGATE (AUTO-HEALING): Detecção de Falha de Inversão de Colunas
        # =========================================================================
        taxa_inversao = (df['bid'] > df['ask']).mean()
        
        if taxa_inversao > 0.50:
            print(f"  [!] ALERTA ESTRUTURAL: {taxa_inversao:.1%} do dataset possui Bid > Ask.")
            print("  [!] Diagnóstico: Falha de cruzamento de colunas detectada na extração original.")
            print("  [*] Executando Auto-Correção (Rescue): Restaurando a ordem correta das colunas...")
            
            temp_bid = df['bid'].copy()
            df['bid'] = df['ask'].copy()
            df['ask'] = temp_bid
            print("  [+] Auto-Correção concluída. Prosseguindo com a validação de spread...")
        # =========================================================================

        linhas_antes_spread = len(df)
        df = df[df['ask'] >= df['bid']]
        removidos_spread = linhas_antes_spread - len(df)
        
        if removidos_spread > 0:
            print(f"  [!] Anomalia: {removidos_spread} ticks reais com spread invertido expurgados.")
            
        if not df['timestamp'].is_monotonic_increasing:
            print("  [!] Anomalia: Timestamps desordenados. Reordenando de forma otimizada...")
            df = df.sort_values('timestamp')
            
        # Limpa o índice fragmentado da RAM antes de salvar
        df = df.reset_index(drop=True)
            
        linhas_finais = len(df)
        linhas_removidas = total_linhas_iniciais - linhas_finais
        
        clean_path = file_path.parent / f"{file_path.stem}_clean.parquet"
        
        # O parâmetro index=False barra a gravação da coluna oculta __index_level_0__
        df.to_parquet(clean_path, engine='pyarrow', compression='snappy', index=False)
        
        if linhas_removidas > 0:
            print(f"  [+] Limpeza concluída. {linhas_removidas} anomalias removidas no total.")
        else:
            print("  [+] Integridade verificada. Nenhuma anomalia encontrada.")
            
        print(f"  [+] Arquivo validado salvo em: {clean_path.name}")
        print(f"  [*] Registros finais: {linhas_finais} ticks operacionais recuperados e validados.")
        return True
        
    except Exception as e:
        print(f"[!] Erro crítico ao processar o arquivo {file_path.name}: {e}")
        return False

def menu_selecao_arquivo(diretorio: Path):
    if not diretorio.exists():
        print(f"[!] O diretório '{diretorio}' não existe.")
        return
        
    arquivos = [f for f in diretorio.glob("*.parquet") if "_clean" not in f.name and "parciais" not in f.parts]
    
    if not arquivos:
        print(f"[!] Nenhum arquivo Parquet bruto encontrado na pasta '{diretorio}'.")
        return
        
    print(f"\n[*] Arquivos disponíveis para limpeza em '{diretorio}':")
    for i, arquivo in enumerate(arquivos):
        print(f"  [{i}] {arquivo.name}")
        
    print(f"  [{len(arquivos)}] Limpar TODOS os arquivos listados")
    
    try:
        escolha = int(input("\nDigite o número do arquivo que deseja auditar/limpar: "))
        if escolha == len(arquivos):
            for arq in arquivos:
                validate_tick_data(arq)
        elif 0 <= escolha < len(arquivos):
            validate_tick_data(arquivos[escolha])
        else:
            print("[!] Opção inválida.")
    except ValueError:
        print("[!] Entrada inválida. Digite um número.")

if __name__ == "__main__":
    print("="*60)
    print(" Auditoria e Limpeza de Dados Forex (PIMAT DoD) ")
    print("="*60)
    
    dir_padrao = Path("data/forex")
    dir_input = input(f"Diretório dos arquivos (Enter para '{dir_padrao}'): ").strip()
    
    diretorio_alvo = Path(dir_input) if dir_input else dir_padrao
    menu_selecao_arquivo(diretorio_alvo)