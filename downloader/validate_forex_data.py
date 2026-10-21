import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

def validate_tick_data(file_path: Path):
    print(f"\n[*] Iniciando auditoria de alta performance no arquivo: {file_path.name}")
    
    try:
        # 1. Inspeção de Schema (Zero RAM)
        # Lê apenas os metadados do Parquet para descobrir quais colunas existem
        parquet_file = pq.ParquetFile(file_path)
        colunas_disponiveis = parquet_file.schema.names
        
        # Mapeia dinamicamente apenas as colunas que importam
        colunas_para_ler = ['timestamp']
        if 'bid' in colunas_disponiveis: colunas_para_ler.append('bid')
        elif 'bidPrice' in colunas_disponiveis: colunas_para_ler.append('bidPrice')
        
        if 'ask' in colunas_disponiveis: colunas_para_ler.append('ask')
        elif 'askPrice' in colunas_disponiveis: colunas_para_ler.append('askPrice')
        
        for vol_col in ['bid_volume', 'ask_volume', 'bidVolume', 'askVolume', 'volume']:
            if vol_col in colunas_disponiveis: colunas_para_ler.append(vol_col)
            
        # 2. Leitura Seletiva (Carrega apenas o essencial, economizando I/O e RAM)
        df = pd.read_parquet(file_path, engine='pyarrow', columns=colunas_para_ler)
        total_linhas_iniciais = len(df)
        
        # Normalização de nomes
        renames = {'bidPrice': 'bid', 'askPrice': 'ask'}
        df.rename(columns={k: v for k, v in renames.items() if k in df.columns}, inplace=True)
            
        # Tratamento de Volume vetorizado
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
                
        # Descarta as colunas de volume antigas liberando RAM
        df = df[colunas_exigidas]
        
        # 3. Limpeza de Nulos (In-place para economizar memória)
        if df.isnull().any().any():
            linhas_antes = len(df)
            df.dropna(inplace=True)
            print(f"  [!] Anomalia: {linhas_antes - len(df)} linhas com valores nulos removidas.")
            
        # 4. Filtragem Vetorizada de Spread Invertido (Substitui o lento .drop())
        linhas_antes_spread = len(df)
        df = df[df['ask'] >= df['bid']] # Mantém apenas o que é válido logicamente
        removidos_spread = linhas_antes_spread - len(df)
        if removidos_spread > 0:
            print(f"  [!] Anomalia: {removidos_spread} ticks com spread invertido expurgados.")
            
        # 5. Verificação de Ordenação Cronológica (In-place)
        if not df['timestamp'].is_monotonic_increasing:
            print("  [!] Anomalia: Timestamps desordenados. Reordenando de forma otimizada...")
            df.sort_values('timestamp', inplace=True)
            
        # 6. Geração do Arquivo Final Validado
        linhas_finais = len(df)
        linhas_removidas = total_linhas_iniciais - linhas_finais
        
        clean_path = file_path.parent / f"{file_path.stem}_clean.parquet"
        
        # Mantém compressão snappy nativa
        df.to_parquet(clean_path, engine='pyarrow', compression='snappy')
        
        if linhas_removidas > 0:
            print(f"  [+] Limpeza concluída. {linhas_removidas} anomalias removidas no total.")
        else:
            print("  [+] Integridade verificada. Nenhuma anomalia encontrada.")
            
        print(f"  [+] Arquivo validado salvo em: {clean_path.name}")
        print(f"  [*] Registros finais: {linhas_finais} ticks operacionais.")
        return True
        
    except Exception as e:
        print(f"[!] Erro crítico ao processar o arquivo {file_path.name}: {e}")
        return False