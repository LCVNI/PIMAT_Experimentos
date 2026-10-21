import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path
import dukascopy_python
import time
import pyarrow.parquet as pq

def fetch_pair_data(instrument: str, start: datetime, end: datetime):
    try:
        df = dukascopy_python.fetch(
            instrument=instrument,
            interval=dukascopy_python.INTERVAL_TICK,
            offer_side=dukascopy_python.OFFER_SIDE_BID,
            start=start,
            end=end
        )
        if df is not None and not df.empty:
            return df
    except Exception:
        pass
    return None

def extract_chunk(base_currency: str, quote_currency: str, chunk_start: datetime, chunk_end: datetime, partials_dir: Path):
    pair_direct = f"{base_currency}/{quote_currency}"
    pair_reversed = f"{quote_currency}/{base_currency}"
    filename = f"{base_currency}{quote_currency}_{chunk_start.strftime('%Y%m')}_ticks.parquet"
    output_file = partials_dir / filename
    
    print(f"  -> Processando lote: {chunk_start.date()} a {chunk_end.date()}...")
    
    if output_file.exists():
        print(f"     [✓] Arquivo já existe. Pulando download...")
        return True

    df = fetch_pair_data(pair_direct, chunk_start, chunk_end)
    is_inverted = False
    
    if df is None:
        df = fetch_pair_data(pair_reversed, chunk_start, chunk_end)
        is_inverted = True
        
    if df is None or df.empty:
        print(f"     [!] Sem dados disponíveis para este período na API.")
        return False

    if 'timestamp' not in df.columns:
        df = df.reset_index(names='timestamp')
        
    if is_inverted:
        if 'bid' in df.columns and 'ask' in df.columns:
            # 1. Troca os zeros (falha de liquidez) por NaN
            df[['bid', 'ask']] = df[['bid', 'ask']].replace(0, np.nan)
            
            # 2. Preenche gaps com o último preço conhecido (Forward Fill)
            # E se o primeiro registro já for nulo, usa o próximo conhecido (Backward Fill)
            df[['bid', 'ask']] = df[['bid', 'ask']].ffill().bfill()
            
            # 3. Inversão matemática segura
            df['bid'] = 1.0 / df['bid']
            df['ask'] = 1.0 / df['ask']
            
            # 4. Transpõe os nomes (cruzamento do spread)
            df.rename(columns={'ask': 'bid', 'bid': 'ask'}, inplace=True)
        else:
            # Fallback de segurança para outras colunas de preço caso o layout mude
            price_columns = [col for col in df.columns if col not in ['timestamp', 'ask_volume', 'bid_volume', 'volume']]
            for col in price_columns:
                df[col] = df[col].replace(0, np.nan).ffill().bfill()
                df[col] = 1.0 / df[col]

    df.to_parquet(output_file, engine='pyarrow', compression='snappy')
    print(f"     [+] Novo lote salvo: {filename} ({len(df)} ticks)")
    
    time.sleep(2)
    return True

def consolidate_parquet_files(base_currency: str, quote_currency: str, start_date: datetime, end_date: datetime, partials_dir: Path, main_dir: Path):
    print(f"\n[*] Iniciando consolidação dos arquivos mensais...")
    
    file_pattern = f"{base_currency}{quote_currency}_??????_ticks.parquet"
    files_to_consolidate = sorted(list(partials_dir.glob(file_pattern)))
    
    if not files_to_consolidate:
        print("[!] Nenhum arquivo parcial encontrado na pasta para consolidar.")
        return

    print(f"[*] {len(files_to_consolidate)} lotes encontrados em '{partials_dir.name}'. Consolidando...")
    
    try:
        dataset = pq.ParquetDataset(files_to_consolidate)
        df_consolidated = dataset.read().to_pandas()
        
        df_consolidated.sort_values(by='timestamp', inplace=True)
        df_consolidated.reset_index(drop=True, inplace=True)
        
        final_filename = f"{base_currency}{quote_currency}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_consolidated_ticks.parquet"
        final_output_file = main_dir / final_filename
        
        df_consolidated.to_parquet(final_output_file, engine='pyarrow', compression='snappy')
        print(f"[+] Consolidação concluída com sucesso!\n    Salvo em: {final_output_file}")
        
    except Exception as e:
        print(f"[!] Erro crítico durante a consolidação: {e}")

def extract_and_save_tick_data_batched(base: str, quote: str, start_date: datetime, end_date: datetime, main_output_path: Path):
    partials_dir_name = f"{base}{quote}-parciais"
    partials_dir = main_output_path / partials_dir_name
    
    main_output_path.mkdir(parents=True, exist_ok=True)
    partials_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[*] Iniciando Pipeline para {base}/{quote}")
    print(f"[*] Parciais serão salvas em: {partials_dir}")
    
    current_start = start_date
    
    while current_start < end_date:
        current_end = current_start + relativedelta(months=1)
        if current_end > end_date:
            current_end = end_date
            
        extract_chunk(base, quote, current_start, current_end, partials_dir)
        current_start = current_end

    print("\n[+] Verificação e extração de lotes finalizada!")
    
    consolidate_parquet_files(base, quote, start_date, end_date, partials_dir, main_output_path)

def get_valid_date(prompt: str) -> datetime:
    while True:
        date_str = input(prompt).strip()
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            print("[!] Formato inválido. Use AAAA-MM-DD (ex: 2020-01-01).")

if __name__ == "__main__":
    print("="*60)
    print(" Pipeline Resiliente de Ticks Forex (Idempotente) ")
    print("="*60)
    
    base = input("Moeda Base (ex: USD): ").strip().upper()
    quote = input("Moeda de Cotação (ex: GBP): ").strip().upper()
    
    print("\n[Período]")
    start = get_valid_date("Data de Início: ")
    end = get_valid_date("Data de Fim: ")
    
    print("\n[Diretório]")
    dir_input = input("Diretório base (Enter para 'data/forex'): ").strip()
    main_dir = Path(dir_input) if dir_input else Path("data/forex")
    
    if start >= end:
        print("\n[!] Erro: A Data de Início deve ser anterior à Data de Fim.")
    else:
        extract_and_save_tick_data_batched(base, quote, start, end, main_dir)