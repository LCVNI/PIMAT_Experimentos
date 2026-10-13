import pandas as pd
from pathlib import Path

def validate_tick_data(file_path: Path):
    print(f"\n[*] Iniciando auditoria no arquivo: {file_path.name}")
    
    try:
        # 1. Carregar os dados brutos
        df = pd.read_parquet(file_path)
        total_linhas_iniciais = len(df)
        
        # 2. Normalização (Adequando ao DoD do PIMAT e dados da Dukascopy)
        # Verifica se precisa renomear (caso venha de outra fonte que não a Dukascopy)
        renames = {}
        if 'bidPrice' in df.columns: renames['bidPrice'] = 'bid'
        if 'askPrice' in df.columns: renames['askPrice'] = 'ask'
        if renames:
            df = df.rename(columns=renames)
            
        # Tratamento de Volume (Dukascopy gera ask_volume e bid_volume)
        if 'bid_volume' in df.columns and 'ask_volume' in df.columns:
            df['volume'] = df['bid_volume'] + df['ask_volume']
        elif 'bidVolume' in df.columns and 'askVolume' in df.columns:
            df['volume'] = df['bidVolume'] + df['askVolume']
            
        # Se o volume não existir por algum motivo, cria zerado para não quebrar o motor
        if 'volume' not in df.columns:
            df['volume'] = 0.0
            
        # Filtra estritamente para as 4 colunas exigidas
        colunas_exigidas = ['timestamp', 'ask', 'bid', 'volume']
        
        # Garante que as colunas existem antes de filtrar
        for col in colunas_exigidas:
            if col not in df.columns:
                print(f"[!] Erro fatal: Coluna essencial '{col}' não encontrada no arquivo.")
                return False
                
        df = df[colunas_exigidas]
        
        # 3. Verificação de Valores Nulos (NaN)
        nulos = df.isnull().sum().sum()
        if nulos > 0:
            print(f"  [!] Anomalia: {nulos} valores nulos detectados. Limpando...")
            df = df.dropna()
            
        # 4. Verificação de Spread Invertido (Bid > Ask)
        spread_invertido = df[df['bid'] > df['ask']]
        if not spread_invertido.empty:
            print(f"  [!] Anomalia: {len(spread_invertido)} ticks com spread invertido (Bid > Ask). Expurgando...")
            df = df.drop(spread_invertido.index)
            
        # 5. Verificação de Ordenação Cronológica
        if not df['timestamp'].is_monotonic_increasing:
            print("  [!] Anomalia: Timestamps fora de ordem cronológica. Reordenando...")
            df = df.sort_values('timestamp')
            
        # 6. Geração do Arquivo Final Validado
        linhas_finais = len(df)
        linhas_removidas = total_linhas_iniciais - linhas_finais
        
        clean_path = file_path.parent / f"{file_path.stem}_clean.parquet"
        
        # Otimização para Big Data: garante que timestamp continue sendo datetime
        df.to_parquet(clean_path, engine='pyarrow', compression='snappy')
        
        if linhas_removidas > 0:
            print(f"  [+] Limpeza concluída. {linhas_removidas} anomalias removidas.")
        else:
            print("  [+] Integridade verificada. Nenhuma anomalia matemática encontrada.")
            
        print(f"  [+] Arquivo final formatado salvo em: {clean_path.name}")
        print(f"  [*] Registro validado: {linhas_finais} ticks operacionais.")
        return True
        
    except Exception as e:
        print(f"[!] Erro crítico ao processar o arquivo {file_path.name}: {e}")
        return False

def menu_selecao_arquivo(diretorio: Path):
    """Busca arquivos Parquet no diretório e permite ao usuário escolher qual limpar."""
    if not diretorio.exists():
        print(f"[!] O diretório '{diretorio}' não existe.")
        return
        
    # Busca arquivos parquet, ignorando os que já tem '_clean' para não limpar duas vezes
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