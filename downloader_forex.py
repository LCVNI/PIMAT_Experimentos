import requests
import sys
import pandas as pd


API_KEY = "JC26TC1MTTWA2NOD"

BASE_URL = "https://www.alphavantage.co/query"


TIMEFRAMES = {
    "daily": "FX_DAILY",
    "weekly": "FX_WEEKLY",
    "monthly": "FX_MONTHLY"
}


def baixar_forex(
    from_currency,
    to_currency,
    timeframe,
    data_inicio,
    data_fim
):

    # Verifica o timeframe
    if timeframe not in TIMEFRAMES:
        print("Timeframe inválido.")
        print("Opções: daily, weekly ou monthly")
        return

    # Converte as datas
    try:
        data_inicio = pd.to_datetime(data_inicio)
        data_fim = pd.to_datetime(data_fim)

    except ValueError:
        print("Formato de data inválido.")
        print("Use o formato YYYY-MM-DD.")
        return

    # Verifica se o intervalo é válido
    if data_inicio > data_fim:
        print("Erro: a data inicial não pode ser maior que a data final.")
        return

    params = {
        "function": TIMEFRAMES[timeframe],
        "from_symbol": from_currency,
        "to_symbol": to_currency,
        "outputsize": "full",
        "datatype": "csv",
        "apikey": API_KEY
    }

    print("Baixando dados da Alpha Vantage...")

    response = requests.get(
        BASE_URL,
        params=params
    )

    if response.status_code != 200:
        print(f"Erro HTTP: {response.status_code}")
        return

    # Salva temporariamente a resposta
    arquivo_temporario = "temp_forex.csv"

    with open(
        arquivo_temporario,
        "w",
        encoding="utf-8"
    ) as arquivo:

        arquivo.write(response.text)

    # Tenta ler o CSV
    try:
        df = pd.read_csv(arquivo_temporario)

    except Exception as erro:
        print("Não foi possível interpretar a resposta da API.")
        print(response.text)
        print(erro)
        return

    # Verifica se a API retornou dados válidos
    if "timestamp" not in df.columns:
        print("A API não retornou dados válidos.")
        print(df)
        return

    # Converte a coluna de data
    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    # Filtra pelo período escolhido
    df = df[
        (df["timestamp"] >= data_inicio) &
        (df["timestamp"] <= data_fim)
    ]

    # Ordena pela data
    df = df.sort_values("timestamp")

    # Nome do arquivo
    nome_arquivo = (
        f"{from_currency}_{to_currency}_"
        f"{timeframe}_"
        f"{data_inicio.strftime('%Y%m%d')}_"
        f"{data_fim.strftime('%Y%m%d')}.csv"
    )

    # Salva o resultado
    df.to_csv(
        nome_arquivo,
        index=False
    )

    # Remove o arquivo temporário
    import os
    os.remove(arquivo_temporario)

    print()
    print("Download concluído!")
    print(f"Registros encontrados: {len(df)}")
    print(f"Arquivo: {nome_arquivo}")


if __name__ == "__main__":

    if len(sys.argv) != 6:

        print(
            "Uso:\n"
            "python forex.py "
            "MOEDA1 MOEDA2 TIMEFRAME "
            "DATA_INICIO DATA_FIM"
        )

        print()
        print("Exemplo:")
        print(
            "python forex.py "
            "EUR USD daily "
            "2020-01-01 2024-12-31"
        )

        sys.exit(1)

    from_currency = sys.argv[1].upper()
    to_currency = sys.argv[2].upper()
    timeframe = sys.argv[3].lower()
    data_inicio = sys.argv[4]
    data_fim = sys.argv[5]

    baixar_forex(
        from_currency,
        to_currency,
        timeframe,
        data_inicio,
        data_fim
    )