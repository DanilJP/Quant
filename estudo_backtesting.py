import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# ============================== #
#     Indicadores e Métricas     #
# ============================== #

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def sharpe_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    excess_ret = returns - rf
    return np.sqrt(252) * excess_ret.mean() / excess_ret.std()

def sortino_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    excess_ret = returns - rf
    downside_std = excess_ret[excess_ret < 0].std()
    return np.sqrt(252) * excess_ret.mean() / downside_std

def calmar_ratio(cagr_value: float, drawdown: float) -> float:
    return cagr_value / abs(drawdown) if drawdown != 0 else np.nan

def cagr(df: pd.DataFrame) -> float:
    anos = (df.index[-1] - df.index[0]) / 365.25
    retorno_total = df['portfolio_value'].iloc[-1] / df['portfolio_value'].iloc[0]
    return retorno_total ** (1 / anos) - 1

# ============================== #
#     Lógica de Estratégias      #
# ============================== #

def calcular_estrategias(ativos: list[str], capital_inicial: float = 100000):
    df = yf.download(ativos[0], start="2015-01-01")['Close'].reset_index()
    df.rename(columns={ativos[0]: 'preco'}, inplace=True)
    df['retorno'] = df['preco'].pct_change().fillna(0)

    # Sinais de estratégias
    df['media_curta'] = df['preco'].rolling(20).mean()
    df['media_longa'] = df['preco'].rolling(100).mean()
    df['rsi'] = compute_rsi(df['preco'])
    df['vol'] = df['retorno'].rolling(20).std()

    sinais = {
        'Momentum': df['retorno'].rolling(180).mean() > 0,
        'Médias Móveis': df['media_curta'] > df['media_longa'],
        'Reversão à Média': df['preco'] < df['media_curta'],
        'RSI': df['rsi'] < 30,
        'Breakout': df['preco'] > df['preco'].rolling(20).max().shift(1),
        'Volatilidade': df['vol'] < df['vol'].rolling(50).mean(),
    }

    # MLP Classifier
    df['target'] = (df['retorno'].shift(-1) > 0).astype(int)
    features = ['retorno', 'rsi', 'vol']
    X = StandardScaler().fit_transform(df[features].fillna(0))
    y = df['target']
    mlp = MLPClassifier(hidden_layer_sizes=(150, 100, 50), max_iter=5000, random_state=42)
    mlp.fit(X[:-1], y[:-1])
    sinais['MLP Classifier'] = pd.Series(mlp.predict(X), index=df.index).astype(bool)

    # Avaliação de estratégias
    resultados = {}
    metricas = []

    for nome, sinal in sinais.items():
        df_copy = df.copy()
        df_copy['position'] = sinal.shift(1).fillna(False).astype(int)
        df_copy['cash'], df_copy['shares'], df_copy['portfolio_value'] = capital_inicial, 0, capital_inicial

        for i in range(1, len(df_copy)):
            preco = df_copy.loc[i, 'preco']
            prev_position = df_copy.loc[i-1, 'position']
            curr_position = df_copy.loc[i, 'position']
            if curr_position and not prev_position:
                qtd = df_copy.loc[i-1, 'cash'] // preco
                df_copy.at[i, 'shares'] = qtd
                df_copy.at[i, 'cash'] = df_copy.loc[i-1, 'cash'] - qtd * preco
            elif not curr_position and prev_position:
                qtd = df_copy.loc[i-1, 'shares']
                df_copy.at[i, 'cash'] = df_copy.loc[i-1, 'cash'] + qtd * preco
                df_copy.at[i, 'shares'] = 0
            else:
                df_copy.at[i, 'shares'] = df_copy.loc[i-1, 'shares']
                df_copy.at[i, 'cash'] = df_copy.loc[i-1, 'cash']
            df_copy.at[i, 'portfolio_value'] = df_copy.loc[i, 'shares'] * preco + df_copy.loc[i, 'cash']

        df_copy['strategy_return'] = df_copy['portfolio_value'].pct_change().fillna(0)
        df_copy['cummax'] = df_copy['portfolio_value'].cummax()
        df_copy['drawdown'] = (df_copy['portfolio_value'] - df_copy['cummax']) / df_copy['cummax']

        sharpe = sharpe_ratio(df_copy['strategy_return'])
        sortino = sortino_ratio(df_copy['strategy_return'])
        max_dd = df_copy['drawdown'].min()
        total_ret = df_copy['portfolio_value'].iloc[-1] / capital_inicial - 1
        cagr_val = cagr(df_copy)
        calmar = calmar_ratio(cagr_val, max_dd)

        resultados[nome] = {'df': df_copy, 'sharpe': sharpe, 'sortino': sortino,
                            'drawdown': max_dd, 'retorno_total': total_ret,
                            'cagr': cagr_val, 'calmar': calmar}

        metricas.append({
            'Estratégia': nome, 'Sharpe': sharpe, 'Sortino': sortino,
            'Máx Drawdown': max_dd, 'Retorno Total': total_ret,
            'CAGR': cagr_val, 'Calmar': calmar
        })

    # Estratégia Buy and Hold
    df_bh = df.copy()
    preco_inicial = df_bh['preco'].iloc[0]
    df_bh['shares'] = capital_inicial // preco_inicial
    df_bh['cash'] = capital_inicial - df_bh['shares'] * preco_inicial
    df_bh['portfolio_value'] = df_bh['shares'] * df_bh['preco'] + df_bh['cash']
    df_bh['strategy_return'] = df_bh['portfolio_value'].pct_change().fillna(0)
    df_bh['cummax'] = df_bh['portfolio_value'].cummax()
    df_bh['drawdown'] = (df_bh['portfolio_value'] - df_bh['cummax']) / df_bh['cummax']

    sharpe = sharpe_ratio(df_bh['strategy_return'])
    sortino = sortino_ratio(df_bh['strategy_return'])
    max_dd = df_bh['drawdown'].min()
    total_ret = df_bh['portfolio_value'].iloc[-1] / capital_inicial - 1
    cagr_val = cagr(df_bh)
    calmar = calmar_ratio(cagr_val, max_dd)

    resultados['Buy and Hold'] = {'df': df_bh, 'sharpe': sharpe, 'sortino': sortino,
                                  'drawdown': max_dd, 'retorno_total': total_ret,
                                  'cagr': cagr_val, 'calmar': calmar}

    metricas.append({
        'Estratégia': 'Buy and Hold', 'Sharpe': sharpe, 'Sortino': sortino,
        'Máx Drawdown': max_dd, 'Retorno Total': total_ret,
        'CAGR': cagr_val, 'Calmar': calmar
    })

    return df, resultados, pd.DataFrame(metricas)

# ============================== #
#     Interface com Streamlit    #
# ============================== #

st.title("Análise de Estratégias Quantitativas")

tickers = st.text_input("Digite o ticker da ação (ex: 'VALE3.SA')", 'VALE3.SA').split(',')
df, resultados, df_metricas = calcular_estrategias(tickers)

# Gráfico de comparação
st.subheader("Comparação de Estratégias")
fig, ax = plt.subplots(figsize=(14, 8))
for nome, r in resultados.items():
    ax.plot(r['df']['portfolio_value'], label=nome)
ax.set_title("Evolução do Portfólio")
ax.set_ylabel("Valor (R$)")
ax.legend()
st.pyplot(fig)

# Tabela de métricas
st.subheader("Métricas de Desempenho")
st.dataframe(df_metricas.round(4))

# Gráfico de barras das métricas
st.subheader("Gráfico Comparativo")
fig, ax = plt.subplots(figsize=(10, 6))
df_metricas.set_index('Estratégia').plot(kind='bar', ax=ax)
ax.set_title("Métricas das Estratégias")
ax.set_ylabel("Valor")
ax.set_xlabel("Estratégia")
st.pyplot(fig)
