import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.stats import norm
from scipy.optimize import brentq
from datetime import datetime

st.set_page_config(page_title="Sorriso de Volatilidade", layout="wide")
st.title("📈 Sorriso de Volatilidade de Opções")

# Funções auxiliares
def black_scholes_price(S, K, T, r, sigma, tipo='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if tipo == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def implied_volatility(price_market, S, K, T, r, tipo='call'):
    try:
        retorno = brentq(lambda sigma: black_scholes_price(S, K, T, r, sigma, tipo) - price_market, 1e-6, 5)
        return retorno
    except:
        return np.nan

# Entradas do usuário
ticker = st.selectbox("Escolha o ticker (ex: AAPL, MSFT, PETR4.SA)",['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'PETR4.SA', 'VALE3.SA', 'ITUB4.SA'])
r = st.number_input("Taxa livre de risco (anual)", value=0.05)
tipo_opcao = st.selectbox("Tipo de opção", ['call', 'put'])

# Coleta de dados
st.write("🔎 Coletando dados do Yahoo Finance...")
asset = yf.Ticker(ticker)
data = asset.option_chain(asset.options[0])  # Primeiro vencimento
options = data.calls if tipo_opcao == 'call' else data.puts
spot_price = asset.history(period="1d")['Close'][-1]
maturity_date = datetime.strptime(asset.options[10], "%Y-%m-%d")
T = (maturity_date - datetime.now()).days / 365.0

# Calcula volatilidade implícita
ivs = []
strikes = []
precos_bs = []
for _, row in options.iterrows():
    K = row['strike']
    market_price = row['lastPrice']
    iv = implied_volatility(market_price, spot_price, K, T, r, tipo=tipo_opcao)
    if not np.isnan(iv):
        strikes.append(K)
        ivs.append(iv)
        precos_bs.append(black_scholes_price(spot_price, K, T, r, iv, tipo_opcao))

# Gráfico do sorriso
st.subheader("📊 Sorriso de Volatilidade")
fig = go.Figure()
fig.add_trace(go.Scatter(x=strikes, y=ivs, mode='markers+lines', name='Volatilidade Implícita'))
fig.update_layout(xaxis_title="Strike", yaxis_title="Volatilidade", height=500)
st.plotly_chart(fig, use_container_width=True)

# Comparação de preços
st.subheader("📉 Comparação de preços de opções")
preco_constante = black_scholes_price(spot_price, spot_price, T, r, np.mean(ivs), tipo_opcao)
st.markdown(f"""
- **Preço atual da ação:** {spot_price:.2f}  
- **Strike at-the-money (aprox):** {spot_price:.2f}  
- **Vol. implícita média:** {np.mean(ivs):.2%}  
- **Preço usando Black-Scholes (vol. média):** ${preco_constante:.2f}
""")