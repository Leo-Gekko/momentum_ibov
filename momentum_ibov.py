#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importando bibliotecas

import pandas as pd
import numpy as np
import investpy 
import yfinance as yf
import matplotlib.pyplot as plt
import datetime as dt
from pandas.tseries.offsets import MonthEnd
import statsmodels.api as sm


# In[2]:


#encontrando população de ativos dentro do ibovespa
def carteira_ibov(indice):
    url = f'http://bvmf.bmfbovespa.com.br/indices/ResumoCarteiraTeorica.aspx?Indice={indice.upper()}&idioma-pt-br'
    return pd.read_html(url, decimal=',', thousands='.', index_col = 'Código')[0][:-1]


# In[3]:


#ajustando tickers para buscar dados
ibov = carteira_ibov('ibov')
acoes_ibov = ibov.index.to_list()
for i in range (0, len(acoes_ibov),1):
    acoes_ibov [i] += '.SA'


# In[4]:


#dados históricos
dados = yf.download(acoes_ibov, start = '2016-12-1', end = '2022-1-1')['Adj Close']
dados.index = pd.to_datetime(dados.index)


# In[5]:


#encontrando retorno mensal dos ativos
retorno_mensal = dados.pct_change().resample('M').agg(lambda x: (x+1).prod()-1).dropna()

#definindo quando a carteira será montada
ultimo_mes = (retorno_mensal+1).rolling(1).apply(np.prod)-1

#quando a estratégia começa
formacao = dt.datetime(2017,1,31)


# In[6]:


#criando ferramentas para backtest
fim_amostragem = formacao - MonthEnd(1)
ret_ult = ultimo_mes.loc[fim_amostragem]
ret_ult = ret_ult.reset_index()


# In[7]:


#criando ranking mensal
amostra = len(ret_ult.index)-1
ret_ult['Ranking']=pd.qcut(ret_ult.iloc[:,1],amostra,labels = False,duplicates='drop')


# In[8]:


#criando carteira mensal
acoes_carteira = ret_ult.nlargest(n=10,columns=['Ranking'])


# In[9]:


acoes_retorno = retorno_mensal.loc[formacao + MonthEnd(1),retorno_mensal.columns.isin(acoes_carteira['index'])]


# In[12]:


#datas onde a carteira é criada
for i in range((12*5)):
    print(formacao + MonthEnd(i))


# In[14]:


#montando carteiras_10 acoes

def momentum(formacao):
    fim_amostragem = formacao - MonthEnd(1)
    ret_ult = ultimo_mes.loc[fim_amostragem]
    ret_ult = ret_ult.reset_index()
    ret_ult['Ranking']=pd.qcut(ret_ult.iloc[:,1],amostra,labels = False,duplicates='drop')
    acoes_carteira_10 = ret_ult.nlargest(n=10,columns = ['Ranking'])
    acoes_10 = retorno_mensal.loc[formacao+MonthEnd(1), retorno_mensal.columns.isin(acoes_carteira_10['index'])]
    retornos_estrategia_10 = acoes_10.mean()
    return retornos_estrategia_10

#backtest_10
retornos_10 = []
datas = []

for i in range ((12*5)-1):
    retornos_10.append(momentum(formacao + MonthEnd(i)))
    datas.append(formacao + (MonthEnd(i)))


# In[15]:


#montando carteiras_15 acoes

def momentum(formacao):
    fim_amostragem = formacao - MonthEnd(1)
    ret_ult = ultimo_mes.loc[fim_amostragem]
    ret_ult = ret_ult.reset_index()
    ret_ult['Ranking']=pd.qcut(ret_ult.iloc[:,1],amostra,labels = False,duplicates='drop')
    acoes_carteira_15 = ret_ult.nlargest(n=15,columns = ['Ranking'])
    acoes_15 = retorno_mensal.loc[formacao+MonthEnd(1), retorno_mensal.columns.isin(acoes_carteira_15['index'])]
    retornos_estrategia_15 = acoes_15.mean()
    return retornos_estrategia_15

#backtest_15
retornos_15 = []
datas = []

for i in range ((12*5)-1):
    retornos_15.append(momentum(formacao + MonthEnd(i)))
    datas.append(formacao + MonthEnd(i))


# In[16]:


#montando carteiras_20 acoes

def momentum(formacao):
    fim_amostragem = formacao - MonthEnd(1)
    ret_ult = ultimo_mes.loc[fim_amostragem]
    ret_ult = ret_ult.reset_index()
    ret_ult['Ranking']=pd.qcut(ret_ult.iloc[:,1],amostra,labels = False,duplicates='drop')
    acoes_carteira_20 = ret_ult.nlargest(n=20,columns = ['Ranking'])
    acoes_20 = retorno_mensal.loc[formacao+MonthEnd(1), retorno_mensal.columns.isin(acoes_carteira_20['index'])]
    retornos_estrategia_20 = acoes_20.mean()
    return retornos_estrategia_20

#backtest_20
retornos_20 = []
datas = []

for i in range ((12*5)-1):
    retornos_20.append(momentum(formacao + MonthEnd(i)))
    datas.append(formacao + MonthEnd(i))


# In[17]:


#coletando dados de benchmark
ibov = yf.download('^BVSP',start = '2017-1-1', end = '2022-1-1')['Adj Close']
ibov_mensal = ibov.pct_change().resample('M').agg(lambda x: (x+1).prod() -1)


# In[18]:


#retornos de portfolio e benchmark em dataframe
df_10 = pd.DataFrame(retornos_10)
df_15 = pd.DataFrame(retornos_15)
df_20 = pd.DataFrame(retornos_20)
df_ibov = pd.DataFrame(ibov_mensal.values)


# In[19]:


#criando um só dataframe e retorno acumulado
df = df_10
df['15 ações'] = df_15
df['20 ações'] = df_20
df['Ibov'] = df_ibov

cota_10 = (1+df_10).cumprod()
cota_15 = (1+df_15).cumprod()
cota_20 = (1+df_20).cumprod()
ibov_cota = (1+df_ibov).cumprod()


# In[20]:


#comparando retornos
retornos = cota_10
retornos['Ibov'] = ibov_cota


# In[21]:


#ajustando nomes
retornos.columns = ['10 ações', '15 ações', '20 ações', 'Ibov' ]


# In[22]:


#incluindo datas
retornos['Datas'] = pd.DataFrame(datas)


# In[23]:


#incluindo datas 2
retornos.set_index('Datas')


# In[25]:


#gráfico de comparacao entre portfolios

plt.figure(figsize=(13,8))

plt.plot(retornos.Datas,retornos.loc[:,'10 ações'],label= '10 ações',color='cyan')
plt.plot(retornos.Datas,retornos.loc[:,'15 ações'],label= '15 ações',color='white')
plt.plot(retornos.Datas,retornos.loc[:,'20 ações'],label= '20 ações',color='gold')

plt.title('Comparativo de performance', fontsize=12)
plt.legend(fontsize=14)
plt.xlabel('Datas')
plt.ylabel('Performance',fontsize=12)
plt.style.use('dark_background')


# In[26]:


#calculando vol ano 

log_10 = np.log(retornos.loc[:,'10 ações']/retornos.loc[:,'10 ações'].shift(1)).dropna()
vol_mensal_10 = log_10.std()
vol_ano_10 = vol_mensal_10*np.sqrt(12)

log_15 = np.log(retornos.loc[:,'15 ações']/retornos.loc[:,'15 ações'].shift(1)).dropna()
vol_mensal_15 = log_15.std()
vol_ano_15 = vol_mensal_15*np.sqrt(12)

log_20 = np.log(retornos.loc[:,'20 ações']/retornos.loc[:,'20 ações'].shift(1)).dropna()
vol_mensal_20 = log_20.std()
vol_ano_20 = vol_mensal_20*np.sqrt(12)

log_ibov = np.log(retornos.loc[:,'Ibov']/retornos.loc[:,'Ibov'].shift(1)).dropna()
vol_mensal_ibov = log_ibov.std()
vol_ano_ibov = vol_mensal_ibov*np.sqrt(12)


print(vol_ano_10, vol_ano_15, vol_ano_20, vol_ano_ibov)


# In[27]:


from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.offline as pyo


# In[32]:


#gráfico de comparacao entre portfolios e ibov

plt.figure(figsize=(17,9))

plt.plot(retornos.Datas,retornos.loc[:,'10 ações'],label= '10 ações',color='cyan')
plt.plot(retornos.Datas,retornos.loc[:,'15 ações'],label= '15 ações',color='white')
plt.plot(retornos.Datas,retornos.loc[:,'20 ações'],label= '20 ações',color='gold')
plt.plot(retornos.Datas,retornos.Ibov,label = 'ibov',linestyle='dashed',linewidth=2,color='green')

plt.title('Comparativo de performance', fontsize=18)
plt.ylabel('Desempenho', fontsize=12)
plt.legend(fontsize=14)
plt.style.use('dark_background')


# In[33]:


#calculando drawdown maximo

drawdown_10 = retornos['10 ações']
pico_10 = drawdown_10.expanding(min_periods=1).max()
drawdown_max10 = ((drawdown_10/pico_10)-1)*100
drawdown_maximo_10 = drawdown_max10.min()

drawdown_15 = retornos['15 ações']
pico_15 = drawdown_15.expanding(min_periods=1).max()
drawdown_max15 = ((drawdown_15/pico_15)-1)*100
drawdown_maximo_15 = drawdown_max15.min()

drawdown_20 = retornos['20 ações']
pico_20 = drawdown_20.expanding(min_periods=1).max()
drawdown_max20 = ((drawdown_20/pico_20)-1)*100
drawdown_maximo_20 = drawdown_max20.min()

drawdown_ibov = retornos['Ibov']
pico_ibov = drawdown_ibov.expanding(min_periods=1).max()
drawdown_max_ibov = ((drawdown_ibov/pico_ibov)-1)*100
drawdown_maximo_ibov = drawdown_max_ibov.min()


print(drawdown_maximo_10,drawdown_maximo_15,drawdown_maximo_20, drawdown_maximo_ibov)


# In[34]:


#criando df para o drawdown
drawdown_comparacao = pd.DataFrame({
    '10 ações':[(drawdown_maximo_10)],
     '15 ações':[(drawdown_maximo_15)],
     '20 ações':[(drawdown_maximo_20)],
     'Ibov':[(drawdown_maximo_ibov)]
    
    
})


# In[35]:


#anualizando retornos
retorno_ano_10 = retornos['10 ações'].iloc[-1]-retornos['10 ações'].iloc[0]/retornos['10 ações'].iloc[0]
retorno_ano_10 = ((1+retorno_ano_10)**(12/59)-1)

retorno_ano_15 = retornos['15 ações'].iloc[-1]-retornos['15 ações'].iloc[0]/retornos['15 ações'].iloc[0]
retorno_ano_15 = ((1+retorno_ano_15)**(12/59)-1)

retorno_ano_20 = retornos['20 ações'].iloc[-1]-retornos['20 ações'].iloc[0]/retornos['20 ações'].iloc[0]
retorno_ano_20 = ((1+retorno_ano_20)**(12/59)-1)

retorno_ano_ibov = retornos['Ibov'].iloc[-1]-retornos['Ibov'].iloc[0]/retornos['Ibov'].iloc[0]
retorno_ano_ibov = ((1+retorno_ano_ibov)**(12/59)-1)

#criando df com retornos anaulizados
retornos_anualizados = pd.DataFrame({
    '10 ações':[(retorno_ano_10)],
    '15 ações':[(retorno_ano_15)],
    '20 ações':[(retorno_ano_20)],
    'Ibov':[(retorno_ano_ibov)],
})


# In[36]:


#gráfico do drawdown
drawdown_grafico = go.Figure(go.Bar(x=drawdown_comparacao.iloc[-1], y = ['10 açoes', '15 ações', '20 ações', 'ibov'], orientation = 'h'))
drawdown_grafico.show()


# In[37]:


#gráfico do retorno anualizado
retorno_anualizado_grafico = go.Figure(go.Bar(x=retornos_anualizados.iloc[-1], y = ['10 açoes', '15 ações', '20 ações', 'ibov'], orientation = 'h'))
retorno_anualizado_grafico.show()


# In[42]:


histograma = make_subplots(rows=2, cols =2)

dez_ativos = go.Histogram(x=retornos_10, name = '10 ações')
quinze_ativos = go.Histogram(x=retornos_15, name = '15 ações')
vinte_ativos = go.Histogram(x=retornos_20, name = '20 ações')
ibov_retornos = go.Histogram(x=ibov_mensal, name = 'Ibov')

histograma.append_trace(dez_ativos,1 ,1)
histograma.append_trace(quinze_ativos,1 ,2)
histograma.append_trace(vinte_ativos,2 ,1)
histograma.append_trace(ibov_retornos,2 ,2)


histograma.update_layout(autosize = False, width=1000, height = 550, title = 'Histograma de retornos e vol anualizada',
                  xaxis = dict(title='Volatilidade com 10 ações: '+str(np.round(vol_ano_10*100,1))),
xaxis2 = dict(title='Volatilidade com 15 ações: '+str(np.round(vol_ano_15*100,1))),
xaxis3 = dict(title='Volatilidade com 20 ações: '+str(np.round(vol_ano_20*100,1))),
xaxis4 = dict(title='Volatilidade do Ibov: '+str(np.round(vol_ano_ibov*100,1))),)





histograma.show()


# In[ ]:




