# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 11:21:41 2023

@author: REGIS CARDOSO
"""

######################################################################################################
## ANÁLISE DE ANOMALIAS EM UM SINAL DE PRESSÃO EM TUBULAÇÃO DE GAS ###
######################################################################################################

## IMPORTAR AS BIBLIOTECAS UTILIZADAS ###


import pandas as pd
import numpy as np
import statistics
import math
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from datetime import timedelta, datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn.ensemble import IsolationForest
from array import array


## IMPORTANDO DADOS ###

df_inicial = pd.read_csv('Pressao.csv', sep=',')


## ORGANIZANDO O DATASET EM ORDEM CRESCENTE DE DATA ###

df_inicial = df_inicial.sort_values(by=['Data'])


## EXCLUINDO COLUNA INDESEJADA E RESETANDO O INDEX DO DATASET ###

df_inicial.drop(["Unnamed: 0"], axis=1, inplace=True)

df_inicial = df_inicial.reset_index()
df_inicial.drop(["index"], axis=1, inplace=True)


## VERIFICANDO O SINAL ###

plt.figure(figsize=(15, 8))
plt.plot(df_inicial['Pressao'], color='blue', alpha=0.4)
plt.show()


# CONVERTENDO AS DATAS SOMENTE EM DIAS DE FORMA SEQUENCIAL ###

string_data = []

for i in range(len(df_inicial)):
    if i == 0:
        Calc = ((pd.to_datetime(df_inicial['Data'].loc[i])) - (pd.to_datetime(df_inicial['Data'].loc[i])))
    else:
        Calc = ((pd.to_datetime(df_inicial['Data'].loc[i])) - (pd.to_datetime(df_inicial['Data'].loc[i - 1])))
    string_data.append(Calc)

string_data_convertida = []
convet = 0

for i in range(len(string_data)):
    new_value_convert = ((np.timedelta64(string_data[i], 'D').astype(int)) + (
                np.timedelta64(string_data[i], 'h').astype(int) / 24) + (
                                 np.timedelta64(string_data[i], 'm').astype(int) / (24 * 60)) + (
                                 np.timedelta64(string_data[i], 's').astype(int) / (24 * 60 * 60)))
    convet = convet + new_value_convert

    string_data_convertida.append(convet)

df_inicial['Data_Dias'] = string_data_convertida

df_inicial.drop(["Data"], axis=1, inplace=True)


df_inicial['Pressao_2'] = df_inicial['Pressao']

df_inicial.drop(["Pressao"], axis=1, inplace=True)


## CRIANDO MODELO USANDO O ALGORITMO DE ONECLASSSVM PARA DETECÇÃO DE POSSÍVEIS ANOMALIAS ###

from sklearn.svm import OneClassSVM

# TREINANDO O MODELO PARA DETECÇÃO DE ANOMALIAS

clf = OneClassSVM(gamma='auto').fit(df_inicial)

df_treino = df_inicial['Pressao_2']

model=OneClassSVM(gamma='auto')
model.fit(df_inicial[['Pressao_2']])


# ADICIONANDO OS SCORES E AS ANOMALIAS NO DATASET

df_inicial['Scores']=model.decision_function(df_inicial[['Pressao_2']])
df_inicial['Anomalias']=model.predict(df_inicial[['Pressao_2']])


### SEPARANDO OS DATASETS EM DATASET COM E SEM ANOMALIAS ###

# DATASET COM ANOMALIAS

df_mask = df_inicial['Anomalias'] == -1
anomalias = df_inicial[df_mask]


# DATASET SEM ANOMALIAS

df_mask = df_inicial['Anomalias'] == 1
filtro_anomalias = df_inicial[df_mask]


### VISUALIZANDO OS DADOS E AS ANOMALIAS DETECTADAS COM ONECLASSSVM

plt.figure(figsize=(15, 8))
plt.title('Tendência', fontsize=24)
plt.plot(df_inicial['Data_Dias'],df_inicial['Pressao_2'], label='Anomalias Dados Normais', color='blue', alpha=0.4)
plt.plot(anomalias['Data_Dias'], anomalias['Pressao_2'], 'o', label='Anomalias Detectadas', color='red', alpha=0.9)
plt.ylabel("Amplitude", fontsize=24)
plt.xlabel("Tempo (dias)", fontsize=24)
plt.show()





## CRIANDO MODELO USANDO O ALGORITMO DE ISOLATIONFOREST PARA DETECÇÃO DE POSSÍVEIS ANOMALIAS ###

# TREINANDO O MODELO PARA DETECÇÃO DE ANOMALIAS

df_inicial2 = []
df_inicial2 = pd.DataFrame(df_inicial2)
df_inicial2['Data_Dias'] = df_inicial['Data_Dias']
df_inicial2['Pressao_2'] = df_inicial['Pressao_2']

model=IsolationForest(n_estimators=50, max_samples='auto', contamination=float(0.1),max_features=1.0)
model.fit(df_inicial2[['Pressao_2']])


# ADICIONANDO OS SCORES E AS ANOMALIAS NO DATASET

df_inicial2['Scores']=model.decision_function(df_inicial2[['Pressao_2']])
df_inicial2['Anomalias']=model.predict(df_inicial2[['Pressao_2']])


# DATASET COM ANOMALIAS

df_mask = df_inicial2['Anomalias'] == -1
anomalias2 = df_inicial2[df_mask]

# DATASET SEM ANOMALIAS

df_mask = df_inicial2['Anomalias'] == 1
filtro_anomalias2 = df_inicial2[df_mask]


### VISUALIZANDO OS DADOS E AS ANOMALIAS DETECTADAS COM ISOLATIONFOREST

plt.figure(figsize=(15, 8))
plt.title('Detecção de Anomalias', fontsize=24)
plt.plot(df_inicial2['Data_Dias'],df_inicial2['Pressao_2'], label='Anomalias Dados Normais', color='blue', alpha=0.4)
plt.plot(anomalias2['Data_Dias'], anomalias2['Pressao_2'], 'o', label='Anomalias Detectadas', color='red', alpha=0.9)
plt.ylabel("Amplitude", fontsize=24)
plt.xlabel("Tempo (dias)", fontsize=24)
plt.show()

