#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import pathlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split, KFold
from tabulate import tabulate
from pandas_profiling import ProfileReport
from sklearn import preprocessing,metrics
from sklearn.feature_selection import VarianceThreshold
from sklearn.cluster import DBSCAN
from collections import Counter
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pylab import rcParams
import scipy.spatial.distance
from pandas import DataFrame
from sklearn.cluster import KMeans


# # Etapas iniciais de limpeza, transformação e seleção de variáveis

# In[21]:


def realiza_por_grupo(df_total):
    metrica = 'euclidean'
    lista_perfis , grupos = pd.DataFrame(columns = df_total.columns), df_total['CO_GRUPO'].unique()
    resultados = pd.DataFrame()
    i = 0
    for grupo in grupos:
        lista_perfis = df_total[df_total['CO_GRUPO'].isin([grupo])]
        kf = KFold(n_splits=10)
        x = lista_perfis.drop(columns = ['INST_CURSO', 'CO_GRUPO'], axis = 1)
        for train_index, test_index in kf.split(x):
            df_treino = lista_perfis.iloc[train_index]
            df_teste = lista_perfis.iloc[test_index]   
            
        print('GRUPO: ', grupo, " Tamanho da amostra:",lista_perfis.shape[0], " CO_IES UNICOS: ",lista_perfis['INST_CURSO'].unique().shape[0])
        df_treino = treino(df_treino)
        resultados = teste(df_treino, df_teste, metrica, resultados,grupo,lista_perfis.shape[0])
    print("Média total de acertos: ", resultados[0].sum()/len(resultados[0]))
    print("Média total de distancias: ", resultados[1].sum()/len(resultados[1]))
    return resultados

def treino(df_treino):         
    df_treino = df_treino.groupby(df_treino.columns.tolist(),as_index=False).size()
    df_treino= df_treino.sort_values('size').drop_duplicates('INST_CURSO',keep='last')
    return df_treino
    
def teste(df_treino, df_teste, metrica,resultados,grupo, tamanho): 
    df_perfis_adaptado = df_treino.drop(labels=['INST_CURSO','size'],axis=1)
    resultado = pd.DataFrame(columns=['Curso_real', 'Curso_definido', 'Distancia', 'Total_amostra', 'Tamanho_perfil'])                    
    i=0
    for index, row in df_teste.iterrows():
        curso_real = row['INST_CURSO']
        row = row.drop(labels=['INST_CURSO'])
        row_adaptada =  row.to_frame().T
        distancia = scipy.spatial.distance.cdist(df_perfis_adaptado,row_adaptada, metric=metrica)                
        similar = df_treino[distancia==distancia.min()].nlargest(1, 'size')
        for jndex, s in similar.iterrows():
            resultado.loc[i] = [curso_real,s['INST_CURSO'],distancia.min(), tamanho,s['size']]
            i=i+1
    mediadistancia = resultado['Distancia'].sum()/len(resultado['Distancia'])
    igual = verifica_taxa(resultado)
    resultados = resultados.append([[igual,mediadistancia, tamanho, grupo, len(resultado['Curso_real'].unique()), len(resultado['Curso_definido'].unique())]],ignore_index=True )
    return resultados


def verifica_taxa(resultado):
    igual = 0
    for index, row in resultado.iterrows():
        if (row['Curso_real'] == row['Curso_definido']):
            igual = igual+1 
    igual=(igual*100)/len(resultado)
    print('ACERTO TOTAL: ',igual)
    return igual


# In[22]:


def exibir( ):
    print(tabulate([['2017', arq17.shape[1],arq17.shape[0]],
                    ['2018', arq18.shape[1],arq18.shape[0]],
                    ['2019', arq19.shape[1],arq19.shape[0]]],
                    headers=['Ano','Coluna', 'Linha'], 
                    tablefmt='orgtbl'))
    
def verifica_nulos(arq):  
    variaveis = arq.columns.tolist()
    print('valores nulos do ano de:',arq['NU_ANO'][0])
    for pos in range(0,len(variaveis)):
        nulos = arq[variaveis[pos]].isna().sum()
        if nulos >0 :
            porcentagem = arq[variaveis[pos]].isna().sum() * 100 / len(arq[variaveis[pos]])
            print(variaveis[pos],';',nulos,";",round(porcentagem,5),"%")
    porcentagem = arq.isna().sum().sum() * 100 / (len(arq)*arq.shape[1])
    print(arq['NU_ANO'][0],';',arq.isna().sum().sum(),";",round(porcentagem,5),"% de um total de :", (len(arq)*arq.shape[1]), "itens")

    
def verifica_nulosSANO(arq):  
    variaveis = arq.columns.tolist()
    print('valores nulos :')
    for pos in range(0,len(variaveis)):
        nulos = arq[variaveis[pos]].isna().sum()
        if nulos >0 :
            porcentagem = arq[variaveis[pos]].isna().sum() * 100 / len(arq[variaveis[pos]])
            print(variaveis[pos],';',nulos,";",round(porcentagem,5),"%")
    porcentagem = arq.isna().sum().sum() * 100 / (len(arq)*arq.shape[1])
   
    
def limites(coluna):
    q1 = coluna.quantile(0.25)
    q3 = coluna.quantile(0.75)
    amplitude = q3 - q1
    return q1 - 1.5 * amplitude, q3 + 1.5 * amplitude

def verifica_perdas(inicial, final):
    linhas_removidas = inicial - final
    porcentagem = (linhas_removidas * 100)/inicial 
    print('{} linhas removidas'.format(linhas_removidas))
    print('{} porcentagem'.format(porcentagem))
    
def visualizacao(coluna,label,base):
    sns.set(style="darkgrid") 
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
    sns.boxplot(coluna, ax=ax_box)
    sns.histplot(data=coluna, x=coluna, ax=ax_hist) 
    ax_box.set(xlabel=label)
    plt.savefig('1.png', bbox_inches='tight')
    plt.show()


def grafico_barra(coluna):     
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(15, 5)
    sns.boxplot(x=coluna, ax=ax1)
    ax2.set_xlim(limites(coluna))
    sns.boxplot(x=coluna, ax=ax2)
    
    plt.figure(figsize=(15,5))
    ax = sns.barplot(x=coluna.value_counts().index, y=coluna.value_counts())
    ax.set_xlim(limites(coluna))
    
    
def histograma(coluna):    
    plt.figure(figsize=(15, 5))    
    plt.ylabel('Densidade')
    sns.distplot(coluna, hist=True)


def deleta_inicial(base):
    base = base.drop(columns = [ 'CO_CURSO','CO_RS_I1', 'CO_RS_I2', 'CO_RS_I3', 'CO_RS_I4', 'CO_RS_I5', 'CO_RS_I6', 
                                'CO_RS_I7', 'CO_RS_I8','CO_ORGACAD','CO_REGIAO_CURSO','CO_RS_I9', 'NT_GER','NT_FG', 
                                'NT_DIS_FG','NT_FG_D1', 'NT_FG_D1_PT','NT_FG_D1_CT', 'NT_FG_D2', 'NT_FG_D2_PT', 
                                'NT_FG_D2_CT', 'NT_CE', 'NT_OBJ_CE', 'NT_DIS_CE','NT_CE_D1', 'NT_CE_D2', 'NT_CE_D3',
                                'NU_ITEM_OFG', 'NU_ITEM_OFG_Z', 'NU_ITEM_OFG_X','NU_ITEM_OFG_N', 'NU_ITEM_OCE',
                                'NU_ITEM_OCE_Z', 'NU_ITEM_OCE_X', 'NU_ITEM_OCE_N','DS_VT_GAB_OFG_ORIG', 'NT_OBJ_FG', 
                                'DS_VT_GAB_OFG_FIN','DS_VT_GAB_OCE_ORIG','DS_VT_GAB_OCE_FIN','DS_VT_ESC_OFG', 
                                'DS_VT_ACE_OFG', 'DS_VT_ESC_OCE', 'DS_VT_ACE_OCE', 'TP_INSCRICAO','TP_INSCRICAO_ADM',
                                'QE_I27','QE_I28', 'QE_I29', 'QE_I30', 'QE_I31', 'QE_I32', 'QE_I33', 'QE_I34', 
                                'QE_I35', 'QE_I36','QE_I37', 'QE_I38', 'QE_I39', 'QE_I40', 'QE_I41', 'QE_I42', 
                                'QE_I43', 'QE_I44', 'QE_I45','QE_I46', 'QE_I47', 'QE_I48', 'QE_I49', 'QE_I50', 
                                'QE_I51', 'QE_I52', 'QE_I53', 'QE_I54','QE_I55', 'QE_I56', 'QE_I57', 'QE_I58', 
                                'QE_I59', 'QE_I60', 'QE_I61', 'QE_I62', 'QE_I63','QE_I64', 'QE_I65', 'QE_I66', 
                                'QE_I67', 'QE_I68', 'TP_PR_OB_FG', 'TP_PR_DI_FG', 'TP_PR_OB_CE', 'TP_PR_DI_CE', 
                                'TP_SFG_D1', 'TP_SFG_D2','TP_SCE_D1', 'TP_SCE_D2','TP_SCE_D3'])
    return base

def delete_faltaram(base, variavel):
    qtde_linhas = base.shape[0]
    base = base.drop(base[base[variavel] != 555].index)
    verifica_perdas(qtde_linhas,base.shape[0])
    base = base.drop(columns = [variavel])
    return base

def deleta_ifallquestionsrnull(arq):
    inicial = arq.shape[0]
    arq = arq.drop(arq[arq.QE_I01.isnull() & arq.QE_I02.isnull() & arq.QE_I03.isnull() & arq.QE_I04.isnull() & 
                       arq.QE_I05.isnull() & arq.QE_I06.isnull() & arq.QE_I07.isnull() & arq.QE_I08.isnull() & 
                       arq.QE_I09.isnull() & arq.QE_I10.isnull() & arq.QE_I11.isnull() & arq.QE_I12.isnull() & 
                       arq.QE_I13.isnull() & arq.QE_I14.isnull() & arq.QE_I15.isnull() & arq.QE_I16.isnull() & 
                       arq.QE_I17.isnull() & arq.QE_I18.isnull() & arq.QE_I19.isnull() & arq.QE_I20.isnull() & 
                       arq.QE_I21.isnull() & arq.QE_I22.isnull() & arq.QE_I23.isnull() & arq.QE_I24.isnull() & 
                       arq.QE_I25.isnull() & arq.QE_I26.isnull()].index)
    print("Deletando todos os que são nulos em todas as questões na base: ", arq['NU_ANO'][0])
    verifica_perdas(inicial, arq.shape[0])
    return arq

def deleta_questionsunsupported(arq):
    inicial = arq.shape[0]
    columns = arq.columns.tolist()
    for c in columns:
        arq = arq.drop(arq[arq[c] == ' '].index)
    verifica_perdas(inicial, arq.shape[0])
    return arq

def listwise(arq):
    inicial = arq.shape[0]
    arq = arq.dropna()
    print("Deletando todos os que são nulos soltos: ", arq['NU_ANO'][0])
    verifica_perdas(inicial, arq.shape[0])
    return arq
                     

def transforma_variaveis(arq):
                     
    arq.loc[arq['CO_TURNO_GRADUACAO']== 3,'CO_TURNO_GRADUACAO']=0
    arq['CO_TURNO_GRADUACAO'].fillna("0.5", inplace=True)
    arq.loc[arq['CO_TURNO_GRADUACAO']!= 0,'CO_TURNO_GRADUACAO']=1
    
    arq.loc[arq['TP_SEXO']=='F', 'TP_SEXO']= 1
    arq.loc[arq['TP_SEXO']=='M', 'TP_SEXO']= 0

    arq.loc[(arq['QE_I01']== 'C') | (arq['QE_I01']== 'D'),'QE_I01'] = 'E'

    arq.loc[(arq['QE_I02']== 'C') | (arq['QE_I02']== 'E'),'QE_I02'] ='F'

    arq.loc[(arq['QE_I06']== 'A') | (arq['QE_I06']== 'D') | (arq['QE_I06']== 'E'),'QE_I06'] ='F'

    arq.loc[arq['QE_I07']== 'A','QE_I07']=0
    arq.loc[arq['QE_I07']== 'B','QE_I07']=1
    arq.loc[arq['QE_I07']== 'C','QE_I07']=2
    arq.loc[arq['QE_I07']== 'D','QE_I07']=3
    arq.loc[arq['QE_I07']== 'E','QE_I07']=4
    arq.loc[arq['QE_I07']== 'F','QE_I07']=5
    arq.loc[arq['QE_I07']== 'G','QE_I07']=6
    arq.loc[arq['QE_I07']== 'H','QE_I07']=7
    
    arq = arq.drop(arq[(arq['QE_I07'] != 0) & (arq['QE_I07'] != 1) & 
                       (arq['QE_I07'] != 2) & (arq['QE_I07'] != 3) &
                       (arq['QE_I07'] != 4) & (arq['QE_I07'] != 5) &
                       (arq['QE_I07'] != 6) & (arq['QE_I07'] != 7)].index)

    arq.loc[(arq['QE_I08']== 'A') | (arq['QE_I08']== 'B'),'QE_I08']='Baixa'
    arq.loc[(arq['QE_I08']== 'C') | (arq['QE_I08']== 'D') | (arq['QE_I08']== 'E'),'QE_I08']='Media'
    arq.loc[(arq['QE_I08']== 'F') | (arq['QE_I08']== 'G'),'QE_I08']='Alta'

    arq.loc[(arq['QE_I11']== 'G') | (arq['QE_I11']== 'H') | (arq['QE_I11']== 'I') |
       (arq['QE_I11']== 'J') | (arq['QE_I11']== 'K'),'QE_I11']='Outros'
    arq.loc[(arq['QE_I11']== 'D') | (arq['QE_I11']== 'F') | (arq['QE_I11']== 'C'),'QE_I11']='E'

    arq.loc[(arq['QE_I17']== 'A') | (arq['QE_I17']== 'D'),'QE_I17']=1
    arq.loc[arq['QE_I17']!=1,'QE_I17']=0
    
    arq.loc[(arq['QE_I19']== 'D') | (arq['QE_I19']== 'E') | (arq['QE_I19']== 'F') | (arq['QE_I19']== 'G'),'QE_I19']='C'

    arq.loc[(arq['QE_I20']== 'E') | (arq['QE_I20']== 'F') | (arq['QE_I20']== 'G') | 
        (arq['QE_I20']== 'H') | (arq['QE_I20']== 'I') | (arq['QE_I20']== 'J') | (arq['QE_I20']== 'D'),'QE_I20']='K'   

    arq.loc[arq['QE_I21']== 'A','QE_I21']=1
    arq.loc[arq['QE_I21']!=1,'QE_I21']=0          

    arq.loc[arq['QE_I22']== 'A','QE_I22']=1
    arq.loc[arq['QE_I22']!=1,'QE_I22']=0 

    arq.loc[arq['QE_I24']== 'E','QE_I24']=1
    arq.loc[arq['QE_I24']!=1,'QE_I24']=0  

    arq.loc[(arq['QE_I25']== 'B') | (arq['QE_I25']== 'F') | (arq['QE_I25']== 'D') | (arq['QE_I25']== 'G'),'QE_I25']='H' 

    arq.loc[(arq['QE_I26']== 'E') | (arq['QE_I26']== 'H') | (arq['QE_I26']== 'D') | (arq['QE_I26']== 'G'),'QE_I26']='I' 

    return arq
                                  
def print_valores(question, base):
    print(tabulate([[question,base[question].isna().sum(), 
                     base[question].value_counts(normalize=False),
                     base[question].value_counts(normalize=True)*100]],
                 headers=['Questão','Nulos','Valor Absoluto', 'Percentual'], 
                 tablefmt='orgtbl'))
    
def outliers_media(base, coluna):
    print(base['NU_ANO'][0],'-->', coluna)    
    print('ANTES')
    print('mínimo:',base[coluna].min() )
    print('máximo:',base[coluna].max() )
   
    resultado_group_by = base.groupby([coluna],as_index=False).size()
    resultado = base.merge(resultado_group_by, left_on=coluna, right_on=coluna)   
    mediana = resultado['size'].median()
    
    resultado2 = resultado.copy()
    resultado2 = resultado2.drop(resultado2[resultado2['size'] < (mediana*0.15)].index)
    media = round(resultado2[coluna].mean())
    
    resultado.loc[resultado['size'] < (mediana*0.15),coluna]= media
    resultado=resultado.drop(columns=['size']) 
    
    print('DEPOIS')
    print('média:', media)
    print('mínimo:',resultado[coluna].min() )
    print('máximo:',resultado[coluna].max() )
    return resultado

def outliers_drop(base, coluna):
    inicial = base.shape[0]
    print(base['NU_ANO'][0],'-->', coluna)    
    print('ANTES')
    print('mínimo:',base[coluna].min() )
    print('máximo:',base[coluna].max() )
   
    resultado_group_by = base.groupby([coluna],as_index=False).size()
    resultado = base.merge(resultado_group_by, left_on=coluna, right_on=coluna)   
    mediana = resultado['size'].median()
    resultado = resultado.drop(resultado[resultado['size'] < (mediana*0.15)].index)
    resultado=resultado.drop(columns=['size']) 
    print('DEPOIS')
    print('média:', round(resultado[coluna].mean()))
    print('mínimo:',resultado[coluna].min() )
    print('máximo:',resultado[coluna].max() )
    verifica_perdas(inicial, resultado.shape[0])
    return resultado

def normalizando(base,question):
    x = base[question].values 
    min_max_scaler = preprocessing.MinMaxScaler()
    x=x.reshape(-1,1)
    x_scaled = min_max_scaler.fit_transform(x)
    base[question] = pd.DataFrame(x_scaled)
    return base

def variance_threshold_selector(data,thresh):
    selector = VarianceThreshold(threshold=thresh)
    selector.fit_transform(data)
    return data[data.columns[selector.get_support(indices=True)]]
 


# In[23]:


arq19 = pd.read_csv(r'C:\Users\elzam\Downloads\TCC\2019.txt' ,sep=';',  encoding='latin-1', low_memory=False)
arq18 = pd.read_csv(r'C:\Users\elzam\Downloads\TCC\2018.txt' ,sep=';',  encoding='latin-1', low_memory=False)
arq17 = pd.read_csv(r'C:\Users\elzam\Downloads\TCC\2017.txt' ,sep=';',  encoding='latin-1', low_memory=False)

arq17=arq17.drop(columns=[ 'QE_I69', 'QE_I70', 'QE_I71', 'QE_I72', 'QE_I73', 'QE_I74', 'QE_I75', 'QE_I76', 'QE_I77', 'QE_I78',
     'QE_I79', 'QE_I80', 'QE_I81'])

exibir()


# In[24]:


arq17 = deleta_inicial(arq17)
arq18 = deleta_inicial(arq18)
arq19 = deleta_inicial(arq19)
exibir()
arq17 = delete_faltaram(arq17, 'TP_PRES')
arq18 = delete_faltaram(arq18, 'TP_PRES')
arq19 = delete_faltaram(arq19, 'TP_PRES')
exibir()
arq17 = delete_faltaram(arq17, 'TP_PR_GER')
arq18 = delete_faltaram(arq18, 'TP_PR_GER')
arq19 = delete_faltaram(arq19, 'TP_PR_GER')
exibir()

arq18['QE_I26'].fillna("I", inplace=True)
arq19['QE_I16'].fillna(99, inplace=True)

arq19['CO_TURNO_GRADUACAO'].fillna("0.5", inplace=True)

exibir()

arq17 = listwise(arq17) 
arq18 = listwise(arq18) 
arq19 = listwise(arq19) 
exibir()
arq19 = deleta_questionsunsupported(arq19)

arq17 = arq17.drop(columns = [ 'CO_MODALIDADE','QE_I03','QE_I12', 'QE_I13', 'QE_I14', 'QE_I15', 'QE_I18'])
arq18 = arq18.drop(columns = [ 'CO_MODALIDADE','QE_I03','QE_I12', 'QE_I13', 'QE_I14', 'QE_I15', 'QE_I18'])
arq19 = arq19.drop(columns = [ 'CO_MODALIDADE','QE_I03','QE_I12', 'QE_I13', 'QE_I14', 'QE_I15', 'QE_I18'])


arq17 = outliers_media(arq17, 'NU_IDADE')
arq17 = outliers_media(arq17, 'ANO_FIM_EM')
arq17 = outliers_media(arq17, 'ANO_IN_GRAD')

arq18 = outliers_media(arq18, 'NU_IDADE')
arq18 = outliers_media(arq18, 'ANO_FIM_EM')
arq18 = outliers_media(arq18, 'ANO_IN_GRAD')

arq19 = outliers_media(arq19, 'NU_IDADE')
arq19 = outliers_media(arq19, 'ANO_FIM_EM')
arq19 = outliers_media(arq19, 'ANO_IN_GRAD')


arq18.loc[(arq18['CO_CATEGAD']== 118) | (arq18['CO_CATEGAD']== 120) | (arq18['CO_CATEGAD']== 121) | 
    (arq18['CO_CATEGAD']== 10005) | (arq18['CO_CATEGAD']== 10006) | (arq18['CO_CATEGAD']== 10007) | 
    (arq18['CO_CATEGAD']== 10008) | (arq18['CO_CATEGAD']== 10009) | (arq18['CO_CATEGAD']== 17634), 'CO_CATEGAD'] = 0
arq18.loc[arq18['CO_CATEGAD']!=0, 'CO_CATEGAD']= 1

arq19.loc[(arq19['CO_CATEGAD']== 118) | (arq19['CO_CATEGAD']== 120) | (arq19['CO_CATEGAD']== 121) | 
    (arq19['CO_CATEGAD']== 10005) | (arq19['CO_CATEGAD']== 10006) | (arq19['CO_CATEGAD']== 10007) | 
    (arq19['CO_CATEGAD']== 10008) | (arq19['CO_CATEGAD']== 10009) | (arq19['CO_CATEGAD']== 17634), 'CO_CATEGAD'] = 0
arq19.loc[arq19['CO_CATEGAD']!=0, 'CO_CATEGAD']= 1

arq17.loc[(arq17['CO_CATEGAD']== 4) | (arq17['CO_CATEGAD']== 5) | (arq17['CO_CATEGAD']== 7), 'CO_CATEGAD']= 0
arq17.loc[arq17['CO_CATEGAD']!=0, 'CO_CATEGAD']= 1

arq17 = transforma_variaveis(arq17)
arq18 = transforma_variaveis(arq18)
arq19 = transforma_variaveis(arq19)

print(verifica_nulos(arq17))
print(verifica_nulos(arq18))
print(verifica_nulos(arq19))


# In[35]:


together["INST_CURSO"] = together["CO_IES"].astype(str) + '_' + together["CO_MUNIC_CURSO"].astype(str) + '_' + together["CO_UF_CURSO"].astype(str)

together=together.drop(columns=['CO_TURNO_GRADUACAO','NU_IDADE','NU_ANO','CO_IES','CO_CATEGAD','CO_MUNIC_CURSO', 'CO_UF_CURSO']) 
print("Base após deletar variáveis", together.columns)


together = normalizando(together, 'QE_I07')
together['DIFERENCA'] = together['ANO_IN_GRAD'] - together['ANO_FIM_EM']
#together.loc[(together['DIFERENCA']< 1) ,'DIFERENCA']=0
#together.loc[(together['DIFERENCA']== 2) ,'DIFERENCA']=2
#together.loc[(together['DIFERENCA']== 3) ,'DIFERENCA']=3
#together.loc[(together['DIFERENCA']== 4) ,'DIFERENCA']=4
#together.loc[(together['DIFERENCA']== 1) ,'DIFERENCA']=1
#together.loc[(together['DIFERENCA']> 4) ,'DIFERENCA']=5

together = normalizando(together, 'DIFERENCA')
together = together.drop(columns=['ANO_IN_GRAD', 'ANO_FIM_EM'])
cleanup_nums = {"QE_I08":     {"Baixa": 0, "Media": 0.5, "Alta": 1},
                "QE_I10":     {"A": 0, "B": 0.25,"C": 0.5,"D": 0.75,"E": 1}}
together = together.replace(cleanup_nums)

dumm = pd.get_dummies(together,columns=['QE_I01', 'QE_I02','QE_I04', 'QE_I05', 'QE_I06','QE_I09', 'QE_I11',
                                       'QE_I16','QE_I19', 'QE_I20', 'QE_I23','QE_I25', 'QE_I26'])

together['DIFERENCA'] = 0
print(dumm.columns)
print(dumm)

