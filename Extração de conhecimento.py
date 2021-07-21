#!/usr/bin/env python
# coding: utf-8

# # Funções que geram os grupos utilizando K-means

# In[ ]:


def classificador(df_total):
    metrica = 'euclidean'
    lista_perfis , grupos = pd.DataFrame(columns = df_total.columns), df_total['CO_GRUPO'].unique()
    i = 0
    resultado = pd.DataFrame() 
    for grupo in grupos:
        taxa = pd.DataFrame()
        lista_perfis = df_total[df_total['CO_GRUPO'].isin([grupo])]
        lista_perfis = lista_perfis.drop(columns = ['CO_GRUPO'], axis = 1)
        n = len(lista_perfis['INST_CURSO'].unique())
        gerador(lista_perfis,n, grupo)

def gerador(df_treino,n,grupo):
    df_treino_full = df_treino.copy()
    kmeans_kwargs = {
       "init": "random",
       "n_init": 20,
       "max_iter": 300,
       "random_state": 42,
       }
    kmeans = KMeans(n_clusters=n, **kmeans_kwargs)
    kmeans.fit(df_treino)
    df_centroides = pd.DataFrame(kmeans.cluster_centers_, columns = df_treino.columns)
    df_treino['labels'] = kmeans.labels_
    df_centroides['labels'] = range(0,n) 
    
    name = str(grupo)
    df_centroides.to_csv(r'C:\Users\elzam\Downloads\TCC\centroids\_' + name + '.csv',sep=";", index=False)
    df_treino.to_csv(r'C:\Users\elzam\Downloads\TCC\perfis\_' + name + '.csv',sep=";", index=False)
    


# # Sistema de Recomendação em Python:
# (Alternativa que pode ser usada como alternativa à aplicação web)

# In[ ]:


def IndicaK(df_treino,df_centroide,df_teste,metrica, resultado):
    i = 0
    df_centroide_adaptado = df_centroide.drop(labels=['labels', 'INST_CURSO'],axis=1)     
    row_adaptada =  df_teste 
    distancia = scipy.spatial.distance.cdist(df_centroide_adaptado,row_adaptada, metric=metrica)  
    df_centroide['Distancia'] = distancia
    centroides_min =df_centroide.nsmallest(3,'Distancia')
    for index2, row2 in centroides_min.iterrows():
        label = row2['labels']
        df_teste_split = df_treino[df_treino['labels'].isin([label])]
        df_teste_split_adaptado = df_teste_split.drop(labels=['labels', 'INST_CURSO'],axis=1)  
        distancia = scipy.spatial.distance.cdist(df_teste_split_adaptado,row_adaptada, metric='euclidean')
        df_teste_split['Distancia'] = distancia
        df_teste_split = df_teste_split.sort_values(by='Distancia', ascending=True)
        df_teste_split = df_teste_split.drop_duplicates(subset=['INST_CURSO'], keep='first') 
        alunos_minimo =df_teste_split.nsmallest(3,'Distancia')
        for jndex, s in alunos_minimo.iterrows():
                resultado.loc[i] = [s['INST_CURSO'],s['Distancia'], df_teste_split.shape[0], df_treino.shape[0]]
                i=i+1
    resultado =resultado.nsmallest(3,'Distancia') 
    return resultado

