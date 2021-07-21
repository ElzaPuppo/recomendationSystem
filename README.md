# Trabalho de Conclusão de Curso (TCC) 
:books: Apresentado na [Universidade Tecnológica Federal do Paraná](https://www.ifg.edu.br/luziania), campus Pato Branco como requisito parcial à conclusão do curso de **Engenharia de Computação** :mortar_board:

# SISTEMA DE RECOMENDAÇÃO DE INSTITUIÇÃO DE ENSINO SUPERIOR PARA ATENUAR EVASÃO UTILIZANDO ANÁLISE DE PERFIS DE EGRESSOS
 

# Resumo
A evasão no ensino superior é um desperdício social, acadêmico e econômico. Estudos têm sido realizados de maneira a identificar os fatores que levam à evasão e, da mesma forma, como combater tais fatores. Este trabalho propõe a utilização de métodos para Extração de Conhecimento e Sistema de Recomendação com a finalidade de sugerir instituições de ensino a futuros ingressantes. O primeiro processo realizado consistiu em compreender a base do ENADE, limpar, pré-processar, transformar os dados e realizar tarefas de Mineração de Dados para obter perfis dos egressos. As tarefas de Mineração de Dados utilizadas abrangeram técnicas dos métodos de agrupamento e classificação para definir os perfis. O segundo processo executado foi o desenvolvimento de um Sistema de Recomendação, por meio de uma aplicação Web para a coleta dos dados do usuário a fim de gerar um perfil desse. Nesse sistema, o perfil do usuário é comparado com os perfis extraídos no primeiro processo a fim de identificar a similaridade entre eles. Posteriormente, o sistema recomenda as instituições de ensino que possuem um perfil mais similar ao perfil do usuário. 

 # Extração de Conhecimento:
## Tecnologias utilizadas:
- Python
- Jupyter  
:pick: :game_die: 
Com o objetivo de extrair os perfis predominantes de cada curso de cada campus optou-se por definir os atributos CO_IES, CO_MUNIC_CURSO e CO_UF_CURSO juntos como rótulo dos procedimentos.  
As etapas de desenvolvimento do projeto podem ser definidas como pré-mineração, mineração e pós-mineração.  
Na etapa de pré-mineração realizou-se a limpeza dos dados, seleção das variáveis e a transformação dos dados.  
A mineração, efetivamente, foi realizada utilizando algoritmos DBSCAN, K-means e Classificação por Similaridade.  
Optou-se por utilizar os resultados do algoritmo K-means para aplicar no Sistema de Recomendação, utilizando os centróides dos grupos gerados e os perfis de cada grupo.  
Os arquivos [Pré-Mineração.py](https://github.com/ElzaPuppo/recomendationSystem/blob/main/Pr%C3%A9-Minera%C3%A7%C3%A3o.py) e [Extração de conhecimento.py](https://github.com/ElzaPuppo/recomendationSystem/blob/main/Extra%C3%A7%C3%A3o%20de%20conhecimento.py) apresentam os códigos em Python utilizados para as etapas de Extração de Dados.

# Sistema de Recomendação:
O Sistema de Recomendação é uma aplicação web que recomenda as intituições de ensino cujos egressos apresentam maior proximidade de perfil em relação ao usuário.

## Tecnologias utilizadas:
- JavaScript
- HTML
- CSS

## Funcionamento:
:earth_americas: O Sistema de Recomendação utiliza o resultado das etapas de Extração de Conhecimento, que são os arquivos .csv com os centróides e perfis de cada cluster.  
O sistema é composto por uma tela para inserção das informações do usuário. As informações são adaptadas e transformadas em um vetor.   
Calcula-se a distância desse vetor gerado com os vetores dos centróides dos grupos, localizados na pasta centroids, selecionando o grupo que apresenta mais proximidade.  
Seleciona-se, então, dentre os perfis pertencentes a este grupo, os 3 mais próximos ao vetor do usuário.  
O Sistema de Recomendação realiza a recomendação de 3 instituições de ensino.  

O template CSS utilizado foi retirado da plataforma [HTML5 UP](https://html5up.net/), dado que tal tarefa não era o foco do trabalho.  
No [repositório RECOMENDACAO](https://github.com/ElzaPuppo/recomendationSystem/tree/main/CRECOMENDACAO) estão disponíveis os arquivos completos do Sistema de Recomendação, incluindo as pastas centroids e perfis com os arquivos .csv resultantes da etapa de extração de conhecimento.

# Monografia

:memo: A monografia apresentada para este projeto encontra-se em [TCC](https://github.com/ElzaPuppo/recomendationSystem/blob/main/TCC_ElzaPuppo.pdf)
