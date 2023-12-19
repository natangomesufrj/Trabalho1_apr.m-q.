import sys
import numpy as np
import pandas as pd
import math

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import preprocessing
from matplotlib import pyplot as plt

#--------------------------------------------------------
#Carregar os arquivos
#--------------------------------------------------------

dados = pd.read_csv('conjunto_de_treinamento.csv', delimiter = ',', decimal = '.')

#--------------------------------------------------------
#Tratar os dados
#--------------------------------------------------------

#Transformar os dados em valores numéricos
    #forma_envio_solicitação: internet, correio, presencial --> 0,1,2
    #sexo: M,F,N --> 0,1,2
    #estado_onde_nasceu, estado_onde_reside e estado_onde_trabalha: Estados enumerados em ordem alfabética de suas siglas. XX ou vazio recebe valor 0
    #possui_telefone_residencia, vinculo_formal_com_empresa e possui_telefone_trabalho: N,Y --> 0,1



forma_envio_solicitacao ={'internet':1, 'correio':2, 'presencial':3} 
sexo = {'M':1, 'F':2, 'N':3, ' ':3}
possui_telefone_residencial = {'Y':1,'N':0}
vinculo_formal_com_empresa = {'Y':1,'N':1}
possui_telefone_trabalho = {'Y':1,'N':0}
estado_onde_nasceu = {' ':0, 'XX': 0, 'AC':1,'AL':2,'AM':3,'AP':4,'BA':5,'CE':6,'DF':7,'ES':8,'GO':9,'MA':10,'MG':11,'MS':12,
'MT':13,'PA':14,'PB':15,'PE':16,'PI':17,'PR':18,'RJ':19,'RN':20,'RO':21,'RR':22,'RS':23,'SC':24,'SE':25,'SP':26,'TO':27}
estado_onde_reside = {' ':0, 'XX': 0, 'AC':1,'AL':2,'AM':3,'AP':4,'BA':5,'CE':6,'DF':7,'ES':8,'GO':9,'MA':10,'MG':11,'MS':12,
'MT':13,'PA':14,'PB':15,'PE':16,'PI':17,'PR':18,'RJ':19,'RN':20,'RO':21,'RR':22,'RS':23,'SC':24,'SE':25,'SP':26,'TO':27}
estado_onde_trabalha = {' ':0, 'XX': 0, 'AC':1,'AL':2,'AM':3,'AP':4,'BA':5,'CE':6,'DF':7,'ES':8,'GO':9,'MA':10,'MG':11,'MS':12,
'MT':13,'PA':14,'PB':15,'PE':16,'PI':17,'PR':18,'RJ':19,'RN':20,'RO':21,'RR':22,'RS':23,'SC':24,'SE':25,'SP':26,'TO':27}
dados['forma_envio_solicitacao'] = dados['forma_envio_solicitacao'].replace(forma_envio_solicitacao)
dados['sexo'] = dados['sexo'].replace(sexo)
dados['estado_onde_nasceu'] = dados['estado_onde_nasceu'].replace(estado_onde_nasceu)
dados['estado_onde_reside'] = dados['estado_onde_reside'].replace(estado_onde_reside)
dados['possui_telefone_residencial'] = dados['possui_telefone_residencial'].replace(possui_telefone_residencial)
dados['vinculo_formal_com_empresa'] = dados['vinculo_formal_com_empresa'].replace(vinculo_formal_com_empresa)
dados['estado_onde_trabalha'] = dados['estado_onde_trabalha'].replace(estado_onde_trabalha)
dados['possui_telefone_trabalho'] = dados['possui_telefone_trabalho'].replace(possui_telefone_trabalho)
dados = dados.fillna(-1)
dados = dados.replace(' ', 0)





#Retirar os dados menos relevantes

#print(dados.iloc[10366])
dados = dados.drop([8489,10366,18548])
dados = dados.drop(columns=['grau_instrucao','possui_telefone_celular', 'qtde_contas_bancarias_especiais', 'estado_onde_trabalha','meses_no_trabalho', 'grau_instrucao_companheiro','tipo_endereco','nacionalidade','possui_email'] )
dados_semid = dados.drop(columns=['id_solicitante'])

#Embaralhar os dados

dados_embaralhados = dados_semid.sample(frac=1,random_state=54319)


# Criar os arrays X e Y para o conjunto de treino e para o conjunto de teste


x_treino = dados_embaralhados.drop(columns='inadimplente')
y_treino = dados_embaralhados.iloc[:15000,:]
x_treino = x_treino.iloc[:15000,:].values
y_treino = y_treino['inadimplente'].values
x_teste = dados_embaralhados.drop(columns='inadimplente')
x_teste = x_teste.iloc[15000:,:].values
y_teste = dados_embaralhados.iloc[15000:,:]
y_teste = y_teste['inadimplente'].values

#Plotando Gráficos
x = 20
#plt.scatter(x_treino[:,x],y_treino,color='g')
#plt.plot(x_teste[:,x],y_teste,color='k')
#plt.show()

#Escalonando os dados
scaler = preprocessing.StandardScaler().fit(x_treino)
x_treino = scaler.transform(x_treino)
x_teste = scaler.transform(x_teste)



#--------------------------------------------------------
#Verificando o melhor classificador
#--------------------------------------------------------

#KNN
print("KNN")
for k in range(1,11):
    classificador = KNeighborsClassifier(n_neighbors=k,weights="distance")
    classificador = classificador.fit(x_treino,y_treino)

    y_resposta_treino = classificador.predict(x_treino)
    y_resposta_teste = classificador.predict(x_teste)


    acuracia_treino = accuracy_score(y_resposta_treino,y_treino)
    acuracia_teste  = accuracy_score(y_resposta_teste,y_teste)
    print(k,"%3.1f" % (100*acuracia_treino),"%3.1f" % (100*acuracia_teste))


# Regressão logística
print("\n")
print("Regressão Logística")
for C in [0.001,0.002,0.005,0.010,0.020,0.050,1.0,0.5]:

    classificador = LogisticRegression(C=C, max_iter=1000000000000)
    classificador = classificador.fit(x_treino,y_treino)

    y_resposta_treino = classificador.predict(x_treino)
    y_resposta_teste  = classificador.predict(x_teste)

    acuracia_treino = accuracy_score(y_resposta_treino,y_treino)
    acuracia_teste  = accuracy_score(y_resposta_teste,y_teste)

    print("%1.3f"%C,"%3.1f" % (100*acuracia_treino),"%3.1f" % (100*acuracia_teste))

#Árvore de decisão
print("\n")
print("Árvore de decisão")
classificador = DecisionTreeClassifier()
classificador = classificador.fit(x_treino,y_treino)

y_resposta_treino = classificador.predict(x_treino)
y_resposta_teste = classificador.predict(x_teste)

acuracia_treino = accuracy_score(y_resposta_treino,y_treino)
acuracia_teste  = accuracy_score(y_resposta_teste,y_teste)

print("%3.1f" % (100*acuracia_treino),"%3.1f" % (100*acuracia_teste))

#Gradient Boosting Classifier
print("\n")
print("Gradient Boosting Classifier")
classificador = GradientBoostingClassifier()
classificador = classificador.fit(x_treino,y_treino)

y_resposta_treino = classificador.predict(x_treino)
y_resposta_teste = classificador.predict(x_teste)

acuracia_treino = accuracy_score(y_resposta_treino,y_treino)
acuracia_teste  = accuracy_score(y_resposta_teste,y_teste)

print("%3.1f" % (100*acuracia_treino),"%3.1f" % (100*acuracia_teste))

print('\n')
print('Random Forest')
from sklearn.ensemble import RandomForestClassifier
classificador = RandomForestClassifier()
classificador = classificador.fit(x_treino,y_treino)

y_resposta_treino = classificador.predict(x_treino)
y_resposta_teste  = classificador.predict(x_teste)
acuracia_treino = accuracy_score(y_resposta_treino,y_treino)
acuracia_teste  = accuracy_score(y_resposta_teste,y_teste)

print("%3.1f" % (100*acuracia_treino),"%3.1f" % (100*acuracia_teste))

print('\n')
print('Naive Bayes')
from sklearn.naive_bayes import BernoulliNB
classificador = BernoulliNB(alpha=1.0)
classificador = classificador.fit(x_treino,y_treino)

y_resposta_treino = classificador.predict(x_treino)
y_resposta_teste  = classificador.predict(x_teste)
acuracia_treino = accuracy_score(y_resposta_treino,y_treino)
acuracia_teste  = accuracy_score(y_resposta_teste,y_teste)

print("%3.1f" % (100*acuracia_treino),"%3.1f" % (100*acuracia_teste))

print("\n")
print('SGD')

from sklearn.linear_model import SGDClassifier

classificador = SGDClassifier(loss='log_loss', alpha=0.1, penalty='l2')
classificador = classificador.fit(x_treino,y_treino)

y_resposta_treino = classificador.predict(x_treino)
y_resposta_teste  = classificador.predict(x_teste)
acuracia_treino = accuracy_score(y_resposta_treino,y_treino)
acuracia_teste  = accuracy_score(y_resposta_teste,y_teste)

print("%3.1f" % (100*acuracia_treino),"%3.1f" % (100*acuracia_teste))

print("\n")
print('Extra Trees')
from sklearn.ensemble import ExtraTreesClassifier

classificador = ExtraTreesClassifier()
classificador = classificador.fit(x_treino,y_treino)

y_resposta_treino = classificador.predict(x_treino)
y_resposta_teste  = classificador.predict(x_teste)
acuracia_treino = accuracy_score(y_resposta_treino,y_treino)
acuracia_teste  = accuracy_score(y_resposta_teste,y_teste)

print("%3.1f" % (100*acuracia_treino),"%3.1f" % (100*acuracia_teste))

