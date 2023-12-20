import sys
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

#--------------------------------------------------------
#Carregar os arquivos
#--------------------------------------------------------

dados = pd.read_csv('conjunto_de_treinamento.csv', delimiter = ',', decimal = '.')
final = pd.read_csv('conjunto_de_teste.csv', delimiter = ',', decimal = '.')


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
vinculo_formal_com_empresa = {'Y':1,'N':0}
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
final['forma_envio_solicitacao'] = final['forma_envio_solicitacao'].replace(forma_envio_solicitacao)
final['sexo'] = final['sexo'].replace(sexo)
final['possui_telefone_residencial'] = final['possui_telefone_residencial'].replace(possui_telefone_residencial)
final['vinculo_formal_com_empresa'] = final['vinculo_formal_com_empresa'].replace(vinculo_formal_com_empresa)
final['possui_telefone_trabalho'] = final['possui_telefone_trabalho'].replace(possui_telefone_trabalho)
final['estado_onde_nasceu'] = final['estado_onde_nasceu'].replace(estado_onde_nasceu)
final['estado_onde_reside'] = final['estado_onde_reside'].replace(estado_onde_reside)
final['estado_onde_trabalha'] = final['estado_onde_trabalha'].replace(estado_onde_trabalha)
dados = dados.fillna(-1)
dados = dados.replace(' ', 0)
final = final.fillna(-1)
final = final.replace(' ', 0)




#Retirar os dados menos relevantes

dados = dados.drop([8489,10366,18548])
dados = dados.drop(columns=['grau_instrucao','possui_telefone_celular', 'qtde_contas_bancarias_especiais', 'estado_onde_trabalha','meses_no_trabalho', 'grau_instrucao_companheiro','tipo_endereco','nacionalidade','possui_email'] )
final = final.drop(columns=['grau_instrucao','possui_telefone_celular', 'qtde_contas_bancarias_especiais', 'estado_onde_trabalha','meses_no_trabalho', 'grau_instrucao_companheiro','tipo_endereco','nacionalidade','possui_email'] )
dados_semid = dados.drop(columns=['id_solicitante'])
final_semid = final.drop(columns=['id_solicitante'])





#Embaralhar os dados

dados_embaralhados = dados_semid.sample(frac=1,random_state=54319)


# Criar os arrays X e Y para o conjunto de treino e para o conjunto de teste

x_treino = dados_embaralhados.iloc[:,:-1].values
y_treino = dados_embaralhados.iloc[:,-1].values
x_final = final_semid.values

scaler = preprocessing.StandardScaler().fit(x_treino)
x_treino = scaler.transform(x_treino)
x_final = scaler.transform(x_final)


#--------------------------------------------------------
#O classificador
#--------------------------------------------------------

#Treinando
classificador = GradientBoostingClassifier()
classificador = classificador.fit(x_treino,y_treino)

#Resultado

y_final_respostas = classificador.predict(x_final)


#--------------------------------------------------------
#Escrevendo no arquivo final
#--------------------------------------------------------

id = []
for a in range(20001,25001):
    id.append(a)

id = np.array(id)

resultado = pd.DataFrame({'id_solicitante': id, 'inadimplente': y_final_respostas})

resultado.to_csv('resultado_1.csv', index=False)
