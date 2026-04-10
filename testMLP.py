import math
import random
import matplotlib.pyplot as plt
import numpy as np

print('-'*30)
#cenarios
X = [
    [0,0],
    [0,1],
    [1,0],
    [1,1]
]

#respostas
y = [0, 0, 0, 1]

for i in range(4):
    print(X[i], y[i])
print("-" * 30)

pesos_hidden = []
bias_hidden = []
#taxa_aprendizagem = random.uniform(0,1)
taxa_aprendizagem = 0.1
neuron = 3

print('Total de Neuronios: ', neuron)
print('-'*30)

for i in range(neuron):
    pesos_hidden.append([random.random(), random.random()])
    bias_hidden.append(random.random())

for i in range(neuron):
    print(f"Neurônio {i}:")
    print(f"  Pesos: {pesos_hidden[i]}")
    print(f"  Bias:  {bias_hidden[i]}")
    print("-" * 30)

pesos_saida = []

#calculo de y_in
def calcNeuron(pesos_hidden, bias_hidden, x, j):
    y_in = 0

    #entradas * seus pesos neuronio
    for i in range(len(x)):
        y_in += x[i] * pesos_hidden[j][i]

    #soma do bias respectivo neuronio
    y_in += bias_hidden[j]
    return y_in

#calculo da funcao de ativacao (fazer prompt para escolher qual funcao usar?)
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

print('tax_aprendizagem inicial: ', taxa_aprendizagem)
print("-" * 30)

for epoca in range (5): #condicao de parada qualquer
    print('epoca = ', epoca)
    print('*'*80)
    for i in range(4): #para cada s:t

        x1 = X[i][0]
        x2 = X[i][1]
        target = y[i]

        print(X[i])

        saidas_hidden = []
        funcao_ativacao_hidden = []

        for j in range(neuron):
             n = calcNeuron(pesos_hidden, bias_hidden, X[i], j)
             s = sigmoid(n)
             saidas_hidden.append(n)
             funcao_ativacao_hidden.append(s)

        print('saidas_hidden = ', saidas_hidden)
        print('sigmoid_hidden = ', funcao_ativacao_hidden)


        print("="*80)


"""         print('=' * 30)
        
        print(X[i], saida)

        print('w1 = ', w1)
        print('w2 = ', w2)
        print('bias = ', bias)

        sinal_erro = target - saida

        print('target: ', target)
        print('sinal de erro: ', target,'-', saida, '=', sinal_erro)
        #sinal erro
        valor_instantaneo_erro = 1/2 * sinal_erro * sinal_erro

        #define o gradiente
        gradiente = sinal_erro * saida *  (1 - saida)

        #definindo pesos New
        w1 = w1 + taxa_aprendizagem*gradiente*x1
        w2 = w2 + taxa_aprendizagem*gradiente*x2
        bias = bias + taxa_aprendizagem*gradiente

        print("-" * 30)
        print('new w1 = ', w1)
        print('new w2 = ', w2)
        print('new bias = ', bias) """
