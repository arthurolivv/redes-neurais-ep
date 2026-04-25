import math
import random
import matplotlib.pyplot as plt
import numpy as np

#=========================
# 1. Entradas e Respostas
#=========================
print('-'*30)
X = [
    [0,0],
    [0,1],
    [1,0],
    [1,1]
]

y = [0, 0, 0, 1]

for i in range(4):
    print(X[i], y[i])
print("-" * 30)

#taxa_aprendizagem = random.uniform(0,1)
taxa_aprendizagem = 0.1

#=========================
# 2. Camada Oculta
#=========================
neuron = 3
pesos_hidden = []
bias_hidden = []
for i in range(neuron):
    pesos_hidden.append([random.random(), random.random()])
    bias_hidden.append(random.random())

#=========================
# X. Console
#=========================
print('Total de Neuronios: ', neuron)
print('tax_aprendizagem inicial: ', taxa_aprendizagem)
print('-'*30)

for i in range(neuron):
    print(f"Neurônio {i}:")
    print(f"  Pesos: {pesos_hidden[i]}")
    print(f"  Bias:  {bias_hidden[i]}")
    print("-" * 30)

#=========================
# 3. Camada de Saida
#=========================
pesos_saida = []
bias_saida = random.random()
for i in range(neuron):
    pesos_saida.append(random.random())

print('pesos saida: ', pesos_saida)
print('bias saida: ', bias_saida)
print("-" * 30)

#=========================
# 4. Funcoes Auxiliares
#=========================
#neuron=None indica que o parametro e opcional
def calcNeuron(entradas, pesos, bias, neuron=None):
    
    if isinstance(pesos[0], (list,tuple)):
        # Se um neurônio específico foi solicitado
            if neuron is None:
                raise ValueError("Para múltiplos neurônios, informe o índice 'neuron'.")
            if neuron < 0 or neuron >= len(pesos):
                raise IndexError("Índice de neurônio inválido.")
            if len(entradas) != len(pesos[neuron]):
                raise ValueError("Número de entradas incompatível com os pesos.") 

            y_in = 0

            for i in range(len(entradas)):
                y_in += pesos[neuron][i] * entradas[i]
        
            y_in += bias[neuron]
        
            return y_in

    else:
        if len(entradas) != len(pesos):
            raise ValueError("Número de entradas incompatível com os pesos.")
        
        y_in = 0

        for i in range(len(entradas)):
            y_in += entradas[i] * pesos[i]
        
        return y_in + bias

#calculo da funcao de ativacao (fazer prompt para escolher qual funcao usar?)
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

#=========================
# 5. Backpropagation
#=========================
for epoca in range (5): #condicao de parada qualquer
    for i in range(4): #para cada s:t
        erro_total = 0
        #------------------------- Entradas e Target -------------------------
        x1 = X[i][0]
        x2 = X[i][1]
        target = y[i]
        print(X[i])
        print('-'*80)
        
        #------------------------- Feedforward: Camada Oculta -------------------------
        saidas_hidden = []
        funcao_ativacao_hidden = []

        for j in range(neuron):
             n = calcNeuron(X[i], pesos_hidden, bias_hidden, neuron=j)
             s = sigmoid(n)
             saidas_hidden.append(n)
             funcao_ativacao_hidden.append(s)

        print('saidas hidden: ', saidas_hidden)
        print('funcao ativada ocultas: ', funcao_ativacao_hidden)

         #------------------------- Feedforward: Camada de Saida -------------------------
        saida = calcNeuron(funcao_ativacao_hidden, pesos_saida, bias_saida)
        neuronio_saida = sigmoid(saida)

        print('saida: ', saida)
        print('neuronio saida: ', neuronio_saida)

        #------------------------- Calculo Erro -------------------------
        erro_saida = target - neuronio_saida
        erro_total += 0.5 * erro_saida**2

        #------------------------- Backpropagation -------------------------
        #delta da saida
        delta_saida = erro_saida * neuronio_saida * (1 - neuronio_saida)
        
        #delta da camada oculta
        delta_hidden = []
        for j in range(neuron):
            h = funcao_ativacao_hidden[j]
            delta = h * (1 - h) * pesos_saida[j] * delta_saida
            delta_hidden.append(delta)

        #------------------------- Atualizacao de pesos -------------------------
        #camada de saida
        for j in range(neuron):
            pesos_saida[j] = pesos_saida[j] * taxa_aprendizagem * delta_saida * funcao_ativacao_hidden[j]
        bias_saida = bias_saida * taxa_aprendizagem * delta_saida

        #camada oculta
        for j in range(neuron):
            for k in range(2):
                pesos_hidden[j][k] = pesos_hidden[j][k] + taxa_aprendizagem * delta_hidden[j] * X[i][k]
            bias_hidden[j] = bias_hidden[j] + taxa_aprendizagem * delta_hidden[j]

        
        #------------------------- Imprimir no Console -------------------------

        print("\nResultados após o treinamento: ")
        print('-'*50)
        print('Entrada  | Saida Esperada   |  Saida Prevista')
        print('-'*50)
        for i in range(len(X)):
            print(f"{X[i]}  |   saida esperada  |   saida prevista")

        print("="*80)

        
