# -*- coding: utf-8 -*-
import math
import random
import matplotlib.pyplot as plt
import numpy as np

#Precisamos polir mais o código, mas a estrutura geral já está funcionando.

X = [
    [0,0],
    [0,1],
    [1,0],
    [1,1]
]
Y = [0, 0, 0, 1]
taxa_aprendizagem = 0.1
rede_neural = []

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def derivadaSigmoid(Y):
    return math.exp(-Y) / ((1 + math.exp(-Y)) ** 2)
    
def criaCamada(entradas, neuronios):
    pesos_rede_neural = []
    bias_rede_neural = []
    
    for n in range(neuronios):
        pesos_rede_neural.append(
            [random.random() for _ in range(entradas)])
        bias_rede_neural.append(random.random())
    
    return {
        "pesos": pesos_rede_neural,
        "bias": bias_rede_neural
    }

def printRedeNeural(rede_neural):
    for l in range(len(rede_neural)):
        print(f"\nCamada {l}:")
        print("Pesos:", rede_neural[l]["pesos"])
        print("Bias:", rede_neural[l]["bias"])
        print("\n")
        
        for n in range(len(rede_neural[l]["pesos"])):
            print(f"Neurônio {n}:")
            print("Pesos:", rede_neural[l]["pesos"][n])
            print("Bias:", rede_neural[l]["bias"][n])       

# #neuron=none indica que o parametro e opcional
# def calculasomatorioneuronio(entradas, pesos, bias, neuronio=none):
    
#     # 1. modo camada oculta (múltiplos neurônios)
#     if isinstance(pesos[0], list):
#         # validações de segurança
#         if neuronio is none:
#             raise valueerror("para múltiplos neurônios, informe o índice 'neuronio'.")
#         if neuronio < 0 or neuronio >= len(pesos):
#             raise indexerror("índice de neurônio inválido.")
#         if len(entradas) != len(pesos[neuronio]):
#             raise valueerror("número de entradas incompatível com os pesos.") 

#         soma_ponderada = 0

#         # multiplica cada entrada pelo peso correspondente deste neurônio específico
#         for i in range(len(entradas)):
#             soma_ponderada += pesos[neuronio][i] * entradas[i]
        
#         # adiciona o bias no final
#         soma_ponderada += bias[neuronio]
        
#         return soma_ponderada

#     # 2. modo camada de saída (neurônio único)
#     else:
#         # validação de segurança
#         if len(entradas) != len(pesos):
#             raise valueerror("número de entradas incompatível com os pesos.")
        
#         soma_ponderada = 0

#         # multiplica cada entrada pelo peso correspondente (lista reta)
#         for i in range(len(entradas)):
#             soma_ponderada += entradas[i] * pesos[i]
        
#         # adiciona o bias da camada de saída no final
#         soma_ponderada += bias
        
#         return soma_ponderada

def calculaSomatorioNeuronio(entradas, pesos_neuronio, bias_neuronio):
    # pesos_neuronio = pesos de um neurônio específico (lista simples)
    # bias_neuronio = bias desse neurônio (valor escalar)
    
    soma_ponderada = 0
    for i in range(len(entradas)):
        soma_ponderada += entradas[i] * pesos_neuronio[i]
    soma_ponderada += bias_neuronio
    
    return soma_ponderada

def erroSaida(target, saida):
    return target - saida

def erroQuadraticoMedio(target, saida):
    return 1/2 * (erroSaida(target, saida)) ** 2

def delta(target, saida):
    return (erroSaida(target, saida)) * derivadaSigmoid(saida)

def backpropagation(X, Y, rede_neural, taxa_aprendizagem, epocas):
    camada_oculta = rede_neural[0]
    pesos_hidden = camada_oculta["pesos"]
    bias_hidden  = camada_oculta["bias"]
    
    camada_saida  = rede_neural[1]
    pesos_saida  = camada_saida["pesos"]
    bias_saida   = camada_saida["bias"]

    neuron = len(pesos_hidden)  # número de neurônios na camada oculta

    for epoca in range(epocas):
        erro_total = 0  # reseta o erro a cada época

        for i in range(len(X)):

            #------------------------- Feedforward: Camada Oculta -------------------------
            # input_j = somatório ponderado (antes da ativação) de cada neurônio oculto
            # z_j     = saída do neurônio oculto após aplicar a sigmoid
            saidas_hidden = []           # guarda os input_j (antes da ativação)
            funcao_ativacao_hidden = []  # guarda os z_j (depois da ativação)

            for j in range(neuron):
                input_j = calculaSomatorioNeuronio(X[i], pesos_hidden[j], bias_hidden[j])
                z_j     = sigmoid(input_j)
                saidas_hidden.append(input_j)
                funcao_ativacao_hidden.append(z_j)

            #------------------------- Feedforward: Camada de Saída -------------------------
            # input_saida = somatório ponderado da camada de saída (antes da ativação)
            # y_previsto  = saída final da rede após aplicar a sigmoid
            input_saida = calculaSomatorioNeuronio(funcao_ativacao_hidden, pesos_saida[0], bias_saida[0])
            y_previsto  = sigmoid(input_saida)

            #------------------------- Cálculo do Erro -------------------------
            # acumula o erro quadrático médio de cada amostra na época
            erro_total += erroQuadraticoMedio(Y[i], y_previsto)

            #------------------------- Delta da Camada de Saída -------------------------
            # delta_saida = (target - y_previsto) * derivada_sigmoid(input_saida)
            # representa o erro ponderado pela sensibilidade do neurônio de saída
            delta_saida = delta(Y[i], y_previsto)

            #------------------------- Delta da Camada Oculta -------------------------
            # delta_j = derivada_sigmoid(input_j) * peso_conexao_saida * delta_saida
            # propaga o erro da saída de volta para cada neurônio oculto j
            delta_hidden = []
            for j in range(neuron):
                delta_j = derivadaSigmoid(saidas_hidden[j]) * pesos_saida[0][j] * delta_saida
                delta_hidden.append(delta_j)

            #------------------------- Atualização: Camada de Saída -------------------------
            # regra delta: novo_peso = peso_atual + taxa * delta_saida * z_j
            # bias também é atualizado como se sua entrada fosse sempre 1
            for j in range(neuron):
                pesos_saida[0][j] += taxa_aprendizagem * delta_saida * funcao_ativacao_hidden[j]
            bias_saida[0] += taxa_aprendizagem * delta_saida

            #------------------------- Atualização: Camada Oculta -------------------------
            # regra delta: novo_peso = peso_atual + taxa * delta_j * x_k
            # bias também é atualizado como se sua entrada fosse sempre 1
            for j in range(neuron):
                for k in range(len(X[i])):
                    pesos_hidden[j][k] += taxa_aprendizagem * delta_hidden[j] * X[i][k]
                bias_hidden[j] += taxa_aprendizagem * delta_hidden[j]

        print(f"Época {epoca+1}/{epocas} - Erro Total: {erro_total:.6f}")

    #------------------------- Resultados Finais -------------------------
    print("\nResultados após o treinamento:")
    print('-' * 50)
    print('Entrada  | Saída Esperada | Saída Prevista')
    print('-' * 50)
    for idx in range(len(X)):
        input_saida_test = calculaSomatorioNeuronio(
            [sigmoid(calculaSomatorioNeuronio(X[idx], pesos_hidden[j], bias_hidden[j])) for j in range(neuron)],
            pesos_saida[0], bias_saida[0]
        )
        saida_final = sigmoid(input_saida_test)
        print(f"{X[idx]}  |      {Y[idx]}       |   {saida_final:.4f}")

def main():
    print("Rede Neural")
    print("XOR:")
    for i in range(len(X)):
        print(X[i], Y[i])
    print('Taxa de Aprendizagem: ', taxa_aprendizagem)
    print("\n")

    mapeamento = [2, 4, 1]
    for m in range(len(mapeamento) - 1):
        entrada = mapeamento[m]
        saida = mapeamento[m+1]
        
        nova_camada = criaCamada(entrada, saida)
        rede_neural.append(nova_camada)
        
    print(f"Camadas Criadas (ignorando entrada): {len(rede_neural)}")
    printRedeNeural(rede_neural)
    
    print('Camada Saida: ')
    print('Pesos Saida: ', rede_neural[-1]["pesos"])
    print('Bias Saida: ', rede_neural[-1]["bias"])
    print("-" * 30)

    backpropagation(X, Y, rede_neural, taxa_aprendizagem, epocas=1500)

if __name__ == "__main__":
    main()