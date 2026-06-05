# -*- coding: utf-8 -*-
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re

rede_neural = []
taxa_aprendizagem = 0.1
pixels = 63

def carrega_dataset(caminho_x, caminho_y):
    #---------------------------[Processamento do Arquivo X]---------------------------
    #caminho_x contém os dados de entrada numéricos do X.txt fornecido na pasta "CARACTERES COMPLETO"
    with open(caminho_x, 'r', encoding='utf-8') as f:
        texto_completo = f.read()
    
    #extrai todos os números no X.txt usando regex, convertendo-os para float e os armazenando em uma lista
    valores_numerais = [float(f) for f in re.findall(r'-?\d+', texto_completo)]
    
    #total de elementos lidos do arquivo X.txt
    total_elementos = len(valores_numerais)
    
    #verifica se o total de elementos é multiplo de 63 (número de pixels), caso contrário, remove os elementos excedentes
    sobra = total_elementos % pixels
    if sobra !      = 0:
        valores_numerais = valores_numerais[:total_elementos - sobra]
    
    #reorganiza os valores em uma matriz onde cada linha é um caracter e cada coluna é um pixel, depois converte essa matriz para uma lista de listas que serão utilizadas como entrada para a rede neural e outras funções logo abaixo no código
    X = np.array(valores_numerais).reshape(-1, pixels).tolist()
    
    #---------------------------[Processamento do Arquivo Y]---------------------------
    #carrega o arquivo Y_letra.txt usando pandas para uma coluna sem cabeçalho e nomeada'Letra'
    df_y = pd.read_csv(caminho_y, header=None, names=['Letra'])
    df_y['Letra'] = df_y['Letra'].str.strip().str.upper()
    
    #mapeia todas as letras presentes no arquivo Y_letra.txt
    categorias = sorted(df_y['Letra'].dropna().unique())
    print(f"DEBUG: Letras detectadas no arquivo Y_letra.txt: {categorias}")
    
    #define o dataframe da coluna 'Letra' como categórico, garantindo que todas as letras sejam reconhecidas como categorias diferentes
    df_y['Letra'] = pd.Categorical(df_y['Letra'], categories=categorias)
    
    #gera uma matriz One-Hot (0.0 e 1.0) para todas as categorias detectadas, ou seja, as letras
    Y = pd.get_dummies(df_y['Letra'], dtype=float).values.tolist()
    
    tamanho_minimo = min(len(X), len(Y))
    
    print(f"DEBUG: Dados alinhados com sucesso!")
    print(f"DEBUG: Total de amostras prontas para o treino: {tamanho_minimo}")
    print("-" * 50)
    
    return X[:tamanho_minimo], Y[:tamanho_minimo], categorias


#Função de Ativação Sigmoid
#Épsilon da Máquina no Python é 1,11e-16, ou seja, o menor número positivo que pode ser representado. Para evitar overflow em exp(-x) quando x é muito grande, limitamos x entre -36 e 36, pois exp(-36) é aproximadamente 2.3e-16, próximo do limite de precisão do Python. É possível chegar nesse valor igualando exp(-x) a 1,11e-16 e resolvendo a igualdade para x, o que dá aproximadamente 36.

def sigmoid(x):
    x = max(-36, min(36, x))
    return 1 / (1 + math.exp(-x))

#Derivada da Função Sigmoid é calculada usando a propriedade de que a derivada de sigmoid(x) pode ser expressa em termos do próprio sigmoid(x), ou seja, sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x)). Isso é útil porque durante o backpropagation, já temos o valor de sigmoid(x) que é o Y calculado, então podemos usar esse valor para calcular a derivada de forma eficiente sem precisar recalcular a função sigmoid para somente depois fazer a derivada.
def derivadaSigmoid(Y):
    return Y * (1.0 - Y)
    
def criaCamada(entradas, neuronios):
    pesos_rede_neural = []
    bias_rede_neural = []
    
    for n in range(neuronios):
        #cria uma lista de pesos para cada neurônio de entrada, onde cada peso é um número aleatório entre -0.1 e 0.1, e adiciona essa lista de pesos à lista geral de pesos da rede neural. Além disso, para cada neurônio, também é criado um bias aleatório entre -0.1 e 0.1, que é adicionado à lista de bias da rede neural.
        pesos_rede_neural.append([random.uniform(-0.1, 0.1) for _ in range(entradas)])
        bias_rede_neural.append(random.uniform(-0.1, 0.1))
    
    return {
        "pesos": pesos_rede_neural,
        "bias": bias_rede_neural
    }

#Calcula a soma ponderada das entradas multiplicadas pelos pesos correspondentes para um neurônio específico, e adiciona o bias desse neurônio à soma.
def calculaSomatorioNeuronio(entradas, pesos_neuronio, bias_neuronio):
    soma_ponderada = 0
    for i in range(len(entradas)):
        soma_ponderada += entradas[i] * pesos_neuronio[i]
    soma_ponderada += bias_neuronio
    
    return soma_ponderada


def backpropagation(X, Y, rede_neural, taxa_aprendizagem, epocas, mapeamento_letras):
    camada_oculta = rede_neural[0]
    pesos_hidden = camada_oculta["pesos"]
    bias_hidden  = camada_oculta["bias"]
    
    camada_saida  = rede_neural[1]
    pesos_saida  = camada_saida["pesos"]
    bias_saida   = camada_saida["bias"]

    neuron_hidden = len(pesos_hidden)
    neuron_saida = len(pesos_saida)

    # Lista para salvar o histórico de erros (Requisito de entrega!)
    historico_erros = []

    for epoca in range(epocas):
        erro_total = 0

        for i in range(len(X)):

            #------------------------- Feedforward: Camada Oculta -------------------------
            funcao_ativacao_hidden = []
            for j in range(neuron_hidden):
                input_j = calculaSomatorioNeuronio(X[i], pesos_hidden[j], bias_hidden[j])
                funcao_ativacao_hidden.append(sigmoid(input_j))

            #------------------------- Feedforward: Camada de Saída -------------------------
            y_previsto = []
            for o in range(neuron_saida):
                input_saida = calculaSomatorioNeuronio(funcao_ativacao_hidden, pesos_saida[o], bias_saida[o])
                y_previsto.append(sigmoid(input_saida))

            #------------------------- Cálculo do Erro Total -------------------------
            erro_total += 0.5 * sum((Y[i][o] - y_previsto[o]) ** 2 for o in range(neuron_saida))

            #------------------------- Delta da Camada de Saída -------------------------
            delta_saida = []
            for o in range(neuron_saida):
                d_out = (Y[i][o] - y_previsto[o]) * derivadaSigmoid(y_previsto[o])
                delta_saida.append(d_out)

            #------------------------- Delta da Camada Oculta -------------------------
            delta_hidden = []
            for j in range(neuron_hidden):
                soma_erro = sum(delta_saida[o] * pesos_saida[o][j] for o in range(neuron_saida))
                d_hid = derivadaSigmoid(funcao_ativacao_hidden[j]) * soma_erro
                delta_hidden.append(d_hid)

            #------------------------- Atualização: Camada de Saída -------------------------
            for o in range(neuron_saida):
                for j in range(neuron_hidden):
                    pesos_saida[o][j] += taxa_aprendizagem * delta_saida[o] * funcao_ativacao_hidden[j]
                bias_saida[o] += taxa_aprendizagem * delta_saida[o]

            #------------------------- Atualização: Camada Oculta -------------------------
            for j in range(neuron_hidden):
                for k in range(len(X[i])):
                    pesos_hidden[j][k] += taxa_aprendizagem * delta_hidden[j] * X[i][k]
                bias_hidden[j] += taxa_aprendizagem * delta_hidden[j]

        historico_erros.append(erro_total)
        if (epoca + 1) % 100 == 0 or epoca == 0:
            print(f"Época {epoca+1}/{epocas} - Erro Total: {erro_total:.6f}")

    # Salva o arquivo contendo os erros por iteração conforme critério de entrega
    np.savetxt("erros_treinamento.txt", historico_erros, fmt="%.6f")

    # ------------------------- Resultados Finais Dinâmicos -------------------------
    print("\nResultados após o treinamento:")
    print('-' * 75)
    print('Amostra | Letra Esperada | Letra Predita | Confiança')
    print('-' * 75)
    for idx in range(len(X)):
        f_hid = [sigmoid(calculaSomatorioNeuronio(X[idx], pesos_hidden[j], bias_hidden[j])) for j in range(neuron_hidden)]
        y_prev = [sigmoid(calculaSomatorioNeuronio(f_hid, pesos_saida[o], bias_saida[o])) for o in range(neuron_saida)]
        
        idx_esperado = Y[idx].index(max(Y[idx]))
        idx_previsto = y_prev.index(max(y_prev))
        
        letra_esperada = mapeamento_letras[idx_esperado]
        letra_prevista = mapeamento_letras[idx_previsto]
        
        print(f"Letra {idx+1:02d} | Em classe: {letra_esperada}      | Predita: {letra_prevista}      | Confiança: {max(y_prev):.4f}")
    print("-" * 75 + "\n")

def main():
    diretorio_do_script = os.path.dirname(os.path.abspath(__file__))
    
    caminho_x = os.path.join(diretorio_do_script, 'files-sarajane', 'CARACTERES COMPLETO', 'X.txt')
    caminho_y = os.path.join(diretorio_do_script, 'files-sarajane', 'CARACTERES COMPLETO', 'Y_letra.txt')
    
    # X_dados, Y_dados e a lista mapeada de strings das letras detectadas
    X_dados, Y_dados, mapeamento_letras = carrega_dataset(caminho_x, caminho_y)
    
    quantidade_entradas = len(X_dados[0])
    quantidade_saidas = len(Y_dados[0])
    
    print(f"--> Entradas extraídas (Atributos): {quantidade_entradas}")
    print(f"--> Saídas extraídas (Classes mapeadas): {quantidade_saidas}\n")

    # Ajustado para 30 neurônios ocultos para dar conta de uma complexidade maior (26 classes)
    mapeamento = [quantidade_entradas, 30, quantidade_saidas]
    
    global rede_neural
    rede_neural = []
    for m in range(len(mapeamento) - 1):
        entrada = mapeamento[m]
        saida = mapeamento[m+1]
        nova_camada = criaCamada(entrada, saida)
        rede_neural.append(nova_camada)
        
    # Executa o backpropagation passando o mapeamento das letras
    backpropagation(X_dados, Y_dados, rede_neural, taxa_aprendizagem, epocas=1000, mapeamento_letras=mapeamento_letras)

if __name__ == "__main__":
    main()