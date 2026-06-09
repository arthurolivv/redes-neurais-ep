# -*- coding: utf-8 -*-

#-----------------------------------------------------------------------------------------------------------------
#Universidade de São Paulo - Escola de Artes, Ciências e Humanidades
#Disciplina: Inteligência Artificial
#Docente: Sarajane Marques Peres
#Projeto: Reconhecimento de Caracteres utilizando Redes Neurais Artificiais (MLP) - Multilayer Perceptron
#Arthur Jacintho de Oliveira Santos - 15635041
#Diogo Leonel dos Santos -15580980
#Tae Jin Chun - 15675241
#Ygor Araujo da Silva - 15506033
#-----------------------------------------------------------------------------------------------------------------

import math
import os
import random
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

rede_neural = []
taxa_aprendizagem = 0.1
#imagens png são 12x10, totalizando 120 pixels por caracter, ou seja, 120 atributos de entrada para a rede neural
pixels = 120 

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
    if sobra != 0:
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
    
    print(f"DEBUG: Arquivos {caminho_x} e {caminho_y} carregados com sucesso!")
    print("-" * 50)
    print(f"DEBUG: Total de amostras prontas para o treino: {tamanho_minimo}")

    return X[:tamanho_minimo], Y[:tamanho_minimo], categorias

def separar_dados_treino_teste(X, Y, proporcao_treino=0.8):
    #unificando X e Y para embaralhar mantendo a correspondencia entre as amostras e as letras que representam
    dados_combinados = list(zip(X, Y))
    random.shuffle(dados_combinados)
    
    tamanho_treino = int(len(dados_combinados) * proporcao_treino)
    
    treino = dados_combinados[:tamanho_treino]
    teste = dados_combinados[tamanho_treino:]
    
    #desempacota de volta para X e Y os dados
    X_treino, Y_treino = zip(*treino)
    X_teste, Y_teste = zip(*teste)
    
    return list(X_treino), list(Y_treino), list(X_teste), list(Y_teste)


def separar_validacao(X_treino, Y_treino, proporcao_validacao=0.1):
    
    dados_combinados = list(zip(X_treino, Y_treino))
    random.shuffle(dados_combinados)

    tamanho_val = int(len(dados_combinados) * proporcao_validacao)
    val = dados_combinados[:tamanho_val]
    treino_red = dados_combinados[tamanho_val:]

    X_val, Y_val = zip(*val)
    X_treino_red, Y_treino_red = zip(*treino_red)

    return list(X_treino_red), list(Y_treino_red), list(X_val), list(Y_val)

#Funções de Ativação e suas Derivadas para o Backpropagation
#Sigmoid
def sigmoid(Z_in):
    if Z_in < -700: return 0.0
    return 1.0 / (1.0 + math.exp(-Z_in))

def derivada_sigmoid(y_ativado):
    return y_ativado * (1.0 - y_ativado)

#ReLU
def relu(Z_in):
    return max(0.0, Z_in)

def derivada_relu(y_ativado):
    return 1.0 if y_ativado > 0.0 else 0.0

#SoftPlus
def softplus(Z_in):
    # A funcao do logaritmo natural ln em Python e chamada via math.log
    if Z_in > 700: return Z_in
    return math.log(1.0 + math.exp(Z_in))

def derivada_softplus(y_ativado):
    return 1.0 - math.exp(-y_ativado)

#Tanh
def tanh(Z_in):
    return math.tanh(Z_in)

def derivada_tanh(y_ativado):
    return 1.0 - (y_ativado ** 2)

#ELU
def elu(Z_in, alpha=1.0):
    return Z_in if Z_in > 0.0 else alpha * (math.exp(Z_in) - 1.0)

def derivada_elu(y_ativado, alpha=1.0):
    return 1.0 if y_ativado > 0.0 else y_ativado + alpha

#SoftSign
def softsign(Z_in):
    return Z_in / (1.0 + abs(Z_in))

def derivada_softsign(y_ativado):
    return (1.0 - abs(y_ativado)) ** 2

#Hard Tanh
def hard_tanh(Z_in):
    return max(-1.0, min(1.0, Z_in))

def derivada_hard_tanh(y_ativado):
    return 1.0 if -1.0 < y_ativado < 1.0 else 0.0

#SELU
def selu(Z_in):
    alpha = 1.67326
    scale = 1.0507
    return scale * Z_in if Z_in > 0.0 else scale * alpha * (math.exp(Z_in) - 1.0)

def derivada_selu(y_ativado):
    alpha = 1.67326
    scale = 1.0507
    return scale if y_ativado > 0.0 else y_ativado + (scale * alpha)

#Hard Shrink
def hard_shrink(Z_in, lambd=0.5):
    return Z_in if Z_in > lambd or Z_in < -lambd else 0.0

def derivada_hard_shrink(y_ativado):
    return 1.0 if y_ativado != 0.0 else 0.0

#Soft Shrink
def soft_shrink(Z_in, lambd=0.5):
    if Z_in > lambd: return Z_in - lambd
    if Z_in < -lambd: return Z_in + lambd
    return 0.0

def derivada_soft_shrink(y_ativado):
    return 1.0 if y_ativado != 0.0 else 0.0

#Hard Sigmoid
def hard_sigmoid(Z_in):
    return max(0.0, min(1.0, (Z_in + 3.0) / 6.0))

def derivada_hard_sigmoid(y_ativado):
    return (1.0 / 6.0) if 0.0 < y_ativado < 1.0 else 0.0

#Catalogo Universal de Funções de Ativação
#camada_saida identifica se a função de ativação é adequada para a camada de saída (True) ou apenas para camadas ocultas (False), garantindo que o usuário escolha uma função de ativação apropriada para a camada de saída, como Sigmoid ou Hard Sigmoid, que são adequadas para problemas de classificação, enquanto as outras funções são mais indicadas somente para as camadas ocultas, pois retornam valores somente no intervalo [0,1]
catalogo_ativacoes = {
    "1": {"nome": "Sigmoid", "funcao": sigmoid, "derivada": derivada_sigmoid, "camada_saida": True},
    "2": {"nome": "ReLU", "funcao": relu, "derivada": derivada_relu, "camada_saida": False},
    "3": {"nome": "SoftPlus", "funcao": softplus, "derivada": derivada_softplus, "camada_saida": False},
    "4": {"nome": "Tanh", "funcao": tanh, "derivada": derivada_tanh, "camada_saida": False},
    "5": {"nome": "ELU", "funcao": elu, "derivada": derivada_elu, "camada_saida": False},
    "6": {"nome": "SoftSign", "funcao": softsign, "derivada": derivada_softsign, "camada_saida": False},
    "7": {"nome": "Hard Tanh", "funcao": hard_tanh, "derivada": derivada_hard_tanh, "camada_saida": False},
    "8": {"nome": "SELU", "funcao": selu, "derivada": derivada_selu, "camada_saida": False},
    "9": {"nome": "Hard Shrink", "funcao": hard_shrink, "derivada": derivada_hard_shrink, "camada_saida": False},
    "10": {"nome": "Soft Shrink", "funcao": soft_shrink, "derivada": derivada_soft_shrink, "camada_saida": False},
    "11": {"nome": "Hard Sigmoid", "funcao": hard_sigmoid, "derivada": derivada_hard_sigmoid, "camada_saida": False}
}
      
#Função auxiliar que recebe o número de entradas e o número de neurônios para criar uma camada da rede neural, onde os pesos são inicializados aleatoriamente entre -0.1 e 0.1, e os bias também são inicializados aleatoriamente no mesmo intervalo. A função retorna um dicionário contendo as listas de pesos e bias para a camada criada.
def criaCamada(entradas, neuronios):
    pesos_rede_neural = []
    bias_rede_neural = []
    
    for n in range(neuronios):
        #cria uma lista de pesos para cada neurônio de entrada, onde cada peso é um número aleatório entre -0.1 e 0.1, e adiciona essa lista de pesos à lista geral de pesos da rede neural. Além disso, para cada neurônio, também é criado um bias aleatório entre -0.1 e 0.1, que é adicionado à lista de bias da rede neural.
        #pesos_rede_neural é uma lista de listas
        pesos_rede_neural.append([random.uniform(-0.1, 0.1) for _ in range(entradas)])
        bias_rede_neural.append(random.uniform(-0.1, 0.1))
    
    return {
        "pesos": pesos_rede_neural,
        "bias": bias_rede_neural
    }

def copiar_rede(rede):
    """Retorna uma cópia independente de todos os pesos e bias da rede."""
    copia = []
    for camada in rede:
        copia.append({
            "pesos": [list(neuronio) for neuronio in camada["pesos"]],
            "bias": list(camada["bias"])
        })
    return copia


def restaurar_rede(rede, copia):
    """Sobrescreve os pesos e bias da rede com os valores da cópia."""
    for idx_camada, camada in enumerate(rede):
        for idx_n, _ in enumerate(camada["pesos"]):
            camada["pesos"][idx_n] = list(copia[idx_camada]["pesos"][idx_n])
        camada["bias"] = list(copia[idx_camada]["bias"])


def calcular_erro_conjunto(X, Y, rede_neural, funcao_ativacao_oculta, funcao_ativacao_saida):
    """Calcula o erro quadrático total sobre um conjunto (sem backprop)."""
    pesos_hidden = rede_neural[0]["pesos"]
    bias_hidden  = rede_neural[0]["bias"]
    pesos_saida  = rede_neural[1]["pesos"]
    bias_saida   = rede_neural[1]["bias"]

    neuron_hidden = len(pesos_hidden)
    neuron_saida  = len(pesos_saida)
    erro_total = 0.0

    for i in range(len(X)):
        f_hid = [
            funcao_ativacao_oculta(calculaSomatorioNeuronio(X[i], pesos_hidden[j], bias_hidden[j]))
            for j in range(neuron_hidden)
        ]
        y_prev = [
            funcao_ativacao_saida(calculaSomatorioNeuronio(f_hid, pesos_saida[o], bias_saida[o]))
            for o in range(neuron_saida)
        ]
        erro_total += calculaSomaErrosQuadraticos(Y[i], y_prev)

    return erro_total


def calcular_acuracia_conjunto(X, Y, rede_neural, funcao_ativacao_oculta, funcao_ativacao_saida):
    """Retorna a acurácia percentual (0–100) sobre um conjunto de amostras."""
    pesos_hidden = rede_neural[0]["pesos"]
    bias_hidden  = rede_neural[0]["bias"]
    pesos_saida  = rede_neural[1]["pesos"]
    bias_saida   = rede_neural[1]["bias"]

    neuron_hidden = len(pesos_hidden)
    neuron_saida  = len(pesos_saida)
    acertos = 0

    for i in range(len(X)):
        f_hid = [
            funcao_ativacao_oculta(calculaSomatorioNeuronio(X[i], pesos_hidden[j], bias_hidden[j]))
            for j in range(neuron_hidden)
        ]
        y_prev = [
            funcao_ativacao_saida(calculaSomatorioNeuronio(f_hid, pesos_saida[o], bias_saida[o]))
            for o in range(neuron_saida)
        ]
        if y_prev.index(max(y_prev)) == Y[i].index(max(Y[i])):
            acertos += 1

    return (acertos / len(X)) * 100.0

#Função auxiliar para salvar os hiperparâmetros da arquitetura e do treinamento da rede neural em um arquivo de texto
def salvar_hiperparametros(caminho_arquivo, entradas, neuronios_ocultos, neuronios_saidas, taxa_aprendizagem, epocas,
                           funcao_ativacao_oculta, funcao_ativacao_saida,
                           paciencia_early_stopping, delta_minimo_early_stopping):
    with open(caminho_arquivo, "w", encoding="utf-8") as arquivo:
        arquivo.write("--- HIPERPARAMETROS DA ARQUITETURA E TREINAMENTO DA REDE NEURAL ---\n\n")
        
        arquivo.write("--- Estrutura da Rede ---\n")
        arquivo.write(f"Neurônios na Camada de Entrada: {entradas}\n")
        arquivo.write(f"Neurônios na Camada Oculta: {neuronios_ocultos}\n")
        arquivo.write(f"Neurônios na Camada de Saída: {neuronios_saidas}\n\n")
        
        arquivo.write("--- Configurações de Aprendizado ---\n")
        arquivo.write(f"Taxa de Aprendizagem (Alpha): {taxa_aprendizagem}\n")
        arquivo.write(f"Total de Épocas (máximo): {epocas}\n")
        arquivo.write(f"Função de Ativação na Camada Oculta: {funcao_ativacao_oculta}\n")
        arquivo.write(f"Função de Ativação na Camada de Saída: {funcao_ativacao_saida}\n\n")

        arquivo.write("--- Configurações de Early Stopping ---\n")
        arquivo.write(f"Paciência (épocas sem melhora): {paciencia_early_stopping}\n")
        arquivo.write(f"Delta Mínimo (melhora mínima aceitável): {delta_minimo_early_stopping}\n")
    
    print(f"DEBUG: Hiperparâmetros salvos em '{caminho_arquivo}'!")
        
#Função auxiliar para salvar os pesos iniciais da rede neural em um arquivo de texto, organizando os pesos por camada e por neurônio, e formatando os valores com 6 casas decimais para melhor legibilidade.
def salvar_pesos(nome_arquivo, rede_neural):
    with open(nome_arquivo, "w") as arquivo:
        #Camada Oculta
        arquivo.write("Camada Oculta\n")
        pesos_hidden = rede_neural[0]["pesos"]
        for i in range(len(pesos_hidden)):
            linha_pesos = " ".join(f"{w:.6f}" for w in pesos_hidden[i])
            arquivo.write(f"Pesos Neuronio {i}: {linha_pesos}\n")
        
        arquivo.write("\n")
        
        #Camada de Saída
        arquivo.write("Camada Saída\n")
        pesos_saida = rede_neural[1]["pesos"]
        for i in range(len(pesos_saida)):
            linha_pesos = " ".join(f"{w:.6f}" for w in pesos_saida[i])
            arquivo.write(f"Pesos Neuronio {i}: {linha_pesos}\n")
            
#Calcula a soma ponderada das entradas multiplicadas pelos pesos correspondentes para um neurônio específico, e adiciona o bias desse neurônio à soma.
def calculaSomatorioNeuronio(entradas, pesos_neuronio, bias_neuronio):
    soma_ponderada = 0
    for i in range(len(entradas)):
        soma_ponderada += entradas[i] * pesos_neuronio[i]
    soma_ponderada += bias_neuronio
    
    return soma_ponderada

#Função auxiliar para calcular a soma dos erros quadráticos entre os valores reais e os valores previstos pela rede neural. A função itera sobre cada elemento das listas de valores reais e previstos, calcula a diferença ao quadrado para cada elemento, e retorna a soma total multiplicada por 0.5, que é a fórmula do erro quadrático médio (MSE) utilizado como critério de avaliação do desempenho da rede neural durante o treinamento.
def calculaSomaErrosQuadraticos(y_real, y_previsto):
    return 0.5 * sum((y_real[k] - y_previsto[k]) ** 2 for k in range(len(y_real)))


def plotar_grafico_erro(historico_erros, historico_erros_val=None, epoca_parada=None,
                        historico_acuracia_treino=None, historico_acuracia_val=None):
    diretorio_do_script = os.path.dirname(os.path.abspath(__file__))

    # ---- Gráfico 1: Erro ----
    plt.figure(figsize=(10, 6))
    plt.plot(historico_erros, color='blue', linewidth=2, label='Erro de Treino')

    if historico_erros_val:
        plt.plot(historico_erros_val, color='orange', linewidth=2, linestyle='--', label='Erro de Validação')

    if epoca_parada is not None:
        plt.axvline(x=epoca_parada, color='red', linestyle=':', linewidth=1.5,
                    label=f'Early Stopping (época {epoca_parada + 1})')

    plt.title('Decaimento do Erro Total ao Longo das Épocas')
    plt.xlabel('Épocas')
    plt.ylabel('Erro Total')
    plt.legend()
    plt.grid(True)

    caminho_erro = os.path.join(diretorio_do_script, 'grafico_decaimento_erro_treinamento.png')
    plt.savefig(caminho_erro, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"DEBUG: Gráfico de Erros ao longo das Épocas de Treinamento salvo em '{caminho_erro}'!")

    # ---- Gráfico 2: Acurácia (apenas se os históricos foram fornecidos) ----
    if historico_acuracia_treino:
        plt.figure(figsize=(10, 6))
        plt.plot(historico_acuracia_treino, color='blue', linewidth=2, label='Acurácia de Treino')

        if historico_acuracia_val:
            plt.plot(historico_acuracia_val, color='orange', linewidth=2, linestyle='--', label='Acurácia de Validação')

        if epoca_parada is not None:
            plt.axvline(x=epoca_parada, color='red', linestyle=':', linewidth=1.5,
                        label=f'Early Stopping (época {epoca_parada + 1})')

        plt.title('Acurácia Média ao Longo das Épocas')
        plt.xlabel('Épocas')
        plt.ylabel('Acurácia (%)')
        plt.ylim(0, 105)
        plt.legend()
        plt.grid(True)

        caminho_acuracia = os.path.join(diretorio_do_script, 'grafico_acuracia_treinamento.png')
        plt.savefig(caminho_acuracia, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"DEBUG: Gráfico de Acurácia ao longo das Épocas salvo em '{caminho_acuracia}'!")

#Função auxiliar para plotar a matriz de confusão utilizando a biblioteca Seaborn. A matriz de confusão é gerada a partir das listas de valores esperados e previstos, e é exibida como um mapa de calor, facilitando a visualização do desempenho da rede neural na classificação das letras.
def plotar_matriz_confusao(lista_esperados, lista_previstos):
    matriz = pd.crosstab(pd.Series(lista_esperados, name='Esperado'), pd.Series(lista_previstos, name='Previsto'))
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Matriz de Confusão dos Resultados do Teste')
    
    diretorio_do_script = os.path.dirname(os.path.abspath(__file__))
    caminho_arquivo = os.path.join(diretorio_do_script, 'matriz_confusao_teste.png')
    
    plt.savefig(caminho_arquivo, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"DEBUG: Matriz de Confusão dos Resultados do Teste salva em '{caminho_arquivo}'!")


def backpropagation(X, Y, rede_neural, taxa_aprendizagem, epocas, mapeamento_letras,
                    funcao_ativacao_oculta, funcao_derivada_oculta,
                    funcao_ativacao_saida, funcao_derivada_saida,
                    X_val=None, Y_val=None,
                    paciencia=10, delta_minimo=1e-4):

    camada_oculta = rede_neural[0]
    pesos_hidden = camada_oculta["pesos"]
    bias_hidden  = camada_oculta["bias"]
    
    camada_saida  = rede_neural[1]
    pesos_saida  = camada_saida["pesos"]
    bias_saida   = camada_saida["bias"]

    neuron_hidden = len(pesos_hidden)
    neuron_saida = len(pesos_saida)

    # Listas para salvar o histórico de erros e acurácias
    historico_erros            = []
    historico_erros_val        = []
    historico_acuracia_treino  = []
    historico_acuracia_val     = []

    # ---- Estado do Early Stopping ----
    usar_early_stopping  = (X_val is not None and Y_val is not None)
    melhor_erro_val      = float('inf')   # menor erro de validação visto até agora
    contador_paciencia   = 0              # épocas consecutivas sem melhora
    melhores_pesos       = None           # snapshot dos pesos da melhor época
    melhor_epoca         = 0             # índice da melhor época (0-based)

    for epoca in range(epocas):
        erro_total = 0

        for i in range(len(X)):
            # ------------------------- Feedforward: Camada Oculta -------------------------
            funcao_ativacao_hidden = []
            for j in range(neuron_hidden):
                Z_in_hidden = calculaSomatorioNeuronio(X[i], pesos_hidden[j], bias_hidden[j])
                funcao_ativacao_hidden.append(funcao_ativacao_oculta(Z_in_hidden))

            # ------------------------- Feedforward: Camada de Saida -------------------------
            y_previsto = []
            for o in range(neuron_saida):
                Z_in_saida = calculaSomatorioNeuronio(funcao_ativacao_hidden, pesos_saida[o], bias_saida[o])
                y_previsto.append(funcao_ativacao_saida(Z_in_saida))

            # ------------------------- Calculo do Erro Total -------------------------
            erro_total += calculaSomaErrosQuadraticos(Y[i], y_previsto)

            # ------------------------- Delta da Camada de Saida -------------------------
            delta_saida = []
            for o in range(neuron_saida):
                d_out = (Y[i][o] - y_previsto[o]) * funcao_derivada_saida(y_previsto[o])
                delta_saida.append(d_out)

            # ------------------------- Delta da Camada Oculta -------------------------
            delta_hidden = []
            for j in range(neuron_hidden):
                soma_erro = sum(delta_saida[o] * pesos_saida[o][j] for o in range(neuron_saida))
                d_hid = funcao_derivada_oculta(funcao_ativacao_hidden[j]) * soma_erro
                delta_hidden.append(d_hid)  

            # ------------------------- Atualização: Camada de Saída -------------------------
            for o in range(neuron_saida):
                for j in range(neuron_hidden):
                    pesos_saida[o][j] += taxa_aprendizagem * delta_saida[o] * funcao_ativacao_hidden[j]
                bias_saida[o] += taxa_aprendizagem * delta_saida[o]

            # ------------------------- Atualização: Camada Oculta -------------------------
            for j in range(neuron_hidden):
                for k in range(len(X[i])):
                    pesos_hidden[j][k] += taxa_aprendizagem * delta_hidden[j] * X[i][k]
                bias_hidden[j] += taxa_aprendizagem * delta_hidden[j]

        historico_erros.append(erro_total)

        # ---- Acurácia do conjunto de treino nesta época ----
        acuracia_treino = calcular_acuracia_conjunto(
            X, Y, rede_neural, funcao_ativacao_oculta, funcao_ativacao_saida
        )
        historico_acuracia_treino.append(acuracia_treino)

        # ---- Avaliação no Conjunto de Validação (Early Stopping) ----
        if usar_early_stopping:
            erro_val = calcular_erro_conjunto(
                X_val, Y_val, rede_neural, funcao_ativacao_oculta, funcao_ativacao_saida
            )
            historico_erros_val.append(erro_val)

            # Acurácia de validação nesta época
            acuracia_val = calcular_acuracia_conjunto(
                X_val, Y_val, rede_neural, funcao_ativacao_oculta, funcao_ativacao_saida
            )
            historico_acuracia_val.append(acuracia_val)

            # Verifica se houve melhora significativa
            if erro_val < melhor_erro_val - delta_minimo:
                melhor_erro_val    = erro_val
                contador_paciencia = 0
                melhores_pesos     = copiar_rede(rede_neural)   # salva snapshot
                melhor_epoca       = epoca
            else:
                contador_paciencia += 1

            if (epoca + 1) % 100 == 0 or epoca == 0:
                print(f"Época {epoca+1}/{epocas} - Erro Treino: {erro_total:.6f} | "
                      f"Acurácia Treino: {acuracia_treino:.2f}% | "
                      f"Erro Validação: {erro_val:.6f} | "
                      f"Acurácia Validação: {acuracia_val:.2f}% | "
                      f"Paciência: {contador_paciencia}/{paciencia}")

            # Critério de parada: paciência esgotada
            if contador_paciencia >= paciencia:
                print(f"\n[Early Stopping] Treinamento interrompido na época {epoca+1}.")
                print(f"[Early Stopping] Melhor época: {melhor_epoca+1} | "
                      f"Melhor erro de validação: {melhor_erro_val:.6f}")
                # Restaura os pesos da melhor época antes de encerrar
                restaurar_rede(rede_neural, melhores_pesos)
                print("[Early Stopping] Pesos restaurados para a melhor época.")
                break
        else:
            if (epoca + 1) % 100 == 0 or epoca == 0:
                print(f"Época {epoca+1}/{epocas} - Erro Total: {erro_total:.6f} | "
                      f"Acurácia Treino: {acuracia_treino:.2f}%")

    # Salva o arquivo contendo os erros por épocas
    np.savetxt("erros_treinamento.txt", historico_erros, fmt="%.6f")
    
    # Chama a função visual do decaimento do erro (passa a época de parada quando aplicável)
    epoca_parada = melhor_epoca if (usar_early_stopping and contador_paciencia >= paciencia) else None
    plotar_grafico_erro(
        historico_erros,
        historico_erros_val if usar_early_stopping else None,
        epoca_parada,
        historico_acuracia_treino,
        historico_acuracia_val if usar_early_stopping else None
    )

    # ------------------------- Resultados Finais Dinâmicos -------------------------
    print("\nResultados após o treinamento:")
    print('-' * 75)
    print('Amostra | Letra Esperada | Letra Predita | Confiança')
    print('-' * 75)
    
    lista_esperados = []
    lista_previstos = []
    
    for idx in range(len(X)):
        f_hid = [funcao_ativacao_oculta(calculaSomatorioNeuronio(X[idx], pesos_hidden[j], bias_hidden[j])) for j in range(neuron_hidden)]
        y_prev = [funcao_ativacao_saida(calculaSomatorioNeuronio(f_hid, pesos_saida[o], bias_saida[o])) for o in range(neuron_saida)]
        
        idx_esperado = Y[idx].index(max(Y[idx]))
        idx_previsto = y_prev.index(max(y_prev))
        
        letra_esperada = mapeamento_letras[idx_esperado]
        letra_prevista = mapeamento_letras[idx_previsto]
        
        lista_esperados.append(letra_esperada)
        lista_previstos.append(letra_prevista)
        
        print(f"Letra {idx+1:02d} | Em classe: {letra_esperada}      | Predita: {letra_prevista}      | Confiança: {max(y_prev):.4f}")
    print(f"DEBUG: Dados de treino processados em {epocas} épocas com sucesso!")

def testar_rede(X_teste, Y_teste, rede_neural, mapeamento_letras, funcao_ativacao_oculta, funcao_ativacao_saida):
    camada_oculta = rede_neural[0]
    pesos_hidden = camada_oculta["pesos"]
    bias_hidden  = camada_oculta["bias"]
    
    camada_saida  = rede_neural[1]
    pesos_saida  = camada_saida["pesos"]
    bias_saida   = camada_saida["bias"]

    neuron_hidden = len(pesos_hidden)
    neuron_saida = len(pesos_saida)

    lista_esperados = []
    lista_previstos = []
    linhas_arquivo_saida = []
    acertos_teste = 0

    print("-" * 75)
    print("Resultados do Conjunto de Teste:")
    print("Amostra | Letra Esperada | Letra Predita | Confiança")

    for idx in range(len(X_teste)):
        # ------------------------- Feedforward Camada Oculta -------------------------
        funcao_ativacao_hidden = []
        for j in range(neuron_hidden):
            Z_in_hidden = calculaSomatorioNeuronio(X_teste[idx], pesos_hidden[j], bias_hidden[j])
            funcao_ativacao_hidden.append(funcao_ativacao_oculta(Z_in_hidden))

        # ------------------------- Feedforward Camada de Saida -------------------------
        y_previsto = []
        for o in range(neuron_saida):
            Z_in_saida = calculaSomatorioNeuronio(funcao_ativacao_hidden, pesos_saida[o], bias_saida[o])
            y_previsto.append(funcao_ativacao_saida(Z_in_saida))

        # ------------------------- Avaliação da Amostra -------------------------
        idx_esperado = Y_teste[idx].index(max(Y_teste[idx]))
        idx_previsto = y_previsto.index(max(y_previsto))
        
        letra_esperada = mapeamento_letras[idx_esperado]
        letra_prevista = mapeamento_letras[idx_previsto]
        confianca = max(y_previsto)
        
        lista_esperados.append(letra_esperada)
        lista_previstos.append(letra_prevista)

        if letra_prevista == letra_esperada:
            acertos_teste += 1
        
        print(f"Teste {idx+1:02d}  | Esperada: {letra_esperada}      | Predita: {letra_prevista}      | Confiança: {confianca:.4f}")
        
        # Prepara a linha para salvar no arquivo de log
        linhas_arquivo_saida.append(f"Amostra {idx+1}: Esperada={letra_esperada}, Predita={letra_prevista}, Confiança={confianca:.4f}\n")

    # ------------------------- Acurácia Final do Teste -------------------------
    acuracia_teste = (acertos_teste / len(X_teste)) * 100
    print("-" * 75)
    print(f"Acurácia no Conjunto de Teste: {acertos_teste}/{len(X_teste)} amostras corretas ({acuracia_teste:.2f}%)")

    # ------------------------- Exportação de Resultados -------------------------
    with open("saidas_teste.txt", "w") as arquivo_teste:
        arquivo_teste.writelines(linhas_arquivo_saida)
        arquivo_teste.write(f"\nAcurácia Final: {acertos_teste}/{len(X_teste)} ({acuracia_teste:.2f}%)\n")

    print("DEBUG: Resultados do teste salvos em 'saidas_teste.txt'!")
   
    #Gera a matriz de confusão
    plotar_matriz_confusao(lista_esperados, lista_previstos)

def main():
    print("Rede Neural - MLP Multilayer Perceptron Detector de Caracteres\n")
    diretorio_do_script = os.path.dirname(os.path.abspath(__file__))
    
    caminho_x = os.path.join(diretorio_do_script, 'files-sarajane', 'CARACTERES COMPLETO', 'X.txt')
    caminho_y = os.path.join(diretorio_do_script, 'files-sarajane', 'CARACTERES COMPLETO', 'Y_letra.txt')
    
    #X_dados, Y_dados e a lista mapeada de strings das letras detectadas
    X_dados, Y_dados, mapeamento_letras = carrega_dataset(caminho_x, caminho_y)
    
    #Separa os dados em 80% para treino e 20% para teste, mantendo a correspondencia entre as amostras e as letras que representam
    X_treino, Y_treino, X_teste, Y_teste = separar_dados_treino_teste(X_dados, Y_dados, proporcao_treino=0.8)

    # ---- Separa uma fatia de validação (10% do treino) para o early stopping ----
    # O conjunto de validação é retirado do treino, não do teste, preservando a
    # avaliação final imparcial sobre dados completamente não vistos.
    X_treino, Y_treino, X_val, Y_val = separar_validacao(X_treino, Y_treino, proporcao_validacao=0.1)
    
    quantidade_entradas = len(X_dados[0])
    quantidade_saidas = len(Y_dados[0])
    
    print(f"--> Entradas extraídas (Atributos): {quantidade_entradas}")
    print(f"--> Saídas extraídas (Classes mapeadas): {quantidade_saidas}")
    print(f"--> Amostras de Treino (após separar validação): {len(X_treino)}")
    print(f"--> Amostras de Validação (early stopping): {len(X_val)}")
    print(f"--> Amostras de Teste: {len(X_teste)}")
    print("-" * 50)

    print("Escolha uma função de ativação abaixo para a rede neural:")
    for chave, config in catalogo_ativacoes.items():
        print(f"{chave}: {config['nome']}")
    
    escolha = input("Digite o número correspondente da sua escolha: ")
    
    while escolha not in catalogo_ativacoes:
        print("Opção inválida. Digite o número correspondente novamente.")
        escolha = input("Digite o número correspondente: ")
        
    config_selecionada = catalogo_ativacoes[escolha]
    nome_ativacao = config_selecionada["nome"]
    
    funcao_ativacao_oculta = config_selecionada["funcao"]
    funcao_derivada_oculta = config_selecionada["derivada"]
    
    if config_selecionada["camada_saida"] is True:
        funcao_ativacao_saida = config_selecionada["funcao"]
        funcao_derivada_saida = config_selecionada["derivada"]
        print(f"\nFunção de Ativação {nome_ativacao} aplicada em todas as camadas com sucesso!")
    else:
        funcao_ativacao_saida = sigmoid
        funcao_derivada_saida = derivada_sigmoid
        print(f"\nAviso Arquitetural:")
        print(f"A função {nome_ativacao} é incompatível com a saída, pois não retorna valores estritos no intervalo [0, 1]. Portanto, a função de ativação da camada de saída foi automaticamente definida como Sigmoid para garantir a correta classificação das letras, pois a função sigmoid retorna apenas valores no intervalo ]0, 1[. Logo, a função {nome_ativacao} será utilizada apenas na camada oculta.")
        print(f"-> Função de Ativação da Camada Oculta: {nome_ativacao}")
        print(f"-> Função de Ativação da Camada de Saída: Sigmoid")
    
    print(f"Função {nome_ativacao} selecionada com sucesso!")

    # ---- Hiperparâmetros do Early Stopping ----
    # paciencia: quantas épocas sem melhora o treino tolera antes de parar.
    # delta_minimo: redução mínima no erro de validação para contar como melhora.
    paciencia_early_stopping    = 10
    delta_minimo_early_stopping = 1e-4

    neuronios_ocultos = 60
    mapeamento = [quantidade_entradas, neuronios_ocultos, quantidade_saidas]
    
    global rede_neural
    rede_neural = []
    for m in range(len(mapeamento) - 1):
        entrada = mapeamento[m]
        saida = mapeamento[m+1]
        nova_camada = criaCamada(entrada, saida)
        rede_neural.append(nova_camada)
        
    caminho_hiperparametros = os.path.join(diretorio_do_script, "hiperparametros.txt")
    epocas_treino = 100
    print('-' * 75)
    salvar_hiperparametros(
        caminho_hiperparametros, quantidade_entradas, neuronios_ocultos,
        quantidade_saidas, taxa_aprendizagem, epocas_treino,
        funcao_ativacao_oculta.__name__, funcao_ativacao_saida.__name__,
        paciencia_early_stopping, delta_minimo_early_stopping
    )
    
    salvar_pesos("pesos_iniciais.txt", rede_neural)
    print(f"DEBUG: Pesos iniciais salvos em '{diretorio_do_script}+\\pesos_iniciais.txt'!") 
    print('-' * 75)   
    
    inicio_treino = time.perf_counter()
    print("Iniciando o Treinamento da Rede Neural\n")
    backpropagation(
        X_treino, Y_treino, rede_neural, taxa_aprendizagem,
        epocas_treino, mapeamento_letras,
        funcao_ativacao_oculta, funcao_derivada_oculta,
        funcao_ativacao_saida, funcao_derivada_saida,
        # Parâmetros do Early Stopping:
        X_val=X_val,
        Y_val=Y_val,
        paciencia=paciencia_early_stopping,
        delta_minimo=delta_minimo_early_stopping
    )
    fim_treino = time.perf_counter()
    tempo_total_treino = fim_treino - inicio_treino
    minutos_treino = int(tempo_total_treino // 60)
    segundos_treino = tempo_total_treino % 60
    print(f"Tempo gasto no treinamento: {minutos_treino} {'minuto' if minutos_treino == 1 else 'minutos'} e {segundos_treino:.3f} {'segundo' if round(segundos_treino, 3) == 1.0 else 'segundos'}")
    print("-" * 75)
    
    salvar_pesos("pesos_finais.txt", rede_neural)
    print(f"DEBUG: Pesos finais salvos em '{diretorio_do_script}+\\pesos_finais.txt'!")
    
    inicio_teste = time.perf_counter()
    testar_rede(X_teste, Y_teste, rede_neural, mapeamento_letras, funcao_ativacao_oculta, funcao_ativacao_saida)
    fim_teste = time.perf_counter()
    tempo_total_teste = fim_teste - inicio_teste
    minutos_teste = int(tempo_total_teste // 60)
    segundos_teste = tempo_total_teste % 60
    print(f"Tempo gasto no teste: {minutos_teste} {'minuto' if minutos_teste == 1 else 'minutos'} e {segundos_teste:.3f} {'segundo' if round(segundos_teste, 3) == 1.0 else 'segundos'}")
    print("-" * 75)  
    
if __name__ == "__main__":
    main()