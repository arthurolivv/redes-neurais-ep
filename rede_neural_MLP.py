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
import random
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re

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
    
    print(f"DEBUG: Dados alinhados com sucesso!")
    print(f"DEBUG: Total de amostras prontas para o treino: {tamanho_minimo}")
    print("-" * 50)
    
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

#Função auxiliar para salvar os hiperparâmetros da arquitetura e do treinamento da rede neural em um arquivo de texto
def salvar_hiperparametros(caminho_arquivo, entradas, ocultos, saidas, taxa, epocas):
    with open(caminho_arquivo, "w", encoding="utf-8") as arquivo:
        arquivo.write("--- HIPERPARAMETROS DA ARQUITETURA E TREINAMENTO ---\n\n")
        
        arquivo.write("--- Estrutura da Rede ---\n")
        arquivo.write(f"Neurônios na Camada de Entrada: {entradas}\n")
        arquivo.write(f"Neurônios na Camada Oculta: {ocultos}\n")
        arquivo.write(f"Neurônios na Camada de Saída: {saidas}\n\n")
        
        arquivo.write("--- Configurações de Aprendizado ---\n")
        arquivo.write(f"Taxa de Aprendizagem (Alpha): {taxa}\n")
        arquivo.write(f"Total de Épocas: {epocas}\n")
        arquivo.write("Função de Ativação: Sigmoide\n")
        
    print(f"Sucesso: Hiperparâmetros salvos em '{caminho_arquivo}'!")

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
            
    print(f"Sucesso: O arquivo '{nome_arquivo}' foi gerado na pasta do projeto!")

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

#Função auxiliar para plotar o gráfico de erro total ao longo das épocas, utilizando a biblioteca Matplotlib. O gráfico exibe o decaimento do erro total, permitindo visualizar a convergência do treinamento da rede neural.
def plotar_grafico_erro(historico_erros):
    plt.figure(figsize=(10, 6))
    plt.plot(historico_erros, color='blue', linewidth=2)
    plt.title('Decaimento do Erro Total ao Longo das Épocas')
    plt.xlabel('Épocas')
    plt.ylabel('Erro Total')
    plt.grid(True)
    plt.show()

#Função auxiliar para plotar a matriz de confusão utilizando a biblioteca Seaborn. A matriz de confusão é gerada a partir das listas de valores esperados e previstos, e é exibida como um mapa de calor, facilitando a visualização do desempenho da rede neural na classificação das letras.
def plotar_matriz_confusao(lista_esperados, lista_previstos):
    matriz = pd.crosstab(pd.Series(lista_esperados, name='Esperado'), pd.Series(lista_previstos, name='Previsto'))
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Matriz de Confusão')
    plt.show()

def backpropagation(X, Y, rede_neural, taxa_aprendizagem, epocas, mapeamento_letras):
    camada_oculta = rede_neural[0]
    pesos_hidden = camada_oculta["pesos"]
    bias_hidden  = camada_oculta["bias"]
    
    camada_saida  = rede_neural[1]
    pesos_saida  = camada_saida["pesos"]
    bias_saida   = camada_saida["bias"]

    neuron_hidden = len(pesos_hidden)
    neuron_saida = len(pesos_saida)

    #Lista para salvar o histórico de erros (Requisito de entrega!)
    historico_erros = []

    for epoca in range(epocas):
        erro_total = 0

        for i in range(len(X)):

            #------------------------- Feedforward: Camada Oculta -------------------------
            funcao_ativacao_hidden = []
            for j in range(neuron_hidden):
                Z_in_hidden = calculaSomatorioNeuronio(X[i], pesos_hidden[j], bias_hidden[j])
                funcao_ativacao_hidden.append(sigmoid(Z_in_hidden))

            #------------------------- Feedforward: Camada de Saída -------------------------
            y_previsto = []
            for o in range(neuron_saida):
                Z_in_saida = calculaSomatorioNeuronio(funcao_ativacao_hidden, pesos_saida[o], bias_saida[o])
                y_previsto.append(sigmoid(Z_in_saida))

            #------------------------- Cálculo do Erro Total -------------------------
            erro_total += calculaSomaErrosQuadraticos(Y[i], y_previsto)

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

    # Salva o arquivo contendo os erros por épocas
    np.savetxt("erros_treinamento.txt", historico_erros, fmt="%.6f")
    
    # Chama a função visual do decaimento do erro
    plotar_grafico_erro(historico_erros)

    # ------------------------- Resultados Finais Dinâmicos -------------------------
    print("\nResultados após o treinamento:")
    print('-' * 75)
    print('Amostra | Letra Esperada | Letra Predita | Confiança')
    print('-' * 75)
    
    lista_esperados = []
    lista_previstos = []
    
    for idx in range(len(X)):
        f_hid = [sigmoid(calculaSomatorioNeuronio(X[idx], pesos_hidden[j], bias_hidden[j])) for j in range(neuron_hidden)]
        y_prev = [sigmoid(calculaSomatorioNeuronio(f_hid, pesos_saida[o], bias_saida[o])) for o in range(neuron_saida)]
        
        idx_esperado = Y[idx].index(max(Y[idx]))
        idx_previsto = y_prev.index(max(y_prev))
        
        letra_esperada = mapeamento_letras[idx_esperado]
        letra_prevista = mapeamento_letras[idx_previsto]
        
        lista_esperados.append(letra_esperada)
        lista_previstos.append(letra_prevista)
        
        print(f"Letra {idx+1:02d} | Em classe: {letra_esperada}      | Predita: {letra_prevista}      | Confiança: {max(y_prev):.4f}")
    print("-" * 75 + "\n")


def testar_rede(X_teste, Y_teste, rede_neural, mapeamento_letras):
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

    print("\nResultados do Conjunto de Teste:")
    print("=" * 75)
    print("Amostra | Letra Esperada | Letra Predita | Confiança")
    print("=" * 75)

    for idx in range(len(X_teste)):
        # ------------------------- Feedforward: Camada Oculta -------------------------
        funcao_ativacao_hidden = []
        for j in range(neuron_hidden):
            Z_in_hidden = calculaSomatorioNeuronio(X_teste[idx], pesos_hidden[j], bias_hidden[j])
            funcao_ativacao_hidden.append(sigmoid(Z_in_hidden))

        # ------------------------- Feedforward: Camada de Saída -------------------------
        y_previsto = []
        for o in range(neuron_saida):
            Z_in_saida = calculaSomatorioNeuronio(funcao_ativacao_hidden, pesos_saida[o], bias_saida[o])
            y_previsto.append(sigmoid(Z_in_saida))

        # ------------------------- Avaliação da Amostra -------------------------
        idx_esperado = Y_teste[idx].index(max(Y_teste[idx]))
        idx_previsto = y_previsto.index(max(y_previsto))
        
        letra_esperada = mapeamento_letras[idx_esperado]
        letra_prevista = mapeamento_letras[idx_previsto]
        confianca = max(y_previsto)
        
        lista_esperados.append(letra_esperada)
        lista_previstos.append(letra_prevista)
        
        print(f"Teste {idx+1:02d}  | Esperada: {letra_esperada}      | Predita: {letra_prevista}      | Confiança: {confianca:.4f}")
        
        # Prepara a linha para salvar no arquivo de log
        linhas_arquivo_saida.append(f"Amostra {idx+1}: Esperada={letra_esperada}, Predita={letra_prevista}, Confianca={confianca:.4f}\n")

    print("=" * 75 + "\n")

    # ------------------------- Exportação de Resultados -------------------------
    with open("saidas_teste.txt", "w") as arquivo_teste:
        arquivo_teste.writelines(linhas_arquivo_saida)

    #Gera a matriz de confusão para o vídeo
    plotar_matriz_confusao(lista_esperados, lista_previstos)

def main():
    diretorio_do_script = os.path.dirname(os.path.abspath(__file__))
    
    caminho_x = os.path.join(diretorio_do_script, 'files-sarajane', 'CARACTERES COMPLETO', 'X.txt')
    caminho_y = os.path.join(diretorio_do_script, 'files-sarajane', 'CARACTERES COMPLETO', 'Y_letra.txt')
    
    #X_dados, Y_dados e a lista mapeada de strings das letras detectadas
    X_dados, Y_dados, mapeamento_letras = carrega_dataset(caminho_x, caminho_y)
    
    #Separa os dados em 80% para treino e 20% para teste, mantendo a correspondencia entre as amostras e as letras que representam
    X_treino, Y_treino, X_teste, Y_teste = separar_dados_treino_teste(X_dados, Y_dados, proporcao_treino=0.8)
    
    quantidade_entradas = len(X_dados[0])
    quantidade_saidas = len(Y_dados[0])
    
    print(f"--> Entradas extraídas (Atributos): {quantidade_entradas}")
    print(f"--> Saídas extraídas (Classes mapeadas): {quantidade_saidas}")
    print(f"--> Amostrar de Treino: {len(X_treino)}")
    print(f"--> Amostrar de Teste: {len(X_teste)}\n")

    # Ajustado 30 neuronios que serão utilizados na camada oculta
    neuronios_ocultos = 30
    mapeamento = [quantidade_entradas, neuronios_ocultos, quantidade_saidas]
    
    global rede_neural
    rede_neural = []
    for m in range(len(mapeamento) - 1):
        entrada = mapeamento[m]
        saida = mapeamento[m+1]
        nova_camada = criaCamada(entrada, saida)
        rede_neural.append(nova_camada)
        
    #Salva os hiperparâmetros da arquitetura e do treinamento da rede neural em um arquivo de texto para documentação
    caminho_hiperparametros = os.path.join(diretorio_do_script, "hiperparametros.txt")
    epocas_treino = 1000
    salvar_hiperparametros(
        caminho_hiperparametros, 
        quantidade_entradas, 
        neuronios_ocultos, 
        quantidade_saidas, 
        taxa_aprendizagem, 
        epocas_treino
    )
    
    #Salva os pesos iniciais gerados aleatoriamente na rede neural antes de iniciar o treinamento
    salvar_pesos("pesos_iniciais.txt", rede_neural)
        
    #Executa o backpropagation passando o mapeamento das letras
    backpropagation(X_treino, Y_treino, rede_neural, taxa_aprendizagem, epocas=epocas_treino, mapeamento_letras=mapeamento_letras)
    
    #Salva os pesos finais da rede neural após o treinamento
    salvar_pesos("pesos_finais.txt", rede_neural)
    
    #Avaliação final utilizando os dados de teste
    testar_rede(X_teste, Y_teste, rede_neural, mapeamento_letras)
    
if __name__ == "__main__":
    main()