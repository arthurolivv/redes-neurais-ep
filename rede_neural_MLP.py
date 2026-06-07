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
<<<<<<< HEAD:testMLP.py
rede_neural = []

=======
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
>>>>>>> 706925d (Separação dos dados de treino e teste com exportação automática de hiperparâmetros, pesos iniciais e finais, gráfico de decaimento de erro, matriz de confusão e saídas pós teste.):rede_neural_MLP.py
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

<<<<<<< HEAD:testMLP.py
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

=======
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
>>>>>>> 706925d (Separação dos dados de treino e teste com exportação automática de hiperparâmetros, pesos iniciais e finais, gráfico de decaimento de erro, matriz de confusão e saídas pós teste.):rede_neural_MLP.py
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

<<<<<<< HEAD:testMLP.py
    #------------------------- Resultados Finais -------------------------
=======
    # Salva o arquivo contendo os erros por épocas
    np.savetxt("erros_treinamento.txt", historico_erros, fmt="%.6f")
    
    # Chama a função visual do decaimento do erro
    plotar_grafico_erro(historico_erros)

    # ------------------------- Resultados Finais Dinâmicos -------------------------
>>>>>>> 706925d (Separação dos dados de treino e teste com exportação automática de hiperparâmetros, pesos iniciais e finais, gráfico de decaimento de erro, matriz de confusão e saídas pós teste.):rede_neural_MLP.py
    print("\nResultados após o treinamento:")
    print('-' * 50)
    print('Entrada  | Saída Esperada | Saída Prevista')
    print('-' * 50)
    for idx in range(len(X)):
<<<<<<< HEAD:testMLP.py
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
=======
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
>>>>>>> 706925d (Separação dos dados de treino e teste com exportação automática de hiperparâmetros, pesos iniciais e finais, gráfico de decaimento de erro, matriz de confusão e saídas pós teste.):rede_neural_MLP.py
    for m in range(len(mapeamento) - 1):
        entrada = mapeamento[m]
        saida = mapeamento[m+1]
        
        nova_camada = criaCamada(entrada, saida)
        rede_neural.append(nova_camada)
        
<<<<<<< HEAD:testMLP.py
    print(f"Camadas Criadas (ignorando entrada): {len(rede_neural)}")
    printRedeNeural(rede_neural)
    
    print('Camada Saida: ')
    print('Pesos Saida: ', rede_neural[-1]["pesos"])
    print('Bias Saida: ', rede_neural[-1]["bias"])
    print("-" * 30)

    backpropagation(X, Y, rede_neural, taxa_aprendizagem, epocas=1500)

=======
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
    
>>>>>>> 706925d (Separação dos dados de treino e teste com exportação automática de hiperparâmetros, pesos iniciais e finais, gráfico de decaimento de erro, matriz de confusão e saídas pós teste.):rede_neural_MLP.py
if __name__ == "__main__":
    main()