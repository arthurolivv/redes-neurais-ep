import math
import matplotlib.pyplot as plt
import numpy as np

X = [
    [0,0],
    [0,1],
    [1,0],
    [1,1]
]

y = [0, 1, 1, 0]

for i in range(4):
    print(X[i], y[i])

w1 = 0.2
w2 = 0.56
bias = 0.15

def sigmoid(x):
    return 1 / (1+ math.exp(-x))


taxa_aprendizagem = 0.1

for epoca in range (5000): #condicao de parada qualquer
    for i in range(4): #para cada s:t

        x1 = X[i][0]
        x2 = X[i][1]
        target = y[i]

        y_in = x1*w1 + x2*w2 + bias
        saida = sigmoid(y_in)

        print('===========================')
        
        print(X[i], saida)

        print('w1 = ', w1)
        print('w2 = ', w2)
        print('bias = ', bias)

        sinal_erro = target - saida

        print('target: ', target)
        print('sinal de erro: ', target,'-', saida, '=', sinal_erro)
        #valor_instantaneo_erro = 1/2 * sinal_erro * sinal_erro

        gradiente = sinal_erro * saida *  (1 - saida)

        #definindo pesos New
        w1 = w1 + taxa_aprendizagem*gradiente*x1
        w2 = w2 + taxa_aprendizagem*gradiente*x2
        bias = bias + taxa_aprendizagem*gradiente

        print('new w1 = ', w1)
        print('new w2 = ', w2)
        print('new bias = ', bias)

# cria uma grade de pontos
x_values = np.linspace(0, 1, 100)
y_values = np.linspace(0, 1, 100)

Z = []

for x in x_values:
    linha = []
    for y_ in y_values:
        y_in = x * w1 + y_ * w2 + bias
        saida = sigmoid(y_in)
        linha.append(saida)
    Z.append(linha)

Z = np.array(Z)

# plot
plt.imshow(Z, extent=(0,1,0,1), origin='lower')
plt.colorbar()

# pontos reais
for i in range(4):
    x1, x2 = X[i]
    plt.scatter(x1, x2)

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Superfície aprendida pela rede")

plt.savefig("grafico.png")
print("Gráfico salvo como grafico.png")