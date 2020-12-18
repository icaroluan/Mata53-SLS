# -*- coding: UTF-8 -*-

class Erro(Exception):
    pass

import numpy as np
#O NumPy oferece funcoes matematicas abrangentes, geradores de numeros aleatorios, rotinas de algebra linear, transformadas de Fourier e muito mais.
#Oferece suporte a uma ampla variedade de plataformas de hardware e computacao e funciona bem com bibliotecas distribuidas, GPU e de matriz esparsa.

#Classe para conter os elementos do algoritmo: Matriz-custo Cnxn
class Hungaro:

    def __init__(self, entradaM=None, p_matriz=False):
        if entradaM is not None:
            matriz = np.array(entradaM)
            self._entradaM = np.array(entradaM)
            self.colunas_maximas = matriz.shape[1]
            self.linhas_maximas = matriz.shape[0]
            
            #Elementos da Matriz
            tamanho_matriz = max(self.colunas_maximas, self.linhas_maximas)
            col = tamanho_matriz - self.linhas_maximas
            li = tamanho_matriz - self.colunas_maximas
            matriz = np.pad(matriz, ((0,col),(0,li)), 'constant', constant_values=(0))
            
            #Funcao para associar o custo a matriz
            if p_matriz:
                matriz = self.criar_custo(matriz)

            self.custo = matriz
            self.tamanho = len(matriz)
            self.formato = matriz.shape

            self.resultados = []
            self.potencial = 0
        else:
            self.custo = None

    def pegar_resultados(self):
        return self.resultados

    def pegar_potencial(self):
        return self.potencial

    def calcular(self, entradaM=None, p_matriz=False):
        if entradaM is None and self.custo is None:
            raise Erro("Invalid input")
        elif entradaM is not None:
            self.__init__(entradaM, p_matriz)

        matrizResultados = self.custo.copy()
        
        #Faz a etapa de subtrair o menor valor presente em cada linha
        for index, linha in enumerate(matrizResultados):
            matrizResultados[index] -= linha.min()
       
        #Faz a etapa de subtrair o menor valor presente em cada coluna
        for index, coluna in enumerate(matrizResultados.T):
            matrizResultados[:, index] -= coluna.min()
        
        #Linhas 60-71 encontra todos os elementos cobertos presentes da matriz que ja esta processada apos atribuirmos a subtracao das etapas anteriores, valor min linha e coluna    
        cobertos = 0
        while cobertos < self.tamanho:
            zeros_cobertos = Cobrir_Zeros(matrizResultados)
            linhas_cobertas = zeros_cobertos.acessar_linhascobertas()
            colunas_cobertas = zeros_cobertos.acessar_colunascobertas()
            cobertos = len(linhas_cobertas) + len(colunas_cobertas)

            if cobertos < self.tamanho:
                matrizResultados = self.matriz_descoberta_minima(matrizResultados, linhas_cobertas, colunas_cobertas)
        
        #Pega o menor número dos elementos não cobertos para poder subtrair os elementos não cobertos por este número encontrado
        resulta = min(self.colunas_maximas, self.linhas_maximas)
        locaiszeros = (matrizResultados == 0)
        
        while len(self.resultados) != resulta:

            if not locaiszeros.any():
                raise Erro("Incapaz de encontrar os resultados. Algoritmo falhou.")

            linhas_coincedem, colunas_coincidem = self.encontrar_coinc(locaiszeros)
            
            #Encontra valores que são cobertos 2x na matriz
            bateu = len(linhas_coincedem) + len(colunas_coincidem)
            
            if bateu == 0:
                linhas_coincedem, colunas_coincidem = self.selecionar_coincidencia(locaiszeros)

            for linha in linhas_coincedem:
                locaiszeros[linha] = False
            for coluna in colunas_coincidem:
                locaiszeros[:, coluna] = False

            self.setar_resultados(zip(linhas_coincedem, colunas_coincidem))

        valor = 0
        for linha, coluna in self.resultados:
            valor += self._entradaM[linha, coluna]
        self.potencial = valor

    @staticmethod
    def criar_custo(matriz_p):
        formato_matriz = matriz_p.shape
        ind_matriz = np.ones(formato_matriz, dtype=int) * matriz_p.max()
        custoo_matriz = ind_matriz - matriz_p
        return custoo_matriz

    def matriz_descoberta_minima(self, matrizResultados, linhas_cobertas, colunas_cobertas):
        elements = []
        for index_linhas, linha in enumerate(matrizResultados):
            if index_linhas not in linhas_cobertas:
                for index, element in enumerate(linha):
                    if index not in colunas_cobertas:
                        elements.append(element)
        numero_desc_min = min(elements)

        matriz_ajustada = matrizResultados
        for linha in linhas_cobertas:
            matriz_ajustada[linha] += numero_desc_min
        for coluna in colunas_cobertas:
            matriz_ajustada[:, coluna] += numero_desc_min

        m_matrix = np.ones(self.formato, dtype=int) * numero_desc_min
        matriz_ajustada -= m_matrix

        return matriz_ajustada

    def encontrar_coinc(self, locaiszeros):
        criar_linhas = np.array([], dtype=int)
        colunas_cobertas = np.array([], dtype=int)

        for index, linha in enumerate(locaiszeros):
            index_linhas = np.array([index])
            if np.sum(linha) == 1:
                index_colunas, = np.where(linha)
                criar_linhas, colunas_cobertas = self.linhas_colunas_cobertas(criar_linhas, colunas_cobertas, index_linhas,
                                                                           index_colunas)

        for index, coluna in enumerate(locaiszeros.T):
            index_colunas = np.array([index])
            if np.sum(coluna) == 1:
                index_linhas, = np.where(coluna)
                criar_linhas, colunas_cobertas = self.linhas_colunas_cobertas(criar_linhas, colunas_cobertas, index_linhas,
                                                                           index_colunas)

        return criar_linhas, colunas_cobertas

    @staticmethod
    def linhas_colunas_cobertas(criar_linhas, colunas_cobertas, index_linhas, index_colunas):
        novas_linhas = criar_linhas
        novas_colunas = colunas_cobertas
        if not (criar_linhas == index_linhas).any() and not (colunas_cobertas == index_colunas).any():
            novas_linhas = np.insert(criar_linhas, len(criar_linhas), index_linhas)
            novas_colunas = np.insert(colunas_cobertas, len(colunas_cobertas), index_colunas)
        return novas_linhas, novas_colunas

    @staticmethod
    def selecionar_coincidencia(locaiszeros):
        linhas, colunas = np.where(locaiszeros)
        quantidade_zeros = []
        for index, linha in enumerate(linhas):
            total_zeros = np.sum(locaiszeros[linha]) + np.sum(locaiszeros[:, colunas[index]])
            quantidade_zeros.append(total_zeros)

        # Get the linha coluna combination with the minimum number of zeros.
        indices = quantidade_zeros.index(min(quantidade_zeros))
        linha = np.array([linhas[indices]])
        coluna = np.array([colunas[indices]])

        return linha, coluna

    def setar_resultados(self, lista_resp):
        for result in lista_resp:
            linha, coluna = result
            if linha < self.linhas_maximas and coluna < self.colunas_maximas:
                novo_resultado = (int(linha), int(coluna))
                self.resultados.append(novo_resultado)


class Cobrir_Zeros:
    def __init__(self, matr):
        self.alocacoes_de_zeros = (matr == 0)
        self.formato = matr.shape

        self.escolhas = np.zeros(self.formato, dtype=bool)

        self._linhas_marcas = []
        self._colunas_marcadas = []

        self._calcular()

        self._cobertas_linhas = list(set(range(self.formato[0])) - set(self._linhas_marcas))
        self._cobertas_colunas = self._colunas_marcadas

    def acessar_linhascobertas(self):
        return self._cobertas_linhas

    def acessar_colunascobertas(self):
        return self._cobertas_colunas

    def _calcular(self):
        while True:
            self._linhas_marcas = []
            self._colunas_marcadas = []

            for index, linha in enumerate(self.escolhas):
                if not linha.any():
                    self._linhas_marcas.append(index)

            if not self._linhas_marcas:
                return True

            nnumer_colunas_marcadas = self.novas_colunas_marcadas_zero_linhas()

            if nnumer_colunas_marcadas == 0:
                return True

            while self.decisao_colunas_cobertas():
                linhas_marc_numeros = self.colunas_marcadas_decisao()

                if linhas_marc_numeros == 0:
                    return True

                nnumer_colunas_marcadas = self.novas_colunas_marcadas_zero_linhas()

                if nnumer_colunas_marcadas == 0:
                    return True

            colunas_escolhidas = self.encontrar_marcacoes()

            while colunas_escolhidas is not None:
                escolha_indice_linhas = self.linhasnao_escolhidas(colunas_escolhidas)

                nova_escolhas_colunas_indic = None
                if escolha_indice_linhas is None:
                    escolha_indice_linhas, nova_escolhas_colunas_indic = \
                        self.melhor_Escolha(colunas_escolhidas)

                    self.escolhas[escolha_indice_linhas, nova_escolhas_colunas_indic] = False

                self.escolhas[escolha_indice_linhas, colunas_escolhidas] = True

                colunas_escolhidas = nova_escolhas_colunas_indic

    def novas_colunas_marcadas_zero_linhas(self):
        nnumer_colunas_marcadas = 0
        for index, coluna in enumerate(self.alocacoes_de_zeros.T):
            if index not in self._colunas_marcadas:
                if coluna.any():
                    row_indices, = np.where(coluna)
                    zer_linhas_marcadas = (set(self._linhas_marcas) & set(row_indices)) != set([])
                    if zer_linhas_marcadas:
                        self._colunas_marcadas.append(index)
                        nnumer_colunas_marcadas += 1
        return nnumer_colunas_marcadas

    def colunas_marcadas_decisao(self):
        linhas_marc_numeros = 0
        for index, linha in enumerate(self.escolhas):
            if index not in self._linhas_marcas:
                if linha.any():
                    index_colunas, = np.where(linha)
                    if index_colunas in self._colunas_marcadas:
                        self._linhas_marcas.append(index)
                        linhas_marc_numeros += 1
        return linhas_marc_numeros

    def decisao_colunas_cobertas(self):
        for index_colunas in self._colunas_marcadas:
            if not self.escolhas[:, index_colunas].any():
                return False
        return True

    def encontrar_marcacoes(self):
        for index_colunas in self._colunas_marcadas:
            if not self.escolhas[:, index_colunas].any():
                return index_colunas

    def linhasnao_escolhidas(self, colunas_escolhidas):
        row_indices, = np.where(self.alocacoes_de_zeros[:, colunas_escolhidas])
        for index_linhas in row_indices:
            if not self.escolhas[index_linhas].any():
                return index_linhas

        return None

    def melhor_Escolha(self, colunas_escolhidas):
        row_indices, = np.where(self.alocacoes_de_zeros[:, colunas_escolhidas])
        for index_linhas in row_indices:
            column_indices, = np.where(self.escolhas[index_linhas])
            index_colunas = column_indices[0]
            if self.linhasnao_escolhidas(index_colunas) is not None:
                return index_linhas, index_colunas

        from random import shuffle

        shuffle(row_indices)
        index_colunas, = np.where(self.escolhas[row_indices[0]])
        return row_indices[0], index_colunas[0]

if __name__ == '__main__':
    
    custo_matriz =  [[26, 35, 74, 20], [26, 36, 72, 22], [35, 53, 80, 40], [38, 48, 81, 41]]
    hungaro = Hungaro(custo_matriz)
    hungaro.calcular()

    print("Resultado Final: ", hungaro.pegar_potencial())
    print("Alocacao: ", hungaro.pegar_resultados())
    print("-" * 80)