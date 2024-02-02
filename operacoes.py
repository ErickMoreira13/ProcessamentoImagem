import cv2
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from statistics import mean
import math

def gerar_dois_histograma(imgA, imgB, title1, title2):
    ranges = [0, 255]    


    plt.figure(figsize = ((12, 6)))
    plt.subplot(1, 2, 1)
    plt.title(title1, fontsize = 16)
    plt.hist(imgA.ravel(), 256, ranges)

    plt.subplot(1, 2, 2)
    plt.title(title2, fontsize = 16)
    plt.suptitle("Gráficos", fontsize = 20)
    plt.hist(imgB.ravel(), 256, ranges)

    plt.show()

def gerar_histograma(img, title):
    channels = [0]
    histSize = [256]
    ranges = [0, 255]


    hist = cv2.calcHist([img], channels, None, histSize, ranges)
    plt.hist(img.ravel(), 256, ranges)
    plt.title(title)
    plt.show()


def padarray(img, n):
    top = bottom = left = right = int(math.floor((n-1)/2))
    img_out = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, None, 0)
    return img_out

#1. a) Operações Algébricas
def dissolve(imgA, imgB, t, tipo):

    h1, w1, c1 = imgA.shape #x, y, z da img
    h2, w2, c2 = imgB.shape #x, y, z da img

    if ( (h1 == h2) and (w1 == w2) ): # Se a dimensão das img's for igual
        img_out = np.zeros((h1, w1, 3), dtype = np.uint8) # Cria uma "img"

        if (tipo == 1): # Tipo Uniforme
            for i in range(h1):
                for j in range(w1):
                    img_out[i, j] = (1-t)*imgA[i, j] + t*imgB[i, j]

            return img_out       
        
        elif (tipo == 2): # Tipo Não-Uniforme
            h3, w3 = t.shape
            if ( (h1 == h3) and (w1 == w3)):
                for i in range(h1):
                    for j in range(w1):
                        if ( (t[i, j] > 0) and (t[i, j] < 1)):
                            img_out[i, j] = ((1-t[i, j])*imgA[i, j]) + t[i,j]*imgB[i, j]
            
            return img_out


#1. b) Transformação de intensidade     
def negativo(img):
    h, w, c = img.shape
    img_out = np.zeros_like(img)

    for i in range(h):
        for j in range(w):
            img_out[i, j] = img.max() - img[i, j]

    return img_out

def alargamento_contraste(img, r_min, r_max, s_min, s_max):
    h, w, c = img.shape #x, y, z da img
    img_out = np.zeros((h, w, 3), dtype = np.uint8) # Cria uma "img"

    for i in range(h):
        for j in range(w):
            for k in range(c):
                img_out[i, j, k] = ((s_max - s_min)/(r_max - r_min))*( (img[i, j, k]) - r_min) + s_min

    return img_out

def limiarizacao(img, t):
    h, w, c = img.shape #x, y, z da img
    img_out = np.zeros((h, w, 3), dtype = np.uint8) # Cria uma "img"
    
    imgG = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    for i in range(h):
        for j in range(w):
            if (imgG[i, j] > t):
                img_out[i, j] = 255
            else:
                img_out[i, j] = 0
    
    return img_out

def gamma(img, cnst, intensidade):
    h, w, c = img.shape #x, y, z da img
    img_out = np.zeros((h, w, 3), dtype = np.uint8) # Cria uma "img"

    # Normalizando a img
    imgN = cv2.normalize(img, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    img_out = imgN.copy()

    for i in range(h):
        for j in range(w):
            for k in range(c):
                img_out[i, j, k] = cnst*((imgN[i, j, k])**intensidade)
                #cnst*(((img[i, j, k])/255)**intensidade)*255
    
    
    return img_out

def logaritmo(img, cnt):
    h, w, c = img.shape #x, y, z da img
    img_out = np.zeros((h, w, 3), dtype = np.uint8) # Cria uma "img"

    # Normalizando a img
    imgN = cv2.normalize(img, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    img_out = imgN.copy()

    for i in range(h):
        for j in range(w):
            for k in range(c):
                img_out[i, j, k] = cnt*(math.log(1 + (imgN[i, j, k])) )

    return img_out


#1. c) Expansão e equalização do histograma
def expansao_histograma(img):
    h, w, c = img.shape #x, y, z da img
    img_out = np.zeros((h, w, 3), dtype = np.uint8) # Cria uma "img"
    r_min = np.min(img) # Pegando o min da img
    r_max = np.max(img) # Pegando o max da img
    L = 256

    for i in range(h):
        for j in range(w):
            for k in range(c):
                img_out[i, j, k] = round( (((img[i, j, k]) - r_min)/(r_max - r_min))*(L-1) )

    
    return img_out  

def equalizacao_histograma(img):
    h, w, chn = img.shape #x, y, z da img
    img_out = np.zeros((h, w, 3), dtype = np.uint8) # Cria uma "img"
    imgG = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Cria uma img em escala cinza

    total_pixels = h*w
    L = 256
    histograma = np.zeros((1, L))

    # Calcula o histograma da imagem
    for i in range(h):
        for j in range(w):
            intensidade = imgG[i, j] + 1
            histograma[0, intensidade] += 1

    # Calcula a função de distribuição acumulativa (CDF)
    cdf = np.cumsum(histograma)/total_pixels

    valores_transformados = np.round((L - 1) * cdf).astype(int) # Aplicando a fórmula aos valores
    valores_transformados = np.clip(valores_transformados, 0, 255) # Mantendo os resultados entre 0 a 255

    # Transferindo os valores para os pixels da imagem de saida
    for i in range(h):
        for j in range(w):
            intensidade = imgG[i, j] + 1
            img_out[i, j] = valores_transformados[intensidade]

    
    return img_out


#1. d) Controle de contraste adaptativo
def controle_adaptativo_contraste(img, c, n):
    h, w, chn = img.shape #x, y, z da img
    img_out = np.zeros((h, w, 3), dtype = np.uint8) # Cria uma "img"
    imgG = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Cria uma img em escala cinza
    
    add = int(math.floor((n-1)/2))
    inicio = (n-add)-1

    imgP = padarray(imgG, n)

    for i in range(inicio, h-add):
        for j in range(inicio, w-add):
            x1 = i - add
            x2 = i + add
            y1 = j - add
            y2 = j + add
        
            bloco = imgP[x1:x2, y1:y2]
            
            media = np.average(bloco) # Tirando a média
            desvio = np.std(bloco) # Calculando o desvio
            

            if (desvio != 0):
                img_out[i, j] = media + (c/desvio)*(img[i,j] - media)
            else:
                img_out[i, j] = img[i, j] 
    
    return img_out


#1. e) Transformação geométrica
# Ampliacao por replicacao
def ampliacao(img, F):
    h, w, chn = img.shape #x, y, z da img
    new_h = F*h-1
    new_w = F*w-1
    img_out = np.zeros((new_h, new_w, 3), dtype = np.uint8) # Cria uma "img"

    # Para cada ponto da img original, a img_out recebe este ponto em coordenadas relativas ao fator (F) de ampliação ("passo" de cada pixel)
    img_out[0:new_h:F, 0:new_w:F] = img[0:h, 0:w]

    return img_out

# Redução por media
def reducao_media(img, F):
    h, w, chn = img.shape #x, y, z da img
    
    nova_linha = math.floor(w/F)
    nova_coluna = math.floor(h/F)

    img_out = np.zeros((nova_linha, nova_coluna, 3), dtype = np.uint8) # Cria uma "img"

    for i in range(nova_linha):
        for j in range(nova_coluna):
            # Dimensoes do bloco
            linha_inicio = (i)*F + 1
            linha_fim = linha_inicio + F - 1
            coluna_inicio = (j)*F + 1
            coluna_fim = coluna_inicio + F - 1

            bloco = img[linha_inicio:linha_fim, coluna_inicio:coluna_fim]

            max_array = [] #Lista para armazenar os valores maximos
            for x in bloco:
                max_array.append(x[:, :].max())

            Mx = mean(max_array) # tirando a media dos valores maximos

            bloco_reduzido = Mx 

            img_out[i, j] = bloco_reduzido
    
    return img_out

def cisalhamento(img, s):
    h, w, chn = img.shape #x, y, z da img
    img_out = np.zeros((h, w, 3), dtype = np.uint8) # Cria uma "img"

    # Inicia a matriz de transformacao
    t = [
        (1, s),
        (0, 1)
    ]


    for i in range(h):
        for j in range(w):
            r = [ [i], [j] ]
            res = [[0], [0]]
            res = np.dot(t, r)
            new_i = res[0][0]
            new_j = res[1][0]

            if ( (new_i > 0 and new_i < h) and (new_j > 0 and new_j < w) ): # Garantindo que as coordenadas nao saiam do estimado
                img_out[new_i, new_j] = img[i, j]

    return img_out

def rebater(img):
    h, w, chn = img.shape #x, y, z da img
    img_out = img.copy()

    for i in range(h):
        for j in range(w):
            img_out[j, i] = img[i, j]

    
    return img_out

def rotacionar(img, graus):
    h, w, chn = img.shape #x, y, z da img
    img_out = np.zeros((h, w, 3), dtype = np.uint8) # Cria uma "img"
    ic = h/2
    jc = w/2

    for i in range(h):
        for j in range(w):
            # Calculando novo indice "i"
            new_i = round ( (i - ic)*math.cos(math.radians(graus)) - (j - jc)*math.sin(math.radians(graus)) + ic ) 
            # Calculando o novo indice "j"
            new_j = round ( (i - ic)*math.sin(math.radians(graus)) + (j - jc)*math.cos(math.radians(graus)) + jc )

            if ( (new_i > 0 and new_i < h) and (new_j > 0 and new_j < w) ): # Garantindo que as coordenadas nao saiam do estimado
                img_out[new_i, new_j] = img[i, j]

    return img_out

def pinch(img, tipo, ip, Fmax):
    h, w, chn = img.shape #x, y, z da img
    img_out = np.zeros((h, w, 3), dtype = np.uint8) # Cria uma "img"

    # Pinch na vertical
    if (tipo == "vertical"):
        for i in range(h):
            for j in range(w):
                if (i <= ip):
                    F = ((Fmax - 1)/(h-1-ip))*(i-ip) + Fmax
                else:
                    F = -((Fmax - 1)/(h-1-ip))*(i-ip) + Fmax

                new_j = round( ((j-ip)/F) + ip )

                if ( (new_j > 0 and new_j < w) ): # Garantindo que as coordenadas nao saiam do estimado
                    img_out[i, new_j] = img[i, j]

    # Pinch de bordas
    if (tipo == "bordas"): 
        for i in range(h):
            for j in range(w):
                if (i <= ip):
                    F = -((Fmax - 1)/(h-1-ip))*(i-ip) + 1
                else:
                    F = ((Fmax - 1)/(h-1-ip))*(i-ip) + 1

                new_j = round( ((j-ip)/F) + ip )

                if ( (new_j > 0 and new_j < w) ): # Garantindo que as coordenadas nao saiam do estimado
                    img_out[i, new_j] = img[i, j]

    return img_out


#1. f) Filtragem linear e não-linear  
def filtragem_media(img, n):
    h, w, c = img.shape #x, y, z da img
    imgP = padarray(img, n)
    h2, w2, c2 = imgP.shape #x, y, z da img
    img_out = np.zeros((h, w, 3), dtype = np.uint8) # Cria uma "img"

    add = int(math.floor((n-1)/2)) # Calcula quantos "layer's"/camadas serão add na img
    inicio = (n-add)-1 # Calcula onde comeca a img

    for i in range(inicio, h-add):
        for j in range(inicio, w-add):
            # Calcula a regiao do bloco
            x1 = i - add
            x2 = i + add
            y1 = j - add
            y2 = j + add
            
            bloco = imgP[x1:x2, y1:y2]

            lin, col, z = bloco.shape # Tamanho do bloco

            # Pegar todos os valores de cada dimensão (R, G, B)
            valores1 = []
            valores2 = []
            valores3 = []

            for m in range(lin):
                for n in range(col):
                    valores1.append(bloco[m, n, 0])

            for m in range(lin):
                for n in range(col):
                    valores2.append(bloco[m, n, 1])
                    
            
            for m in range(lin):
                for n in range(col):
                    valores3.append(bloco[m, n, 2])
            
            # Tira a media de cada dimensao (R, G, B)
            media1 = np.mean(valores1)
            media2 = np.mean(valores2)
            media3 = np.mean(valores3)

            img_out[i, j, 0] = media1
            img_out[i, j, 1] = media2
            img_out[i, j, 2] = media3
            
    return img_out

def filtragem_mediana(img, n):
    h, w, c = img.shape #x, y, z da img
    imgP = padarray(img, n)
    h2, w2, c2 = imgP.shape #x, y, z da img
    img_out = np.zeros((h2, w2, 3), dtype = np.uint8) # Cria uma "img"

    add = int(math.floor((n-1)/2))
    inicio = (n-add)-1

    for i in range(inicio, h):
        for j in range(inicio, w):
            # Calcula a regiao do bloco
            x1 = i - add
            x2 = i + add
            y1 = j - add
            y2 = j + add
            
            bloco = imgP[x1:x2, y1:y2]

            lin, col, z = bloco.shape # Tamanho do bloco

            # Pegar todos os valores de cada dimensão (R, G, B)
            valores1 = []
            valores2 = []
            valores3 = []

            for m in range(lin):
                for n in range(col):
                    valores1.append(bloco[m, n, 0])

            for m in range(lin):
                for n in range(col):
                    valores2.append(bloco[m, n, 1])
                    
            
            for m in range(lin):
                for n in range(col):
                    valores3.append(bloco[m, n, 2])

            # Ordenando os valores em cada lista
            valores1 = sorted(valores1)
            valores2 = sorted(valores2)
            valores3 = sorted(valores3)

            # Tira a mediana de cada dimensao (R, G, B)
            mediana1 = math.floor(np.median(valores1))
            mediana2 = math.floor(np.median(valores2))
            mediana3 = math.floor(np.median(valores3))

            img_out[i, j, 0] = mediana1
            img_out[i, j, 1] = mediana2
            img_out[i, j, 2] = mediana3
            
    return img_out


#1. g) Detecção de bordas
def sobel(img):
    t1 = 245
    t2 = 150
    imgG = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Normaliza a imagem para o intervalo [0, 1]
    imgG_normalized = imgG.astype(float)/255

    imgP = np.pad(imgG_normalized, ((1, 1), (1, 1)), mode='constant')
    h2, w2 = imgP.shape
    img_out = np.zeros((h2, w2), dtype=np.uint8)

    for i in range(1, h2 - 1):
        for j in range(1, w2 - 1):
            gradiente = np.abs(imgP[i + 1, j - 1] + 2 * imgP[i + 1, j] + imgP[i + 1, j + 1] -
                               imgP[i - 1, j - 1] - 2 * imgP[i - 1, j] - imgP[i - 1, j + 1]) \
                         + np.abs(imgP[i - 1, j + 1] + imgP[i, j + 1] + imgP[i + 1, j + 1] -
                                  imgP[i - 1, j - 1] - 2 * imgP[i, j - 1] - imgP[i + 1, j - 1])

            gradiente = np.clip(gradiente * 255, 0, 255)

            if gradiente >= t1:
                img_out[i, j] = 255
            elif gradiente >= t2:
                img_out[i, j] = 0

    return img_out
   

#1. h) Aguçamento de bordas e high boost
def agucamento_bordas(img, k):
    h, w, c = img.shape
    img_out = np.zeros((h, w, 3), dtype = np.uint8)

    m = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ])
    imgBordas = convolucao(img, m)

    for i in range(h):
        for j in range(w):
            valor = img[i, j] + (k*imgBordas[i, j])
            valor = np.clip(valor, 0, 255)
            img_out[i, j] = valor
            
    

    return img_out

def high_boost(img, k):
    n = 3
    h, w, c = img.shape #x, y, z da img
    imgC = img.copy()
    imgC = padarray(imgC, n)
    imgM = filtragem_media(img, n)
    h2, w2, c2 = imgM.shape
    img_res =  np.zeros((h2, w2, 3), dtype = np.uint8)

    add = int(math.floor((n-1)/2))
    inicio = (n-add)-1

    for i in range(inicio, h):
        for j in range(inicio, w):
            img_res[i, j] = imgC[i, j] - imgM[i, j]

    img_out = img_res[0:(w+add), 0:(h+add)].copy()
    
    for i in range(h-1):
        for j in range(w-1):
            img_out[i, j] = (k*img_out[i, j]) + img[i, j]
    
    return img_out
    
#1. i) Convolução entre uma imagem f e uma máscara h
def convolucao(img, kernel):
    offset = 1
    (iH, iW) = img.shape[:2]
    (kH, kW) = kernel.shape[:2]
    pad = (kW - 1) // 2

    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE) # Adicionando as bordas a mais para a matriz | "padarray"
    img_out = np.zeros((iH, iW, 3), dtype = "uint8")
    kernel = np.flipud(np.fliplr(kernel)) # Invertendo a matriz
    
    for i in np.arange(pad, iH + pad):
        for j in np.arange(pad, iW + pad):
            roi = img[i - pad:i + pad + 1, j - pad:j + pad + 1]
            valor = (roi * kernel).sum() + offset
            valor = np.clip(valor, 0, 255)
            img_out[i - pad, j - pad] = valor
            
    
    return img_out
