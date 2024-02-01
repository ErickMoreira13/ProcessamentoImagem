from CTkMessagebox import CTkMessagebox
from  operacoes import *
import cv2


#1. a) Operações Algébricas
def gerarDissolveCruzadoUniforme(imgA, imgB, t, img_out):
    def chamarDissolveCruzado(imgA, imgB, t, img_out):
        img_out = dissolve(imgA, imgB, t, 1)
        cv2.imshow("Resultado", img_out)

    if ((len(imgA) > 1) and (len(imgB) > 1)):
        return chamarDissolveCruzado(imgA, imgB, t, img_out)
    else:
        CTkMessagebox(title="Erro!", message="A imagem 1 e imagem 2 tem que ser definidas!")

def gerarDissolveCruzadoNaoUniforme(imgA, imgB, t, img_out):
    def chamarDissolveCruzado(imgA, imgB, t, img_out):
        img_out = dissolve(imgA, imgB, t, 2)
        cv2.imshow("Resultado", img_out)

    if ((len(imgA) > 1) and (len(imgB) > 1)):
        return chamarDissolveCruzado(imgA, imgB, t, img_out)
    else:
        CTkMessagebox(title="Erro!", message = "A imagem 1 e imagem 2 tem que ser definidas!")


#1. b) Transformação de intensidade 
def gerarNegativo(imgA, img_out):
    def chamarNegativo(imgA, img_out):
        img_out = negativo(imgA)
        cv2.imshow("Resultado", img_out)

    if ((len(imgA) > 1)):
        return chamarNegativo(imgA, img_out)
    else:
        CTkMessagebox(title="Erro!", message = "A imagem 1 tem que ser definida!")

def gerarAlargamento_contraste(imgA, r_min, r_max, s_min, s_max, img_out):
    def chamarAlargamento_contraste(imgA, r_min, r_max, s_min, s_max, img_out):
        img_out = alargamento_contraste(imgA, r_min, r_max, s_min, s_max)
        cv2.imshow("Resultado", img_out)

    if ((len(imgA) > 1)):
        return chamarAlargamento_contraste(imgA, r_min, r_max, s_min, s_max, img_out)
    else:
        CTkMessagebox(title="Erro!", message = "A imagem 1 tem que ser definida!")
    
def gerarLimiarizacao(imgA, t, img_out):
    def chamarLimiarizacao(imgA, t, img_out):
        img_out = limiarizacao(imgA, t)
        cv2.imshow("Resultado", img_out)

    if ((len(imgA) > 1)):
        return chamarLimiarizacao(imgA, t, img_out)
    else:
        CTkMessagebox(title="Erro!", message = "A imagem 1 tem que ser definida!")

def gerarGamma(imgA, cnst, intensidade, img_out):
    def chamarGama(imgA, cnst, intensidade, img_out):
        img_out = gamma(imgA, cnst, intensidade)
        cv2.imshow("Resultado", img_out)

    if ((len(imgA) > 1)):
        return chamarGama(imgA, cnst, intensidade, img_out)
    else:
        CTkMessagebox(title="Erro!", message = "A imagem 1 tem que ser definida!")

def gerarLogaritmo(imgA, cnt, img_out):
    def chamarLogaritmo(imgA, cnt, img_out):
        img_out = logaritmo(imgA, cnt)
        cv2.imshow("Resultado", img_out)

    if ((len(imgA) > 1)):
        return chamarLogaritmo(imgA, cnt, img_out)
    else:
        CTkMessagebox(title="Erro!", message = "A imagem 1 tem que ser definida!")


#1. c) Expansão e equalização do histograma
def gerarExpansao_histograma(imgA, img_out):
    def chamarExpansao_histograma(imgA, img_out):
        img_out = expansao_histograma(imgA)
        cv2.imshow("Original", imgA)
        cv2.imshow("Resultado", img_out)
        
        gerar_dois_histograma(imgA, img_out, "Original", "Resultado")

    if ((len(imgA) > 1)):
        return chamarExpansao_histograma(imgA, img_out)
    else:
        CTkMessagebox(title="Erro!", message = "A imagem 1 tem que ser definida!")

def gerarEqualizacao_histograma(imgA, img_out):
    def chamarEqualizacao_histograma(imgA, img_out):
        img_out = equalizacao_histograma(imgA)
        cv2.imshow("Resultado", img_out)
        gerar_histograma(img_out, "Resultado")

    if ((len(imgA) > 1)):
        return chamarEqualizacao_histograma(imgA, img_out)
    else:
        CTkMessagebox(title="Erro!", message = "A imagem 1 tem que ser definida!")


#1. d) Controle de contraste adaptativo
def gerarControle_adaptativo_contraste(imgA, c, n, img_out):
    def chamarControle_adaptativo_contraste(imgA, c, n, img_out):
        img_out = controle_adaptativo_contraste(imgA, c, n)
        cv2.imshow("Resultado", img_out)

    if ((len(imgA) > 1)):
        return chamarControle_adaptativo_contraste(imgA, c, n, img_out)
    else:
        CTkMessagebox(title="Erro!", message = "A imagem 1 tem que ser definida!")


#1. e) Transformação geométrica
def gerarAmpliacao(imgA, F, img_out):
    def chamarAmpliacao(imgA, F, img_out):
        img_out = ampliacao(imgA, F)
        cv2.imshow("Resultado", img_out)

    if ((len(imgA) > 1)):
        return chamarAmpliacao(imgA, F, img_out)
    else:
        CTkMessagebox(title="Erro!", message = "A imagem 1 tem que ser definida!")

def gerarReducao_media(imgA, F, img_out):
    def chamarReducao_media(imgA, F, img_out):
        img_out = reducao_media(imgA, F)
        cv2.imshow("Resultado", img_out)

    if ((len(imgA) > 1)):
        return chamarReducao_media(imgA, F, img_out)
    else:
        CTkMessagebox(title="Erro!", message = "A imagem 1 tem que ser definida!")

def gerarCisalhamento(imgA, s, img_out):
    def chamarCisalhamento(imgA, s, img_out):
        img_out = cisalhamento(imgA, s)
        cv2.imshow("Resultado", img_out)

    if ((len(imgA) > 1)):
        return chamarCisalhamento(imgA, s, img_out)
    else:
        CTkMessagebox(title="Erro!", message = "A imagem 1 tem que ser definida!")

def gerarRebatimento(imgA, img_out):
    def chamarRebater(imgA, img_out):
        img_out = rebater(imgA)
        cv2.imshow("Resultado", img_out)

    if ((len(imgA) > 1)):
        return chamarRebater(imgA, img_out)
    else:
        CTkMessagebox(title="Erro!", message="A imagem 1 tem que ser definida!")

def gerarRotacionar(imgA, graus, img_out):
    def chamarRotacionar(imgA, graus, img_out):
        img_out = rotacionar(imgA, graus)
        cv2.imshow("Resultado", img_out)

    if ((len(imgA) > 1)):
        return chamarRotacionar(imgA, graus, img_out)
    else:
        CTkMessagebox(title="Erro!", message="A imagem 1 tem que ser definida!")

def gerarPinch(imgA, tipo, ip, Fmax, img_out):
    if (tipo == "vertical" or tipo == "bordas"):
        def chamarPinch(imgA, tipo, ip, Fmax, img_out):
            img_out = pinch(imgA, tipo, ip, Fmax)
            cv2.imshow("Resultado", img_out)

        if ((len(imgA) > 1)):
            return chamarPinch(imgA, tipo, ip, Fmax, img_out)
        else:
            CTkMessagebox(title="Erro!", message="A imagem 1 tem que ser definida!")

    else:
        CTkMessagebox(title="Erro!", message="Tipos: 'vertical' ou 'bordas'")


#1. f) Filtragem linear e não-linear
def gerarFiltragem_media(imgA, n, img_out):
    def chamarFiltragem_media(imgA, n, img_out):
        img_out = filtragem_media(imgA, n)
        print(img_out.shape)
        cv2.imshow("Resultado", img_out)

    if ((len(imgA) > 1)):
        return chamarFiltragem_media(imgA, n, img_out)
    else:
        CTkMessagebox(title="Erro!", message="A imagem 1 tem que ser definida!")

def gerarFiltragem_mediana(imgA, n, img_out):
    def chamarFiltragem_mediana(imgA, n, img_out):
        img_out = filtragem_mediana(imgA, n)
        cv2.imshow("Resultado", img_out)

    if ((len(imgA) > 1)):
        return chamarFiltragem_mediana(imgA, n, img_out)
    else:
        CTkMessagebox(title="Erro!", message="A imagem 1 tem que ser definida!")


#1. g) Detecção de bordas
def gerarSobel(imgA, img_out):
    def chamarSobel(imgA, img_out):
        img_out = sobel(imgA)
        cv2.imshow("Resultado", img_out)

    if ((len(imgA) > 1)):
        return chamarSobel(imgA, img_out)
    else:
        CTkMessagebox(title="Erro!", message="A imagem 1 tem que ser definida!")


#1. h) Aguçamento de bordas e high boost
def gerarAgucamento_bordas(imgA, k, img_out):
    def chamarAgucamento_bordas(imgA, k, img_out):
        img_out = agucamento_bordas(imgA, k)
        cv2.imshow("Resultado", img_out)

    if ((len(imgA) > 1)):
        return chamarAgucamento_bordas(imgA, k, img_out)
    else:
        CTkMessagebox(title="Erro!", message="A imagem 1 tem que ser definida!")

def gerarHigh_boost(imgA, k, img_out):
    def chamarHigh_boost(imgA, k, img_out):
        img_out = high_boost(imgA, k)
        cv2.imshow("Resultado", img_out)

    if ((len(imgA) > 1)):
        return chamarHigh_boost(imgA, k, img_out)
    else:
        CTkMessagebox(title="Erro!", message="A imagem 1 tem que ser definida!")


#1. i) Convolução entre uma imagem f e uma máscara h
def gerarConvolucao(imgA, a, b, c, d, e, f, g, h, i, img_out):
    kernel = np.array([
        [a, b, c],
        [d, e, f],
        [g, h, i]
    ])

    def chamarConvolucao(imgA, kernel, img_out):
        img_out = convolucao(imgA, kernel)
        cv2.imshow("Resultado", img_out)

    if ((len(imgA) > 1)):
        return chamarConvolucao(imgA, kernel, img_out)
    else:
        CTkMessagebox(title="Erro!", message="A imagem 1 tem que ser definida!")
