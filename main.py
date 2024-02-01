import customtkinter as ctk
from customtkinter import *
from tkinter import *
from customtkinter import filedialog
from CTkMessagebox import CTkMessagebox
import numpy as np
import cv2
from operacoesInterface import *

class Application():
    def __init__(self):
        # Inicializando as variáveis e argumentos
        self.caminho1 = ""
        self.caminho2 = ""
        self.imgA = np.empty((1,1))
        self.imgB = np.empty((1,1))
        self.img_out = np.empty((1,1, 3))

        self.mainWindow = ctk.CTk()
        self.telaConfig()
        self.criarAreaEnviarImg()
        self.criarAbas()
        self.preencherAbas()
        self.mainWindow.mainloop()
        
    def telaConfig(self):
        self.mainWindow.title("Processamento de imagem")
        self.mainWindow.geometry("1080x1000")
        self.mainWindow.maxsize(width = 1200, height = 1000)
        self.mainWindow.minsize(width = 1080, height = 950)
        self.mainWindow._set_appearance_mode("system")

    def lerArquivo1(self):
        self.caminho1 = filedialog.askopenfilename(initialdir = "/Desktop", 
                                              title = "Selecione um arquivo", 
                                              filetypes = (("Imagens", ["*.png", "*.jpg", "*.tif", "*.tiff"]),) )
        
        self.imgA = cv2.imread(self.caminho1)

    def lerArquivo2(self):
        self.caminho2 = filedialog.askopenfilename(initialdir = "/Desktop", 
                                              title = "Selecione um arquivo", 
                                              filetypes = (("Imagens", ["*.png", "*.jpg", "*.tif", "*.tiff"]),) )
        
        self.imgB = cv2.imread(self.caminho2)
        
    def mostrarImgs(self):
        if ( (len(self.imgA) == 1) and (len(self.imgB) == 1)):
            CTkMessagebox(title="Erro", message="Pelo menos uma imagem deve ser definida!", icon="cancel")

        elif( (len(self.imgA) > 1) and (len(self.imgB) > 1)):
            cv2.imshow("Imagem 1", self.imgA)
            cv2.imshow("Imagem 2", self.imgB)
            cv2.waitKey(0)
            
        elif ((len(self.imgA) > 1)):
            cv2.imshow("Imagem 1", self.imgA)
            cv2.waitKey(0)

        elif ((len(self.imgB) > 1)):
            cv2.imshow("Imagem 2", self.imgB)
            cv2.waitKey(0)

        else:
            CTkMessagebox(title="Erro!", message="Há algo estranho...", icon="cancel")        

    def criarAreaEnviarImg(self):
        self.text1 = ctk.CTkLabel(self.mainWindow, text = "Insira a imagem 1:", font = ("arial", 20)).pack(pady = 8)
        self.btn_img1 = ctk.CTkButton(self.mainWindow, text = "Enviar", command = self.lerArquivo1).pack()
        
        self.text2 = ctk.CTkLabel(self.mainWindow, text = "Insira a imagem 2:", font = ("arial", 20)).pack(pady= 8)
        self.btn_img2 = ctk.CTkButton(self.mainWindow, text = "Enviar", command = self.lerArquivo2).pack()

        self.btn_show = ctk.CTkButton(self.mainWindow, text = "Mostrar", command = self.mostrarImgs).pack(pady = 10)
        
    def criarAbas(self):
        self.tabview = ctk.CTkTabview(self.mainWindow, width = 800, height = 700)
        self.tabview.pack()
        self.tabview.add("Operações Algébricas").grid_columnconfigure(0, weight = 1)
        self.tabview.add("Transformações").grid_columnconfigure(0, weight = 1)

        # Aba de Tranformações -> Intensidade | Gemométrica 
        self.tabTransform = ctk.CTkTabview(self.tabview.tab("Transformações"), width = 600, height = 600,border_width = 1, border_color = "blue")
        self.tabTransform.pack()
        self.tabTransform.add("Intensidade").grid_columnconfigure(0, weight = 1)
        self.tabTransform.add("Geométrica").grid_columnconfigure(0, weight = 1)
        

        self.tabview.add("Histograma").grid_columnconfigure(0, weight = 1)
        self.tabview.add("Controle de contraste adaptativo").grid_columnconfigure(0, weight = 1)
        self.tabview.add("Filtragem").grid_columnconfigure(0, weight = 1)
        self.tabview.add("Bordas e High boost").grid_columnconfigure(0, weight = 1)
        self.tabview.add("Convolução").grid_columnconfigure(0, weight = 1)

    def preencherAbas(self):
        self.preencherAbaOperacoesAlgebricas()
        self.preencherAbaTransformações()
        self.preencherAbaHistograma()
        self.preencherAbaControle_contraste_adaptativo()
        self.preencherAbaFiltragem()
        self.preencherAbaBordasHighBoost()
        self.preencherAbaConvolucao()

    def preencherAbaOperacoesAlgebricas(self):
        self.t = DoubleVar(self.tabview.tab("Operações Algébricas"))
        self.text_op_algebricas = ctk.CTkLabel(self.tabview.tab("Operações Algébricas"), text = "Dissolve cruzado Uniforme:", font = ("arial", 40)).pack(pady = 20)

        self.text_t = ctk.CTkEntry(self.tabview.tab("Operações Algébricas"), textvariable=self.t).pack()
        

        self.op_algebricas = ctk.CTkButton(self.tabview.tab("Operações Algébricas"), text = "Aplicar", font = ("arial bold", 20), corner_radius = 8,
                                             command = lambda: gerarDissolveCruzadoUniforme(self.imgA, self.imgB, float(self.t.get()), self.img_out))
        self.op_algebricas.pack(pady = 20)

    def preencherAbaTransformações(self):
        self.preencherAbaIntensidade()
        self.preencherAbaGeometrica()

    def preencherAbaIntensidade(self):
        # Negativo
        self.text_intensidade = ctk.CTkLabel(self.tabTransform.tab("Intensidade"), text = "Negativo:", font = ("arial", 15))
        self.text_intensidade.grid_configure(row = 0, column = 0, pady = 10)

        self.btn_intensidade = ctk.CTkButton(self.tabTransform.tab("Intensidade"), text = "Aplicar", command = lambda: gerarNegativo(self.imgA, self.img_out))
        self.btn_intensidade.grid_configure(row = 0, column = 5, padx=80, pady=30)

        # Alargamento contraste

        # r_min, r_max, s_min, s_max
        self.text_vars_alarg_contraste = ctk.CTkLabel(self.tabTransform.tab("Intensidade"), text = "Variáveis = r_min | r_max | s_min | s_max:", font = ("arial", 15))
        self.text_vars_alarg_contraste.grid_configure(row = 1, column = 0, pady = 4, padx= 4, columnspan=2)

        # Txt Alargamento de contraste
        self.text_alarg_contraste = ctk.CTkLabel(self.tabTransform.tab("Intensidade"), text = "Alargamento de contraste:", font = ("arial", 12))
        self.text_alarg_contraste.grid_configure(row = 2, column = 0, pady = 4, padx= 4)
        
        
        
        self.r_min = DoubleVar(self.tabTransform.tab("Intensidade"))
        self.r_max = DoubleVar(self.tabTransform.tab("Intensidade"))
        self.s_min = DoubleVar(self.tabTransform.tab("Intensidade"))
        self.s_max = DoubleVar(self.tabTransform.tab("Intensidade"))

        self.text_r_min = ctk.CTkEntry(self.tabTransform.tab("Intensidade"), textvariable = self.r_min).grid_configure(row = 2, column = 1, padx = 4)
        self.text_r_max = ctk.CTkEntry(self.tabTransform.tab("Intensidade"), textvariable = self.r_max).grid_configure(row = 2, column = 2, padx = 4)
        self.text_s_min = ctk.CTkEntry(self.tabTransform.tab("Intensidade"), textvariable = self.s_min).grid_configure(row = 2, column = 3, padx = 4)
        self.text_s_max = ctk.CTkEntry(self.tabTransform.tab("Intensidade"), textvariable = self.s_max).grid_configure(row = 2, column = 4, pady = 4)

        self.btn_alarg_contraste = ctk.CTkButton(self.tabTransform.tab("Intensidade"), text = "Aplicar", 
                                                 command = lambda: gerarAlargamento_contraste(self.imgA, self.r_min.get(), self.r_max.get(),
                                                                                              self.s_min.get(), self.s_max.get(),
                                                                                               self.img_out )).grid_configure(row = 2, column = 5, padx = 4, pady = 30)
        
        # Limiarizacao
        self.text_limiarizacao = ctk.CTkLabel(self.tabTransform.tab("Intensidade"), text = "Limiarização:", font = ("arial", 15))
        self.text_limiarizacao.grid_configure(row = 3, column = 0, pady = 10, padx= 4)

        self.text_limiarizacao_t = ctk.CTkLabel(self.tabTransform.tab("Intensidade"), text = "Valor t (limiar):", font = ("arial", 15))
        self.text_limiarizacao_t.grid_configure(row = 3, column = 3, pady = 10, padx= 3)

        self.limiarizacao_t = IntVar(self.tabTransform.tab("Intensidade"))
        self.entry_limiarizacao_t = ctk.CTkEntry(self.tabTransform.tab("Intensidade"), textvariable = self.limiarizacao_t).grid_configure(row = 3, column = 4, pady = 4)



        self.btn_limiarizacao = ctk.CTkButton(self.tabTransform.tab("Intensidade"), text = "Aplicar", 
                                                 command = lambda: gerarLimiarizacao(self.imgA, self.limiarizacao_t.get(), 
                                                                                     self.img_out)).grid_configure(row = 3, column = 5, padx = 4, pady = 30)

        # Gamma

        # cnst, intensidade
        self.text_vars_gamma = ctk.CTkLabel(self.tabTransform.tab("Intensidade"), text = "Variáveis Gamma = constante | intensidade:", font = ("arial", 15))
        self.text_vars_gamma.grid_configure(row = 4, column = 0, pady = 4, padx= 4, columnspan=2)

        self.text_gamma= ctk.CTkLabel(self.tabTransform.tab("Intensidade"), text = "Gamma:", font = ("arial", 15))
        self.text_gamma.grid_configure(row = 5, column = 0, pady = 10, padx= 4)

        self.gamma_cnst = DoubleVar(self.tabTransform.tab("Intensidade"))
        self.entry_gamma_cnst = ctk.CTkEntry(self.tabTransform.tab("Intensidade"), textvariable = self.gamma_cnst).grid_configure(row = 5, column = 3, pady = 4)

        self.gamma_intensidade = DoubleVar(self.tabTransform.tab("Intensidade"))
        self.entry_gamma_intensidade = ctk.CTkEntry(self.tabTransform.tab("Intensidade"), textvariable = self.gamma_intensidade).grid_configure(row = 5, column = 4, pady = 4)

        self.btn_gamma = ctk.CTkButton(self.tabTransform.tab("Intensidade"), text = "Aplicar", 
                                                 command = lambda: gerarGamma(self.imgA, self.gamma_cnst.get(), self.gamma_intensidade.get(),
                                                                              self.img_out)).grid_configure(row = 5, column = 5, padx = 4)


        # Logaritmo

        # cnt
        self.text_logaritmo= ctk.CTkLabel(self.tabTransform.tab("Intensidade"), text = "Logaritmo:", font = ("arial", 15))
        self.text_logaritmo.grid_configure(row = 6, column = 0, pady = 10, padx= 4)

        self.text_logaritmo_cnst= ctk.CTkLabel(self.tabTransform.tab("Intensidade"), text = "Constante:", font = ("arial", 15))
        self.text_logaritmo_cnst.grid_configure(row = 6, column = 3, pady = 10, padx= 4)

        self.logaritmo_cnst = DoubleVar(self.tabTransform.tab("Intensidade"))
        self.entry_logaritmo_cnst = ctk.CTkEntry(self.tabTransform.tab("Intensidade"), textvariable = self.logaritmo_cnst).grid_configure(row = 6, column = 4, pady = 4)

        self.btn_logaritmo = ctk.CTkButton(self.tabTransform.tab("Intensidade"), text = "Aplicar", 
                                                 command = lambda: gerarGamma(self.imgA, float(self.gamma_cnst.get()), float(self.gamma_intensidade.get()),
                                                                              self.img_out)).grid_configure(row = 6, column = 5, padx = 4, pady = 30)

    def preencherAbaGeometrica(self):
        # Ampliação

        # F
        self.text_ampliacao= ctk.CTkLabel(self.tabTransform.tab("Geométrica"), text = "Ampliação:", font = ("arial", 15))
        self.text_ampliacao.grid_configure(row = 0, column = 0, pady = 10, padx= 4)

        self.text_ampliacao_f= ctk.CTkLabel(self.tabTransform.tab("Geométrica"), text = "Fator:", font = ("arial", 15))
        self.text_ampliacao_f.grid_configure(row = 0, column = 3, pady = 10, padx= 4)

        self.ampliacao_f= IntVar(self.tabTransform.tab("Geométrica"))
        self.entry_ampliacao_f = ctk.CTkEntry(self.tabTransform.tab("Geométrica"), textvariable = self.ampliacao_f).grid_configure(row = 0, column = 4, pady = 4)

        self.btn_ampliacao = ctk.CTkButton(self.tabTransform.tab("Geométrica"), text = "Aplicar", 
                                                 command = lambda: gerarAmpliacao(self.imgA, self.ampliacao_f.get(), self.img_out))
        
        self.btn_ampliacao.grid_configure(row = 0, column = 5, padx = 4, pady = 30)
        
        # Reducao
        
        # F
        self.text_reducao= ctk.CTkLabel(self.tabTransform.tab("Geométrica"), text = "Redução:", font = ("arial", 15))
        self.text_reducao.grid_configure(row = 1, column = 0, pady = 10, padx= 4)

        self.text_reducao_f= ctk.CTkLabel(self.tabTransform.tab("Geométrica"), text = "Fator:", font = ("arial", 15))
        self.text_reducao_f.grid_configure(row = 1, column = 3, pady = 10, padx= 4)

        self.reducao_f = IntVar(self.tabTransform.tab("Geométrica"))
        self.entry_reducao_f = ctk.CTkEntry(self.tabTransform.tab("Geométrica"), textvariable = self.reducao_f).grid_configure(row = 1, column = 4, pady = 4)

        self.btn_reducao = ctk.CTkButton(self.tabTransform.tab("Geométrica"), text = "Aplicar", 
                                                 command = lambda: gerarReducao_media(self.imgA, self.reducao_f.get(), self.img_out))
        
        self.btn_reducao.grid_configure(row = 1, column = 5, padx = 4, pady = 30)

        # Cisalhamento
        
        # S
        self.text_cisalhamento= ctk.CTkLabel(self.tabTransform.tab("Geométrica"), text = "Cisalhamento:", font = ("arial", 15))
        self.text_cisalhamento.grid_configure(row = 2, column = 0, pady = 10, padx= 4)

        self.text_cisalhamento_s= ctk.CTkLabel(self.tabTransform.tab("Geométrica"), text = "s:", font = ("arial", 15))
        self.text_cisalhamento_s.grid_configure(row = 2, column = 3, pady = 10, padx= 4)

        self.cisalhamento_s= IntVar(self.tabTransform.tab("Geométrica"))
        self.entry_cisalhamento_s = ctk.CTkEntry(self.tabTransform.tab("Geométrica"), textvariable = self.cisalhamento_s).grid_configure(row = 2, column = 4, pady = 4)

        self.btn_cisalhamento = ctk.CTkButton(self.tabTransform.tab("Geométrica"), text = "Aplicar", 
                                                 command = lambda: gerarCisalhamento(self.imgA, self.cisalhamento_s.get(), self.img_out))
        
        self.btn_cisalhamento.grid_configure(row = 2, column = 5, padx = 4, pady = 30)

        # Rebatimento
        
        self.text_rebatimento= ctk.CTkLabel(self.tabTransform.tab("Geométrica"), text = "Rebatimento:", font = ("arial", 15))
        self.text_rebatimento.grid_configure(row = 3, column = 0, pady = 10, padx= 4)

        self.btn_rebatimento = ctk.CTkButton(self.tabTransform.tab("Geométrica"), text = "Aplicar", 
                                                 command = lambda: gerarRebatimento(self.imgA, self.img_out))
        
        self.btn_rebatimento.grid_configure(row = 3, column = 5, padx = 4, pady = 30)

        # Rotacionar
        
        # graus
        self.text_rotacionar= ctk.CTkLabel(self.tabTransform.tab("Geométrica"), text = "Rotacionar:", font = ("arial", 15))
        self.text_rotacionar.grid_configure(row = 4, column = 0, pady = 10, padx= 4)

        self.text_rotacionar_graus= ctk.CTkLabel(self.tabTransform.tab("Geométrica"), text = "Graus:", font = ("arial", 15))
        self.text_rotacionar_graus.grid_configure(row = 4, column = 3, pady = 10, padx= 4)

        self.rotacionar_graus= IntVar(self.tabTransform.tab("Geométrica"))
        self.entry_rotacionar_graus = ctk.CTkEntry(self.tabTransform.tab("Geométrica"), textvariable = self.rotacionar_graus).grid_configure(row = 4, column = 4, pady = 4)

        self.btn_rotacionar = ctk.CTkButton(self.tabTransform.tab("Geométrica"), text = "Aplicar", 
                                                 command = lambda: gerarRotacionar(self.imgA, self.rotacionar_graus.get(), self.img_out))
        
        self.btn_rotacionar.grid_configure(row = 4, column = 5, padx = 4, pady = 30)

        # Pinch

        # r_min, r_max, s_min, s_max
        self.text_vars_pinch = ctk.CTkLabel(self.tabTransform.tab("Geométrica"), text = "Variáveis = tipo ('vertical' ou 'bordas') | ip | Fmax:", font = ("arial", 15))
        self.text_vars_pinch.grid_configure(row = 5, column = 0, pady = 4, padx= 4, columnspan = 3)

        # Txt Alargamento de contraste
        self.text_pinch = ctk.CTkLabel(self.tabTransform.tab("Geométrica"), text = "Pinch:", font = ("arial", 20))
        self.text_pinch.grid_configure(row = 6, column = 0, pady = 4, padx= 4)
        
        
        
        self.tipo = StringVar(self.tabTransform.tab("Geométrica")) 
        self.ip = IntVar(self.tabTransform.tab("Geométrica"))
        self.f_max = DoubleVar(self.tabTransform.tab("Geométrica"))
        
        self.text_tipo = ctk.CTkEntry(self.tabTransform.tab("Geométrica"), textvariable = self.tipo).grid_configure(row = 6, column = 1, padx = 4)
        
        self.text_ip = ctk.CTkEntry(self.tabTransform.tab("Geométrica"), textvariable = self.ip).grid_configure(row = 6, column = 2, padx = 4)
        self.text_f_max = ctk.CTkEntry(self.tabTransform.tab("Geométrica"), textvariable = self.f_max).grid_configure(row = 6, column = 3, padx = 4)
        
        self.btn_pinch = ctk.CTkButton(self.tabTransform.tab("Geométrica"), text = "Aplicar", 
                                                 command = lambda: gerarPinch(self.imgA, self.tipo.get(), self.ip.get(), self.f_max.get(), self.img_out))
        
        self.btn_pinch.grid_configure(row = 6, column = 5, padx = 4, pady = 30)

    def preencherAbaHistograma(self):
        # Expansão de histograma

        self.text_space= ctk.CTkLabel(self.tabview.tab("Histograma"), text = "", font = ("arial", 20)).pack(pady = 20)
        
        self.text_expansao= ctk.CTkLabel(self.tabview.tab("Histograma"), text = "Expansão de histograma:", font = ("arial", 20)).pack(pady = 30)
        #self.text_expansao.grid_configure(row = 0, column = 0, pady = 10, padx = 4)

        self.btn_expansao = ctk.CTkButton(self.tabview.tab("Histograma"), text = "Aplicar", 
                                                 command = lambda: gerarExpansao_histograma(self.imgA, self.img_out)).pack()
        
        #self.btn_expansao.grid_configure(row = 0, column = 5, padx = 4, pady = 30)


        # Equalização de histograma

        self.text_equalizacao= ctk.CTkLabel(self.tabview.tab("Histograma"), text = "Equalização de histograma:", font = ("arial", 20)).pack(pady = 30)
        #self.text_equalizacao.grid_configure(row = 1, column = 0, pady = 10, padx = 4)

        self.btn_equalizacao = ctk.CTkButton(self.tabview.tab("Histograma"), text = "Aplicar", 
                                                 command = lambda: gerarEqualizacao_histograma(self.imgA, self.img_out)).pack()
        
        #self.btn_equalizacao.grid_configure(row = 1, column = 5, padx = 4, pady = 30)

    def preencherAbaControle_contraste_adaptativo(self):
        self.text_controle_cont_adap = ctk.CTkLabel(self.tabview.tab("Controle de contraste adaptativo"), text = "Controle contraste adaptativo:", font = ("arial", 30))
        self.text_controle_cont_adap.pack(pady = 50)

        # c, n
        self.adap_cnst = DoubleVar(self.tabview.tab("Controle de contraste adaptativo"))
        self.text_adap_cnst = ctk.CTkLabel(self.tabview.tab("Controle de contraste adaptativo"), text = "Constante:", font = ("arial", 20)).pack()
        self.entry_adap_cnst = ctk.CTkEntry(self.tabview.tab("Controle de contraste adaptativo"), textvariable = self.adap_cnst).pack(pady = 20)

        self.adap_n = IntVar(self.tabview.tab("Controle de contraste adaptativo"))
        self.text_n = ctk.CTkLabel(self.tabview.tab("Controle de contraste adaptativo"), text = "Tamanho da matriz:", font = ("arial", 20)).pack()
        self.entry_adap_n = ctk.CTkEntry(self.tabview.tab("Controle de contraste adaptativo"), textvariable = self.adap_n).pack(pady = 20)

        self.btn_adap = ctk.CTkButton(self.tabview.tab("Controle de contraste adaptativo"), text = "Aplicar", 
                                      command = lambda: gerarControle_adaptativo_contraste(self.imgA, self.adap_cnst.get(), self.adap_n, self.img_out)).pack()

    def preencherAbaFiltragem(self):
        self.text_space1 = ctk.CTkLabel(self.tabview.tab("Filtragem"), text = "", font = ("arial", 30)).pack(pady = 20)

        # n
        self.text_media= ctk.CTkLabel(self.tabview.tab("Filtragem"), text = "Média:", font = ("arial", 30)).pack(pady = 10)
        #self.text_media.grid_configure(row = 0, column = 0, pady = 10, padx= 4)

        self.text_media_n= ctk.CTkLabel(self.tabview.tab("Filtragem"), text = "Tamanho da matriz n x n ('n' somente):", font = ("arial", 20)).pack(pady = 10)
        #self.text_media_n.grid_configure(row = 0, column = 3, pady = 10, padx= 4)

        self.media_n = IntVar(self.tabview.tab("Filtragem"))
        self.entry_media_n = ctk.CTkEntry(self.tabview.tab("Filtragem"), textvariable = self.media_n).pack(pady = 10)
        #.grid_configure(row = 0, column = 4, pady = 4)

        self.btn_media = ctk.CTkButton(self.tabview.tab("Filtragem"), text = "Aplicar", 
                                                 command = lambda: gerarFiltragem_media(self.imgA, self.media_n.get(), self.img_out)).pack(pady = 10)
        
        #self.btn_media.grid_configure(row = 0, column = 5, padx = 4, pady = 30)

        self.text_space2 = ctk.CTkLabel(self.tabview.tab("Filtragem"), text = "", font = ("arial", 30)).pack(pady = 20)

        # n
        self.text_mediana= ctk.CTkLabel(self.tabview.tab("Filtragem"), text = "Mediana:", font = ("arial", 30)).pack(pady = 10)
        
        self.text_mediana_n= ctk.CTkLabel(self.tabview.tab("Filtragem"), text = "Tamanho da matriz n x n ('n' somente):", font = ("arial", 20)).pack(pady = 10)
       
        self.mediana_n = IntVar(self.tabview.tab("Filtragem"))
        self.entry_mediana_n = ctk.CTkEntry(self.tabview.tab("Filtragem"), textvariable = self.mediana_n).pack(pady = 10)

        self.btn_mediana = ctk.CTkButton(self.tabview.tab("Filtragem"), text = "Aplicar", 
                                                 command = lambda: gerarFiltragem_mediana(self.imgA, self.mediana_n.get(), self.img_out)).pack(pady = 10)

    def preencherAbaBordasHighBoost(self):
        self.text_space = ctk.CTkLabel(self.tabview.tab("Bordas e High boost"), text = "", font = ("arial", 15))
        self.text_space.grid_configure(row = 0, column = 0, pady = 80)

        self.text_sobel = ctk.CTkLabel(self.tabview.tab("Bordas e High boost"), text = "Gradiente de Sobel:", font = ("arial", 15))
        self.text_sobel.grid_configure(row = 1, column = 0, pady = 10, padx= 4)

        self.btn_sobel = ctk.CTkButton(self.tabview.tab("Bordas e High boost"), text = "Aplicar", 
                                                 command = lambda: gerarSobel(self.imgA, self.img_out))
        
        self.btn_sobel.grid_configure(row = 1, column = 5, padx = 4, pady = 30)
        
        # Aguçamento de bordas

        
        self.text_aguc_bordas = ctk.CTkLabel(self.tabview.tab("Bordas e High boost"), text = "Aguçamento de bordas:", font = ("arial", 15))
        self.text_aguc_bordas.grid_configure(row = 2, column = 0, pady = 10, padx= 4)

        self.text_aguc_bordas_k = ctk.CTkLabel(self.tabview.tab("Bordas e High boost"), text = "Constante:", font = ("arial", 15))
        self.text_aguc_bordas_k.grid_configure(row = 2, column = 3, pady = 10, padx= 4)

        # k
        self.aguc_bordas_k = DoubleVar(self.tabview.tab("Bordas e High boost"))
        self.entry_laguc_bordas_k = ctk.CTkEntry(self.tabview.tab("Bordas e High boost"), textvariable = self.aguc_bordas_k).grid_configure(row = 2, column = 4, pady = 4)

        self.btn_logaritmo = ctk.CTkButton(self.tabview.tab("Bordas e High boost"), text = "Aplicar", 
                                                 command = lambda: gerarAgucamento_bordas(self.imgA, self.aguc_bordas_k.get(), self.img_out))
        
        self.btn_logaritmo.grid_configure(row = 2, column = 5, padx = 4, pady = 30)


        # High boost

        self.text_h_boost = ctk.CTkLabel(self.tabview.tab("Bordas e High boost"), text = "High boost:", font = ("arial", 15))
        self.text_h_boost.grid_configure(row = 3, column = 0, pady = 10, padx= 4)

        self.text_h_boost_k = ctk.CTkLabel(self.tabview.tab("Bordas e High boost"), text = "Constante:", font = ("arial", 15))
        self.text_h_boost_k.grid_configure(row = 3, column = 3, pady = 10, padx= 4)

        # k
        self.h_boost_k = DoubleVar(self.tabview.tab("Bordas e High boost"))
        self.entry_h_boost_k = ctk.CTkEntry(self.tabview.tab("Bordas e High boost"), textvariable = self.h_boost_k).grid_configure(row = 3, column = 4, pady = 4)

        self.btn_h_boost = ctk.CTkButton(self.tabview.tab("Bordas e High boost"), text = "Aplicar", 
                                                 command = lambda: gerarHigh_boost(self.imgA, self.h_boost_k.get(), self.img_out))
        
        self.btn_h_boost.grid_configure(row = 3, column = 5, padx = 4, pady = 30)

    def preencherAbaConvolucao(self):
        self.text_space = ctk.CTkLabel(self.tabview.tab("Convolução"), text = " ", font = ("arial", 12))
        self.text_space.grid_configure(row = 0, column = 0, pady = 4)

        self.text_convolucao = ctk.CTkLabel(self.tabview.tab("Convolução"), text = "Matriz para convolucao:", font = ("arial bold", 20))
        self.text_convolucao.grid_configure(row = 1, column = 0, columnspan = 2, pady = 20)
        
        
        
        self.kernel_a = IntVar(self.tabview.tab("Convolução"))

        if self.kernel_a == "": return
        self.kernel_b = IntVar(self.tabview.tab("Convolução"))
        self.kernel_c = IntVar(self.tabview.tab("Convolução"))

        self.kernel_d = IntVar(self.tabview.tab("Convolução"))
        self.kernel_e = IntVar(self.tabview.tab("Convolução"))
        self.kernel_f = IntVar(self.tabview.tab("Convolução"))

        self.kernel_g = IntVar(self.tabview.tab("Convolução"))
        self.kernel_h = IntVar(self.tabview.tab("Convolução"))
        self.kernel_i = IntVar(self.tabview.tab("Convolução"))
        
        # Linha 1 da matriz
        self.text_kernel_a = ctk.CTkEntry(self.tabview.tab("Convolução"), textvariable = self.kernel_a).grid_configure(row = 2, column = 0)
        self.text_kernel_b = ctk.CTkEntry(self.tabview.tab("Convolução"), textvariable = self.kernel_b).grid_configure(row = 2, column = 1)
        self.text_kernel_c = ctk.CTkEntry(self.tabview.tab("Convolução"), textvariable = self.kernel_c).grid_configure(row = 2, column = 2, padx = 80)

        # Linha 2 da matriz
        self.text_kernel_d = ctk.CTkEntry(self.tabview.tab("Convolução"), textvariable = self.kernel_d).grid_configure(row = 3, column = 0)
        self.text_kernel_e = ctk.CTkEntry(self.tabview.tab("Convolução"), textvariable = self.kernel_e).grid_configure(row = 3, column = 1)
        self.text_kernel_f = ctk.CTkEntry(self.tabview.tab("Convolução"), textvariable = self.kernel_f).grid_configure(row = 3, column = 2, padx = 80)

        # Linha 3 da matriz
        self.text_kernel_g = ctk.CTkEntry(self.tabview.tab("Convolução"), textvariable = self.kernel_g).grid_configure(row = 4, column = 0)
        self.text_kernel_h = ctk.CTkEntry(self.tabview.tab("Convolução"), textvariable = self.kernel_h).grid_configure(row = 4, column = 1)
        self.text_kernel_i = ctk.CTkEntry(self.tabview.tab("Convolução"), textvariable = self.kernel_i).grid_configure(row = 4, column = 2, padx = 80)
        

        self.btn_convolucao = ctk.CTkButton(self.tabview.tab("Convolução"), text = "Aplicar", 
                                            command = lambda: gerarConvolucao(self.imgA, self.kernel_a.get(), self.kernel_b.get(), self.kernel_c.get(), self.kernel_d.get(), 
                                                                                   self.kernel_e.get(), self.kernel_f.get(), self.kernel_g.get(), self.kernel_h.get(), 
                                                                                   self.kernel_i.get(), self.img_out))
        
        self.btn_convolucao.grid_configure(row = 5, column = 1, padx = 4, pady = 30)
        
Application()