#################################################################
from flask import Flask, render_template, redirect, request
from datetime import datetime
import os
import joblib
import cv2
import numpy as np
import pandas as pd
import skimage
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import model_from_json
#################################################################

app = Flask(__name__)

#################################################################
json_path = 'models/inception_v3.json'
model_path = 'models/inception_v3.h5'

classes = []
diretorio_base = "static/imagens/"
pasta_lamina = ""
pasta_existente = False
houve_classificacao = False
img_invalidas = 0
img_removidas = 0
df = pd.DataFrame(columns=['nome_imagem', 'classe_predicao', 'porcentagem_predicao'])
filenames=[]

df_laminas = pd.DataFrame(columns=['nome_lamina', 'classe_predicao', 'data_atualizacao'])
#df_laminas.to_csv(diretorio_base + "dataset.csv", index=False)
df_laminas = pd.read_csv(diretorio_base + "dataset.csv")

img_altura = 299
img_largura = 299

# Lendo o modelo do arquivo JSON
with open(json_path, 'r') as json_file:
    json_modelo_salvo = json_file.read()

# leitura do modelo
model = model_from_json(json_modelo_salvo)

# leitura dos pesos
model.load_weights(model_path)
# print(model.summary())

# Carrrega o modelo do RandomForest para classificação das features do GLCM
# Carrega o modelo salvo a partir do arquivo
loaded_model = joblib.load('models/random_forest_model.joblib')
#################################################################

def resetar_variaveis():
    global classes, pasta_lamina, pasta_existente, img_invalidas, img_removidas, df, filenames

    classes = []
    pasta_lamina = ""
    pasta_existente = False
    houve_classificacao = False
    img_invalidas = 0
    img_removidas = 0
    df = pd.DataFrame(columns=['nome_imagem', 'classe_predicao', 'porcentagem_predicao'])
    filenames = []

@app.route('/')
def index():
    return render_template('index.html', pagina="Dashboard")

@app.route('/index.html')
def home():
    return redirect('/')

@app.route('/dataset')
def dataset():
    return render_template('dataset.html', pagina="Dataset", dataset=df_laminas)

@app.route('/dataset/<lamina>')
def dataset_imagens(lamina):
    global diretorio_base, df

    caminho_csv = f'{diretorio_base + lamina + "/" + lamina}.csv'
    df = pd.read_csv(caminho_csv)  # ler o CSV

    return render_template('imagens_laminas.html', pagina="Dataset", lamina=lamina, dataset=df)

@app.route('/adiciona_lamina.html')
def adiciona_lamina():
    global pasta_existente

    return render_template('adiciona_lamina.html', pagina="Classificação", pasta_existente=pasta_existente)

@app.route('/adiciona_lamina')
def inicia_lamina():
    resetar_variaveis()
    return redirect("/adiciona_lamina.html")

def consulta_predicao_dataset(lamina):
    num_linhas = df_laminas.shape[0]
    predicao_existente = "Indefinida"

    if(num_linhas>0):
        # Verificando a predição com base no nome da lâmina
        predicao_existente = df_laminas.loc[df_laminas['nome_lamina'] == lamina, 'classe_predicao'].values[0]

    return predicao_existente

def atualiza_predicao_dataset(lamina):

    predicao_existente = consulta_predicao_dataset(lamina)

    # Só atualiza se não for positiva
    if not predicao_existente == "Positiva":
        # Verificando se existe classe positiva no DataFrame
        existe_classe_positiva = (df['classe_predicao'] == 'Positiva').any()

        if existe_classe_positiva:
            # Alterando a predição da lâmina para positiva
            df_laminas.loc[df_laminas['nome_lamina'] == lamina, 'classe_predicao'] = "Positiva"
        else:
            # Alterando a predição da lâmina para negativo
            df_laminas.loc[df_laminas['nome_lamina'] == lamina, 'classe_predicao'] = "Negativa"

        # alterando a data
        df_laminas.loc[df_laminas['nome_lamina'] == lamina, 'data_atualizacao'] = datetime.now()

        df_laminas.to_csv(diretorio_base + "dataset.csv", index=False)

def filtrar_dataframe():
    global df

    # Filtrando apenas as predições positivas
    df_positivas = df[df['classe_predicao'] == 'Positiva']

    dez_maiores = None #inicializa variável

    # Selecionando as dez maiores porcentagens de predições
    if (df_positivas.shape[0] > 0):
        dez_maiores = df_positivas.sort_values(by='porcentagem_predicao', ascending=False).head(10)
    else:
        dez_maiores = df.sort_values(by='porcentagem_predicao', ascending=False).head(10)

    if (dez_maiores.shape[0] == 0):
        nova_linha = {
            'nome_imagem': "../static/img/placeholder.jpg",
            'classe_predicao': "",
            'porcentagem_predicao': ""
        }
        dez_maiores = dez_maiores.append(nova_linha, ignore_index=True)

    return dez_maiores

@app.route('/classificacao.html')
def classificacao():

    global classes, img_removidas, img_invalidas, pasta_lamina, houve_classificacao

    if pasta_lamina == "":
        return inicia_lamina()

    dez_maiores = filtrar_dataframe() # seleciona as 10 maiores predições positivas
    num_linhas = dez_maiores.shape[0]

    # Verificando a predição com base no nome da lâmina
    lamina=pasta_lamina.split("/")[-1]
    predicao_existente = consulta_predicao_dataset(lamina)

    return render_template('classificacao.html', pagina="Classificação", houve_classificacao=houve_classificacao, lamina=lamina, predicao_existente=predicao_existente, dataset=dez_maiores, num_linhas=num_linhas, img_removidas=img_removidas, img_invalidas=img_invalidas)

@app.route('/classificacao')
def inicia_classificacao():
    resetar_variaveis()
    return inicia_lamina()

@app.route('/classificacao/<lamina>')
def classificacao_lamina(lamina):
    global pasta_lamina

    resetar_variaveis()
    pasta_lamina = diretorio_base + lamina

    return redirect("/classificacao.html")

def ler_imagem():
    global filenames

    imagens = []
    for filename in filenames:
        img = load_img(filename, target_size=(img_altura, img_largura))
        img = img_to_array(img)  # array numpy
        img = np.expand_dims(img, axis=0)  # formato de um tensor
        img = preprocess_input(img)  # entradas no padrão da rede
        imagens.append(img)

    return imagens

#Entensões permitidas png, jpg e jpeg
EXTENSOES = set(['jpg', 'jpeg', 'png', 'tif'])
def formatos_permitidos(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in EXTENSOES

# Extrai características com o GLCM
def calculate_glcm_features(image):
    # Converta a imagem para escala de cinza
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calcule a matriz GLCM
    glcm = skimage.feature.graycomatrix(gray_image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

    # Calcule as propriedades da matriz GLCM
    contrast = skimage.feature.graycoprops(glcm, 'contrast')
    dissimilarity = skimage.feature.graycoprops(glcm, 'dissimilarity')
    homogeneity = skimage.feature.graycoprops(glcm, 'homogeneity')
    energy = skimage.feature.graycoprops(glcm, 'energy')
    correlation = skimage.feature.graycoprops(glcm, 'correlation')

    return contrast, dissimilarity, homogeneity, energy, correlation

# Verifica se a imagem é de microscopia, com base no calculo GLCM. Remove a imagem caso não seja de microscopia
def verifica_imagem_upload():
    global filenames, img_removidas
    imagens = filenames.copy()
    img_removidas = 0

    for filename in imagens:
        image = cv2.imread(filename)
        features = calculate_glcm_features(image) # Extrai as características GLCM da imagem
        features = np.reshape(features, (1, -1)) # Redimensiona as características para que tenham a mesma forma usada durante o treinamento
        prediction = loaded_model.predict(features) # Faz a predição usando o modelo treinado
        print("Predição Ramdom Forest: ", prediction)

        # Classificar com base no limiar
        if prediction >= 0.5: # limiar
            continue
        else:
            print("A imagem [{}] não é uma imagem de microscopia.".format(filename))
            filenames.remove(filename) # remove a imagem da lista
            os.remove(filename) # remove a imagem da pasta
            img_removidas+=1

@app.route('/add_lamina', methods=['POST'])
def add_lamina():

    global pasta_lamina, pasta_existente, df, df_laminas

    nome_lamina = request.form['nome_lamina']
    pasta_lamina = diretorio_base + nome_lamina.upper()
    caminho_csv = f'{pasta_lamina + "/" + pasta_lamina.split("/")[-1]}.csv'

    if os.path.isdir(pasta_lamina):  # pasta ja existe
        pasta_existente = True
        df = pd.read_csv(caminho_csv) # ler o CSV
        return redirect("/adiciona_lamina.html")
    else:
        os.mkdir(pasta_lamina)  # cria a pasta
        df.to_csv(caminho_csv, index=False) # cria o CSV

        # Adiciona uma linha no CSV de dataset
        df_laminas = df_laminas.append({
            'nome_lamina': nome_lamina.upper(),
            'classe_predicao': "Indefinida", #lâmina não classificada
            'data_atualizacao': datetime.now()
        }, ignore_index=True)
        df_laminas.to_csv(diretorio_base + "dataset.csv", index=False) # escreve em disco

        pasta_existente = False
        return redirect("/classificacao.html")

@app.route('/predict', methods=['POST'])
def predict():
    global classes, filenames, model, img_invalidas, pasta_lamina, df, houve_classificacao

    arquivos = request.files.getlist('arquivo')  # Obtem uma lista de arquivos enviados

    filenames = []
    img_invalidas = 0
    for arquivo in arquivos:
        if arquivo and formatos_permitidos(arquivo.filename):  # Verifica o formato da imagem
            nome_arquivo = arquivo.filename
            arquivo_path = os.path.join(pasta_lamina, nome_arquivo)
            arquivo.save(arquivo_path)
            filenames.append(arquivo_path)
        else:
            img_invalidas += 1 # formato inválido

    print(filenames)

    # Verifica se a imagem é de microscopia, com base no calculo GLCM. Remove a imagem caso não seja de microscopia
    verifica_imagem_upload()

    imagens = ler_imagem()  # Pré-processamento das imagens

    classe=""
    porcentagem=""
    nova_linha={}
    for i, img in enumerate(imagens):
        prediction = model.predict(img)  # Predição
        temp = prediction
        prediction = (prediction > 0.5).astype(np.uint8)
        if prediction[[0]] == 1:
            classe = "Positiva"
            porcentagem = round(temp[0][0] * 100, 2)
            #classe = f"Classe: Positiva ({porcentagem}%)"
        else:
            classe = "Negativa"
            porcentagem = round((1 - temp[0][0]) * 100, 2)

        # Dicionário com os dados da nova linha a ser adicionada
        nova_linha = {
            'nome_imagem': filenames[i],
            'classe_predicao': classe,
            'porcentagem_predicao': porcentagem
        }

        # Adicionando a nova linha ao DataFrame
        df = df.append(nova_linha, ignore_index=True)

    # Exibindo o DataFrame atualizado
    print(df)

    # atualiza predição da lâmina no dataset
    lamina = pasta_lamina.split("/")[-1]
    atualiza_predicao_dataset(lamina)

    # Exportar predições dos campos de lâmina para um arquivo CSV
    df.to_csv(f'{pasta_lamina + "/" + lamina}.csv', index=False)

    houve_classificacao = True

    return redirect('/classificacao.html')




if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)