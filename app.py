#################################################################
from flask import Flask, render_template, redirect, request
import os
import joblib
import cv2
import numpy as np
import skimage
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import model_from_json
#################################################################

app = Flask(__name__)

#################################################################
json_path = 'models/inception_v3.json'
model_path = 'models/inception_v3.h5'

classes = ""
img_invalidas = 0
img_removidas = 0
filenames = ["../static/img/placeholder.jpg"]

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

@app.route('/')
def index():
    return render_template('index.html', pagina="Dashboard")

@app.route('/index.html')
def home():
    return redirect('/')

@app.route('/classificacao.html')
def classificacao():

    global classes, filenames, img_removidas, img_invalidas

    return render_template('classificacao.html', pagina="Classificação", predict=classes, img_removidas=img_removidas, img_invalidas=img_invalidas, len_filenames=len(filenames), filenames=zip(filenames, range(0, len(filenames))))

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
            filenames.remove(filename)
            img_removidas+=1

@app.route('/predict', methods=['POST'])
def predict():
    global classes, filenames, model, img_invalidas

    arquivos = request.files.getlist('arquivo')  # Obtem uma lista de arquivos enviados

    filenames = []
    img_invalidas = 0
    for arquivo in arquivos:
        if arquivo and formatos_permitidos(arquivo.filename):  # Verifica o formato da imagem
            nome_arquivo = arquivo.filename
            arquivo_path = os.path.join('static/imagens', nome_arquivo)
            arquivo.save(arquivo_path)
            filenames.append(arquivo_path)
        else:
            img_invalidas += 1 # formato inválido

    print(filenames)

    # Verifica se a imagem é de microscopia, com base no calculo GLCM. Remove a imagem caso não seja de microscopia
    verifica_imagem_upload()

    imagens = ler_imagem()  # Pré-processamento das imagens

    classes = []
    for img in imagens:
        prediction = model.predict(img)  # Predição
        temp = prediction
        prediction = (prediction > 0.5).astype(np.uint8)
        if prediction[[0]] == 1:
            classe = f"Classe: Positiva ({round(temp[0][0] * 100, 2)}%)"
        else:
            classe = f"Classe: Negativa ({round((1 - temp[0][0]) * 100, 2)}%)"
        classes.append(classe)
    print(classes)

    # Caso não tenha nenhuma imagem na lista
    if len(filenames) == 0:
        filenames.append("../static/img/placeholder.jpg")

    return redirect("/classificacao.html")

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)