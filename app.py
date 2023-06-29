#################################################################
from flask import Flask, render_template, redirect, request
import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.models import model_from_json
#################################################################

app = Flask(__name__)

#################################################################
json_path = 'models/inception_v3.json'
model_path = 'models/inception_v3.h5'

classe = ""
arquivo_path = "../static/img/placeholder.jpg"

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
#################################################################

@app.route('/')
def index():
    return render_template('index.html', pagina="Dashboard")

@app.route('/index.html')
def home():
    return redirect('/')

@app.route('/classificacao.html')
def classificacao():

    global classe

    return render_template('classificacao.html', pagina="Classificação", predict=classe, imagem_upload=arquivo_path)

def ler_imagem(filename):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_altura, img_altura)) # redimensiona
    img = img_to_array(img) # array numpy
    img = np.expand_dims(img, axis = 0) # formato de um tensor
    img = preprocess_input(img) # entradas no padrão da rede
    return img

#Entensões permitidas png, jpg e jpeg
EXTENSOES = set(['jpg' , 'jpeg' , 'png'])
def formatos_permitidos(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in EXTENSOES

@app.route('/predict', methods=['POST'])
def predict():
    global classe, arquivo_path, model
    arquivo = request.files['arquivo']
    if arquivo and formatos_permitidos(arquivo.filename): # Verifica o formato da imagem
        nome_arquivo = arquivo.filename
        arquivo_path = os.path.join('static/imagens', nome_arquivo)
        arquivo.save(arquivo_path)
        img = ler_imagem(arquivo_path) #pré-processamento
        prediction = model.predict(img) #predição
        temp = prediction
        prediction = (prediction > 0.5).astype(np.uint8)
        if prediction[[0]] == 1:
            classe = f"Classe: Positiva ({round(temp[0][0]*100,2)}%)"
        else:
            classe = f"Classe: Negativa ({round((1-temp[0][0])*100,2)}%)"

    return redirect("/classificacao.html")

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)