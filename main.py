# Importações
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from flask import Flask, jsonify, render_template
import os   
from dotenv import load_dotenv
from datetime import datetime, timedelta
import base64
from io import BytesIO
import pickle   

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Classe para Coleta de Dados
class ColetorDeDados:
    """Classe para coleta de dados históricos de ações."""

    def __init__(self, ticker, data_inicio, data_fim):
        self.ticker = ticker
        self.data_inicio = data_inicio
        self.data_fim = data_fim

    def coletar_dados_historicos(self):
        """Coleta dados históricos de fechamento de ações usando a biblioteca yfinance."""
        dados = yf.download(self.ticker, start=self.data_inicio, end=self.data_fim)
        dados = dados[['Close']]
        return dados

    def coletar_dados_tempo_real(self):
        dados = yf.download(self.ticker, start=self.data_inicio, end=self.data_fim)
        dados = dados[['Close']]
        return dados


# Funções de Pré-processamento
def explorar_dados(dados):
    """Explora os dados com estatísticas descritivas e visualizações."""
    print(dados.describe())
    plt.figure(figsize=(14, 7))
    plt.plot(dados['data_pregao'], dados['fechamento'], label='Preço de Fechamento')
    plt.xlabel('Data')
    plt.ylabel('Preço')
    plt.title('Preço de Fechamento ao Longo do Tempo')
    plt.legend()
    plt.show()

# Função das janelas temporais
def cria_janelas_temporais(data, time_steps=30):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps, 0])
        y.append(data[i+time_steps, 0])
    return np.array(X), np.array(y)


def pre_processar_dados(dados):

    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(dados)

    X, Y = cria_janelas_temporais(df_scaled)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Divisão dos dados de treino e teste
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
    
    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test.shape)

    return X_train, X_test, Y_train, Y_test, scaler

# Funções de Modelagem
def treinar_modelo(X_treino, y_treino):
    """Treina o modelo LSTM."""

    model = Sequential()
    model.add(LSTM( units=100,
                    activation='relu',
                #  return_sequences=True,
                    input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))

    #model.add(LSTM(units=100, activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Treinamento
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    model.fit(X_treino,
            y_treino,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0)
 
    return model



def avaliar_modelo(modelo, X_test, Y_test, scaler):
    """Avalia o modelo."""

    Y_pred = modelo.predict(X_test, verbose=0)
    Y_test_rescaled = scaler.inverse_transform(np.column_stack((np.zeros_like(Y_teste), Y_teste)))[:, 1]
    Y_pred_rescaled = scaler.inverse_transform(np.column_stack((np.zeros_like(Y_pred), Y_pred)))[:, 1]

    loss = modelo.evaluate(X_test, Y_test, verbose=0)
    mae = mean_absolute_error(Y_test_rescaled, Y_pred_rescaled)
    rmse = np.sqrt(root_mean_squared_error(Y_test_rescaled, Y_pred_rescaled))
    mape = mean_absolute_percentage_error(Y_test_rescaled, Y_pred_rescaled)*100
    
    resultados = {
        "loss": loss,
        "mae": mae,
        "rmse": rmse,
        "mape": mape}
   
    print(f'Loss: {loss}')
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {(mape):.2f}%")

    return resultados


def gerar_grafico_historico(previsao=None):
    """Gera o gráfico de histórico de preços com a previsão de amanhã."""
    dados = ler_dados_rds("goog_historico")
    if dados is None or dados.empty:
        return None  # Retorna None em caso de erro ou dados vazios

    plt.figure(figsize=(10, 5))
    plt.plot(dados['data_pregao'], dados['fechamento'], label='Preço de Fechamento', color='blue')
    
    # Adicionar o ponto da previsão de amanhã, se disponível
    if previsao:
        plt.scatter(datetime.now() + timedelta(days=1), previsao, color='red', label='Previsão de Amanhã')
    
    plt.xlabel('Data')
    plt.ylabel('Preço (USD)')
    plt.title('Histórico de Preços de GOOG com Previsão')
    plt.legend()
    plt.grid(True)

    # Salvar o gráfico em um buffer na memória
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_url = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()

    return plot_url

def gerar_grafico_previsao():
    """Obtém a previsão do preço de amanhã."""
    ticker = "KO"
    hoje = datetime.now().strftime("%Y-%m-%d")
    coletor = ColetorDeDados(ticker, hoje, hoje)
    dados_recentes = coletor.coletar_dados_tempo_real()

    if dados_recentes.empty:
        return None, "Não há dados recentes para gerar a previsão."

    previsao_response, status_code = prever_acao()

    if status_code != 200:
        return None, f"Erro ao obter a previsão: {previsao_response.get_json()['mensagem']}"

    previsao = previsao_response.get_json()['previsao']
    return None, f"Previsão para amanhã: ${previsao:.2f}"

# API Flask

app = Flask(__name__)

# Importa as rotas

@app.route('/prever', methods=['POST']) 
def prever_acao():
    """Realiza a previsão do valor da ação com base nos dados fornecidos."""
    
    # Coletar os dados da ação em tempo real
    ticker = "KO"
    coletor = ColetorDeDados(ticker, datetime.now().strftime("%Y-%m-%d"), datetime.now().strftime("%Y-%m-%d"))
    dados_reais = coletor.coletar_dados_tempo_real()

    # Validar os dados de entrada
    if dados_reais.empty:
        return jsonify({"mensagem": "Não foram encontrados dados diarios da Ação"}), 400


    dados_reais['media_movel'] = dados_reais['fechamento'].rolling(window=20).mean()
    dados_reais['retorno_diario'] = dados_reais['fechamento'].pct_change()
    dados_reais.dropna(inplace=True)
  
    # # Verifica se o modelo e o scaler estão carregados
    with open('modelo.pkl', 'rb') as arquivo_modelo:
        modelo = pickle.load(arquivo_modelo)
    with open('scaler.pkl', 'rb') as arquivo_scaler:
        scaler = pickle.load(arquivo_scaler)

    # Certifique-se de que a ordem e os nomes das chaves correspondem às features do modelo
    features = ['fechamento', 'volume', 'media_movel', 'retorno_diario']

    # Extrair os últimos valores das features
    input_data = [dados_reais[feature].iloc[-1] for feature in features]

    # Verificar se algum valor está ausente
    if any(pd.isna(value) for value in input_data):
        return jsonify({"mensagem": "Dados de entrada incompletos"}), 400
    
    # Transformar os dados de entrada usando o mesmo scaler usado no treinamento
    input_scaled = scaler.transform([input_data])

    # Realizar a previsão
    previsao = modelo.predict(input_scaled)

    # Inverter a escala da previsão para a escala original
    previsao_original = scaler.inverse_transform(input_scaled)[:, 0][0]

    return jsonify({"previsao": previsao_original}), 200


@app.route('/treinarmodelo', methods=['GET'])
def coletar_dados_treinar():
    print("Iniciando o treinamento do modelo...")
    
    # Coletar e processar dados históricos até o dia anterior
    ticker = "KO"
    data_inicio = "2010-01-01"
    data_fim = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"Coletando dados de {ticker} de {data_inicio} a {data_fim}...")

    coletor = ColetorDeDados(ticker, data_inicio, data_fim)
    dados = coletor.coletar_dados_historicos()
    
    # Explorar e pré-processar dados
    #explorar_dados(dados)
    X_treino, X_teste, y_treino, y_teste, scaler = pre_processar_dados(dados)
    
    # Treinar e avaliar o modelo
    modelo = treinar_modelo(X_treino, y_treino)
    resultados_avaliacao = avaliar_modelo(modelo, X_teste, y_teste, scaler)

    os.makedirs("Models", exist_ok=True)
    modelo.save(f"Models/modelKO.keras")
    joblib.dump(scaler, f'Models/scalerKO.pkl')


    return jsonify("Dados coletados e modelo treinado!"), 200


@app.route("/")
def dashboard():
    """Rota principal do dashboard."""
    grafico_previsao, mensagem_previsao = gerar_grafico_previsao()
    grafico_historico = gerar_grafico_historico(previsao=float(mensagem_previsao.split('$')[1]))

    return render_template(
        "dashboard.html",
        grafico_historico=grafico_historico,
        mensagem_previsao=mensagem_previsao
    )

# Exemplo de Uso (para teste)
if __name__ == '__main__':

    # Iniciar a API
    app.run(debug=False, use_reloader=False)