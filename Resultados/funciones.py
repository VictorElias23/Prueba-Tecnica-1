#Importar bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Carga y Normalización, se especifica el separador y el decimal de los números
def cargar_y_normalizar_csv(path, sep=",", decimal=".", dayfirst=False):
    df = pd.read_csv(path, sep=sep, decimal=decimal) #Lectura de csv
    df.columns = df.columns.str.lower() #minúsculas

    if not {"date", "price"}.issubset(df.columns):
        raise ValueError(f"El archivo {path} no contiene las columnas date o price.")

    #Ordenamiento de columnas por facilidad
    df = df[["date", "price"]]
    
    #Transformamos la columna date a datetime
    df["date"] = pd.to_datetime(df["date"], dayfirst=dayfirst, errors="coerce")
    #Transformamos la columna price a numérico
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df

#Análisis descriptivo
#Resumen de los datos
def resumen_estadistico(df, nombre=""):
    print(f"\nResumen estadístico {nombre}")
    estadistica = df["price"].describe()
    print(estadistica)
    rango = df["price"].max() - df["price"].min()
    print(f"Rango (max - min): {rango:.2f}")
    return estadistica

#Boxplots
def boxplots_precios(dfs, etiquetas, titulo="Distribución de precios"):
    data = [df["price"].dropna() for df in dfs]
    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=etiquetas, showfliers=True)
    plt.title(titulo)
    plt.ylabel("Precio")
    plt.show()

#Serie de tiempo
def plot_serie_temporal(df, nombre=""):
    df_plot = df.sort_values("date")
    plt.figure(figsize=(12, 4))
    plt.plot(df_plot["date"], df_plot["price"])
    plt.title(f"Precio Materia Prima {nombre}")
    plt.xlabel("Fecha")
    plt.ylabel("Precio")
    plt.tight_layout()
    plt.show()


#Split de los datos para el entrenamiento de modelos (80%, 20%)
def division_train_test(df, train_size=0.8):
    df = df.sort_values("date") #Ordenamos los datos en orden de la fecha
    n = len(df)
    split_idx = int(n * train_size)
    train = df.iloc[:split_idx] #train
    test = df.iloc[split_idx:] #test
    return train, test


#Métricas y forecast
from sklearn.metrics import mean_squared_error

def forecast_model(model, steps):
    return model.forecast(steps=steps)

#Raíz del error cuadrático medio
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

#Modelo ARMA
def entrenar_arma(train_prices, order=(1,0,1)):
    return ARIMA(train_prices, order=order).fit()

#Modelo ARIMA
def entrenar_arima(train_prices, order=(1,1,1)):
    return ARIMA(train_prices, order=order).fit()

#Modelo SARIMA
def entrenar_sarima(train_prices, order=(1,1,1), seasonal_order=(1,1,1,12)):
    return SARIMAX(
        train_prices,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)

#Visualización de la serie de tiempo con los datos train, test y pred
def plot_train_test_pred(train_df, test_df, pred, titulo=""):
    plt.figure(figsize=(12, 4))
    plt.plot(train_df["date"], train_df["price"], label="Train")
    plt.plot(test_df["date"], test_df["price"], label="Test")
    plt.plot(test_df["date"], pred, label="Predicción")
    plt.title(titulo)
    plt.xlabel("Fecha")
    plt.ylabel("Precio")
    plt.legend()
    plt.tight_layout()
    plt.show()

#Construcción de df para poder una visualización de la serie de tiempo
def construir_df_forecast(serie_full, forecast):
    last_date = serie_full.index.max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(forecast), freq="D")
    return pd.DataFrame({"date": future_dates, "price": forecast.values})

#Construcción de los 3 escenarios posibles
def escenarios_desde_forecast(forecast):
    valores = np.array(forecast)
    return {
        "optimista": np.percentile(valores, 25),
        "promedio": np.mean(valores),
        "conservador": np.percentile(valores, 75)
    }

#Costos base finales con los 3 escenarios posibles
def costos_equipos_por_escenario(esc_x, esc_y, esc_z):
    resultados = {}
    for escenario in ["optimista", "promedio", "conservador"]:
        px, py, pz = esc_x[escenario], esc_y[escenario], esc_z[escenario]
        resultados[escenario] = {
            "equipo_1": 0.2 * px + 0.8 * py,
            "equipo_2": (px + py + pz) / 3
        }
    return resultados
