from funciones import *
import warnings
from statsmodels.tools.sm_exceptions import ValueWarning

warnings.filterwarnings("ignore", category=ValueWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Paths
PATHS = {
    "X": "Datos/X.csv",
    "Y": "Datos/Y.csv",
    "Z": "Datos/Z.csv"
}

def main(PATHS):
    # --------1: Carga--------
    dfs = {
        "X": cargar_y_normalizar_csv(PATHS["X"]),
        "Y": cargar_y_normalizar_csv(PATHS["Y"], sep=";", decimal=",", dayfirst=True),
        "Z": cargar_y_normalizar_csv(PATHS["Z"])
    }

    # --------2: EDA--------
    for k, df in dfs.items():
        resumen_estadistico(df, k)
        plot_serie_temporal(df, k)

    boxplots_precios([dfs["X"], dfs["Y"], dfs["Z"]], ["X", "Y", "Z"])

    # --------3: Split --------
    splits = {}
    for k, df in dfs.items():
        train, test = division_train_test(df)
        splits[k] = {"train": train, "test": test}

    # --------4: Modelos--------
    modelos = {}
    preds = {}
    rmses = {}

    for k in ["X", "Y", "Z"]:
        train_df = splits[k]["train"]
        test_df = splits[k]["test"]

        entrada = train_df["price"]
        h = len(test_df)

        arma = entrenar_arma(entrada)
        arima = entrenar_arima(entrada)
        sarima = entrenar_sarima(entrada)

        pred_arma = forecast_model(arma, h)
        pred_arima = forecast_model(arima, h)
        pred_sarima = forecast_model(sarima, h)

        modelos[k] = {"arma": arma, "arima": arima, "sarima": sarima}
        preds[k] = {"arma": pred_arma, "arima": pred_arima, "sarima": pred_sarima}

        rmses[k] = {
            "arma": rmse(test_df["price"], pred_arma),
            "arima": rmse(test_df["price"], pred_arima),
            "sarima": rmse(test_df["price"], pred_sarima),
        }

        print(f"RMSE {k}:",
              rmses[k]["arma"],
              rmses[k]["arima"],
              rmses[k]["sarima"])

        # Visualizaciones
        plot_train_test_pred(train_df, test_df, pred_arma, f"ARMA - Materia Prima {k}")
        plot_train_test_pred(train_df, test_df, pred_arima, f"ARIMA - Materia Prima {k}")
        plot_train_test_pred(train_df, test_df, pred_sarima, f"SARIMA - Materia Prima {k}")

    # --------5: Reentrenar SARIMA con datos históricos--------
    forecasts_full = {}
    historicos = {}

    for k, df in dfs.items():
        serie_full = df.sort_values("date").set_index("date")["price"]
        sarima_full = entrenar_sarima(serie_full)

        h = 360  # 1 año
        forecast = sarima_full.forecast(steps=h)

        historicos[k] = df.sort_values("date")[["date", "price"]]
        forecasts_full[k] = {
            "serie_full": serie_full,
            "forecast": forecast,
            "df_forecast": construir_df_forecast(serie_full, forecast)
        }

        plot_train_test_pred(
            historicos[k],
            forecasts_full[k]["df_forecast"],
            forecasts_full[k]["df_forecast"]["price"],
            f"SARIMA (reentrenado) - Materia Prima {k}"
        )

    # --------6: Escenarios--------
    esc = {
        k: escenarios_desde_forecast(forecasts_full[k]["forecast"])
        for k in ["X", "Y", "Z"]
    }

    resultados_costos = costos_equipos_por_escenario(
        esc["X"], esc["Y"], esc["Z"]
    )

    df_resultados = (
        pd.DataFrame(resultados_costos)
        .T.reset_index()
        .rename(columns={"index": "escenario"})
    )

    return df_resultados


if __name__ == "__main__":
    resultados = main(PATHS)
    print(resultados)
