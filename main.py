from fastapi import FastAPI
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import ast



app = FastAPI(title="Proyecto I: predicción precio steamgames",description="API carrera data science Henry")

# Variables globales para almacenar la data
df = None
rmse_train_static = None
df_modelo_entrenado = None
tree_model = None
label_encoder = None

# Load data on startup
@app.on_event("startup")
async def load_data_and_model():
    global df, df_modelo_entrenado, tree_model, label_encoder,rmse_train_static

    # Cargar los datos del CSV usado para las funciones, resultado del ETL
    df = pd.read_csv('steam_games_limpio.csv')

    # Cargar los datos del CSV usado para el entrenamiento del modelo, resultado del EDA
    df_modelo_entrenado = pd.read_csv('df_modelo_entrenado.csv')

    # Cargar el LabelEncoder usado para los genres desde el archivo .pkl
    label_encoder = joblib.load('label_encoder.pkl')

    # Feature engineering
    feature_cols = ["early_access","genres_encoded","metascore","año"]
    X = df_modelo_entrenado[feature_cols]
    y = df_modelo_entrenado["price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    tree_model = DecisionTreeRegressor(max_depth=20, min_samples_leaf=1, min_samples_split=2)

    tree_model.fit(X_train, y_train)

    y_train_pred = tree_model.predict(X_train)
    y_test_pred = tree_model.predict(X_test)

    rmse_train_static = mean_squared_error(y_train, y_train_pred, squared=False)
    
# Función que retorna el top 5 the generos para un año dado
@app.get("/genres/")
async def genres(year: int):
    # Filtrar el DataFrame para obtener solo los datos del año específico
    df_year = df[df["año"] == year]
    
    # Crear un diccionario para contar la cantidad de veces que aparece cada género
    genre_counts = {}
    for genres_list in df_year['genres']:
        for genre in genres_list.split(','):
            genre_counts[genre.strip()] = genre_counts.get(genre.strip(), 0) + 1
    
    # Ordenar el diccionario por los valores (cantidad de ventas) en orden descendente
    sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Obtener los 5 géneros más vendidos
    top_genres = [genre for genre, _ in sorted_genres[:5]]
    
    # Crear un diccionario para el resultado final
    result_dict = {}
    for genre in top_genres:
        # Obtener la cantidad de juegos de cada género en el año específico
        count = genre_counts.get(genre, 0)
        result_dict[genre] = count
    
    return result_dict

# Función que retorna una lista de juegos lanzados en un año dado
@app.get("/juegos/")
async def juegos(year: int):
    # Filtrar el DataFrame para obtener solo los datos del año específico
    df_year = df[df["año"] == int(year)]

    # Obtener la lista de juegos del año específico
    juegos_list = df_year['title'].tolist()

    juegos_dict = {year: juegos_list}

    return juegos_dict


# Función que retorna una lista con las 5 especificaciones de juegos que más se vendieron
@app.get("/specs/")
async def specs(year: int):
    # Filtrar el DataFrame para obtener solo los datos del año específico
    df_year = df[df["año"] == int(year)]
    
    # Crear un diccionario para contar la cantidad de veces que aparece cada género
    spec_counts = {}
    for specs_list in df_year['specs']:
        try:
            specs_list = ast.literal_eval(specs_list)  # Convertir la cadena a una lista
        except (ValueError, SyntaxError):
            continue  # Omitir el elemento si no es una lista válida
        
        for spec in specs_list:
            spec_counts[spec] = spec_counts.get(spec, 0) + 1
    
    # Ordenar el diccionario por los valores (cantidad de ventas) en orden descendente
    sorted_specs = sorted(spec_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Obtener los 5 specs más vendidos
    top_specs = [spec for spec, _ in sorted_specs[:5]]
    
    return {"top_specs": top_specs}


# Función que retorna la cantidad de juegos que tuvieron early access en un año dado
@app.get("/earlyacces/")
async def early_access(year: int):
    # Filtrar el DataFrame para obtener solo los datos del año específico y que tengan Early Access
    df_year_early_access = df[(df["año"] == int(year)) & (df["early_access"] == True)]
    
    # Obtener la cantidad de juegos con Early Access
    cantidad_juegos_early_access = len(df_year_early_access)
    
    return {"year": year, "early_access": cantidad_juegos_early_access}

# Función que retorna la cantidad de registros que se encuentren categorizados con un análisis de sentimiento para 
# el año dado
@app.get("/sentiment/")
async def sentiment(year: int):
    # Filtrar el DataFrame para obtener solo los datos del año específico
    df_year = df[df["año"] == int(year)]
    
    # Contar la cantidad de registros para cada categoría de sentimiento y lo convertimos en un diccionario
    sentiment_counts = df_year['sentiment'].value_counts().to_dict()
    
    return sentiment_counts

# Función que retorna los 5 juegos con mayor metascore para el año dado
@app.get("/metascore/")
async def metascore(year: int):
    # Filtrar el DataFrame para obtener solo los datos del año específico
    df_year = df[df["año"] == int(year)]
    
    # Ordenar el DataFrame por el metascore en orden descendente
    df_sorted = df_year.sort_values(by='metascore', ascending=False)
    
    # Obtener los 5 juegos con mayor metascore
    top_5_games = df_sorted.head(5)["title"].to_list()
    
    return {"top5_metascore": top_5_games}

# Función que predice el precio de un juego según los parámetros dados por el usuario
@app.get("/prediction/")
async def prediction(genre: str, early_access: bool, metascore: int, year: int):
    # Verificar que el género ingresado esté presente en el LabelEncoder
    if genre not in label_encoder.classes_:
        genres_list = ", ".join(label_encoder.classes_)
        print(f"Error: El género '{genre}' no está presente en el dataset.")
        print(f"Los géneros disponibles son: {genres_list}")
        return None, None
    
    # Obtener el valor codificado del género usando el LabelEncoder
    genre_encoded = label_encoder.transform([genre])[0]
    
    # Verificar que el metascore ingresado esté presente en el dataset
    if metascore not in df_modelo_entrenado["metascore"].unique():
        metascores_list = ", ".join(map(str, df_modelo_entrenado["metascore"].unique()))
        print(f"Error: El metascore '{metascore}' no está presente en el dataset.")
        print(f"Los metascores disponibles son: {metascores_list}")
        return None, None
    
    # Verificar que el año ingresado esté presente en el dataset
    if year not in df_modelo_entrenado["año"].unique():
        min_year = df_modelo_entrenado["año"].min()
        max_year = df_modelo_entrenado["año"].max()
        print(f"Error: El año '{year}' no está presente en el dataset.")
        print(f"El rango de años disponibles es de {min_year} a {max_year}.")
        return None, None
    
    # Crear un DataFrame con las características ingresadas
    data = pd.DataFrame({
        "early_access": [early_access],
        "genres_encoded": [genre_encoded],
        "metascore": [metascore],
        "año": [year]
    })
    
    # Realizar la predicción del precio utilizando el modelo entrenado
    price_pred = tree_model.predict(data)[0]
    
    return {"predicción_de_precio": price_pred, "rmse": rmse_train_static}

