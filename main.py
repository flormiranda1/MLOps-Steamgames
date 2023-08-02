from fastapi import FastAPI
import pandas as pd

app = FastAPI(title="Proyecto I: predicción precio steamgames",description="API carrera data science Henry")

# Global variables to store the data
df = None

# Load data on startup
@app.on_event("startup")
async def load_data():
    global df

    # Load the data from the CSV file
    df = pd.read_csv('steam_games_limpio.csv')
    
# Function that returns the top 5 genres for a given year
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
    
    return {"year": year, "top_genres": top_genres}

# Function that returns a list of games released in a given year
@app.get("/juegos/")
async def juegos(year: int):
    # Filtrar el DataFrame para obtener solo los datos del año específico
    df_year = df[df["año"] == int(year)]
    
   # Crear un diccionario para almacenar los juegos
    juegos_dict = {}
    
    # Iterar sobre el DataFrame y agregar los juegos al diccionario
    for index, juego in df_year['title'].items():
        juegos_dict[index] = juego
    
    return juegos_dict

@app.get("/specs/")
async def specs(year: int):
    # Filtrar el DataFrame para obtener solo los datos del año específico
    df_year = df[df["año"] == int(year)]
    
    # Crear un diccionario para contar la cantidad de veces que aparece cada género
    spec_counts = {}
    for specs_list in df_year['specs']:
        for spec in specs_list:
            spec_counts[spec] = spec_counts.get(spec, 0) + 1
    
    # Ordenar el diccionario por los valores (cantidad de ventas) en orden descendente
    sorted_specs = sorted(spec_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Obtener los 5 géneros más vendidos
    top_specs = [spec for spec, _ in sorted_specs[:5]]
    
    return {"year": year, "top_specs": top_specs}

@app.get("/early_access/")
async def early_access(year: int):
    # Filtrar el DataFrame para obtener solo los datos del año específico y que tengan Early Access
    df_year_early_access = df[(df["año"] == int(year)) & (df["early_access"] == True)]
    
    # Obtener la cantidad de juegos con Early Access
    cantidad_juegos_early_access = len(df_year_early_access)
    
    return {"year": year, "early_access": cantidad_juegos_early_access}

@app.get("/sentiment/")
async def sentiment(year: int):
    # Filtrar el DataFrame para obtener solo los datos del año específico
    df_year = df[df["año"] == int(year)]
    
    # Contar la cantidad de registros para cada categoría de sentimiento y lo convertimos en un diccionario
    sentiment_counts = df_year['sentiment'].value_counts().to_dict()
    
    return sentiment_counts

@app.get("/metascore/")
async def metascore(year: int):
    # Filtrar el DataFrame para obtener solo los datos del año específico
    df_year = df[df["año"] == int(year)]
    
    # Ordenar el DataFrame por el metascore en orden descendente
    df_sorted = df_year.sort_values(by='metascore', ascending=False)
    
    # Obtener los 5 juegos con mayor metascore
    top_5_games = df_sorted.head(5)["title"].to_list()
    
    return {"year": year, "metascore": top_5_games}