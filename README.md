![Henry](henry.jpg)
# Proyecto-SteamGames-FastAPI 🎮

El proyecto de MLOps de Steam Games consiste en un sistema de Machine Learning centrado en el análisis y predicción del precio de los juegos disponibles en la plataforma Steam. Utilizando técnicas de Data Science, se procesa y explora un conjunto de datos que incluye diversas características de los juegos, como género, fecha de lanzamiento, metascore, entre otras.

## Flujo de trabajo 🔨💻

El flujo de trabajo del proyecto se enfoca en primera instancia en un proceso de transformación y preprocesamiento de los datos para conformar unas funciones que se detallarán más adelante. 

Luego, se pasa a la etapa del modelo de Machine Learning, empezando con un EDA y posterior construcción, entrenamiento y despliegue de distintos modelos de Machine Learning para predecir el precio de los juegos en base a las características proporcionadas hasta conseguir el que mejor score nos ofrece. Para ello, se emplean librerías populares como pandas, seaborn, scikit-learn y joblib.

El sistema se despliega como una API utilizando FastAPI, lo que permite a los usuarios interactuar con el modelo mediante solicitudes HTTP. La API ofrece una funcionalidad para obtener recomendaciones de géneros de juegos más populares para un año específico y, a su vez, utiliza el modelo entrenado para predecir el precio de los juegos basado en las características proporcionadas.

Además, el proyecto implementa técnicas de MLOps para garantizar la reproducibilidad y mantenibilidad del modelo. Se utilizan entornos virtuales para aislar las dependencias, y se automatizan las tareas de entrenamiento y despliegue mediante scripts y comandos de terminal.


### 1) [Preprocesamiento de datos](https://github.com/flormiranda1/Proyecto-steamgames/blob/main/Transformaciones_funciones.ipynb) ✂️📄

Los datos utilizados en este proyecto provienen de un archivo json que contiene información de la plataforma Steam [Dataset](https://github.com/flormiranda1/Proyecto-steamgames/blob/main/steam_games.json)
* Se trató la columna precio de manera especial, tratando de perder la menor cantidad de datos posibles ya que será nuestra variable dependiente en el modelo de predicción. Se reemplazaron los juegos "Free, Free to Play, etc" por precio 0 y el se eliminaron nulos que no era un número significativo de datos
* En esta primera instancia se reemplazaron los nulos de la columna metascore por ceros para no eliminar gran cantidad de datos, luego en el EDA se decidirá como tratarlos
* Misma situación para la columna sentiment, se agruparon aquellos registros con pocos reviews en una sola característica "pocos reviews" y a los nulos se los dejó como "unknown" momentáneamente para no perder información.
* Se eliminaron los nulos de la columna release_date ya que no eran significantes y no sería posible asignarle una fecha específica. Además se hizo el cambio a tipo de dato datetime y se agregó una columna año que tendrá más utilidad a futuro en nuestro análisis.
* Se eliminaron duplicados de las columnas title y id ya que no sería valioso tener información repetida de un mismo videojuego.
* Se hizo una comparación entre la columna title vs. app_name y publisher vs. developer para ver con cual quedarnos, ya que a simple vista parecían tener información similar.
* Se eliminaron columnas innecesarias.
[Dataset modificado](https://github.com/flormiranda1/Proyecto-steamgames/blob/main/steam_games_limpio.csv)


### 2) [Funciones de recomendación](https://github.com/flormiranda1/Proyecto-steamgames/blob/main/Transformaciones_funciones.ipynb) 

Se crean algunas funciones de recomendación en base a un año que el usuario ingresa en relación a diferentes variables:
📌 Top 5 de géneros según el año que se ingrese

📌 Lista de juegos en el año que se ingrese

📌 Juegos con early_access en el año que se ingrese

📌 Distribución de sentimientos en el año ingresado

📌 Top 5 de juegos con mayor metascore en el año solicitado


### 3) MLOps: modelo de predicción de precios 🔎📊

* Se realizan algunos ajustes más al dataset, eliminando columnas innecesarias para el modelo de predicción y se realizan análisis exploratirios con la ayuda de diagramas como el pairplot, heatmap e histogramas. Se abordan dos enfoques diferentes para el tratamiento de los datos en busca de conseguir los mejores resultados (score y rmse) en el modelo de ML.

El [primer enfoque](https://github.com/flormiranda1/Proyecto-steamgames/blob/main/An%C3%A1lisis%20EDA%20y%20ML%201.ipynb) pensado para este dataset fue eliminar los nulos (ceros) de metascore ya que pensando intuitivamente, la gente cuando decide dar una puntuación, es generalmente cuando se lleva una mala experiencia y los que no dan puntuación, es más probable que su experiencia haya sido regular o buena, por lo que no podemos imputar ceros en los scores vacios. En base a estos datos restantes, se aplica además una codificación label_encoder para la columna genres y se entrenan 3 modelos diferentes de regresión para ver sus resultados (DecisionTreeRegressor, RandomForest, GradientBoostingRegressor y GreedSearch en los dos últimos). Además se entrenan los modelos probando dos alternativas extras (balancear los datos de early_access o no hacerlo). Se obtienen muy buenos resultados con el dataset de datos balanceados en el feature early_access y sin el balanceo de datos los resultados siguen siendo aceptables.

El [segundo enfoque](https://github.com/flormiranda1/Proyecto-steamgames/blob/main/An%C3%A1lisis%20EDA%20y%20ML%202.ipynb) utilizado es conservar todos los nulos de metascore (guardados como ceros) y además usar el método de get dummies para la separación de la columna genres. Se realiza todo el proceso EDA y el entrenamiento de tres modelos diferentes (DecisionTreeRegressor, RandomForest, GradientBoostingRegressor y GreedSearch en los dos últimos). No se consiguen buenos resultados, scores muy bajos y rmse en el rango de 10 y 15.

Se decide tomar como [modelo definitivo](https://github.com/flormiranda1/Proyecto-steamgames/blob/main/modelo.ipynb) el RandomForest con los mejores hiperparámetros arrojados por el GridSearch. Se alimenta en nuestra API desde el [dataset reducido](https://github.com/flormiranda1/Proyecto-steamgames/blob/main/df_modelo_entrenado.csv) luego del proceso EDA.


## Links útiles 📎🎬 
- Repositorio (Github): https://github.com/flormiranda1/Proyecto-steamgames
- Deploy del Proyecto (Render): https://fastapi-steam-games.onrender.com/docs#/
- Video explicativo (Drive): https://drive.google.com/file/d/1APmfoHEV_t3M7IVa-lC1199Npne_axDE/view?usp=sharing
