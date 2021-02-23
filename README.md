# Data Science

En el repositorio se encuentran varios proyectos realizados con el objetivo de estudiar distintos tipos de algoritmos de Machine Learning y Deep Learning, aplicando diferentes modelos y técnicas para su visualización, así como análisis estadísticos y descripción de datasets.

*Nota: Todos los análisis realizados se encuentran en inglés ya que fueron compartidos en la plataforma Kaggle para recibir feedback.*

* [Perfil de Kaggle](https://www.kaggle.com/agustinpugliese)

* [Perfil de LinkedIn](https://www.linkedin.com/in/agust%C3%ADnpugliese7/)

* Contacto: agustin.pugliese@hotmail.com

## Contenido

### Aprendizaje Supervisado

### :moneybag: [Predicción de compra de un producto.](https://nbviewer.jupyter.org/github/AgusTP/Data-Science/blob/master/Purchase%20classification%20algorithms/Social%20network%20product%20purchase.ipynb)
*Algoritmos de clasificación.* 
Se realizó un análisis exploratorio de datos para evaluar la posibilidad de compra o no de un producto publicitado en una red social, teniendo en cuenta características como edad, salario y género de una persona. Luego se utilizaron distintos modelos de clasificación, comparando métricas para evaluar cuál de todos presenta la mayor precisión, analizando también las curvas CAP y ROC y sus áreas bajo la curva.

### :car: [Precio de venta de autos.](https://nbviewer.jupyter.org/github/AgusTP/Data-Science/blob/master/Audi_price_Kaggle_Task/Audi_Kaggle_price%2896%25%20Score%29.ipynb)
*Algoritmos de regresión. Feature selection.*
El dataset contiene distintos modelos de autos de la marca Audi, así como su año, tipo de transmisión y tipo de combustible, entre otros. Primero se hizo un análisis exploratorio de datos y luego se compararon distintos modelos de regresión (Lineal, Lasso, SVR, Decision Tree, entre otros) para alcanzar el mayor valor de precisión. Se utilizó el método SelectKBest de sklearn para poder obtener un número acorde de variables a utilizar en la predicción. 

### :notebook: [Admisión a beca.](https://nbviewer.jupyter.org/github/AgusTP/Data-Science/blob/master/Admission%20prediction/Admission%20Prediction.ipynb)
*Algoritmos de regresión. Model tuning.* 
EDA y visualización del dataset, para poder predecir el porcentaje de admisión a una beca dadas notas de diversos exámenes. Utilización de K-Fold Cross Validation para comparar y tener una mejor información sobre la precisión del modelo y utilización de GridSearch para obtener los mejores parámetros de los modelos predictivos.

### :tropical_fish: [Predicción del peso de distintas especies.](https://nbviewer.jupyter.org/github/AgusTP/Data-Science/blob/master/Fish%20weight%20analysis/Fish_weight_prediction.ipynb)
*Algoritmos de regresión.*
Se hizo un análisis visual del set de datos, que contiene información sobre dimensiones de diferentes especies de peces. Luego se comprobaron las hipótesis que permiten trabajar con un modelo sencillo de regresión lineal múltiple (tendencia lineal con la variable a predecir, distribución normal de las variables, multicolinealidad, homocedasticidad y autocorrelación) y se compararon los resultados con un modelo polinómico y con uno más robusto ensamblado (Random Forest). También se hizo un análisis para descartar valores outliers con el método IQR.

### Aprendizaje no supervisado

### :barber: [Clustering de compradores.](https://nbviewer.jupyter.org/github/AgusTP/Data-Science/blob/master/Clustering%20comparison/Mall%20Customers%20Clustering.ipynb)
*Algoritmos de Clustering.*
EDA y comparación de distintos algoritmos de clustering (K-Means, Hierarchical, Affinity Propagation y DBSCAN) para obtener resultados y relaciones sobre el dataset. Todas las visualizaciones 2D y 3D fueron realizadas con Plotly para mejorar la experiencia de visualización de las conclusiones.

### Deep Learning

### :bank: [Clasificación con Red Neuronal.](https://nbviewer.jupyter.org/github/AgusTP/Data-Science/blob/master/NN%20classification/ANN%20.ipynb)
*Neural Network.*
Análisis exploratorio de datos y visualizaciones del set, que contiene información sobre clientes o ex clientes de un banco europeo. Explicación detallada de todos los pasos para crear la red neuronal profunda: sus capas de entrada, salida y ocultas, función de optimización, de pérdida e inicialización de parámetros. Análisis de la precisión del modelo y curvas ROC y AUC.

### NLP

### :fork_and_knife: [Reseñas de restaurant.](https://nbviewer.jupyter.org/github/AgusTP/Data-Science/blob/master/NLP%20reviews/NLP%20review%20analysis.ipynb)
*Sentiment Analysis.*
Análisis de distintas valoraciones de un restaurant, utilizando el método bag of words para clasificar las reseñas en positivas o negativas, realizando los procedimientos de tokenización, normalización y limpieza del corpus. Gráfico de unigramas, bigramas, trigramas y sentimientos para las reseñas tanto positivas como negativas.

### Reglas de asociación

### - :hamburger: [Apriori para sets de compras.](https://nbviewer.jupyter.org/github/AgusTP/Data-Science/blob/master/Association%20rule%20learning/Association%20Rules.ipynb)
Utilización del algoritmo Apriori para generar una lista de elementos frecuentes. El dataset incluye distintas transacciones de un supermercado, de las cuales se quiere generar conclusiones para preparar posibles estrategias comerciales de venta con ayuda del modelo. Este algoritmo, tanto como Eclat y FP-growth son herramientas fundamentales en sistemas de recomendación de productos.

### EDAs

### :movie_camera: [Análisis de películas.](https://nbviewer.jupyter.org/github/AgusTP/Data-Science/blob/master/Movie%20EDA%20and%20visualizations/Movie%20EDA%20and%20visualizations.ipynb)
Se hizo un análisis exploratorio completo con diferentes técnicas y módulos de visualización de gráficos para poder obtener conclusiones acerca del set, que contiene información sobre distintos títulos lanzados así como director, año de estreno, actor principal, entre otros. 

### :headphones: [Spotify top 50.](https://nbviewer.jupyter.org/github/AgusTP/Data-Science/blob/master/Spotify_top_50/Spotify%20top%2050%20songs%20EDA.ipynb)
Análisis exploratorio de datos y distintas visualizaciones para poder sacar conclusiones acerca del dataset, que contiene una lista con las 50 canciones más escuchadas en Spotify, con sus autores y popularidades entre otros.

**Herramientas:** Numpy, Pandas, Matplotlib, Seaborn, Sklearn, Keras, XGBoost, Statsmodels, Geopandas, NLTK, re, NRCLex, Mlxtend, Plotly, WordCloud, Warnings.
