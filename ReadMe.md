Ventajas clave de usar Playwright para este caso específico
Necesidad	Playwright	Otras librerías
Soporte para páginas dinámicas (JavaScript)	✅ Excelente	❌ requests + BeautifulSoup no lo procesan
Headless + control total del navegador	✅ Controla Chromium, Firefox, Webkit	Selenium también lo hace pero es más pesado y lento
Fácil extracción de <a> dinámicos	✅ Detecta contenido que aparece después de cargar JS	❌ requests solo ve HTML inicial
Multi-tab, multi-page, manejo de wait_for_timeout()	✅ Sencillo y robusto	Selenium lo puede hacer pero requiere más código
Instalación de dependencias	✅ Solo playwright install	Selenium necesita instalar manualmente drivers como chromedriver o geckodriver

---

## Recomendaciones de Nikhil

1. No utilizar linear Regression, pero no hacerlo muy complicado porque sera menos preciso (Random Forest)
2. La prediccion debe ser mejor que 50%
3. Obtener las noticias, analizarlas, sacar el sentiment, y luego obtener el stock market (NASDAQ) para tener una mejor variabilidad.
4. Con eso poder analizar el cambio en el mercado
5. Como pruebas, obtener las noticias de los ultimos 2-3 anos, entrenar el modelo
6. Por otro lado obtener los stocks para el mismo periodo, entrenar el modelo solo con stocks
7. Luego comparar si al juntar las noticias y stocks el modelo mejora o no




## Optimization of the Regression Model.

#### Round 1.
#### ===== Indicadores de Calidad para Random Forest =====
MSE (Error Cuadrático Medio): 206080.0311
MAE (Error Absoluto Medio): 333.9989
R² (Varianza Explicada): 0.7340
Accuracy Direccional (DA): 40.68%

#### ===== Indicadores de Calidad para XGBoost =====
MSE (Error Cuadrático Medio): 140075.6627
MAE (Error Absoluto Medio): 297.7426
R² (Varianza Explicada): 0.8192
Accuracy Direccional (DA): 40.68%


- Run1 Feature Aggregation: add new features like moving averages or lags to the dataset.
- Run2 Normalization: apply standard scaling to features, even though normalization usually doesn't affect tree models.
- Run3 Hyperparameter Tuning: adjusting parameters.

### VERSION 5 RUN 1 
===== Dataset Loaded =====
Comparison of Model Performance Metrics:
       Run1_Random       Run1_XGB    Run2_Random       Run2_XGB    Run3_Random       Run3_XGB
MSE  163878.034513  133087.179486  206080.031094  140075.662670  172516.136579  401325.344544
MAE     307.618926     284.699384     333.998888     297.742585     313.014860     544.001688
R2        0.788510       0.828246       0.734047       0.819228       0.777362       0.482076
DA       44.827586      48.275862      41.379310      41.379310      41.379310      44.827586

Comparison of Overfitting-Prevention Model Performance Metrics:
           XGB_Reg  XGB_Reg_Early      XGB_DART
MSE  142680.927422   2.699171e+06  3.175299e+06
MAE     295.395028   1.552046e+03  1.675053e+03
R2        0.815865  -2.483373e+00 -3.097833e+00
DA       44.827586   4.482759e+01  4.655172e+01