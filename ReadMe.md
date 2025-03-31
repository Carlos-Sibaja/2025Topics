Ventajas clave de usar Playwright para este caso específico
Necesidad	Playwright	Otras librerías
Soporte para páginas dinámicas (JavaScript)	✅ Excelente	❌ requests + BeautifulSoup no lo procesan
Headless + control total del navegador	✅ Controla Chromium, Firefox, Webkit	Selenium también lo hace pero es más pesado y lento
Fácil extracción de <a> dinámicos	✅ Detecta contenido que aparece después de cargar JS	❌ requests solo ve HTML inicial
Multi-tab, multi-page, manejo de wait_for_timeout()	✅ Sencillo y robusto	Selenium lo puede hacer pero requiere más código
Instalación de dependencias	✅ Solo playwright install	Selenium necesita instalar manualmente drivers como chromedriver o geckodriver

---

## Recomendaciones de Nikhil

1. No utilizar linear Regression, pero no hacerlo muy complicado porque sera menos preciso
2. La prediccion debe ser mejor que 50%
3. Obtener las noticias, analizarlas, sacar el sentiment, y luego obtener el stock market (NASDAQ) para tener una mejor variabilidad.
4. Con eso poder analizar el cambio en el mercado
5. Como pruebas, obtener las noticias de los ultimos 2-3 anos, entrenar el modelo
6. Por otro lado obtener los stocks para el mismo periodo, entrenar el modelo solo con stocks
7. Luego comparar si al juntar las noticias y stocks el modelo mejora o no