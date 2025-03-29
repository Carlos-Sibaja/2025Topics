Ventajas clave de usar Playwright para este caso específico
Necesidad	Playwright	Otras librerías
Soporte para páginas dinámicas (JavaScript)	✅ Excelente	❌ requests + BeautifulSoup no lo procesan
Headless + control total del navegador	✅ Controla Chromium, Firefox, Webkit	Selenium también lo hace pero es más pesado y lento
Fácil extracción de <a> dinámicos	✅ Detecta contenido que aparece después de cargar JS	❌ requests solo ve HTML inicial
Multi-tab, multi-page, manejo de wait_for_timeout()	✅ Sencillo y robusto	Selenium lo puede hacer pero requiere más código
Instalación de dependencias	✅ Solo playwright install	Selenium necesita instalar manualmente drivers como chromedriver o geckodriver