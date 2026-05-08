---
title: RAG Asesor Financiero
emoji: 💰
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 6.14.0
app_file: app.py
pinned: false
---

# Asesor RAG de Libertad Financiera

Asesor RAG de Libertad Financiera es una aplicación web en Python con Gradio para un proyecto final de un curso de IA. Responde preguntas educativas sobre independencia financiera, finanzas personales, fondos de emergencia, inversión a largo plazo, diversificación, gestión de riesgos, presupuesto, manejo de deudas y priorización de inversiones usando un pipeline local de Retrieval-Augmented Generation.

El chatbot recupera fragmentos relevantes desde archivos `.txt` en la carpeta local `docs/`, envía ese contexto a Gemini mediante LangChain y devuelve una respuesta con los archivos fuente utilizados.

## Tecnologías utilizadas

- Python
- Gradio
- Gemini API
- LangChain
- ChromaDB
- Estructura compatible con HuggingFace Spaces

Modelos usados por defecto:

- LLM: `gemini-2.5-flash`
- Embeddings: `gemini-embedding-001`

Opcionalmente puedes cambiar los modelos con variables de entorno:

```bash
export GEMINI_MODEL="gemini-2.5-flash"
export GEMINI_EMBEDDING_MODEL="models/gemini-embedding-001"
```

## Estructura del proyecto

```text
.
├── app.py
├── requirements.txt
├── README.md
└── docs/
    ├── 01_bases_de_libertad_financiera.txt
    ├── 02_presupuesto_y_flujo_de_caja.txt
    ├── 03_fondos_de_emergencia.txt
    ├── 04_manejo_de_deudas.txt
    ├── 05_priorizacion_financiera.txt
    ├── 06_inversion_a_largo_plazo.txt
    ├── 07_diversificacion.txt
    ├── 08_gestion_de_riesgos.txt
    ├── 09_cuentas_de_retiro_e_impuestos.txt
    └── 10_comportamiento_y_mentalidad.txt
```

## Instalación

Crea un entorno virtual e instala las dependencias:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configurar `GEMINI_API_KEY`

La aplicación lee la API key de Gemini desde una variable de entorno. No escribas la API key directamente en el código.

macOS/Linux:

```bash
export GEMINI_API_KEY="tu-api-key-de-gemini"
```

Windows PowerShell:

```powershell
$env:GEMINI_API_KEY="tu-api-key-de-gemini"
```

## Ejecutar localmente

```bash
python app.py
```

Luego abre la URL local de Gradio que aparece en la terminal.

## Notas de despliegue en HuggingFace Spaces

1. Crea un nuevo HuggingFace Space usando el SDK de Gradio.
2. Sube `app.py`, `requirements.txt`, `README.md` y la carpeta `docs/`.
3. En la configuración del Space, agrega un secreto llamado `GEMINI_API_KEY`.
4. Reinicia el Space después de agregar el secreto.

La app construye la base vectorial local de Chroma a partir de `docs/` cuando se hacen preguntas. Esto mantiene el Space simple y evita subir archivos generados de base de datos.

## Preguntas de demostración

- ¿Qué tan grande debería ser mi fondo de emergencia antes de invertir?
- ¿Cómo ayuda la diversificación a manejar el riesgo de inversión?
- ¿En qué orden debería priorizar pagar deudas, ahorrar e invertir?

## Referencias públicas utilizadas

La base de conocimiento local fue escrita como contenido educativo original en español, tomando como referencia estas fuentes públicas de educación financiera:

- [Investor.gov: Asset Allocation and Diversification](https://www.investor.gov/introduction-investing/getting-started/assessing-your-risk-tolerance)
- [Investor.gov: Asset Allocation, Diversification, and Rebalancing 101](https://www.investor.gov/index.php/introduction-investing/getting-started/asset-allocation)
- [SEC: Beginners' Guide to Asset Allocation, Diversification, and Rebalancing](https://www.sec.gov/investor/pubs/assetallocation.htm)
- [FINRA: Risk](https://www.finra.org/investors/investing/investing-basics/risk)
- [FINRA: Asset Allocation and Diversification](https://www.finra.org/investors/investing/investing-basics/asset-allocation-diversification)
- [Consumer Financial Protection Bureau: Budgeting resources](https://www.consumerfinance.gov/consumer-tools/budgeting/)

## Notas

Esta aplicación es solo educativa. No ofrece asesoría financiera, fiscal, legal o de inversión personalizada.
