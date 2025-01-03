import os
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from fpdf import FPDF
import matplotlib.pyplot as plt
import csv
from io import StringIO

# Función para generar un informe ejecutivo en PDF
def generate_executive_report(data, insights, regression_results, fig_path, file_name="informe_ejecutivo.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Portada
    pdf.cell(200, 10, txt="Informe Ejecutivo - Admisión Universitaria en Rusia", ln=True, align="C")
    pdf.ln(10)

    # Introducción
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 10, "Introducción:", ln=True)
    pdf.multi_cell(0, 10, "Este informe presenta un análisis detallado sobre las admisiones universitarias en Rusia "
                          "entre 2014 y 2023. Incluye estadísticas descriptivas, visualizaciones y resultados del "
                          "modelo de regresión.")
    pdf.ln(5)

    # Resumen Estadístico
    pdf.cell(0, 10, "Resumen Estadístico del Dataset:", ln=True)
    for column in data.describe().T.itertuples():
        pdf.cell(0, 10, f"{column.Index}: Media={column.mean:.2f}, Desviación Estándar={column.std:.2f}", ln=True)
    pdf.ln(5)

    # Insights
    pdf.cell(0, 10, "Principales Insights:", ln=True)
    for insight in insights:
        pdf.multi_cell(0, 10, f"- {insight}")
    pdf.ln(5)

    # Resultados del Modelo de Regresión
    pdf.cell(0, 10, "Resultados del Modelo de Regresión:", ln=True)
    pdf.multi_cell(0, 10, regression_results)
    pdf.ln(5)

    # Gráfico de Predicciones
    pdf.add_page()
    pdf.cell(0, 10, "Gráfico de Predicciones vs Valores Reales:", ln=True)
    pdf.image(fig_path, x=10, y=30, w=190)

    # Conclusiones
    pdf.add_page()
    pdf.cell(0, 10, "Conclusiones y Recomendaciones:", ln=True)
    pdf.multi_cell(0, 10, "En base a los análisis realizados, se recomienda:\n"
                          "- Analizar más variables para mejorar las predicciones.\n"
                          "- Considerar tendencias históricas para estimar los próximos años.")

    # Guardar el archivo
    pdf.output(file_name)

    # Retornar el archivo para la descarga
    return file_name

# Configuración inicial
st.set_page_config(page_title="Análisis de Admisión Universitaria", layout="wide", page_icon="🎓")
st.sidebar.title("🎓 Opciones de Navegación")
st.sidebar.info("Usa las pestañas para explorar cada sección.")

# Pestañas principales
tab1, tab2, tab3, tab4 = st.tabs(["📂 Cargar Datos", "📊 Análisis Exploratorio", "🔮 Modelo de Regresión", "📄 Informe PDF"])

# Tab 1: Cargar Datos
with tab1:
    st.header("📂 Cargar Datos")
    uploaded_file = st.file_uploader("Sube el archivo CSV con los datos:", type=["csv"])

    if uploaded_file is not None:
        try:
            # Leer el archivo como texto para detectar el separador
            content = uploaded_file.read().decode("utf-8")
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(content)
            sep = dialect.delimiter

            # Cargar el archivo con el separador detectado
            data = pd.read_csv(StringIO(content), sep=sep)

            # Normalizar nombres de columnas
            data.columns = data.columns.str.strip().str.lower().str.replace(" ", "_").str.replace(r"[^\w]", "")

            # Guardar data en session_state
            st.session_state["data"] = data

            st.success("Datos cargados exitosamente.")
            st.dataframe(data.head())
        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")

# Tab 2: Análisis Exploratorio
with tab2:
    st.header("📊 Análisis Exploratorio")
    if "data" in st.session_state:
        data = st.session_state["data"]
        st.write(data.describe())

        st.subheader("Distribución de Estudiantes por Año")
        if "year" in data.columns and "number_of_students" in data.columns:
            fig = px.bar(data, x="year", y="number_of_students", title="Estudiantes por Año")
            st.plotly_chart(fig)

        st.subheader("Relación entre Aplicaciones y Estudiantes")
        if "number_of_applications" in data.columns and "number_of_students" in data.columns:
            fig = px.scatter(data, x="number_of_applications", y="number_of_students", title="Aplicaciones vs Estudiantes")
            st.plotly_chart(fig)
    else:
        st.warning("Por favor, sube un archivo en la pestaña '📂 Cargar Datos'.")

# Tab 3: Modelo de Regresión
with tab3:
    st.header("🔮 Modelo de Regresión")
    if "data" in st.session_state:
        data = st.session_state["data"]
        predictors = st.multiselect("Selecciona las variables predictoras:", ["number_of_applications", "year", "tuition_fees"])
        target = st.selectbox("Selecciona la variable objetivo:", ["number_of_students"])

        if st.button("Entrenar Modelo"):
            if target and predictors:
                valid_predictors = [col for col in predictors if col in data.columns]
                if not valid_predictors:
                    st.error("No se encontraron columnas válidas como predictores.")
                else:
                    data = data.dropna(subset=valid_predictors + [target])
                    X = data[valid_predictors]
                    y = data[target]

                    # Entrenar modelo
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)

                    # Calcular métricas
                    mse = mean_squared_error(y_test, predictions)
                    r2 = r2_score(y_test, predictions)

                    # Guardar en session_state
                    st.session_state["mse"] = mse
                    st.session_state["r2"] = r2

                    st.success("Modelo entrenado exitosamente.")
                    st.write(f"Error Cuadrático Medio (MSE): {mse:.2f}")
                    st.write(f"R² del Modelo: {r2:.2f}")

                    # Gráfico
                    plt.figure(figsize=(8, 6))
                    plt.scatter(y_test, predictions, color="#D8E2DC")
                    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="black", linestyle="--")
                    plt.xlabel("Valores Reales")
                    plt.ylabel("Predicciones")
                    plt.title("Predicciones vs Valores Reales")
                    plt.grid()
                    plt.savefig("plot.png")
                    st.pyplot(plt)

# Tab 4: Informe PDF
with tab4:
    st.header("📄 Generación de Informe PDF")
    if "mse" in st.session_state and "r2" in st.session_state:
        if st.button("Generar Informe Ejecutivo"):
            insights = ["El modelo muestra una relación fuerte entre las aplicaciones y los estudiantes."]
            regression_results = f"MSE: {st.session_state['mse']:.2f}, R²: {st.session_state['r2']:.2f}"
            file_name = generate_executive_report(st.session_state["data"], insights, regression_results, "plot.png")
            with open(file_name, "rb") as file:
                st.download_button("Descargar Informe", file, file_name)
    else:
        st.warning("Primero debes entrenar el modelo en la pestaña '🔮 Modelo de Regresión'.")
