import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import easyocr
from ultralytics import YOLO
import folium
from streamlit_folium import st_folium
from fpdf import FPDF
from datetime import datetime

# Configuración de página
st.set_page_config(page_title="TrazaFiscal Puno", page_icon="🛃", layout="wide")

# Base de datos simulada de SUNAT / Aduanas
DATA_SUNAT = pd.DataFrame([
    {"placa": "Z4B812", "ruc": "20601489211", "contribuyente": "IMPORTACIONES ALTIPLANO S.A.C.", "guia": "T001-004521", "valor_declarado": 25000.0, "estado_sunat": "CONFORME", "puesto": "Desaguadero", "lat": -16.5656, "lon": -69.0417},
    {"placa": "V9A743", "ruc": "20448120934", "contribuyente": "COMERCIALIZADORA SUR PERU E.I.R.L.", "guia": "T001-001108", "valor_declarado": 2000.0, "estado_sunat": "SUBVALUADO", "puesto": "Kasani (Yunguyo)", "lat": -16.2564, "lon": -69.0911},
    {"placa": "X2K905", "ruc": "SIN RUC", "contribuyente": "NO REGISTRADO", "guia": "SIN GUIA", "valor_declarado": 0.0, "estado_sunat": "CONTRABANDO", "puesto": "Tilali", "lat": -15.4083, "lon": -69.3653},
    {"placa": "A1B234", "ruc": "20512345678", "contribuyente": "TRANSPORTES TITICACA S.R.L.", "guia": "T002-008910", "valor_declarado": 18500.0, "estado_sunat": "CONFORME", "puesto": "Desaguadero", "lat": -16.5660, "lon": -69.0425}
])

@st.cache_resource
def cargar_modelos():
    modelo_yolo = YOLO("yolov8n.pt")
    lector_ocr = easyocr.Reader(['es', 'en'], gpu=False)
    return modelo_yolo, lector_ocr

yolo_model, ocr_reader = cargar_modelos()

def generar_pdf(placa, datos, valor_real, igv_omitido, arancel_omitido, total_omitido):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "TRAZAFISCAL PUNO - ACTA DE TRAZABILIDAD CONTABLE Y FISCAL", ln=True, align="C")
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 8, f"Fecha y Hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} | Frontera Puno - Bolivia", ln=True, align="C")
    pdf.ln(8)
    
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 8, "1. DATOS DE INTERVENCION VEHICULAR", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 7, f"Placa Detectada (IA/OCR): {placa}", ln=True)
    pdf.cell(0, 7, f"RUC Asociado: {datos['ruc']} - {datos['contribuyente']}", ln=True)
    pdf.cell(0, 7, f"Guia de Remision Electronica: {datos['guia']} | Estado: {datos['estado_sunat']}", ln=True)
    pdf.cell(0, 7, f"Puesto de Control: {datos['puesto']}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 8, "2. LIQUIDACION DE TRIBUTOS OMITIDOS (AUDITORIA)", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 7, f"Valor Fisico Estimado en Almacen: US$ {valor_real:,.2f}", ln=True)
    pdf.cell(0, 7, f"Valor Declarado en Comprobante: US$ {datos['valor_declarado']:,.2f}", ln=True)
    pdf.cell(0, 7, f"Base Imponible No Sustentada: US$ {max(0, valor_real - datos['valor_declarado']):,.2f}", ln=True)
    pdf.cell(0, 7, f"IGV + IPM Omitido (18%): US$ {igv_omitido:,.2f}", ln=True)
    pdf.cell(0, 7, f"Arancel Ad-Valorem Omitido (6%): US$ {arancel_omitido:,.2f}", ln=True)
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 10, f"TOTAL DEUDA TRIBUTARIA RECUPERABLE: US$ {total_omitido:,.2f}", ln=True)
    
    return pdf.output(dest="S").encode("latin-1")

st.title("🛃 TrazaFiscal Puno")
st.markdown("*Sistema de Trazabilidad Contable y Fiscal contra el Contrabando (Frontera Puno - Bolivia)*")

tab1, tab2, tab3 = st.tabs(["📷 Escáner IA (YOLO + OCR)", "🗺️ Mapa de Control Fronterizo", "📊 Base de Datos SUNAT"])

with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1. Captura de Vehículo / Placa")
        origen = st.radio("Fuente de imagen:", ["Subir foto de vehículo/placa", "Usar cámara web en vivo", "Simular lectura rápida"])
        
        imagen_procesar = None
        placa_detectada = ""
        
        if origen == "Subir foto de vehículo/placa":
            archivo = st.file_uploader("Sube una imagen (JPG/PNG)", type=["jpg", "jpeg", "png"])
            if archivo:
                imagen_procesar = Image.open(archivo)
        elif origen == "Usar cámara web en vivo":
            camara = st.camera_input("Toma una foto a la placa")
            if camara:
                imagen_procesar = Image.open(camara)
        else:
            placa_detectada = st.selectbox("Selecciona una placa captada en puesto:", ["Z4B812", "V9A743", "X2K905", "A1B234", "P9Z999"])

        if imagen_procesar is not None:
            img_np = np.array(imagen_procesar)
            resultados = yolo_model(img_np, classes=[2, 5, 7], conf=0.3)
            img_anotada = resultados[0].plot()
            st.image(img_anotada, caption="Detección Vehicular YOLOv8", use_container_width=True)
            
            textos = ocr_reader.readtext(img_np, detail=0)
            candidatos = [t.replace("-", "").replace(" ", "").upper() for t in textos if len(t.replace(" ", "")) >= 5]
            st.info(f"Texto bruto leído por OCR: {textos}")
            placa_detectada = st.text_input("Placa identificada (puedes corregir si hay reflejo):", value=candidatos[0] if candidatos else "X2K905")

    with col2:
        st.subheader("2. Cruce Contable y Liquidación Fiscal")
        if placa_detectada:
            placa_limpia = placa_detectada.replace("-", "").replace(" ", "").upper()
            registro = DATA_SUNAT[DATA_SUNAT["placa"] == placa_limpia]
            
            if registro.empty:
                datos = {"placa": placa_limpia, "ruc": "SIN RUC", "contribuyente": "NO HABIDO / SIN REGISTRO", "guia": "NINGUNA", "valor_declarado": 0.0, "estado_sunat": "CONTRABANDO", "puesto": "Desaguadero"}
            else:
                datos = registro.iloc[0].to_dict()

            valor_fisico = st.number_input("Valorización física de mercancía intervenida (US$):", min_value=0.0, value=20000.0, step=500.0)
            
            sin_sustento = max(0.0, valor_fisico - datos["valor_declarado"])
            pct_sin_sustento = (sin_sustento / valor_fisico * 100) if valor_fisico > 0 else 0
            igv_omitido = sin_sustento * 0.18
            arancel_omitido = sin_sustento * 0.06
            total_omitido = igv_omitido + arancel_omitido

            if total_omitido == 0:
                st.success(f"✅ VEHÍCULO CONFORME | RUC: {datos['ruc']} ({datos['contribuyente']}) | Guía: {datos['guia']}")
            else:
                st.error(f"🚨 ALERTA DE RIESGO FISCAL: {datos['estado_sunat']} | RUC: {datos['ruc']} | Guía: {datos['guia']}")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Mercancía sin sustento", f"{pct_sin_sustento:.1f}%", f"US$ {sin_sustento:,.0f}")
            m2.metric("IGV + IPM Omitido (18%)", f"US$ {igv_omitido:,.2f}")
            m3.metric("Arancel Omitido (6%)", f"US$ {arancel_omitido:,.2f}")
            
            st.warning(f"*Total Impuesto Recuperable por Auditoría:* US$ {total_omitido:,.2f}")

            pdf_bytes = generar_pdf(placa_limpia, datos, valor_fisico, igv_omitido, arancel_omitido, total_omitido)
            st.download_button(
                label="📄 Descargar Reporte Oficial PDF (SUNAT / Aduanas)",
                data=pdf_bytes,
                file_name=f"Acta_TrazaFiscal_{placa_limpia}.pdf",
                mime="application/pdf"
            )

with tab2:
    st.subheader("Mapa de Vigilancia en Tiempo Real - Frontera Puno - Bolivia")
    mapa = folium.Map(location=[-16.15, -69.15], zoom_start=8)
    for _, row in DATA_SUNAT.iterrows():
        color = "green" if row["estado_sunat"] == "CONFORME" else "red"
        folium.Marker(
            [row["lat"], row["lon"]],
            popup=f"<b>Puesto:</b> {row['puesto']}<br><b>Placa:</b> {row['placa']}<br><b>Estado:</b> {row['estado_sunat']}<br><b>Declarado:</b> US$ {row['valor_declarado']}",
            icon=folium.Icon(color=color, icon="info-sign")
        ).add_to(mapa)
    st_folium(mapa, width=900, height=450)

with tab3:
    st.subheader("Registros Simulados de Guías de Remisión Electrónica (GRE)")
    st.dataframe(DATA_SUNAT, use_container_width=True)