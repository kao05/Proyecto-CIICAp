#Creación del script para probar el pipeline de RAG
""""
Donde realizaremos los siguientes pasos:
1. Extracción de texto del PDF
2. Division del texto en fragmentos
3. Generación de embeddings 
4. Búsqueda semantica 
5. Generación de respuestas con el modelo Gemma3
""" 

import os
import fitz  # PyMuPDF para extracción de texto de PDF
from tqdm import tqdm  # Para barras de progreso (opcional)
import json

Pdf = "Test/PLAN_DE_ESTUDIOS_BIOINGENIERIA.pdf"
Output_text = "Test/TextoExtraido.txt"
output_chunks = "Test/Chunks.json"

#Parte 1 - Extracción de texto del PDF
#En caso que la ruta no sea correcta
if not os.path.exists(Pdf):
    print(f"El archivo {Pdf} no existe. Por favor, verifica la ruta.")
    exit()

#Extracción de texto del PDF
documento = fitz.open(Pdf)
total_paginas = len(documento)

print(f"total de páginas en el PDF: {total_paginas}")
print("Tamaño del PDF: {:.2f} MB".format(os.path.getsize(Pdf) / (1024 * 1024)))

texto_completo = []
paginas_sin_texto = 0

for total_pagina in tqdm(documento, desc="Extrayendo texto del PDF"):
    pagina = total_pagina.get_text()
    if pagina.strip():  # Verificar si el texto no está vacío
        texto_completo.append(pagina)
    else:
        paginas_sin_texto += 1
