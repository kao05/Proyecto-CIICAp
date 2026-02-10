from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def cargar_modelo(model_name):
    """Carga el tokenizador y modelo"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

def generar_embedding(texto, tokenizer, model):
    """Genera embedding para un texto"""
    inputs = tokenizer(texto, return_tensors="pt", padding=True, 
                      truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Usar el token [CLS] como representación de la frase
    embedding = outputs.last_hidden_state[:, 0, :].numpy()
    return embedding

def comparar_similitud(pregunta, contextos, tokenizer, model):
    """Compara similitud entre pregunta y contextos"""
    emb_pregunta = generar_embedding(pregunta, tokenizer, model)
    
    similitudes = []
    for contexto in contextos:
        emb_contexto = generar_embedding(contexto, tokenizer, model)
        similitud = cosine_similarity(emb_pregunta, emb_contexto)[0][0]
        similitudes.append(similitud)
    
    return similitudes

# Ejemplos de preguntas y contextos institucionales
pregunta = "¿Cuál es el proceso de inscripción?"

contextos = [
    "El proceso de inscripción se realiza en línea a través del portal institucional.",
    "La cafetería está ubicada en el edificio principal.",
    "Los requisitos de admisión incluyen certificado de bachillerato.",
    "Para inscribirte debes completar el formulario en la página oficial."
]

# Probar ambos modelos
print("=" * 60)
print("PRUEBA DE MODELOS")
print("=" * 60)

modelos = {
    "RoBERTa": "PlanTL-GOB-ES/roberta-base-bne",
    "BETO": "dccuchile/bert-base-spanish-wwm-cased"
}

for nombre, model_name in modelos.items():
    print(f"\n{'='*60}")
    print(f"Modelo: {nombre}")
    print(f"{'='*60}")
    
    tokenizer, model = cargar_modelo(model_name)
    similitudes = comparar_similitud(pregunta, contextos, tokenizer, model)
    
    print(f"\nPregunta: {pregunta}\n")
    for i, (contexto, sim) in enumerate(zip(contextos, similitudes), 1):
        print(f"{i}. Similitud: {sim:.4f} - {contexto}")
    
    # Encontrar el contexto más relevante
    mejor_idx = np.argmax(similitudes)
    print(f"\n✓ Contexto más relevante: #{mejor_idx + 1}")