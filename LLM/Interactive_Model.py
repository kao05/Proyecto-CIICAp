"""
First for use this model you need to install the following:
1. pip install -U transformers torch accelerate Pillow
2. verify have the version of transformers >= 4.50.0
"""

from dotenv import load_dotenv
import os
import torch
from huggingface_hub import login
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
"""
autoProcessor: Es una clase que se encarga de preprocesar,el encargado de preparar la información ANTES de que el modelo la reciba.
Gemma3ForConditionalGeneration: Es el modelo en sí mismo, el cerebro del chatbot.
Se llama "condicional" porque no genera texto al azar, sino que lo genera basándose en la pregunta o contexto que recibe.
"""



#Configuration
model_name = "google/gemma-3-4b-it"
token = os.getenv("HG_TOKEN")
limit_tokens_model = 256
limit_conversacion = 20
login(token=token)
#charging the model
print("Cargando el modelo...")
model = Gemma3ForConditionalGeneration.from_pretrained(
    model_name,
    device_map="cuda",  # Asignar automáticamente a GPU/CPU
    torch_dtype=torch.float16,  # Usar half precision
    token=token
).eval()  # Poner el modelo en modo evaluación

processor = AutoProcessor.from_pretrained(model_name, token=token)
print("Modelo cargado exitosamente.")
#Diccionario para salir del bucle
Token_salida = ["adios", "adiós", "para", "hemos terminado", "stop", "salir"]

# ================================
# Interacción con el bot
# ================================

print("=" * 50)
print("¡Bienvenido al asistente virtual de la institución CIICAp")
print("=" * 50)
print("para salir, escribe 'adiós', 'para', 'hemos terminado', 'stop' o 'salir'.")
print("=" * 50 + "\n")

historial = []  # Para almacenar el historial de la conversación
"""
por el historial:
- Las preguntas tienen contexto
- El modelo "recuerda" la conversación
- Más natural y útil para el usuario
- Con 20 entradas (10 intercambios) es suficiente para mantener un contexto conversacional coherente sin sacrificar rendimiento.
"""
# Conversación en bucle
while True: 
    #Capturar pregunta del usuario
    pregunta = input("Tú: ").strip()#strip() para eliminar espacios al inicio y al final
    #Checar que no este vacio
    if not pregunta:
        print("Por favor, ingresa una pregunta válida.")
        continue
    #Verificar si el usuario quiere salir
    if any(palabra in pregunta.lower() for palabra in Token_salida): 
        print("Asistente: ¡Hasta luego! Si necesitas ayuda en el futuro, no dudes en volver.")
        break
    #construir el mensaje para el modelo
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": (
                "Eres un asistente virtual de una institución educativa. "
                "Responde de manera clara, concisa y profesional en español."
            )}]
        }
    ]
    #Agregar el historial de la conversación
    messages.extend(historial) 
    #Agregar la pregunta actual del usuario
    messages.append({ #append() para agregar la pregunta al final de la lista de mensajes
        "role": "user",
        "content": [{"type": "text", "text": pregunta}]
    })
    # Procesar y generar respuesta
    inputs = processor.apply_chat_template( 
        messages, 
        add_generation_prompt=True, #Agregar un prompt para que el modelo sepa que debe generar una respuesta
        tokenize=True,
        return_dict=True, #Devolver un diccionario con los tensores necesarios para la generación
        return_tensors="pt" #Devolver tensores en formato PyTorch
    ).to(model.device,dtype=torch.bfloat16) #Mover los tensores al mismo dispositivo que el modelo (GPU o CPU)
    inputs_len = inputs["input_ids"].shape[1] #Obtener la longitud de los tokens de entrada

    with torch.inference_mode(): #Desactivar el cálculo de gradientes para ahorrar memoria y acelerar la generación
        outputs = model.generate(
            **inputs,
            max_new_tokens=limit_tokens_model, #Limitar la respuesta a 256 tokens
            temperature=0.7, #Controlar la creatividad de la respuesta (0.7 es un buen valor para respuestas coherentes pero no demasiado predecibles)
            top_p=0.9, #Usar top-p sampling para mejorar la diversidad de las respuestas
            do_sample=True, #Habilitar el muestreo para generar respuestas más variadas
        )
        generation = outputs[0] [inputs_len:] #Obtener solo los tokens generados por el modelo, excluyendo los tokens de entrada
    respuesta = processor.decode(generation, skip_special_tokens=True).strip() #Decodificar la respuesta generada y eliminar espacios al inicio y al final
    
    print(f"Asistente: {respuesta}\n") #Imprimir la respuesta del asistente
    # Actualizar el historial de la conversación 
    historial.append({ #Agregar la pregunta y la respuesta al historial
        "role": "user",
        "content": [{"type": "text", "text": pregunta}]
    })
    historial.append({
        "role": "assistant",
        "content": [{"type": "text", "text": respuesta}]
    })
    # Limitar el historial a las últimas 20 entradas (10 intercambios)
    if len(historial) > limit_conversacion:
        historial = historial[-limit_conversacion:] #Mantener solo las últimas 20 entradas para no sobrecargar el modelo con demasiada información histórica
