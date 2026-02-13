"""
Para realizar el LLM se penso en 3 modelos en especifico debido a que comparten caracteristicas similares y pero tienen diversos enfoques en su arquitectura y aplicacion que aun esta por elegirse alguno de ellos
1.meta-llama/Llama-3.2-3B-Instruct ó meta-llama/Llama-3.1-8B-Instruct
Pesados pero buenos en calidad para lo que se desea realizar el primero es mas liviano y el segundo es mas pesado pero con mejor calidad, ambos son de la misma familia de modelos y se pueden ajustar a las necesidades del proyecto.
2. mistralai/Mistral-7B-Instruct-v0.3

Un modelo de 7B parámetros que es conocido por su eficiencia y rendimiento en tareas de lenguaje natural.

3. google/gemma-2-2b-it
Un modelo de 2B parámetros desarrollado por Google, diseñado para tareas de lenguaje natural con un enfoque en la eficiencia y la calidad de las respuestas. Es muy ligero y rápido, lo que lo hace ideal para aplicaciones que requieren respuestas rápidas y eficientes, aunque puede no ser tan preciso como los modelos más grandes.
"""
"""
Antes de probar el code verificar si estan descargadas las dependencias
- pip install transformers torch accelerate huggingface_hub
Si se tiene GPU NVIDIA: pip install transformers torch accelerate huggingface_hub bitsandbytes
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenKevin = "lol"
print("Cargando el modelo...")
model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=tokenKevin)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Usar half precision
    device_map="auto",  # Asignar automáticamente a GPU/CPU
    token=tokenKevin
)
print("Modelo cargado exitosamente.")

def hacer_pregunta(pregunta, contexto=""):
    """
    Genera una respuesta del modelo dada una pregunta
    
    Args:
        pregunta (str): La pregunta del usuario
        contexto (str): Contexto adicional opcional
    
    Returns:
        str: Respuesta generada por el modelo
    """
    # Construir el mensaje
    messages = [
        {
            "role": "system", 
            "content": "Eres un asistente virtual de una institución educativa. Responde de manera clara, concisa y precisa en español."
        }
    ]
    
    # Agregar contexto si existe
    if contexto:
        messages.append({
            "role": "system",
            "content": f"Información relevante: {contexto}"
        })
    
    # Agregar la pregunta del usuario
    messages.append({
        "role": "user",
        "content": pregunta
    })
    
    # Aplicar el template de chat de Llama
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenizar
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generar respuesta
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decodificar la respuesta completa
    respuesta_completa = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extraer solo la respuesta del asistente (después del último "assistant")
    if "assistant" in respuesta_completa:
        respuesta = respuesta_completa.split("assistant")[-1].strip()
    else:
        respuesta = respuesta_completa.strip()
    
    return respuesta


# Ejemplo de uso simple
print("\n" + "="*60)
print("EJEMPLO 1: Pregunta simple")
print("="*60)

pregunta1 = "¿Cuáles son los requisitos de admisión para la licenciatura en ingeniería?"
respuesta1 = hacer_pregunta(pregunta1)

print(f"\n🤔 Pregunta: {pregunta1}")
print(f"\n🤖 Respuesta: {respuesta1}")


# Ejemplo con contexto
print("\n" + "="*60)
print("EJEMPLO 2: Pregunta con contexto")
print("="*60)

pregunta2 = "¿Cómo puedo inscribirme?"
contexto2 = """
El proceso de inscripción se realiza en línea a través del portal institucional.
Los pasos son: 1) Crear cuenta, 2) Llenar formulario, 3) Subir documentos, 
4) Pagar cuota de inscripción. El periodo de inscripciones es del 1 al 15 de julio.
"""

respuesta2 = hacer_pregunta(pregunta2, contexto2)

print(f"\n🤔 Pregunta: {pregunta2}")
print(f"\n📄 Contexto: {contexto2.strip()}")
print(f"\n🤖 Respuesta: {respuesta2}")