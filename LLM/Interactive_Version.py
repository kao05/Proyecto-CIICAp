from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 
# Cargar modelo
print("Cargando Llama 3.2-3B-Instruct...")
model_name = "meta-llama/Llama-3.2-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    token=HF_TOKEN
)

print("✓ Modelo cargado!\n")

def chatbot_interactivo():
    """Chatbot interactivo en consola"""
    
    print("="*60)
    print("CHATBOT INSTITUCIONAL - Llama 3.2-3B")
    print("="*60)
    print("Escribe 'salir' para terminar")
    print("Escribe 'contexto' para agregar información de contexto")
    print("="*60 + "\n")
    
    contexto_global = ""
    
    while True:
        # Obtener pregunta del usuario
        pregunta = input("👤 Tú: ").strip()
        
        if pregunta.lower() == 'salir':
            print("\n👋 ¡Hasta luego!")
            break
        
        if pregunta.lower() == 'contexto':
            print("\n📄 Ingresa el contexto (presiona Enter dos veces para terminar):")
            lineas = []
            while True:
                linea = input()
                if linea == "":
                    break
                lineas.append(linea)
            contexto_global = "\n".join(lineas)
            print("✓ Contexto guardado!\n")
            continue
        
        if not pregunta:
            continue
        
        # Construir mensajes
        messages = [
            {
                "role": "system",
                "content": "Eres un asistente virtual de una institución educativa. Responde de manera clara y precisa en español."
            }
        ]
        
        if contexto_global:
            messages.append({
                "role": "system",
                "content": f"Información relevante: {contexto_global}"
            })
        
        messages.append({
            "role": "user",
            "content": pregunta
        })
        
        # Generar respuesta
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        print("\n🤖 Asistente: ", end="", flush=True)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        respuesta_completa = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extraer respuesta
        if "assistant" in respuesta_completa:
            respuesta = respuesta_completa.split("assistant")[-1].strip()
        else:
            respuesta = respuesta_completa.strip()
        
        print(respuesta + "\n")

# Ejecutar chatbot
chatbot_interactivo()