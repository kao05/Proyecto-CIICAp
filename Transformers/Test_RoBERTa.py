#This a script in python for testing chatbot CIICAp
#Model Architecture transformer, compatible with Pytorch and Spanish language 
#Models used (encoder-only):
#1. PlanTL-GOB-ES/roberta-base-bne 
#2. dccuchile/bert-base-spanish-wwm

#Importing necessary libraries
import torch
import embedding
from transformers import AutoTokenizer, AutoModel

#Te abstrae de la arquitectura exacta (BERT, RoBERTa).HuggingFace detecta automáticamente el tipo correcto.

model_name = "PlanTL-GOB-ES/roberta-base-bne" #first model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
#Example input for tokenization and encoding
text = "¿Cuáles son los requisitos de admisión para la licenciatura en ingeniería?"

#Tokenization and encoding 
inputs = tokenizer(text, return_tensors="pt",padding=True, truncation=True, max_length=512) #tokenizer devuelve tensores de PyTorch y se asegura de que la entrada tenga una longitud máxima de 512 tokens, con padding y truncation según sea necesario. 

#Generating embeddings
with torch.no_grad(): #Desactiva el cálculo de gradientes para ahorrar memoria y acelerar la inferencia.
    outputs = model(**inputs) #Pasa los tensores de entrada al modelo para obtener las salidas.

# El embedding de la frase completa (usando el token [CLS])
embeddings = outputs.last_hidden_state[:, 0, :] #Extrae la representación del token [CLS] (primer token) como embedding de la oración completa.

print(f"Shape del embedding: {embeddings.shape}")
print(f"Embedding (primeros 10 valores): {embeddings[0][:10]}")
