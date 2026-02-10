#model_name = "dccuchile/bert-base-spanish-wwm-cased" #second model
from transformers import AutoTokenizer, AutoModel
import torch

# Cargar el tokenizador y modelo
model_name = "dccuchile/bert-base-spanish-wwm-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Ejemplo de uso
texto = "¿Dónde puedo consultar el calendario académico?"

# Tokenizar
inputs = tokenizer(texto, return_tensors="pt", padding=True, truncation=True, max_length=512)

# Generar embeddings
with torch.no_grad():
    outputs = model(**inputs)
    
embedding = outputs.last_hidden_state[:, 0, :].numpy()

print(f"Shape del embedding: {embedding.shape}")
print(f"Embedding (primeros 10 valores): {embedding[0][:10]}")