# Proyecto Chatbot CIICAp — LLM + RAG 🤖
Asistente virtual para la página institucional del CIICAp, diseñado para facilitar la búsqueda de información a estudiantes y visitantes mediante inteligencia artificial.

## Stack de Tecnologías 
- Postgresql 
- Docker
- Fast API LTS 
- REST
- Python 3.11
### ¿Por qué Python 3.11?
Compatibilidad total con todas las librerías: transformers, torch, fastapi, redis, psycopg2, bitsandbytes y accelerate tienen soporte completamente estable en 3.11.
Rendimiento es hasta un 25% más rápido que 3.10 en operaciones generales, y tiene mejor soporte de PyTorch en Windows.
Google Colab actualmente corre Python 3.11 como versión por defecto, lo que significa que si desarrollas en 3.11 local, el código correrá igual en Colab sin problemas.


- 
### Transformers
- https://huggingface.co/PlanTL-GOB-ES/roberta-base-bne
- https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased
  

## Librerias
- BeautifulSoup

## Herramientas
- Web Scraping
  - https://www.youtube.com/watch?v=bK3EwIMHm94
  - https://www.youtube.com/watch?v=yKi9-BfbfzQ
- RAG
  - https://www.youtube.com/watch?v=uAsd9pOIcLg
  - https://www.youtube.com/watch?v=W2YwMuxzyJY
  - https://www.youtube.com/watch?v=tjcMv_CPIxA
  - Qué es?
    - https://www.youtube.com/watch?v=esQ4LMVdbaA&t=210s
    - https://www.youtube.com/watch?v=5Y3a61o0jFQ




## Porqué se ha elegido Gemma 3 4B
este modelo de LLM se escogio debido a que a pesar de tener bastantes datos con los que fue entrenado relativamente no es tan pesado como otros que se pueden llegar a encontrar, a parte al hacer las pruebas tecnicas y empiricas este no presento gran demanda en el software, tambien porque sus respuestas comparadas con otros modelos fueron más acertivas y coherentes.  