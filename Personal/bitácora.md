# Bitácora de Desarrollo - Chatbot CIICAp

**Institución:** Centro de Investigacion en Ingeniería y Ciencias Aplicadas
**Tecnologías:** LLM + RAG
**Desarrollador:** *Kevin Vargas Flores*

---

## Sesión 0: Plan de trabajo del proyecto

**Fecha:** [26/01/26]  

### Enfoque de la Sesión

*Establecer el marco teórico, los objetivos institucionales del chatbot para el CIICAp y definir el stack tecnológico inicial. Se buscó alinear las necesidades de distribución de información de la página con las capacidades de la arquitectura LLM + RAG.*

### Actividades realizadas

1. **Conceptos:** Se aprendieron nuevos conceptos tanto en la materia de *Ingenieria de Software* como de *Programación de lenguaje natural* para el desarrollo del proyecto.
2. **Arquitectura:** Se presento diversas formas en las cuales se podría llevar a cabo el proyecto en forma que existen diversas tecnologias las cuales nos pueden ayudar según corresponda el caso.
3. **Manera de trabajar:** Según los tiempos del Doctor se acordaron horarios y formas en las cuales ibamos a trabajar para mejorar nuestro rendimiento en base de estos.

### Aprendizajes clave

- *Conceptos nuevos:* Conocer nuevos conceptos que me serviran en mi crecimiento como desarrollador Como lo fueron **REST, Redis, RAG, Embeddings**.
- *Tecnologías:* Saber que es lo que nos conviene usar dado un caso especifico segun nuestras necesidades y entorno en el cual se utilice nuestro software.

### Desafíos y Bloqueos

*Ninguno por el momento dado que fue la primera interacción con el proyecto.*

### Reflexión Técnica

- **Análisis:** Evaluación de la viabilidad de modelos de lenguaje frente a las limitantes de hardware. Se determinó que la solución debe ser contenida en Docker para asegurar la portabilidad

- **Decisión:** Se optó por un stack basado en Python 3.13, PostgreSQL (para almacenamiento de vectores), FastAPI para el backend y PyTorch para el manejo de transformadores.

### Evidencias y recursos

- Esquema de arquitectura lógica proporcionado por el asesor (Diagrama de flujo LLM+RAG)

### Objetivos de la semana

- [x] Relacionarme con las tecnologias presentadas.
- [x] Desenpolvar conocimientos acerca de los LLM's.
- [x] Crear un flujo de trabajo.
- [x] Estudiar la manera en la que se puede llevar todo el proyecto de mejor forma con las tecnologías seleccionadas.

---

## Sesión 1: Comprensión del proyecto y bases

**Fecha:** [03/02/26]  

### Enfoque de la Sesión

*Consolidar la comprensión técnica de los componentes del proyecto (LLM + RAG) e inicializar el entorno de desarrollo distribuido para asegurar la integridad del código y la compatibilidad entre librerías.*

### Actividades realizadas

1. **Explicación:** Explicar de que manera he comprendido los conceptos nuevos que me enseño el doctor de manera adecuada, para poder comprender todo lo que estoy realizando.
2. **GitHub:** Creación de un repositorio en `GitHub`, estableciendo la estructura base del proyecto para permitir el desarrollo remoto y el seguimiento de cambios mediante `Git`.
3. **Tecnologías:** Selección rigurosa de versiones del stack tecnológico, optando por `Python 3.13` debido a su estabilidad y rendimiento superior con librerías de IA.
4. **LLM:** Empezar la búsqueda de un LLM que cumpla los requisitos de ser entrenado en idioma español y que sea compatible con `pytorch` facilitando futuras tareas de ajuste fino (fine-tuning).

### Aprendizajes clave

- Comprension en su totalidad del como es que funciona un LLM.
- Comprension de vectorizar en *tokens* las oraciones. 
- Diferenciar tecnologías y su elección según nuestras necesidades.
- Profundizar en la pagina de HuggingFace y su funcionamiento para la busqueda del modelo.
- Explicación de que es lo que realiza `FastAPI`

### Desafíos y Bloqueos

*Ninguno por el momento dado que fue aún no ha existido algún problema.*

### Reflexión Técnica

- **Análisis (REST vs. WebSocket):** Se evaluó el protocolo de comunicación para la interfaz del chatbot. Mientras que *WebSocket* ofrece bidireccionalidad en tiempo real, se determinó que el flujo de peticiones escolares no requiere una conexión persistente de alta frecuencia.
- **Decisión:** Se seleccionó *REST* para la comunicación entre la página institucional y el chatbot. Esto simplifica la integración, mejora la escalabilidad y es totalmente compatible con la arquitectura de `FastAPI`.
  
### Evidencias y recursos

- [GitHub](https://github.com/kao05/Proyecto-CIICAp)
- [HuggingFace](https://huggingface.co/)

### Objetivos de la semana

- [x] Crear una cuenta en HuggingFace para poder utilizar un token para los modelos.
- [x] Crear un repositorio para subir los avances del proyecto.
- [x] Crear un Jira para la asignación de tareas semanalmente.  
- [x] Encontrar cual es el modelo LLM que más nos conviene dado las limitaciones de recursos y especificaciones que necesitamos.

---

## Sesión 2: Evaluación Comparativa y Prototipado del Motor Conversacional 

**Fecha:** [10/02/26]  

### Enfoque de la Sesión

*Ejecutar un proceso de benchmarking cualitativo entre modelos de lenguaje de última generación para seleccionar el núcleo del chatbot. Asimismo, iniciar la fase de implementación mediante el desarrollo de un prototipo interactivo que valide la comunicación con el ecosistema de HuggingFace.*

### Actividades realizadas

1. **Elección del modelo:** valuación comparativa de tres arquitecturas líderes: `Mistral 7B`, `Llama 3.2 3B` y `Gemma 3 4B` con los cuales se interactuo de manera rápida realizando unas cuantas [preguntas base](./Platicas%20con%20modelos/Gemma%204B.pdf) para ver su manera de respuesta simulando un rol de administrador escolar para medir la coherencia, el tono y la capacidad de respuesta en español.
2. **LLM Studio:** Se utilizo la aplicación `LLM Studio` para probar los diversos modelos y versiones de estosm y verificar cual de ellos era lo que más nos convenia.
3. **OpenAPI(Swagger):** Se nos presento estas especificaciones para generar los *endpoints,métodos,parámetros...* que necesitaremos para nuestro modelo.
4. **Interacción Modelo:** Creación del script `Interactive_Model.py` que integra el motor de generación de texto con un ciclo de interacción continua `(while loop)` y gestión de tokens de salida. Se implementó la autenticación mediante la librería `huggingface_hub` para el acceso a modelos protegidos.

### Aprendizajes clave

- Manera más fácil y práctica de interactuar con modelos ya diseñados para su elección en implementaciones.
- Implementación de tokens de acceso con permisos de lectura para la descarga segura de modelos desde repositorios remotos.
- Desarrollo de lógica de control para finalizar sesiones de chat de manera limpia mediante disparadores (tokens de salida) como "adiós" o "salir".

### Desafíos y Bloqueos

- **Error:** Se presentaron dificultades en la autenticación con Hugging Face debido a permisos insuficientes en los tokens generados.
- **Solución:** Se migraron las credenciales a una cuenta verificada y se ajustó el alcance (scope) del token a permisos de lectura exclusivos para modelos gated.

### Reflexión Técnica

- **Análisis:** Elección del modelo por caracteristicas del servidor en el que correra
- **Decisión:** Este modelo de LLM se escogio debido a que a pesar de tener bastantes datos con los que fue entrenado relativamente no es tan pesado como otros que se pueden llegar a encontrar, a parte al hacer las pruebas tecnicas y empiricas este no presento gran demanda en el software, tambien porque sus respuestas comparadas con otros modelos fueron más acertivas y coherentes.  

### Evidencias y recursos

- [LLM Studio](https://lmstudio.ai/)
- [Gemma](https://huggingface.co/google/gemma-3-4b-it)
- [OpenAPI](https://swagger.io/tools/swagger-ui/)
- **Documentación de Pruebas:** `Personal/Platicas con modelos/`
- **Código Fuente:** `LLM/Interactive_Model.py.`

### Objetivos de la semana

- [x] Crear un script de manera que el modelo seleccionado podamos interactuar con el atraves de un codigo propio.
- [x] Ratificación de Python 3.13 como versión estándar por su rendimiento superior y estabilidad con `torch`.
- [x] Resolución definitiva de la comunicación con la API de HuggingFace. 

---

## Sesión 3: Presentación y explicación del proyecto

**Fecha:** [17/02/26]  

### Enfoque de la Sesión

*Normalizar las prácticas de codificación bajo estándares profesionales, implementar protocolos de seguridad para la gestión de credenciales y optimizar el script de inferencia para garantizar la compatibilidad multiplataforma (CPU/GPU).*

### Actividades realizadas

1. **Script sin errores:** Se presento un script sin errores para poder interactuar con el modelo, el cual mantenia problemas por el tipo de dato que devolvia segun lo que se utilizara CPU o GPU.
2. **Normalización de Inferencia (Hardware Agnostic):** Se depuró el script Interactive_Model.py para detectar automáticamente el hardware disponible. Se implementó una lógica de tipos de datos donde se asigna float16 para GPU (CUDA) para mayor rapidez y float32 para CPU para evitar errores de precisión o valores nulos (NaNs).
3. **Entorno Virtual:** Por el proyecto se asentaron las bases de conocimiento en los entornos virtualizados en PC, gracias a ello ahora podemos aislar las versiones de las librerias dependiendo del uso que queramos aprovechar de estas haciendo que se mantengan aisladas, evitando conflictos con el sistema global y facilitando la portabilidad.
4. **Encriptaciones:** Debido que para mantener la seguridad de los tokens GitHub no te permite subir contraseñas como actualizacion del codigo, es por ello que se crea el .env para ahí colocar todo de manera que el codigo pueda buscar la informacion que necesita sin necesidad que este en este.
5. **Políticas de Exclusión en Repositorio:** Configuración exhaustiva del archivo .gitignore para omitir archivos redundantes o sensibles, tales como el entorno virtual (venv/), archivos de caché (__pycache__/) y el archivo de configuración .env.
6. **Código más limpio:** Adopción del estándar `snake_case` para el nombramiento de variables y funciones (ej. limit_tokens_model, tipo_dato), mejorando la legibilidad y el formalismo siguiendo las guías de estilo de Python (PEP 8).

### Aprendizajes clave

- Como dejar de manera local los archivos mediante el `.gitignore`.
- Mayor compresion y manejo de los entornos virtuales.
- Encriptación de contraseñas o código que no deberían subirse a GitHub
- Usar "_" en lugar de espacios para tener un código más limpio
- Comprendí que GitHub bloquea activamente la subida de secretos (tokens) y que la solución profesional es el uso de archivos `.env`.
- Saber donde es que se guarda el modelo una vez descargado de HuggingFace para después sólo descargar los pesos y utilizarse.

### Desafíos y Bloqueos

- **Error:** Bloqueo por parte de GitHub al intentar subir el código con las contraseñas explicitas.
- **Solución:** Se eliminó el token del historial del código, se migró a una variable de entorno gestionada por `os.getenv` y se añadió el archivo `.env` al `.gitignore` para prevenir futuras exposiciones.

### Reflexión Técnica

- **Análisis:** ¿En que parte del almacenamiento se guarda el modelo descargado de HuggingFace?
- **Decisión:** El modelo se almacena en el sistema de archivos local como un "caché". Se determinó que al usar `from_pretrained`, el sistema busca primero en este directorio local; si existe, solo carga los pesos en memoria, reduciendo drásticamente el tiempo de inicio de la API.

### Evidencias y recursos

- **Documentación interna:** Guía de arquitectura RAG proporcionada por el asesor.

### Objetivos de la semana

- [x] Implementar manera de corregir el uso de GPU o CPU en el servidor que se vaya a utilizar
- [x] Iniciar la fase de arquitectura RAG: Configuración de la base de datos de vectores en PostgreSQL.
- [x] Investigar la integración de Redis para el almacenamiento de preguntas frecuentes (caché de respuestas).

---

## Sesión 4: Implementación de ejemplo de RAG

**Fecha:** [19/05/26]  

### Enfoque de la Sesión

*Entender completamente la arquitectura RAG .*

### Actividades realizadas

1. **Aprendizaje RAG teorico:** Durante la sesión de esta semana se repasaron diversos conceptos para poder hacer practica la implementacion del RAG: *Transformers, BD vectorial, embeddings, Retrivel, Similitud del Coseno*, así como las ventajas de en nuestro caso utilizar la arquitectura RAG y no realizar un Fine-Tuning, tambien se vieron diversas técnicas avanzadas para la implementación de RAG que hacen más sencillo que nuestro modelo recupere información.
2. **:** .
3. **:** .

### Aprendizajes clave

- .
- .

### Desafíos y Bloqueos

- **Error:** .
- **Solución:** .

### Reflexión Técnica

- **Análisis:** 
- **Decisión:**

### Evidencias y recursos

### Objetivos de la semana

- [ ] .

---

## Sesión 5: Redis

**Fecha:** [/05/26]  

### Enfoque de la Sesión

*.*

### Actividades realizadas

1. **:** .
2. **:** .
3. **:** .

### Aprendizajes clave

- .
- .

### Desafíos y Bloqueos

- **Error:** .
- **Solución:** .

### Reflexión Técnica

- **Análisis:** 
- **Decisión:**

### Evidencias y recursos

### Objetivos de la semana

- [ ] .

---
