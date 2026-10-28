# DevDesk RAG Copilot API

Sistema de soporte técnico basado en RAG (Retrieval-Augmented Generation) para sugerir soluciones a tickets de TI usando documentación interna en Markdown, ChromaDB y Gemini.

## Descripción general

Este proyecto combina:

- FastAPI como API REST
- ChromaDB para almacenamiento vectorial
- Sentence Transformers para embeddings semánticos
- Google Gemini para generación de respuestas
- PostgreSQL/SQLAlchemy para persistencia relacional
- Documentos Markdown como base de conocimiento

La idea es procesar manuales técnicos internos, indexarlos como chunks semánticos y, cuando llega un ticket, recuperar los fragmentos más relevantes para generar una recomendación técnica con referencias a las fuentes usadas.

## Funcionalidades

- Indexado de documentos Markdown en ChromaDB
- Búsqueda semántica por similitud
- Generación de respuesta contextual con Gemini
- Persistencia del historial de consultas
- Endpoint para consultar tickets o resolver un ticket concreto
- Documentación automática con Swagger en `/docs`

## Arquitectura

```text
Cliente / Frontend
        |
        v
FastAPI /main.py
        |
        +--> ChromaDB (base vectorial de conocimiento)
        |
        +--> PostgreSQL (tickets, sugerencias, historial)
        |
        +--> Gemini API (respuesta generada)
        |
        +--> docs_conocimiento/*.md (manuales internos)
```

## Requisitos

- Python 3.10+
- pip
- PostgreSQL (recomendado para compatibilidad con los queries del proyecto)
- API Key de Google Gemini
- Git

## Estructura del proyecto

```text
rag-copilot/
├─ main.py                  # API principal FastAPI
├─ ingest.py                # Indexado de documentos a ChromaDB
├─ database.py              # Base SQLAlchemy
├─ models.py                # Modelo de historial RAG
├─ requirements.txt          # Dependencias del proyecto
├─ docs_conocimiento/       # Manuales internos en Markdown
├─ chroma_db/               # Base persistente de ChromaDB
├─ .env                     # Variables de entorno locales
├─ .env.example             # Plantilla de configuración
├─ EJERCICIO_GUIA_RAG_HELPDESK.md
├─ sugerencias_rag.sql      # SQL de referencia para esquema de sugerencias
├─ test_models.py           # Validación de modelos
└─ venv/                    # Entorno virtual local
```

## Requisitos de instalación

### 1) Clonar el proyecto

```bash
git clone <url-del-repositorio>
cd rag-copilot
```

### 2) Crear entorno virtual

En Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

En macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3) Instalar dependencias

```bash
pip install -r requirements.txt
```

## Configuración de variables de entorno

Crea un archivo `.env` en la raíz del proyecto con esta estructura:

```env
DATABASE_URL=postgresql+psycopg2://usuario:password@localhost:5432/devdesk_rag
GEMINI_API_KEY=tu_clave_api_de_gemini
GEMINI_MODEL=gemini-3.8-flash
RAG_API_KEY=crea_una_clave_interna_larga_y_aleatoria
FRONTEND_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

> Importante: El proyecto consulta tablas con nombre `public.tickets` y `public.sugerencias_rag`, por lo que se recomienda PostgreSQL. Si usas SQLite, deberás ajustar la estructura y las consultas SQL.

Los endpoints RAG requieren el header `X-RAG-API-Key` con el valor de
`RAG_API_KEY`. Nunca envíes `GEMINI_API_KEY` al frontend. Configura
`FRONTEND_ORIGINS` con los orígenes permitidos separados por comas.

También puedes usar el archivo de ejemplo incluido:

```bash
copy .env.example .env
```

## Base de conocimiento

El proyecto espera que la carpeta `docs_conocimiento/` contenga archivos `.md` con documentación interna. Ejemplos válidos:

- `manual_vpn_remoto.md`
- `soporte_internet.md`
- `solucion_correo.md`
- `soporte_fotocopiadora.md`
- `soporte_perifericos.md`

Estos archivos se dividirán en chunks, se convertirán a embeddings y se almacenarán en ChromaDB.

## Ingesta los documentos

Desde la raíz del proyecto:

```bash
python ingest.py
```

Esto crea o reutiliza la colección `manuales_ti` dentro de `chroma_db/` y guarda los documentos indexados.

## Ejecutar la aplicación

Inicia la API:

```bash
uvicorn main:app --reload --port 8000
```

La documentación interactiva queda disponible en:

- http://localhost:8000/docs
- http://localhost:8000/redoc

## Endpoints principales

### 1) Consultar soporte con RAG

```http
POST /api/rag/consultar
```

Body de ejemplo:

```json
{
  "consulta": "No puedo conectarme a la VPN desde mi casa",
  "usuario_id": 101,
  "ticket_id": null
}
```

### 2) Resolver una sugerencia para un ticket existente

```http
POST /api/tickets/{ticket_id}/sugerir-solucion
```

Este endpoint:

1. lee el ticket desde la base relacional
2. genera un embedding del texto del ticket
3. recupera documentos relevantes en ChromaDB
4. arma un prompt con el contexto encontrado
5. envía la consulta a Gemini
6. guarda la sugerencia y el historial asociado

### 3) Consultar historial

```http
GET /api/rag/historial
```

## Flujo recomendado de trabajo

1. Crear y activar entorno virtual
2. Instalar dependencias
3. Configurar `.env` con `DATABASE_URL` y `GEMINI_API_KEY`
4. Colocar manuales internos en `docs_conocimiento/`
5. Ejecutar `python ingest.py`
6. Iniciar la API con `uvicorn`
7. Probar los endpoints desde Swagger o Postman
8. Revisar el historial en `/api/rag/historial`

## Modelo de datos relevante

El proyecto asume estas entidades principales:

- `tickets`
- `sugerencias_rag`
- `historial_rag`

En `models.py` se define la tabla `HistorialRAG`, y en la lógica principal se persisten sugerencias de respuesta junto con sus fuentes.

## Buenas prácticas

- Mantén los documentos en Markdown limpios y estructurados
- Usa títulos y subtítulos claros en los manuales
- Reindexa cuando agregues o modifiques documentación
- Mantén la consulta y la respuesta dentro del contexto del soporte técnico
- No inventes procedimientos: la respuesta debe basarse solo en los manuales internos

## Solución de problemas

### Error: `Falta DATABASE_URL en el archivo .env`
Asegúrate de que el archivo `.env` exista y contenga la variable `DATABASE_URL`.

### Error: `Falta GEMINI_API_KEY en el archivo .env`
Verifica que la clave sea válida y que no esté vacía.

### Error: no se encuentran documentos para indexar
Comprueba que la carpeta `docs_conocimiento/` tenga archivos `.md` válidos.

### Error: `Ticket no encontrado`
Confirma que el `ticket_id` exista en la base de datos y que la consulta SQL del proyecto sea compatible con tu esquema.

### El servidor no responde
Puedes verificarlo con:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Notas finales

Este repositorio está pensado como una base sólida para un sistema de ayuda técnica basado en IA generativa con contexto documental. Puedes ampliarlo con:

- autenticación JWT
- panel web para tickets
- manejo de usuarios y permisos
- más colecciones de conocimiento
- archivos PDF/DOCX junto con Markdown
- métricas de utilidad de las respuestas

## Licencia

Este proyecto se entrega como base de aprendizaje y desarrollo práctico. Ajusta la licencia según el uso que le des al proyecto.
