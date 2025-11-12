# Reranking con `BAAI/bge-reranker-v2-m3` (Transformers)

Guía práctica para **entender y reutilizar** un *pipeline* de *reranking* con Hugging Face Transformers. Incluye:
- **Explicación línea por línea** del ejemplo base.
- **Varios casos de uso sencillos**, cada uno con una **introducción** y su código listo para copiar/pegar.
- Consejos básicos y errores comunes.

---

## 🧩 ¿Qué es el *reranking*?
Dado una **consulta (query)** y varios **documentos (candidatos)**, un *reranker* puntúa cada documento según su **relevancia semántica** respecto a la consulta y los ordena de mayor a menor.

El modelo **`BAAI/bge-reranker-v2-m3`** es un *cross-encoder* multilingüe muy usado en RAG y búsqueda semántica.

---

## 🚀 Ejemplo base (runnable)

```python
from transformers import pipeline

reranker = pipeline(
    task="text-classification",
    model="BAAI/bge-reranker-v2-m3",
    tokenizer="BAAI/bge-reranker-v2-m3",
    function_to_apply="sigmoid",
    device=-1  # CPU
)

query = "¿Qué es un panda?"
docs = [
    "Hola, ¿cómo estás?",
    "El panda gigante es una especie de oso endémica de China.",
    "Ayer ganó el Madrid en liga."
]
pairs = [{"text": query, "text_pair": d} for d in docs]
scores = reranker(pairs)

ranked = sorted(zip(docs, (r["score"] for r in scores)), key=lambda x: x[1], reverse=True)
for d, s in ranked:
    print(f"{s:.3f} | {d}")
```

### 💡 Requisitos mínimos
```bash
pip install transformers
# (Opcional para GPU) pip install torch --index-url https://download.pytorch.org/whl/cu121
```

> 📝 **Nota:** Cambia `device=-1` por `device=0` si tienes GPU disponible.

---

## 🔎 El mismo código con numeración de líneas (para comentar)

```python
 1| from transformers import pipeline
 2|
 3| reranker = pipeline(
 4|     task="text-classification",
 5|     model="BAAI/bge-reranker-v2-m3",
 6|     tokenizer="BAAI/bge-reranker-v2-m3",
 7|     function_to_apply="sigmoid",
 8|     device=-1  # CPU
 9| )
10|
11| query = "¿Qué es un panda?"
12| docs = [
13|     "Hola, ¿cómo estás?",
14|     "El panda gigante es una especie de oso endémica de China.",
15|     "Ayer ganó el Madrid en liga."
16| ]
17| pairs = [{"text": query, "text_pair": d} for d in docs]
18| scores = reranker(pairs)
19|
20| ranked = sorted(zip(docs, (r["score"] for r in scores)), key=lambda x: x[1], reverse=True)
21| for d, s in ranked:
22|     print(f"{s:.3f} | {d}")
```

### 🧠 Explicación línea por línea

| Línea(s) | Explicación |
|---|---|
| 1 | Importa `pipeline` de `transformers`, una interfaz de alto nivel para crear modelos listos para usar. |
| 3–9 | Crea el *pipeline* de **clasificación de texto** usando el modelo `BAAI/bge-reranker-v2-m3`. `function_to_apply="sigmoid"` mapea la salida a **[0,1]** (probabilidad). `device=-1` = CPU; `0` = primera GPU. |
| 11 | Define la **consulta** (query) a la que quieres encontrar respuestas relevantes. |
| 12–16 | Lista de **documentos candidatos** a ser reordenados por relevancia respecto a la query. |
| 17 | Crea los **pares** requeridos por el *cross-encoder*: `{ "text": query, "text_pair": documento }`. Uno por documento. |
| 18 | Pasa los pares al *pipeline*. Devuelve una lista con un `score` por par. |
| 20 | Combina cada documento con su `score` y ordénalos de mayor a menor relevancia. |
| 21–22 | Imprime el ranking con la puntuación formateada a tres decimales. |

---

## ⚠️ Errores comunes y tips rápidos
- **Entradas demasiado largas**: **recorta o “chunkea”** los documentos largos (p. ej., por frases o párrafos) y reordena los trozos.  
- **Lento en CPU**: prueba con **GPU** (`device=0`) o reduce el número de candidatos.  
- **Score bajo para todo**: revisa el **idioma** y la **calidad del texto**; el modelo es multilingüe pero necesita textos informativos.  
- **Confundir “pandas” (lib) con “panda” (animal)**: aporta **contexto suficiente** en la query o filtra antes por palabras clave.

---

# 🧪 Casos de uso con introducción + ejemplo

## Caso 1 — FAQ Bot (elige la mejor respuesta de una base pequeña)
### Introducción
En centros de ayuda o documentación interna, podemos tener **varias respuestas predefinidas**. El *reranker* selecciona la **más relevante** para la consulta del usuario sin necesidad de una base vectorial.

```python
from transformers import pipeline

reranker = pipeline(
    task="text-classification",
    model="BAAI/bge-reranker-v2-m3",
    tokenizer="BAAI/bge-reranker-v2-m3",
    function_to_apply="sigmoid",
    device=-1
)

faq = [
    ("¿Cómo cambio mi contraseña?",
     "Ve a Ajustes > Seguridad > Cambiar contraseña y sigue los pasos."),
    ("¿Cómo exporto a CSV?",
     "En Reportes, pulsa 'Exportar' y elige 'CSV'."),
    ("¿Cómo activo 2FA?",
     "Ajustes > Seguridad > 2FA. Escanea el código con tu app de autenticación.")
]

def answer(query, faqs, min_score=0.35):
    docs = [a for _, a in faqs]
    pairs = [{"text": query, "text_pair": a} for a in docs]
    scores = reranker(pairs)
    ranked = sorted(zip(faqs, (r["score"] for r in scores)), key=lambda x: x[1], reverse=True)
    (q_ref, best), score = ranked[0][0], ranked[0][1]
    return (best, score) if score >= min_score else ("No encontré una respuesta clara.", score)

query = "Necesito exportar mis datos a CSV, ¿cómo lo hago?"
resp, sc = answer(query, faq)
print(f"[{sc:.3f}] {resp}")
```

---

## Caso 2 — Mini‑RAG con párrafos cortos (chunking muy simple)
### Introducción
Para documentos más largos, cortamos el texto en **párrafos/“chunks”** y dejamos que el *reranker* decida **cuáles son los más útiles**. Ideal para recuperar **fragmentos concretos** antes de resumir o mostrar.

```python
from transformers import pipeline

reranker = pipeline(
    task="text-classification",
    model="BAAI/bge-reranker-v2-m3",
    tokenizer="BAAI/bge-reranker-v2-m3",
    function_to_apply="sigmoid",
    device=-1
)

def chunk_by_dots(text, max_chars=300):
    sents = [s.strip() for s in text.replace("\n"," ").split(".") if s.strip()]
    chunks, cur = [], ""
    for s in sents:
        if len(cur) + len(s) + 1 <= max_chars:
            cur = (cur + " " + s).strip()
        else:
            if cur: chunks.append(cur)
            cur = s
    if cur: chunks.append(cur)
    return chunks

doc_animal = (
    "El panda gigante es un oso nativo de China. "
    "Se alimenta principalmente de bambú. "
    "Habita en bosques templados montanos. "
    "Es un símbolo de conservación."
)
doc_lib = (
    "Pandas es una biblioteca de Python para análisis de datos. "
    "Permite manipular DataFrames, cargar CSV, filtrar y agrupar. "
    "También puede exportar a CSV con to_csv()."
)

corpus = [("panda (animal)", chunk_by_dots(doc_animal)),
          ("pandas (librería)", chunk_by_dots(doc_lib))]

def retrieve(query, top_k=3, min_score=0.25):
    pairs, meta = [], []
    for title, chunks in corpus:
        for ch in chunks:
            pairs.append({"text": query, "text_pair": ch})
            meta.append((title, ch))
    scores = reranker(pairs)
    ranked = sorted(zip(meta, (r["score"] for r in scores)), key=lambda x: x[1], reverse=True)
    return [(title, ch, s) for (title, ch), s in ranked[:top_k] if s >= min_score]

for q in ["¿Qué come un panda?", "¿Cómo exporto un DataFrame a CSV en pandas?"]:
    print("\nQuery:", q)
    for title, ch, s in retrieve(q, top_k=3):
        print(f"{s:.3f} | {title} -> {ch[:90]}{'...' if len(ch)>90 else ''}")
```

---

## Caso 3 — Elegir el snippet de código más útil
### Introducción
En asistentes de programación, podemos **priorizar ejemplos de código** según la intención del usuario. Ej.: si el usuario pregunta por **leer CSV por chunks**, el snippet con `chunksize` debería salir primero.

```python
from transformers import pipeline

reranker = pipeline(
    task="text-classification",
    model="BAAI/bge-reranker-v2-m3",
    tokenizer="BAAI/bge-reranker-v2-m3",
    function_to_apply="sigmoid",
    device=-1
)

query = "leer un CSV grande por chunks en pandas"
snippets = [
    "df = pd.read_csv('data.csv')",
    "for chunk in pd.read_csv('data.csv', chunksize=100000): procesa(chunk)",
    "df.to_csv('out.csv', index=False)"
]

pairs = [{"text": query, "text_pair": s} for s in snippets]
scores = reranker(pairs)
ranked = sorted(zip(snippets, (r["score"] for r in scores)), key=lambda x: x[1], reverse=True)

for s, score in ranked:
    print(f"{score:.3f} | {s}")
```

---

## 🧰 Bloque reutilizable: `rerank(query, docs, top_k=5, min_score=0.0)`
Función compacta que puedes importar en tus scripts.

```python
from transformers import pipeline

reranker = pipeline(
    task="text-classification",
    model="BAAI/bge-reranker-v2-m3",
    tokenizer="BAAI/bge-reranker-v2-m3",
    function_to_apply="sigmoid",
    device=-1
)

def rerank(query, docs, top_k=5, min_score=0.0):
    if not docs:
        return []
    pairs = [{"text": query, "text_pair": d} for d in docs]
    scores = reranker(pairs)
    ranked = sorted(zip(docs, (r["score"] for r in scores)), key=lambda x: x[1], reverse=True)
    out = [(doc, float(score)) for doc, score in ranked[:top_k] if float(score) >= min_score]
    return out

# Ejemplo rápido
if __name__ == "__main__":
    docs = ["hola", "El panda come bambú", "Madrid ganó ayer"]
    print(rerank("¿Qué come un panda?", docs))
```

---

## 📎 Licencia y crédito
- Código de ejemplo con fines educativos.
- Modelo: `BAAI/bge-reranker-v2-m3` (Hugging Face). Revisa su licencia y uso permitido en el *hub* del modelo.
