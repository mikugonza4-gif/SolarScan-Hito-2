# Configuración y despliegue de SolarScan (Hito 2)

## 1. Servicio en Render
1. Comando de inicio: `uvicorn interfaz_solar:app --host 0.0.0.0 --port $PORT`.
2. Instancia recomendada: 1 CPU y 2 GB de RAM (PyTorch, EfficientNet-B0, la LSTM y el índice del manual).
3. Variables de entorno: `GEMINI_API_KEY`, `N8N_WEBHOOK_TOKEN` (un secreto largo y aleatorio) y
   `TABLERO_CSV_URL` (paso 3). Opcional: `GEMINI_MODEL`.
4. Comprobar: `/health` responde `listo`; `/serie/info` lista 22 inversores; `/tablero` carga.

## 2. Índice del manual y comparación de imágenes (GitHub Actions)
El flujo `.github/workflows/modelos-hito2.yml` se ejecuta al subir el código (o manualmente desde la
pestaña Actions) y guarda sus resultados en el repositorio:

- `rag/indice_fragmentos.json`, `rag/embeddings.npy`, `rag/config_rag.json` y `rag/resultados_rag.json`.
- `notebooks/resultados_imagenes/` y el notebook 03 ejecutado.

Para evaluar los embeddings hay que crear el secreto `GEMINI_API_KEY` en Settings > Secrets and
variables > Actions. Sin el secreto se construye el índice y se evalúa solo la regla fija; mientras no
exista `embeddings.npy`, el servicio calcula los vectores en la primera consulta. Si el índice no
existe, `/rag/buscar` devuelve contexto vacío y la recomendación declara "Sin información en los
manuales para este caso".

## 3. Google Sheets
1. Crear una planilla con dos hojas, `Recomendaciones` y `OrdenesTrabajo`, con los encabezados de
   `docs/plantilla_google_sheets.xlsx` (fila 1).
2. Archivo > Compartir > Publicar en la web > hoja `Recomendaciones` > formato CSV. Copiar la URL en
   `TABLERO_CSV_URL` de Render. La hoja publicada no contiene nombres ni correos.

## 4. n8n
1. Importar `n8n/SolarScan_Flujo_Errores.json`, `n8n/SolarScan_Flujo_Principal_Hito2.json` y
   `n8n/SolarScan_Flujo_Monitoreo_Diario.json`.
2. Reemplazar los textos `REEMPLAZAR_...`:
   - `REEMPLAZAR_CORREO_RESPONSABLE`: correo del supervisor (nodos Gmail).
   - `REEMPLAZAR_CHAT_ID_TELEGRAM`: chat del supervisor (nodos Telegram).
   - `REEMPLAZAR_ID_GOOGLE_SHEET`: identificador de la planilla (nodos Google Sheets).
3. Asignar credenciales: Gmail (OAuth), Telegram (token del bot), Google Sheets (OAuth) y, en el nodo
   "LLM · Recomendación integrada (Gemini)", una credencial Header Auth con nombre `X-SolarScan-Token`
   y el mismo valor de `N8N_WEBHOOK_TOKEN`.
4. En la configuración del flujo principal y del monitoreo diario, elegir como Error workflow
   "SolarScan · Avisos de error".
5. Activar los tres flujos y abrir la URL de producción del nodo "Formulario de inspección".

## 5. Prueba de extremo a extremo
| Caso | Inversor y fecha | Resultado esperado |
|---|---|---|
| Foto con polvo, pérdida severa | INV-11 · 2020-06-14 | Limpieza prioritaria, ALTA, pide aprobación |
| Foto limpia, pérdida sostenida | INV-04 · 2020-06-12 | Revisión eléctrica del inversor, ALTA, pide aprobación |
| Foto limpia, generación normal | INV-05 · 2020-06-16 | Seguimiento, BAJA, sin aprobación |
| Foto con daño físico | cualquier inversor | Inspección técnica, ALTA, pide aprobación |
