\# Week 4 – API Output Examples



This file shows example outputs from the FastAPI defect detection service.



\## Health check



Request:



```text

GET /health

```



Response:



```json

{

&#x20; "status": "ok",

&#x20; "model": "models\\\\best\_32epoch.onnx",

&#x20; "task": "detect"

}

```



\## Single-image prediction



Request:



```text

POST /predict

Content-Type: multipart/form-data



file: crazing\_103\_c85c40689d.jpg

annotate: false

```



Example response:



```json

{

&#x20; "filename": "crazing\_103\_c85c40689d.jpg",

&#x20; "inference\_ms": 310.22,

&#x20; "detections\_count": 0,

&#x20; "detections": \[]

}

```



\## Batch prediction



Request:



```text

POST /predict-batch

Content-Type: multipart/form-data



files: \[crazing\_103\_c85c40689d.jpg, scratches\_50.jpg]

annotate: false

```



Example response:



```json

{

&#x20; "files\_processed": 2,

&#x20; "total\_inference\_ms": 620.45,

&#x20; "results": \[

&#x20;   {

&#x20;     "filename": "crazing\_103\_c85c40689d.jpg",

&#x20;     "inference\_ms": 310.22,

&#x20;     "detections\_count": 0,

&#x20;     "detections": \[]

&#x20;   },

&#x20;   {

&#x20;     "filename": "scratches\_50.jpg",

&#x20;     "inference\_ms": 310.23,

&#x20;     "detections\_count": 0,

&#x20;     "detections": \[]

&#x20;   }

&#x20; ]

}

```



\## Logs



Example lines from `api.log`:



```text

2026-09-05 12:21:35,634 | INFO | POST /predict | file=crazing\_103\_c85c40689d.jpg | inference\_ms=7636.59 | detections=0 | annotate=False

2026-09-05 12:21:37,852 | INFO | POST /predict | file=crazing\_103\_c85c40689d.jpg | inference\_ms=277.59 | detections=0 | annotate=False

2026-09-05 12:21:47,949 | INFO | POST /predict | file=crazing\_103\_c85c40689d.jpg | inference\_ms=333.6 | detections=0 | annotate=False

2026-09-05 12:30:10,123 | INFO | POST /predict-batch | files=2 | total\_inference\_ms=140.31 | annotate=False

```

