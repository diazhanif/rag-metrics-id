# rag-metrics-id

CLI dan library tanpa dependensi untuk mengukur kualitas *ranking* hasil retrieval dalam eksperimen RAG. Berikan daftar ID dokumen yang dianggap relevan dan ranking yang dikembalikan retriever; tool menghitung Precision@k, Recall@k, MRR@k, nDCG@k, dan Hit Rate@k.

## Format masukan

```json
[
  {"relevant": ["doc-a", "doc-c"], "retrieved": ["doc-c", "doc-b", "doc-a"]},
  {"relevant": ["doc-x"], "retrieved": ["doc-y"]}
]
```

Jalankan dari stdin:

```sh
cat eval.json | rag-metrics-id --k 5
```

Bisa juga dipanggil tanpa instalasi perintah:

```sh
cat eval.json | python -m rag_metrics_id --k 5
```

Laporan JSON memuat metrik makro (rata-rata per query) dan skor per baris dataset. Query dan ID dokumen tidak disalin ke laporan.

## Catatan metrik

Precision@k memakai `k` sebagai penyebut walau retriever mengembalikan kurang dari `k` hasil. Recall@k membandingkan hasil relevan di top-k dengan semua ID relevan yang diberikan. MRR dan nDCG menghargai dokumen relevan yang muncul lebih awal; nDCG memakai relevansi biner. Dataset harus memberi setidaknya satu ID relevan per query, dan ID dalam tiap daftar harus unik.

## Batasan dan privasi

Tool ini menghitung kecocokan terhadap label relevansi yang Anda sediakan; ia tidak menentukan relevansi semantik, menilai jawaban AI, atau mengirim data ke layanan luar. Hanya metrik dan nomor baris yang dicetak, bukan query atau ID dokumen. Tidak ada API atau dependensi runtime.

Python 3.9+. Instalasi: `python -m pip install .`

Tes: `python -m unittest discover -s tests -v`

## Lisensi

MIT. Lihat [LICENSE](LICENSE).
