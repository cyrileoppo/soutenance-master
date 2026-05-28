# Model directory

The fine-tuned SentenceTransformer checkpoint is not stored in this repository.

Point `MODEL_PATH` to a locally mounted or synchronized Google Drive folder containing the exported model so the application can call:

```python
SentenceTransformer(MODEL_PATH)
```

Typical examples:

- `/content/drive/MyDrive/final_model`
- `G:/Mon Drive/final_model`
- `/Users/name/Library/CloudStorage/GoogleDrive/final_model`
