from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/multilingual-e5-small")

vector = model.encode("Парацетамол применяется для снижения температуры.")

print(f"Vector length: {len(vector)}")
print(f"First 5 values: {vector[:5]}")