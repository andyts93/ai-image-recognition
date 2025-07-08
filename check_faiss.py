import faiss
from config import *

index = faiss.read_index("faiss_indices/index_33.index")

print("Numero vettori: ", index.ntotal)
print("Dimensione dei vettori: ", index.d)