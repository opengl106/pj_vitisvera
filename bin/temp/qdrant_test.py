from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
import numpy as np

from src.utils.consts.services import QDRANT_HOST, QDRANT_PORT

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

if not client.collection_exists("my_collection"):
   client.create_collection(
      collection_name="my_collection",
      vectors_config=VectorParams(size=100, distance=Distance.COSINE),
   )

query_vector = np.random.rand(100)
hits = client.query_points(
   collection_name="my_collection",
   query=query_vector,
   limit=5,
   with_vectors=True
)
[np.dot(query_vector, point.vector) for point in hits.points]

p = client.retrieve(
   collection_name="my_collection",
   ids=[85],
   with_vectors=True
)
p[0].vector
