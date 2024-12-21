from pinecone import Pinecone, ServerlessSpec
import time
import os
from dotenv import load_dotenv

load_dotenv()
# PINECONE_API = 'xxxxx' # Replacing with env


pc = Pinecone(api_key=os.environ.get('PINECONE_KEY'))

index_name = 'index1'

index = pc.Index(index_name)

query = 'Tell me about a tech company known as Apple'

embedding = pc.inference.embed(
    model='multilingual-e5-large',
    inputs=[query],
    parameters={
        'input_type': 'query'
    }
)

results = index.query(
    namespace='ns1',
    vector=embedding[0].values,
    top_k=3,
    include_values=False,
    include_metadata=True
)

print(results)