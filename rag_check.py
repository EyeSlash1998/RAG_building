from pinecone import Pinecone, ServerlessSpec
import time
import os
from dotenv import load_dotenv

load_dotenv()
# PINECONE_API = 'xxxxx' # Replacing with env


pc = Pinecone(api_key=os.environ.get('PINECONE_KEY'))

index_name = 'index1'

index = pc.Index(index_name)

print(index.describe_index_stats())