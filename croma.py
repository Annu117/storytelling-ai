from chromadb import PersistentClient

# Ensure database path is correct
db_path = "chroma_db"  # Change to absolute path if needed

# Reinitialize ChromaDB
client = PersistentClient(path=db_path, tenant="default_tenant", database="default_database")

# Explicitly create a collection before using it
if "stories" not in [c.name for c in client.list_collections()]:
    client.create_collection("stories")
