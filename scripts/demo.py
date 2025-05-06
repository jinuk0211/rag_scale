from generator import retriever
  
if __name__ == "__main__":
  doc = retriever.search_document_demo("What is the relationship between the lifetime of a quantum state and its energy uncertainty?",2)
  for retrieved_document in doc:
      print(retrieved_document['text'])
