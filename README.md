## Usage Example

```
from multirag import MultiModalEncoder, MultiModalIndex, Document, ContextualMemory, MultiModalRAGPipeline
from PIL import Image

# Initialize components
encoder = MultiModalEncoder()
index = MultiModalIndex(
    text_dimension=encoder.text_dimension,
    image_dimension=encoder.image_dimension
)
memory = ContextualMemory()

# Create pipeline
pipeline = MultiModalRAGPipeline(encoder, index, memory)

# Add documents
doc = Document(
    id="doc1",
    text="Sample text",
    image_path="sample.jpg"
)
image = Image.open(doc.image_path)

# Encode and index
doc.text_embedding = encoder.encode_text(doc.text)
doc.image_embedding = encoder.encode_image(image)
index.add_document(doc)

# Search and generate
results = pipeline.search(
    query_text="What is this about?",
    query_image=image
)
response = pipeline.generate("Explain this document", results)
```
