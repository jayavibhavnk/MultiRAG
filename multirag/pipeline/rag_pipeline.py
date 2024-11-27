from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List, Dict, Optional, Union

class MultiModalRAGPipeline:
    def __init__(
        self,
        encoder,
        index,
        memory,
        llm_model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    ):
        self.encoder = encoder
        self.index = index
        self.memory = memory
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(llm_model_name)
        
    def search(
        self,
        query_text: Optional[str] = None,
        query_image: Optional[Any] = None,
        k: int = 3
    ) -> List[Document]:
        results = []
        
        if query_text:
            text_vector = self.encoder.encode_text(query_text)
            D, I = self.index.text_index.search(np.array([text_vector]), k)
            results.extend([self.index.documents[i] for i in I[0]])
            
        if query_image:
            image_vector = self.encoder.encode_image(query_image)
            D, I = self.index.image_index.search(np.array([image_vector]), k)
            results.extend([self.index.documents[i] for i in I[0]])
            
        self.memory.add((query_text, query_image), results)
        return results
    
    def generate(
        self,
        query: str,
        retrieved_docs: List[Document],
        max_length: int = 512
    ) -> str:
        context = self._prepare_context(retrieved_docs)
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
