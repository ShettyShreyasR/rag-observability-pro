
import torch
from llama_index.core import Settings, SummaryIndex, Document
from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata, CompletionResponseGen
from unsloth import FastLanguageModel
from typing import Any

# Stabilized RAG Configuration
def initialize_engine():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Llama-3.2-3B-Instruct",
        max_seq_length = 2048,
        load_in_4bit = True,
    )
    
    class LocalLlama(CustomLLM):
        @property
        def metadata(self) -> LLMMetadata:
            return LLMMetadata(context_window=2048, num_output=256, model_name="Llama-3.2-3B")

        def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    use_cache=False, # Stabilizer for T4 GPU shape mismatch
                    pad_token_id=tokenizer.eos_token_id,
                )
            response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
            return CompletionResponse(text=response_text.strip())

        def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
            yield self.complete(prompt, **kwargs)

    Settings.llm = LocalLlama()
    Settings.embed_model = "local:BAAI/bge-small-en-v1.5"
    return Settings
