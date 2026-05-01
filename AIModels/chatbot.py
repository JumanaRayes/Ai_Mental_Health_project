import gc


import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class MistralChatbot:
    def __init__(self, model_id="mistralai/Mistral-7B-Instruct-v0.2"):
        self.model_id = model_id
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        """Loads the tokenizer and the model based on hardware availability."""
        if self.model is not None and self.tokenizer is not None:
            return  # Already loaded

        print(f"Loading Tokenizer for {self.model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading Model: {self.model_id} on {self.device}")

        if self.device == "cuda":
            # 1. Clear GPU cache
            torch.cuda.empty_cache()
            gc.collect()

            # 2. Configure 4-bit quantization (requires CUDA)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

            #   quantization_config=bnb_config,

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map="auto",
                torch_dtype=torch.float16,  # try float16
                low_cpu_mem_usage=True,
            )
            print("Model loaded with 4-bit quantization on GPU.")
        else:
            # Fallback for CPU / no CUDA
            print(
                "WARNING: CUDA not detected. Loading full model on CPU. This will consume high RAM and be slow."
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, device_map="cpu", torch_dtype=torch.float32
            )
            print("Model loaded on CPU.")

    def generate_response(
        self, user_message: str, max_new_tokens: int = 250, temperature: float = 0.7
    ) -> str:
        """Generates a text response from the model."""
        if self.tokenizer is None or self.model is None:
            raise RuntimeError(
                "Model and tokenizer are not loaded. Call load_model() first."
            )

        prompt = f"<s>[INST] {user_message} [/INST]"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        print("Thinking...")
        with torch.no_grad():  # Saves memory
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True if temperature > 0 else False,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        final_answer = (
            response.split("[/INST]")[1].strip()
            if "[/INST]" in response
            else response.strip()
        )

        return final_answer


def run_gradio():
    """Starts a Gradio Chat Interface if this file is run directly."""
    chatbot = MistralChatbot()
    print("Initializing Chatbot UI...")

    # Let's load model lazily or immediately
    chatbot.load_model()

    def gradio_interface(message, history):
        return chatbot.generate_response(message)

    demo = gr.ChatInterface(
        fn=gradio_interface,
        title="Mental Health Support Chatbot",
        description="This AI assistant provides empathetic emotional support.",
        examples=[
            "I feel so stressed about my exams and graduation.",
            "I've been feeling very lonely lately.",
            "How can I deal with my anxiety?",
        ],
        theme="soft",
    )
    demo.launch(share=True)


if __name__ == "__main__":
    run_gradio()
