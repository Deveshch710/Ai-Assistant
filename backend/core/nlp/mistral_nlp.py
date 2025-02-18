# MistralNLP class with the Mistral-7B model
# This is the original MistralNLP class that uses the Mistral-7B model for generating text responses.Better in terms of gentrating the responses the quality is top notch but it requires a lot of memory and gpu to run this model.
# To try other model, go to the huggingface and add its model address in the model_name parameter.

# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# import torch

# class MistralNLP:
    # def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.3"):
    #     """Initialize the Mistral-7B Model."""
    #     self.device = "cuda" if torch.cuda.is_available() else "cpu"
    #     print(f"Using device: {self.device}")

    #     # Load model & tokenizer
    #     self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    #     self.model = AutoModelForCausalLM.from_pretrained(
    #         model_name,
    #         torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
    #         device_map="auto"
    #     )

    #     # Create text generation pipeline without specifying device
    #     self.generator = pipeline(
    #         "text-generation",
    #         model=self.model,
    #         tokenizer=self.tokenizer
    #     )

    # def generate_text(self, prompt, max_length=200):
    #     """Generate AI response based on user input."""
    #     # Format the prompt according to Mistral's instruction format
    #     formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        
    #     response = self.generator(
    #         formatted_prompt,
    #         max_length=max_length,
    #         num_return_sequences=1,
    #         temperature=0.7,
    #         do_sample=True,
    #         top_k=50,
    #         top_p=0.95
    #     )
        
    #     # Extract the generated text after the prompt
    #     generated_text = response[0]["generated_text"]
    #     # Remove the instruction formatting if present in the output
    #     if "[/INST]" in generated_text:
    #         generated_text = generated_text.split("[/INST]")[1].strip()
        
    #     return generated_text
    






# MistralNLP class with a smaller model for CPU
# This is a low-memory, low-CPU intensive model that can be used on CPU but has the problem that it's not a smart model.
# low cpu intensive model wjich can be used on cpu but have problem taht its not smart model. but with better specification and gpu and meory it can give us the best output in above model
# To try other model, go to the huggingface and add its model address in the model_name parameter.

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

class MistralNLP:
    def __init__(self, model_name="openai-community/gpt2"):  # Using a much smaller model
        """Initialize a smaller LLM suitable for CPU."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Load model & tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for CPU
            low_cpu_mem_usage=True,
            device_map="auto"
        )

        # Create text generation pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )

    def generate_text(self, prompt, max_length=200):
        """Generate AI response based on user input."""
        try:
            # Format the prompt
            formatted_prompt = f"<|system|>You are a helpful AI assistant.</s><|user|>{prompt}</s><|assistant|>"
            
            response = self.generator(
                formatted_prompt,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract the generated text after the prompt
            generated_text = response[0]["generated_text"]
            # Remove the system and user prompts from the output
            if "<|assistant|>" in generated_text:
                generated_text = generated_text.split("<|assistant|>")[1].strip()
            
            return generated_text
        except Exception as e:
            print(f"Error in generate_text: {str(e)}")
            return f"Error generating response: {str(e)}"
