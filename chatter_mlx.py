from mlx_lm import load, generate
from llmlingua import PromptCompressor

class Chatter:
    """
    The `Chatter` class is a Python class that provides a chatbot functionality.
    It uses two pre-trained language models to generate responses based on user prompts and context.
    """

    def __init__(self, model_name: str = "mlx-community/NeuralBeagle14-7B-mlx"):
        """
        Initializes the `Chatter` class with the specified language model.

        Args:
            model_name (str): The name of the language model to use.
        """
        self.model, self.tokenizer = load(model_name)
        self.lingua_compressor = PromptCompressor(device_map="mps")
        self.context_model, self.context_tokenizer = load("mlx-community/NeuralBeagle14-7B-4bit-mlx")

    def chat(self, prompt: str, context: str = None, max_tokens: int = 400):
        """
        Generates a response and context based on the user prompt.

        Args:
            prompt (str): The user prompt.
            context (str): The context. Defaults to None.
            max_tokens (int): The maximum number of tokens for the response. Defaults to 4000.

        Returns:
            tuple[str, str]: The generated response and context.
        """
        response = self.generate_answer(prompt, context, max_tokens=max_tokens)
        if context:
            context_to_compress = "context: " + context
        else:
            context_to_compress = ""
        context_to_compress = context_to_compress + " \n\nprompt: " + prompt + " \n\nresponse: " + response
        compressed_prompt = self.lingua_compressor.compress_prompt(
            context_to_compress, instruction="Only use a few keywords for your answer.", question="What is the main content of the previous messages?", target_token=int(max_tokens/4))
        context = compressed_prompt["compressed_prompt"]
        context = self.generate_context(
            context, max_tokens=int(max_tokens/4))
        return response, context

    def generate_context(self, response: str, max_tokens: int = 100) -> str:
        """
        Generates a context based on the generated response.

        Args:
            response (str): The generated response.
            max_tokens (int): The maximum number of tokens for the context. Defaults to 100.

        Returns:
            str: The generated context.
        """
        # Create the messages list with system and response messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Summarizing messages into a concise context by using only a few keywords."},
            {"role": "message", "content": response},
        ]

        # Generate the chat prompt using the context_tokenizer
        chat_prompt = self.context_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Generate the context using the context_model and context_tokenizer
        context = generate(
            self.context_model,
            self.context_tokenizer,
            prompt=chat_prompt,
            verbose=False,
            max_tokens=max_tokens,
        )

        # Remove everything after "<|im_end|>"" in the context 
        context = context.split("<|im_end|>")[0]
        return context

    def generate_answer(self, prompt, context_vector=None, max_tokens=400):
        """
        Generates a response based on a user prompt and an optional context vector.

        Args:
            prompt (str): The user prompt.
            context_vector (str, optional): The context vector. Defaults to None.
            max_tokens (int, optional): The maximum number of tokens for the response. Defaults to 1000.

        Returns:
            str: The generated response.
        """
        # Combine the context vector with the prompt if the context is not empty
        if context_vector:
            combined_prompt = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "context", "content": context_vector},
                {"role": "user", "content": prompt}
            ]
        else:
            combined_prompt = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]

        combined_prompt = self.tokenizer.apply_chat_template(
            combined_prompt, tokenize=False, add_generation_prompt=True)

        # Generate the response using the model
        response = generate(self.model, self.tokenizer, prompt=combined_prompt,
                            verbose=False, max_tokens=max_tokens)
        # Remove everything after "<|im_end|>"" in the response
        response = response.split("<|im_end|>")[0]
        return response
