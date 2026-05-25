"""Prompt-engineered safe reply generator using pre-trained DistilGPT-2."""
import os
import re
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from src.config import (
    DEVICE, REPLY_MODEL_NAME, REPLY_MAX_LEN, REPLY_TEMPERATURE,
    REPLY_TOP_P, REPLY_TOP_K
)

# Best-performing prompt template (to be validated empirically)
PROMPT_TEMPLATES = [
    "The following message may contain cyberbullying. Write a short, empathetic, de-escalatory, and safe reply to support the victim:\n\nMessage: {message}\n\nSafe Reply:",
    "A person received this hurtful message. Compose a brief, kind, calming response that supports them and discourages further conflict:\n\nMessage: {message}\n\nSupportive Reply:",
    "You are a compassionate moderator. Someone was sent this message. Write a short, safe, and reassuring reply:\n\nMessage: {message}\n\nModerator Reply:",
    "You are a highly trained crisis counselor and mental-health advocate. A vulnerable person has just received the following harmful message online.\n\nYour task:\n1. Acknowledge their pain without repeating slurs or hate speech.\n2. Offer a calming, empathetic, and empowering perspective.\n3. Keep the reply under 30 words.\n4. Do NOT be preachy, robotic, or generic. Be human and warm.\n5. End with a gentle offer of support.\n\nHarmful Message: {message}\n\nCompassionate Reply:",
]

class SafeReplyGenerator:
    def __init__(self, model_name=REPLY_MODEL_NAME, prompt_idx=0):
        self.model_name = model_name
        self.prompt_idx = prompt_idx
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(DEVICE)
        self.model.eval()
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id
    
    def generate(self, message, num_return_sequences=1):
        """Generate safe reply for a single message."""
        prompt = PROMPT_TEMPLATES[self.prompt_idx].format(message=message)
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                max_length=256)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=REPLY_MAX_LEN,
                temperature=REPLY_TEMPERATURE,
                top_p=REPLY_TOP_P,
                top_k=REPLY_TOP_K,
                do_sample=True,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=2,
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract reply after the prompt
        reply = generated_text[len(prompt):].strip()
        
        # Clean up: stop at newline or repetition
        reply = re.split(r"[\n\r]", reply)[0].strip()
        
        # Remove trailing incomplete sentences
        if reply and reply[-1] not in ".!?":
            last_punct = max(reply.rfind("."), reply.rfind("!"), reply.rfind("?"))
            if last_punct > 0:
                reply = reply[:last_punct+1]
        
        return reply if reply else "I'm sorry you had to see that. Please reach out if you need support."
    
    def generate_batch(self, messages, batch_size=8):
        """Generate replies for multiple messages."""
        results = []
        for i in range(0, len(messages), batch_size):
            batch = messages[i:i+batch_size]
            for msg in batch:
                results.append(self.generate(msg))
        return results
