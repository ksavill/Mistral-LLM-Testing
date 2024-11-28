# First install required packages
# pip install torch transformers accelerate bitsandbytes
# pip install --upgrade transformers
# pip install safetensors

import torch
import os
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import logging
import time

"""
Ensure you have a hugging face account and have created an access token.

Accept the terms to use the Mistral model:
https://huggingface.co/mistralai/Mistral-7B-v0.1

Dependencies:
- torch
- transformers
- huggingface_hub
- accelerate
- bitsandbytes
"""

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def get_auth_token():
    """Retrieves the Hugging Face authentication token from environment variables."""
    logger.info("Attempting to retrieve HF_AUTH_TOKEN from environment")
    auth_token = os.getenv("HF_AUTH_TOKEN")
    if not auth_token:
        logger.error("HF_AUTH_TOKEN environment variable not found")
        raise EnvironmentError("HF_AUTH_TOKEN environment variable is not set.")
    logger.info("Successfully retrieved HF_AUTH_TOKEN")
    return auth_token

def get_device_config(use_gpu=False):
    if use_gpu:
        logger.info("Configuring for GPU usage")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,  # Use torch.bfloat16 if your GPU supports it
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=True
        )
        return "auto", quantization_config
    else:
        logger.info("Configuring for CPU-only usage")
        device_map = "cpu"
        quantization_config = None
        return device_map, quantization_config

def setup_mistral(use_gpu=False):
    logger.info(f"Starting Mistral model setup in {'GPU' if use_gpu else 'CPU'} mode")
    start_time = time.time()

    try:
        auth_token = get_auth_token()
        logger.info("Logging in to Hugging Face Hub")
        login(token=auth_token)
    except Exception as e:
        logger.error(f"Failed to authenticate: {str(e)}")
        raise

    model_id = "mistralai/Mistral-7B-v0.1"
    logger.info(f"Loading model: {model_id}")

    try:
        device_map, quantization_config = get_device_config(use_gpu)
        logger.info(f"Loading model with {'GPU' if use_gpu else 'CPU'} configuration")

        max_memory = {
            0: "7GB",    # Allocate up to 7GB of GPU memory
            "cpu": "9GB" # Allocate up to 9GB of system RAM
        }

        model_kwargs = {
            "pretrained_model_name_or_path": model_id,
            "device_map": "auto",
            "max_memory": max_memory,
            "token": auth_token,
            "offload_folder": "offload",
            "low_cpu_mem_usage": True,
            "offload_state_dict": True,
        }

        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["torch_dtype"] = torch.float16
        else:
            model_kwargs["torch_dtype"] = torch.float32

        model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
        logger.info("Model loaded successfully")

        logger.info("Loading tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=auth_token
        )
        logger.info("Tokenizer loaded successfully")

    except Exception as e:
        logger.error(f"Failed to load model or tokenizer: {str(e)}")
        raise

    setup_time = time.time() - start_time
    logger.info(f"Total setup time: {setup_time:.2f} seconds")

    return model, tokenizer

def create_text_pipeline(model, tokenizer):
    """Creates the text generation pipeline."""
    logger.info("Creating text generation pipeline")
    try:
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.1,  # Very low temperature
            top_p=0.1,        # Very selective sampling
            repetition_penalty=1.15
        )
        logger.info("Pipeline created successfully")
        return pipe
    except Exception as e:
        logger.error(f"Failed to create pipeline: {str(e)}")
        raise

def summarize_text(pipe, text):
    logger.info("Starting text summarization")
    prompt = f"""Summarize the following text concisely:
    
    {text}
    
    Summary:"""
    
    try:
        logger.debug(f"Input text length: {len(text)} characters")
        start_time = time.time()
        response = pipe(prompt, max_new_tokens=150, temperature=0.3)[0]['generated_text']
        summary = response.split("Summary:")[-1].strip()
        process_time = time.time() - start_time
        logger.info(f"Summarization completed in {process_time:.2f} seconds")
        return summary
    except Exception as e:
        logger.error(f"Summarization failed: {str(e)}")
        raise

def classify_text(pipe, text, categories):
    logger.info(f"Starting text classification with categories: {categories}")
    prompt = f"""Classify the following text into exactly one of these categories {categories}.
    Only return the category name, nothing else.
    
    Text: {text}
    
    Category:"""
    
    try:
        start_time = time.time()
        response = pipe(prompt, max_new_tokens=50, temperature=0.1)[0]['generated_text']
        category = response.split("Category:")[-1].strip()
        process_time = time.time() - start_time
        logger.info(f"Classification completed in {process_time:.2f} seconds")
        return category
    except Exception as e:
        logger.error(f"Classification failed: {str(e)}")
        raise

def extract_relevant_segments(pipe, text, query):
    logger.info(f"Starting relevant segment extraction for query: {query[:50]}...")
    prompt = f"""Find and extract the most relevant segment from the text that answers the query.
    
    Text: {text}
    Query: {query}
    
    Relevant segment:"""
    
    try:
        start_time = time.time()
        response = pipe(prompt, max_new_tokens=200, temperature=0.1)[0]['generated_text']
        segment = response.split("Relevant segment:")[-1].strip()
        process_time = time.time() - start_time
        logger.info(f"Extraction completed in {process_time:.2f} seconds")
        return segment
    except Exception as e:
        logger.error(f"Segment extraction failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Set use_gpu=True when you want to use GPU, False for CPU-only
    USE_GPU = True  # Easy toggle for GPU/CPU mode
    
    try:
        model, tokenizer = setup_mistral(use_gpu=USE_GPU)
        pipe = create_text_pipeline(model, tokenizer)
        
        # Example text
        sample_text = """
        Climate change poses significant challenges to global ecosystems. Rising temperatures 
        affect wildlife habitats, cause sea level rise, and increase extreme weather events. 
        Scientists project these impacts will intensify unless greenhouse gas emissions are 
        substantially reduced. Many countries have pledged to achieve net-zero emissions by 2050.
        """
        
        # Example uses
        logger.info("Running example tasks")
        
        print("\nSummary:")
        print(summarize_text(pipe, sample_text))
        
        print("\nClassification:")
        categories = ["Environmental", "Political", "Economic", "Social"]
        print(classify_text(pipe, sample_text, categories))
        
        print("\nRelevant Segment:")
        query = "What are countries doing about climate change?"
        print(extract_relevant_segments(pipe, sample_text, query))
        
        logger.info("Application completed successfully")
        
    except Exception as e:
        logger.error(f"Application failed: {str(e)}")
        raise