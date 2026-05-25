"""Efficiency benchmarking utilities."""
import time
import torch
import psutil
import os

def benchmark_inference(model, texts, tokenizer=None, batch_size=32, device="cuda"):
    """
    Measure inference time and peak memory.
    
    Args:
        model: Model with predict() or generate() method.
        texts: List of input texts.
        tokenizer: Required for transformer models.
        batch_size: Batch size for inference.
        device: Device string.
    
    Returns:
        Dict with inference_time_sec, peak_vram_mb, peak_ram_mb.
    """
    process = psutil.Process(os.getpid())
    ram_before = process.memory_info().rss / (1024 ** 2)
    
    if torch.cuda.is_available() and device != "cpu":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    
    start = time.time()
    
    # Run inference
    if hasattr(model, "predict"):
        # Batch inference
        preds = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            preds.extend(model.predict(batch))
    else:
        # Generator
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            _ = model.generate_batch(batch)
    
    if torch.cuda.is_available() and device != "cpu":
        torch.cuda.synchronize(device)
    
    inference_time = time.time() - start
    
    ram_after = process.memory_info().rss / (1024 ** 2)
    peak_ram = ram_after - ram_before
    
    peak_vram = 0
    if torch.cuda.is_available() and device != "cpu":
        peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    
    return {
        "inference_time_sec": float(inference_time),
        "peak_vram_mb": float(peak_vram),
        "peak_ram_mb": float(peak_ram),
        "samples": len(texts)
    }


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)
