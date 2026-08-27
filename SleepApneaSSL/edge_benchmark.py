import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np

try:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
except ImportError:
    print("[!] Error: Missing required ONNX packages.")
    print("    Please run: pip install onnx onnxruntime")
    sys.exit(1)

# Import model architecture
from downstream_finetune import SleepApneaClassifier
from model import STFTEncoder2D

def get_file_size_mb(path):
    """Returns the file size in MB."""
    if not os.path.exists(path):
        return 0.0
    return os.path.getsize(path) / (1024 * 1024)

class EdgeDeploymentWrapper(nn.Module):
    """
    Wraps the classifier to bypass the PyTorch STFT step during ONNX export.
    In edge deployment, STFT is typically computed via hardware DSP (e.g., CMSIS-DSP),
    and the neural network only processes the resulting spectrogram.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, specs):
        features = self.model.encoder(specs, input_is_spec=True)
        logits = self.model.classifier(features)
        return logits.view(-1)

def export_and_benchmark():
    print("=" * 60)
    print("  ONNX EXPORT & INT8 QUANTIZATION BENCHMARK")
    print("=" * 60)

    # 1. Initialize PyTorch Model
    weights_path = r'E:\SleepApnea\SleepApneaSSL\best_downstream_model.pth'
    if not os.path.exists(weights_path):
        weights_path = r'E:\SleepApnea\SleepApneaSSL\clinical_finetuned_model.pth'
        
    print("[+] Loading PyTorch model...")
    encoder = STFTEncoder2D(in_channels=20, embed_dim=128)
    model = SleepApneaClassifier(encoder, embed_dim=128)
    
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
        print(f"    Loaded weights from: {os.path.basename(weights_path)}")
    else:
        print(f"[!] Warning: Downstream weights not found. Using randomly initialized model.")
        
    model.eval()
    model.cpu()
    
    # Wrap for Edge Export
    edge_model = EdgeDeploymentWrapper(model)
    edge_model.eval()

    # 2. ONNX Export (FP32)
    fp32_onnx_path = r'E:\SleepApnea\SleepApneaSSL\sleep_apnea_fp32.onnx'
    int8_onnx_path = r'E:\SleepApnea\SleepApneaSSL\sleep_apnea_int8.onnx'
    
    # Generate dummy raw input and compute the STFT shape dynamically
    dummy_raw = torch.randn(1, 20, 3000, dtype=torch.float32)
    with torch.no_grad():
        dummy_spec = model.encoder.stft(dummy_raw)
    
    print(f"[+] Exporting model to ONNX (FP32) with DSP decoupled...")
    print(f"    Expected Spectrogram Input Shape: {list(dummy_spec.shape)}")
    
    torch.onnx.export(
        edge_model,
        dummy_spec,
        fp32_onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['spectrogram'],
        output_names=['logits'],
        dynamic_axes={'spectrogram': {0: 'batch_size'}, 'logits': {0: 'batch_size'}}
    )
    print(f"    Saved: {os.path.basename(fp32_onnx_path)}")

    # 3. Dynamic INT8 Quantization
    print("[+] Quantizing ONNX model to INT8...")
    quantize_dynamic(
        model_input=fp32_onnx_path,
        model_output=int8_onnx_path,
        weight_type=QuantType.QUInt8
    )
    print(f"    Saved: {os.path.basename(int8_onnx_path)}")

    # 4. Latency & Footprint Benchmarking
    print("[+] Initializing ONNX Runtime for INT8 model...")
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(int8_onnx_path, sess_options, providers=['CPUExecutionProvider'])
    
    input_name = session.get_inputs()[0].name
    
    # Pre-allocate numpy array for dummy inference
    dummy_np = dummy_spec.numpy()
    
    print("[i] Warming up execution provider (50 iterations)...")
    for _ in range(50):
        _ = session.run(None, {input_name: dummy_np})
        
    print("[i] Running latency benchmark (500 iterations)...")
    latencies = []
    num_runs = 500
    
    for _ in range(num_runs):
        start_time = time.perf_counter()
        _ = session.run(None, {input_name: dummy_np})
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000.0) # ms
        
    latencies = np.array(latencies)
    mean_latency = np.mean(latencies)
    p99_latency = np.percentile(latencies, 99)
    
    # Real-Time Factor (RTF) = Total inference time (s) / Length of audio window (s)
    # Window length = 30.0 seconds
    mean_latency_sec = mean_latency / 1000.0
    rtf = mean_latency_sec / 30.0

    # Footprint
    pth_size = get_file_size_mb(weights_path)
    fp32_size = get_file_size_mb(fp32_onnx_path)
    int8_size = get_file_size_mb(int8_onnx_path)
    
    print("\n" + "=" * 60)
    print("  BENCHMARK RESULTS")
    print("=" * 60)
    
    print("\n--- MEMORY FOOTPRINT ---")
    print(f"{'Format':<15} | {'File Size (MB)':<15}")
    print("-" * 35)
    print(f"{'PyTorch (.pth)':<15} | {pth_size:>10.2f} MB")
    print(f"{'ONNX FP32':<15} | {fp32_size:>10.2f} MB")
    print(f"{'ONNX INT8':<15} | {int8_size:>10.2f} MB")
    
    reduction_factor = pth_size / int8_size if int8_size > 0 else 0
    print(f"\n>> Size Reduction: {reduction_factor:.1f}x smaller than original PyTorch model")
    
    print("\n--- LATENCY (Single Window, 30s EEG) ---")
    print(f"{'Metric':<15} | {'Value':<15}")
    print("-" * 35)
    print(f"{'Mean Latency':<15} | {mean_latency:>10.2f} ms")
    print(f"{'P99 Latency':<15} | {p99_latency:>10.2f} ms")
    print(f"{'Real-Time Factor':<15} | {rtf:>10.5f}")
    
    print("\n[+] Edge deployment pipeline verified.")

if __name__ == '__main__':
    export_and_benchmark()
