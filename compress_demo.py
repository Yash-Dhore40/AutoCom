import torch
import torchvision.models as models
import torch.nn.utils.prune as prune
import os

# Helper function to save the model and get its size
def save_and_get_size(model, filename):
    torch.save(model.state_dict(), filename)
    size_mb = os.path.getsize(filename) / 1e6
    return size_mb

print("Loading pre-trained MobileNetV2...")
# Load a pre-trained model
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

# Save baseline model and measure size
baseline_file = "baseline_mobilenet.pth"
baseline_size = save_and_get_size(model, baseline_file)
print(f"Baseline Model Size: {baseline_size:.2f} MB (Saved as: {baseline_file})")

print("\nApplying L1 Unstructured Pruning (removing 30% of connections)...")
# Prune the model (apply to all Conv2d and Linear layers)
for module in model.modules():
    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.3)
        # Remove pruning re-parameterization to make it permanent
        prune.remove(module, 'weight')

print("Applying Post-Training Dynamic Quantization (INT8)...")
# Move to CPU for Quantization (Dynamic Quantization works best on CPU for Linear layers)
model.to('cpu')
quantized_model = torch.ao.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Save compressed model and measure size
compressed_file = "compressed_mobilenet.pth"
compressed_size = save_and_get_size(quantized_model, compressed_file)
print(f"Compressed Model Size: {compressed_size:.2f} MB (Saved as: {compressed_file})")

compression_ratio = baseline_size / compressed_size
print(f"\nCompression Ratio: {compression_ratio:.2f}x")
print("Execution Completed Successfully! You can view the .pth files in your folder.")
