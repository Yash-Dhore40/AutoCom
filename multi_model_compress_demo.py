import torch
import torchvision.models as models
import torch.nn.utils.prune as prune
import os

# Helper function to save the model and get its size
def save_and_get_size(model, filename):
    torch.save(model.state_dict(), filename)
    size_mb = os.path.getsize(filename) / 1e6
    return size_mb

def compress_model(model_name, model):
    print(f"\n{'='*50}")
    print(f"Starting compression for {model_name}...")
    print(f"{'='*50}")
    
    # Save baseline model and measure size
    baseline_file = f"baseline_{model_name.lower()}.pth"
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
    compressed_file = f"compressed_{model_name.lower()}.pth"
    compressed_size = save_and_get_size(quantized_model, compressed_file)
    print(f"Compressed Model Size: {compressed_size:.2f} MB (Saved as: {compressed_file})")
    
    compression_ratio = baseline_size / compressed_size
    print(f"\nCompression Ratio: {compression_ratio:.2f}x")

def main():
    print("AI Model Compression Demo")
    print("=========================")
    print("1. MobileNetV2")
    print("2. ResNet-50")
    print("3. VGG-16")
    print("4. Compress all models")
    print("5. Exit")
    
    choice = input("\nEnter your choice (1-5): ")
    
    models_to_test = {}
    
    if choice == '1':
        print("\nLoading MobileNetV2...")
        models_to_test["MobileNetV2"] = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    elif choice == '2':
        print("\nLoading ResNet-50...")
        models_to_test["ResNet50"] = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    elif choice == '3':
        print("\nLoading VGG-16...")
        models_to_test["VGG16"] = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    elif choice == '4':
        print("\nLoading all models...")
        models_to_test = {
            "MobileNetV2": models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT),
            "ResNet50": models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
            "VGG16": models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        }
    elif choice == '5':
        print("Exiting...")
        return
    else:
        print("Invalid choice. Exiting...")
        return
    
    for name, model in models_to_test.items():
        compress_model(name, model)
        
    print("\nExecution Completed Successfully! You can view the .pth files in your folder.")

if __name__ == "__main__":
    main()
