# AI Model Compression Demo

This project demonstrates how to compress large Deep Learning models using **PyTorch**. The script takes pre-trained models, reduces their size, and optimizes them for deployment without completely sacrificing accuracy.

It currently supports compressing the following models:
- **MobileNetV2**
- **ResNet-50**
- **VGG-16**

## How it works
The demo applies two key compression techniques:
1. **L1 Unstructured Pruning**: Removes 30% of the less important connections (weights) in the Convolutional and Linear layers.
2. **Post-Training Dynamic Quantization**: Converts the 32-bit floating-point (FP32) weights of the Linear layers into 8-bit integers (INT8), significantly reducing the model size and improving inference speed on CPUs.

---

## Step-by-Step Guide to Run the Demo

### Prerequisites
Make sure you have **Python** (3.8 or newer) installed on your system.

### Step 1: Download the Repository
Clone this repository to your local machine using git, or simply download the `.zip` file and extract it.
```bash
git clone <your-github-repo-url>
cd <your-repo-folder>
```

### Step 2: Install Dependencies
The script relies on PyTorch and Torchvision. You can easily install the required libraries using the provided `requirements.txt` file.

Run the following command in your terminal:
```bash
pip install -r requirements.txt
```

### Step 3: Run the Script
Once the dependencies are installed, you can start the compression demo by running:
```bash
python multi_model_compress_demo.py
```

### Step 4: Choose a Model
Upon running the script, you will be presented with an interactive menu:
```text
AI Model Compression Demo
=========================
1. MobileNetV2
2. ResNet-50
3. VGG-16
4. Compress all models
5. Exit

Enter your choice (1-5): 
```
Enter the number corresponding to the model you want to compress (or press `4` to process them all at once).

### Step 5: View the Results
The script will automatically download the pre-trained weights (if it's your first time running it), apply pruning and quantization, and save the models locally.

For each model you compress, two `.pth` files will be saved in the same directory:
- `baseline_<model>.pth` (The original, uncompressed model)
- `compressed_<model>.pth` (The optimized, smaller model)

The console output will display the size comparison in Megabytes (MB) and calculate the total **Compression Ratio**.
