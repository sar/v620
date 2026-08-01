### Inference Getting Started

Since you are doing offline inference, you must perform a **Conversion Step** on a machine with enough RAM (CPU is fine) to load the safetensors and save them as `.hfq`.

Here is the step-by-step process to run your `qwen3-8b` (assuming Qwen2.5 or similar) model:

#### Step 1: Organize your SafeTensors
Ensure your model directory contains the standard files:
*   `config.json`
*   `tokenizer.json`
*   `model.safetensors` (or sharded files like `model-00001-of-0000X.safetensors`)

#### Step 2: Convert/Quantize to `.hfq`
You need to use the `hipfire-quantize` binary that was built in the previous Docker steps.

**Command:**
```bash
# Usage: hipfire-quantize <input_path> <output_path.hfq> [quantization_type]

# Example for your Qwen model:
./hipfire-quantize /path/to/qwen-safetensors/ ./qwen-8b-q4.hfq Q4_0
```

*   **`<input_path>`**: Folder containing your `safetensors` and `config.json`.
*   **`<output_path.hfq>`**: The output file. **Must end in `.hfq`**.
*   **`[quantization_type]`**: This is optional but recommended. Common values are `Q4_0` (fast, 4-bit), `Q8_0` (8-bit), or `F16` (no quantization, just container change). For an 8B model, `Q4_0` is recommended for speed.

*Note: This conversion process is usually CPU-based. It does not require a GPU, but it requires significant RAM to load the model weights.*

#### Step 3: Run Inference
Once you have the `.hfq` file, you can run inference using the `daemon` or the `hipfire` CLI.

**Option A: Using the CLI (One-off)**
```bash
# Place the model in the models directory
mkdir -p ~/.hipfire/models
mv ./qwen-8b-q4.hfq ~/.hipfire/models/

# Run
hipfire run qwen-8b-q4.hfq "Write a hello world program in Rust"
```

**Option B: Using the Daemon (Server Mode)**
This is better if you are running multiple prompts. The daemon keeps the model loaded in VRAM.

1.  **Start the Daemon:**
    ```bash
    ~/.hipfire/bin/daemon --model ~/.hipfire/models/qwen-8b-q4.hfq
    ```
    (It will print output saying it's listening, usually on port 8080 or similar).

2.  **Send Requests:**
    The daemon exposes an OpenAI-compatible API. You can use `curl`:
    ```bash
    curl http://localhost:8080/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "qwen-8b-q4",
        "messages": [{"role": "user", "content": "Hello!"}]
      }'
    ```

### Summary for your Setup
Since you are building on a Cloud VM (no GPU) and running on a Local Machine (with GPU):

1.  **Build**: Use the Docker process we defined to get the `hipfire-quantize` and `daemon` binaries.
2.  **Transfer**: Move the binaries AND your `safetensors` to your Local GPU Machine.
3.  **Quantize**: Run `./hipfire-quantize` on your Local GPU Machine to convert `safetensors` -> `.hfq`. (This is easier than trying to quantize on the cloud VM and transferring the model).
4.  **Run**: Execute `./daemon` or `hipfire run`.