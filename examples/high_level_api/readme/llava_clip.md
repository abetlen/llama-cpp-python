# llava_clip.py

This Python script utilizes the Llama library for image description. It encodes an image into a base64 data URI, initializes a chat handler using the Llava15ChatHandler, and then generates a detailed description of the image using the Llama model.

## Prerequisites

    - llama-cpp-python
    - base64

## Setup

1. Clone the repository:

```bash
git clone https://github.com/abetlen/llama-cpp-python.git
cd llama-cpp-python/examples/high_level_api/
```

2.  Install the required dependencies:

   ```bash
    pip install llama-cpp-python
   ```

3.    Download the Llama model files and place them in the specified folders:

    - Download mmproj-model-f16.gguf and llava_v1.5_7b_q4_k.gguf from llama-models-repo and place them in path/to/your/folder/llava/.

    - Ensure that the image file is located in path/to/your/image.png.

### Usage

make sure to fill model folder path and image path at the top of llava_cli.py file
```python 
llava_folder = "path/to/folder/llava/llava_v1.5_7b/"
file_path = 'path/to/your/image.png'

clip_model_path = f"{llava_folder}mmproj-model-f16.gguf"
model_path = f"{llava_folder}llava_v1.5_7b_q4_k.gguf"
```
image format tested:

    - .png
    - .jpg

Run the script using the following command:

   ```bash
python llava_clip.py
   ```
- The script will load the image
- encode it
- initialize the chat handler
- then generate a description of the image.


## Configuration

    - llava_folder: Path to the folder containing Llama model files.
    - file_path: Path to the image file.
    - clip_model_path: Path to the CLIP model file.
    - model_path: Path to the Llama model file.
    - n_ctx: Number of context tokens for the Llama model.
    - logits_all: Boolean flag needed to be set True to make Llava work.

### Output

The script will print a generated description of the image.

    - first ouput contain the whole response object.
    - second output contain the response text.



## License

This project is licensed under the MIT License.
Acknowledgments

