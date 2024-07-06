# Pickles to Safetensors Converter

This Python script converts PyTorch pickle (.pt) files to the Safetensors format. It's useful for converting both embedding and VAE (Variational Autoencoder) files.

## GitHub Repository

https://github.com/BitsofJeremy/pickles_to_safetensors

## What does this script do?

In simple terms, this script takes PyTorch model files (which end with .pt) and converts them into a format called Safetensors. This new format is safer and more efficient for storing machine learning models.

## Requirements

Before you start, make sure you have:

- Python 3.6 or newer installed on your computer
- pip (Python package installer) - this usually comes with Python

The script needs these Python libraries:
- PyTorch
- safetensors
- numpy

Don't worry if you don't have these yet - we'll install them in the next steps!

## Installation

Follow these steps to get the script ready on your computer:

1. Open your terminal (Mac/Linux) or command prompt (Windows).

2. Clone (download) the repository:
   ```
   git clone https://github.com/BitsofJeremy/pickles_to_safetensors.git
   ```

3. Move into the project folder:
   ```
   cd pickles_to_safetensors
   ```

4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
   This command reads the requirements.txt file and installs all the necessary libraries.

## How to Use

To use the script, you'll run it from the command line with some specific instructions. Here's the basic format:

```
python p2s.py <path> <model_type> [--verbose]
```

Let's break down what each part means:
- `python p2s.py`: This tells Python to run our script.
- `<path>`: Replace this with the location of your .pt file or a folder containing .pt files.
- `<model_type>`: This should be either "embedding" or "vae", depending on what type of model you're converting.
- `[--verbose]`: This is optional. If you include it, the script will give you more detailed information as it runs.

### Examples

Here are some examples to help you get started:

1. Convert a single embedding file:

   On Mac/Linux:
   ```
   python p2s.py /Users/YourName/Documents/my_embedding.pt embedding
   ```

   On Windows:
   ```
   python p2s.py C:\Users\YourName\Documents\my_embedding.pt embedding
   ```

2. Convert all VAE files in a folder:

   On Mac/Linux:
   ```
   python p2s.py /Users/YourName/Documents/vae_folder vae --verbose
   ```

   On Windows:
   ```
   python p2s.py C:\Users\YourName\Documents\vae_folder vae --verbose
   ```

   Replace `/Users/YourName/Documents/` or `C:\Users\YourName\Documents\` with the actual path on your computer.

## What happens after you run the script?

The script will create new files with the .safetensors extension in the same folder as your original .pt files. Don't worry - your original files won't be changed or deleted.

## Implications of Conversion

Converting from PyTorch's .pt format to Safetensors format is generally safe and beneficial, but it's important to understand the implications:

1. **File Size**: Safetensors files are typically slightly larger than their PyTorch counterparts. This is because Safetensors prioritizes safety and fast loading over file size.

2. **Compatibility**: Most modern AI frameworks and libraries support Safetensors, but some older systems might not. Ensure your target system supports Safetensors before converting.

3. **Safety**: Safetensors format is designed to be safer than PyTorch's pickle format, reducing the risk of arbitrary code execution when loading models.

4. **Loading Speed**: Safetensors files typically load faster than PyTorch files, especially for larger models.

5. **Data Preservation**: The conversion process preserves all tensor data and most metadata. However, some PyTorch-specific metadata that isn't relevant to the model weights might not be transferred.

6. **Reversibility**: The conversion to Safetensors is not reversible. While you can convert from Safetensors back to PyTorch format, some PyTorch-specific metadata might be lost in the process.

7. **Model Functionality**: The conversion should not affect the functionality of your model. The weights and architecture remain the same; only the storage format changes.

It's always a good practice to:
- Keep backups of your original .pt files.
- Test the converted Safetensors files in your specific use case to ensure they work as expected.
- If you're working on a critical project, consider keeping both versions until you've thoroughly tested the Safetensors version.

If you encounter any unexpected behavior after conversion, please open an issue on the GitHub repository.

## Troubleshooting

If you run into any issues:
1. Make sure you've installed all the requirements (step 4 in the Installation section).
2. Check that you're in the correct folder when running the script.
3. Double-check your file paths - make sure they're correct and that the files exist.
4. On Mac/Linux, you might need to use `python3` instead of `python` if you have multiple Python versions installed.
5. If you're still having trouble, feel free to open an issue on the GitHub page!

## Contributing

If you have ideas to make this script better, we'd love to hear them! You can:
- Open an issue to report a bug or suggest a feature
- Submit a pull request if you've made improvements to the code

Check out the [issues page](https://github.com/BitsofJeremy/pickles_to_safetensors/issues) to see if your idea has already been suggested.

## License

This project is open source and available under the [GNU Affero General Public License v3.0](LICENSE).

## Author

Created by [BitsofJeremy](https://github.com/BitsofJeremy).

Happy converting!