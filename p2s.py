"""
PT to Safetensors Converter

This script converts PyTorch (.pt) files to the Safetensors format.
It supports converting both embedding and VAE (Variational Autoencoder) files.

Usage:
    python p2s.py <path> <model_type> [--verbose]

    <path>: Path to a .pt file or directory containing .pt files
    <model_type>: Type of model to convert (choices: "embedding" or "vae")
    --verbose: (Optional) Enable verbose output

Author: BitsofJeremy (https://github.com/BitsofJeremy)
"""

import os
import argparse
from typing import Any, Dict

import torch
from safetensors.torch import save_file

# Set the device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def process_pt_files(path: str, model_type: str, verbose: bool = True) -> None:
    """
    Process .pt files in the given path and convert them to Safetensors format.

    Args:
        path (str): Path to a .pt file or directory containing .pt files
        model_type (str): Type of model to convert ("embedding" or "vae")
        verbose (bool): Whether to print detailed information during processing

    Raises:
        FileNotFoundError: If the specified path does not exist
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"The specified path does not exist: {path}")

    if os.path.isdir(path):
        # Path is a directory, process all .pt files in the directory
        for file_name in os.listdir(path):
            if file_name.endswith('.pt'):
                process_file(os.path.join(path, file_name), model_type, verbose)
    elif os.path.isfile(path) and path.endswith('.pt'):
        # Path is a .pt file, process this file
        process_file(path, model_type, verbose)
    else:
        print(f"{path} is not a valid directory or .pt file.")


def process_file(file_path: str, model_type: str, verbose: bool) -> None:
    """
    Process a single .pt file and convert it to Safetensors format.

    Args:
        file_path (str): Path to the .pt file
        model_type (str): Type of model to convert ("embedding" or "vae")
        verbose (bool): Whether to print detailed information during processing

    Raises:
        ValueError: If the model_type is not supported
    """
    # Load the PyTorch model
    model = torch.load(file_path, map_location=device)

    if verbose:
        print(f"Processing file: {file_path}")

    if model_type == 'embedding':
        s_model = process_embedding_file(model, verbose)
    elif model_type == 'vae':
        s_model = process_vae_file(model, verbose)
    else:
        raise ValueError(f"model_type `{model_type}` is not supported!")

    # Save the model with the new extension
    new_file_path = file_path[:-3] + '.safetensors'
    save_file(s_model, new_file_path)
    print(f"Saved converted file: {new_file_path}")


def process_embedding_file(model: Dict[str, Any], verbose: bool) -> Dict[str, torch.Tensor]:
    """
    Process an embedding file and prepare it for Safetensors format.

    Args:
        model (Dict[str, Any]): The loaded PyTorch model
        verbose (bool): Whether to print detailed information during processing

    Returns:
        Dict[str, torch.Tensor]: Processed model ready for Safetensors format
    """
    # Extract the embedding tensors
    model_tensors = model.get('string_to_param', {}).get('*')
    s_model = {'emb_params': model_tensors}

    if verbose:
        # Print the requested training information, if it exists
        if 'sd_checkpoint_name' in model and model['sd_checkpoint_name'] is not None:
            print(f"Trained on {model['sd_checkpoint_name']}.")
        else:
            print("Checkpoint name not found in the model.")

        if 'step' in model and model['step'] is not None:
            print(f"Trained for {model['step']} steps.")
        else:
            print("Step not found in the model.")
        
        print(f"Dimensions of embedding tensor: {model_tensors.shape}")
        print()

    return s_model


def process_vae_file(model: Dict[str, Any], verbose: bool) -> Dict[str, torch.Tensor]:
    """
    Process a VAE file and prepare it for Safetensors format.

    Args:
        model (Dict[str, Any]): The loaded PyTorch model
        verbose (bool): Whether to print detailed information during processing

    Returns:
        Dict[str, torch.Tensor]: Processed model ready for Safetensors format
    """
    # Extract the state dictionary
    s_model = model["state_dict"]
    
    if verbose:
        # Print the requested training information, if it exists
        step = model.get('step', model.get('global_step'))
        if step is not None:
            print(f"Trained for {step} steps.")
        else:
            print("Step not found in the model.")
        print()
    
    return s_model


def main():
    """
    Main function to handle command-line arguments and run the conversion process.
    """
    parser = argparse.ArgumentParser(description="Convert PT files to Safetensors format")
    parser.add_argument("path", help="Path to a .pt file or directory containing .pt files")
    parser.add_argument("model_type", choices=["embedding", "vae"], help="Type of model to convert")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    try:
        process_pt_files(args.path, args.model_type, args.verbose)
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()