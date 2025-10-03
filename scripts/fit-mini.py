
"""
Minimal fit of the combined (ITA+HTA) B->Knunu likelihood using pyhf.

This script demonstrates how to:
1. Load a pyhf workspace from a JSON specification
2. Configure the minimizer for parameter uncertainties
3. Perform a maximum likelihood estimation fit
4. Extract and display the signal strength parameter
"""

import json
from pathlib import Path

import pyhf


def load_likelihood_specification(filename: str = "data/combined_likelihood.json") -> dict:
    """
    Load the combined likelihood specification from HEPData.
    
    Args:
        filename: Path to the JSON file containing the likelihood specification
        
    Returns:
        Dictionary containing the pyhf workspace specification
    """
    filepath = Path(filename)
    if not filepath.exists():
        raise FileNotFoundError(f"Likelihood file not found: {filepath}")
    
    with open(filepath, "r", encoding="utf-8") as file:
        return json.load(file)


def setup_workspace_and_model(spec: dict) -> tuple[pyhf.Model, list]:
    """
    Create pyhf workspace, model, and extract data.
    
    Args:
        spec: pyhf workspace specification dictionary
        
    Returns:
        Tuple of (model, data) for fitting
    """
    # Load pyhf workspace containing the likelihood model and data
    workspace = pyhf.Workspace(spec)
    model = workspace.model()
    data = workspace.data(model)
    
    return model, data


def configure_minimizer() -> None:
    """Configure pyhf to use iminuit optimizer for parameter uncertainties."""
    pyhf.set_backend("numpy", "minuit")


def perform_fit(data: list, model: pyhf.Model) -> tuple:
    """
    Perform maximum likelihood estimation fit.
    
    Args:
        data: Observed data for the fit
        model: pyhf model object
        
    Returns:
        Fit results with parameter values and uncertainties
    """
    return pyhf.infer.mle.fit(
        data,
        model,
        return_uncertainties=True,
    )


def extract_signal_strength(best_fit: tuple, model: pyhf.Model) -> tuple[float, float]:
    """
    Extract signal strength parameter from fit results.
    
    Args:
        best_fit: Results from pyhf.infer.mle.fit
        model: pyhf model object
        
    Returns:
        Tuple of (signal_strength, uncertainty)
    """
    mu_result = best_fit[model.config.par_slice("mu")][0]
    return mu_result[0], mu_result[1]


if __name__ == "__main__":
    # Load the likelihood specification
    spec = load_likelihood_specification()
    
    # Setup workspace and model
    model, data = setup_workspace_and_model(spec)
    
    # Configure the minimizer
    configure_minimizer()
    
    # Perform the fit
    best_fit = perform_fit(data, model)
    
    # Extract and display results
    signal_strength, uncertainty = extract_signal_strength(best_fit, model)
    print(f"The best-fit signal strength is {signal_strength:.1f} +/- {uncertainty:.1f}.")