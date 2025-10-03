"""
Model construction for B->Knunu reinterpretation with Wilson coefficients.

This module builds the statistical model for reinterpreting the Belle II B->Knunu
measurement using the Weak Effective Theory (WET) framework with Wilson coefficients.

Workflow:
    1. Load kinematic distributions (null and alternative hypotheses)
    2. Load experimental number densities for ITA and HTA regions
    3. Extract form factor parameters from EOS analysis
    4. Create custom modifiers for Wilson coefficients and form factors
    5. Build pyhf model with custom theory predictions
    6. Perform maximum likelihood fit

Parameters of Interest:
    cvl : float
        Wilson coefficient Re{cVL} (vector left-handed), range [0.0, 20.0]
    csl : float
        Wilson coefficient Re{cSL} (scalar left-handed), range [0.0, 20.0]
    ctl : float
        Wilson coefficient Re{cTL} (tensor left-handed), range [0.0, 20.0]
    FFK : Tuple[float, ...]
        8 form factor parameters with Gaussian constraint from lattice QCD

    Note: Since the left- and right-handed Wilson coefficients are degenerate in B->Knunu,
    only the left-handed coefficients (cVL, cSL, cTL) are included in the model.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pyhf
from redist import modifier

from utils import analysis, NullPrediction, AlternativePrediction, parameter_cov

log = logging.getLogger(__name__)

cwd = Path(__file__).resolve().parent


"""
Data loading functions
"""


def get_distributions() -> Tuple[NullPrediction, AlternativePrediction]:
    """
    Create kinematic distribution predictors for null and alternative hypotheses.
    
    Returns:
        Tuple containing (null prediction, alternative prediction) instances
    """
    return NullPrediction(), AlternativePrediction()


def get_binning() -> List[float]:
    """
    Define the binning for momentum transfer squared (q2).
    
    Returns:
        List of bin edges for q2 [GeV^2], with sentinel value -1 at the start
    """
    return [-1] + np.linspace(0, 22.885, 100 + 1).tolist()


def get_mappings() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load experimental number density mappings for all analysis regions.
    
    Loads the joint number densities from numpy files for:
        - ITA Region A (Signal Region)
        - ITA Region B (Control Region)
        - HTA (Hadronic Tagging Analysis)
    
    Returns:
        Tuple containing (ITA_A, ITA_B, HTA) number density arrays
    """
    itaA = np.loadtxt(cwd / 'data/number_density_ita_regA.csv', delimiter=',')
    itaB = np.loadtxt(cwd / 'data/number_density_ita_regB.csv', delimiter=',')
    hta = np.loadtxt(cwd / 'data/number_density_hta.csv', delimiter=',')
    return itaA, itaB, hta

"""
Parameter extraction and modifier setup
"""


def get_ff_pars(return_cov: bool = True) -> Tuple:
    """
    Extract form factor parameters from EOS analysis.
    
    Runs the EOS analysis to obtain posterior mean values for the 8 form factor
    parameters (f+_0,1,2, f0_1,2, fT_0,1,2) using the BSZ2015 parameterization
    with lattice QCD constraints from FLAG:2021A and HPQCD:2022A.
    
    Args:
        return_cov: If True, also compute and return covariance matrix from MCMC
        
    Returns:
        If return_cov=True: (parameter tuple, covariance matrix)
        If return_cov=False: parameter tuple only
        
        Parameter tuple contains 8 form factor values:
            (fp0, fp1, fp2, f01, f02, fT0, fT1, fT2)
    """
    ana = analysis()
    
    # Extract optimized form factor parameter values
    fp0 = ana.parameters['B->K::alpha^f+_0@BSZ2015'].evaluate()
    fp1 = ana.parameters['B->K::alpha^f+_1@BSZ2015'].evaluate()
    fp2 = ana.parameters['B->K::alpha^f+_2@BSZ2015'].evaluate()
    f01 = ana.parameters['B->K::alpha^f0_1@BSZ2015'].evaluate()
    f02 = ana.parameters['B->K::alpha^f0_2@BSZ2015'].evaluate()
    fT0 = ana.parameters['B->K::alpha^fT_0@BSZ2015'].evaluate()
    fT1 = ana.parameters['B->K::alpha^fT_1@BSZ2015'].evaluate()
    fT2 = ana.parameters['B->K::alpha^fT_2@BSZ2015'].evaluate()
    
    ff_params = (fp0, fp1, fp2, f01, f02, fT0, fT1, fT2)
    
    if return_cov:
        # Compute covariance from MCMC samples
        cov = parameter_cov(ana, chains=5, samples=5000)
        return ff_params, cov
    else:
        return ff_params


def get_custom_modifiers() -> Tuple[modifier.Modifier, modifier.Modifier, modifier.Modifier]:
    """
    Create custom modifiers for Wilson coefficients and form factors.
    
    Sets up three custom modifiers (one for each analysis region) with parameters:
        - cvl: Vector left-handed Wilson coefficient Re{cVL}
        - csl: Scalar left-handed Wilson coefficient Re{cSL}
        - ctl: Tensor left-handed Wilson coefficient Re{cTL}
        - FFK: 8 form factor parameters with Gaussian constraint
    
    Returns:
        Tuple of (modifier_A, modifier_B, modifier_H) for ITA regions A, B and HTA
    """
    import eos
    
    # Get form factor parameters and covariance from EOS
    ff_pars, cov = get_ff_pars()
    
    # Get Standard Model value for vector Wilson coefficient
    cvlSM = eos.Parameters()['sbnunu::Re{cVL}'].evaluate()

    # Define parameter specifications for custom modifiers
    new_params = {
        'cvl': {
            'inits': (cvlSM,),
            'bounds': ((0.0, 20.0),),
            'paramset_type': 'unconstrained'
        },
        'csl': {
            'inits': (0.0,),
            'bounds': ((0.0, 20.0),),
            'paramset_type': 'unconstrained'
        },
        'ctl': {
            'inits': (0.0,),
            'bounds': ((0.0, 20.0),),
            'paramset_type': 'unconstrained'
        },
        'FFK': {
            'inits': ff_pars,
            'bounds': (),
            'cov': cov,
            'paramset_type': 'constrained_by_normal'
        }
    }

    # Get distributions and data mappings
    null, alt = get_distributions()
    mappingIA, mappingIB, mappingH = get_mappings()
    q2binning = get_binning()
    
    # Create modifier for each analysis region
    cmodA = modifier.Modifier(
        new_params,
        alt.distribution,
        null.distribution,
        mappingIA,
        [q2binning],
        name='regA',
        cutoff=((0.0, 22.885),)
    )
    cmodB = modifier.Modifier(
        new_params,
        alt.distribution,
        null.distribution,
        mappingIB,
        [q2binning],
        name='regB',
        cutoff=((0.0, 22.885),)
    )
    cmodH = modifier.Modifier(
        new_params,
        alt.distribution,
        null.distribution,
        mappingH,
        [q2binning],
        name='hta',
        cutoff=((0.0, 22.885),)
    )
    
    return cmodA, cmodB, cmodH

"""
Model construction and fitting
"""


def make_model() -> Tuple[pyhf.Model, List, List[modifier.Modifier]]:
    """
    Construct the statistical model with custom theory predictions.
    
    Loads the Belle II published likelihood, removes the signal strength parameter
    and old form factor systematics, then adds custom modifiers for Wilson
    coefficients and form factor parameters.
    
    Returns:
        Tuple containing (pyhf model, data, list of custom modifiers)
    """
    # Load Belle II published likelihood specification
    likelihood_file = cwd / "data/combined_likelihood.json"
    with open(likelihood_file, "r", encoding="utf-8") as file:
        spec = json.load(file)

    workspace = pyhf.Workspace(spec)
    
    # Remove signal strength parameter (replaced by Wilson coefficients)
    workspace = workspace.prune(modifiers=['mu'])
    
    # Remove old form factor systematics (replaced by EOS-derived constraints)
    workspace = workspace.prune(['corr_ff_c1', 'corr_ff_c2', 'corr_ff_c3'])

    model = workspace.model(poi_name=None)

    # Define custom modifier specifications for each region
    custom_modA = {
        "name": "regA_theory",
        "type": "regA",
        "data": {"expr": "regA_weight_fn"}
    }

    custom_modB = {
        "name": "regB_theory",
        "type": "regB",
        "data": {"expr": "regB_weight_fn"}
    }

    custom_modH = {
        "name": "hta_theory",
        "type": "hta",
        "data": {"expr": "hta_weight_fn"}
    }
    
    # Get custom modifiers for Wilson coefficients and form factors
    cmodA, cmodB, cmodH = get_custom_modifiers()

    # Merge all custom modifier specifications
    expanded_pyhf = {
        **cmodA.expanded_pyhf,
        **cmodB.expanded_pyhf,
        **cmodH.expanded_pyhf
    }
    
    # Add custom modifiers to model for each analysis region
    model = modifier.add_to_model(
        model,
        ['A__Bsig_H_reconstructed_q2_vs_BDT2_Bplus2Kplus_v42_signal_inefficiency_zoomSR_Y4S'],
        ['signal'],
        expanded_pyhf,
        custom_modA,
        poi_name=None
    )
    model = modifier.add_to_model(
        model,
        ['B__Bsig_H_reconstructed_q2_vs_BDT2_Bplus2Kplus_v42_signal_inefficiency_zoomCR_Y4S'],
        ['signal'],
        expanded_pyhf,
        custom_modB,
        poi_name=None
    )
    model = modifier.add_to_model(
        model,
        ['sgn1_had'],
        ['sgn_1_had'],
        expanded_pyhf,
        custom_modH,
        poi_name=None
    )
    
    # Extract data (without auxiliary data for now)
    data = workspace.data(model, include_auxdata=False)
    
    return model, data, [cmodA, cmodB, cmodH]


def save_model(file: str, model: pyhf.Model, cmods: List[modifier.Modifier], data: List) -> None:
    """
    Save the reinterpretation model specification to a JSON file.
    
    Args:
        file: Output file path for the model specification
        model: pyhf Model instance
        cmods: List of custom modifiers
        data: Model data
    """
    modifier.save(file, model.spec, cmods, data)


def fit(
    data: List,
    model: pyhf.Model,
    return_result_obj: bool = False
) -> Tuple:
    """
    Perform maximum likelihood fit to extract Wilson coefficients.
    
    Uses Minuit optimizer with strategy=2 for robust convergence.
    
    Args:
        data: Observed data (including auxiliary data)
        model: pyhf Model instance with Wilson coefficient parameters
        return_result_obj: If True, return full OptimizeResult object
        
    Returns:
        If return_result_obj=True:
            (best_fit_params, twice_nll, OptimizeResult)
        If return_result_obj=False:
            (best_fit_params, twice_nll, correlation_matrix)
    """
    log.info("FITTING...")
    
    # Configure backend with Minuit optimizer
    pyhf.set_backend("numpy", "minuit")
    
    # Perform maximum likelihood estimation
    best_fit, twice_nll, OptimizeResult = pyhf.infer.mle.fit(
        data,
        model,
        return_uncertainties=True,
        return_fitted_val=True,
        return_result_obj=True
    )
    
    if return_result_obj:
        return best_fit.tolist(), float(twice_nll), OptimizeResult
    
    return best_fit.tolist(), float(twice_nll), OptimizeResult.corr.tolist()


if __name__ == '__main__':
    # Configure logging to display INFO level messages
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    parser = argparse.ArgumentParser(
        description='Build and fit B->Knunu reinterpretation model with Wilson coefficients'
    )
    parser.add_argument(
        '--fit',
        action="store_true",
        help='Perform maximum likelihood fit after building model',
        default=True
    )
    args = parser.parse_args()
    
    # Build the model with custom theory predictions
    log.info("Building model...")
    model_wet, data_wet, cmods_wet = make_model()
    
    # Save reinterpretable model specification
    output_file = 'data/reinterpretation_likelihood_wet.json'
    log.info(f"Saving model to {output_file}...")
    save_model(output_file, model_wet, cmods_wet, data_wet)
    
    # Optionally perform fit
    if args.fit:
        log.info("Performing fit...")
        fit_results = fit(
            data_wet + model_wet.config.auxdata,
            model_wet
        )

        best_fit_params, twice_nll, corr = fit_results
        
        # Extract Wilson coefficient results
        cvl_slice = model_wet.config.par_slice('cvl')
        csl_slice = model_wet.config.par_slice('csl')
        ctl_slice = model_wet.config.par_slice('ctl')
        
        cvl_fit = best_fit_params[cvl_slice][0]
        csl_fit = best_fit_params[csl_slice][0]
        ctl_fit = best_fit_params[ctl_slice][0]
        
        log.info("Fit results:")
        log.info(f"  C_VL+C_VR = {cvl_fit[0]:.2f} +/- {cvl_fit[1]:.2f}")
        log.info(f"  C_SL+C_SR = {csl_fit[0]:.2f} +/- {csl_fit[1]:.2f}")
        log.info(f"  C_TL      = {ctl_fit[0]:.2f} +/- {ctl_fit[1]:.2f}")
        log.info(f"  -2log(L) = {twice_nll:.2f}")
