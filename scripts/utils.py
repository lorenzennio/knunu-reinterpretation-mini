"""
Utility functions and classes for B->Knunu analysis.

This module provides theoretical predictions for B->Knunu decays including:
    - Standard Model (null) predictions using form factors
        - arXiv:2207.12468: Form factor parameterization
        - arXiv:2207.13371: Theoretical predictions
    - Beyond Standard Model (alternative) predictions using EOS
    - EOS analysis setup with form factor constraints
    - Covariance matrix computation for parameter uncertainties
"""

from dataclasses import dataclass
from typing import List, Tuple, Union
import numbers

import numpy as np
import eos


# Physical constants and parameters
@dataclass(frozen=True)
class PhysicalConstants:
    """Physical constants used in the analysis."""
    
    # Fermi constant [GeV^-2]
    G_FERMI: float = 1.1663787e-5
    # Electromagnetic coupling constant (inverse)
    ALPHA_EW_INV: float = 127.952
    # CKM matrix element parameter
    XT: float = 1.468
    # Weinberg angle
    SIN2_THETA: float = 0.23124
    # CKM matrix element
    VV: float = 0.04185
    # B meson mass [GeV]
    M_B: float = 5.279495
    # Kaon mass [GeV]
    M_K: float = 0.4956440
    # B* meson mass [GeV]
    M_BS_STAR: float = 5.4158
    # Bottom quark mass [GeV]
    MB_QUARK: float = 4.87
    # B meson lifetime [GeV^-1] (converted from seconds)
    TAU_B: float = 1.638e-12 / 6.58211e-25


@dataclass(frozen=True)
class FormFactorConfig:
    """Configuration for form factor analysis."""
    
    # Number of form factor parameters
    N_FF_PARAMS: int = 3
    # Logarithmic correction factor
    LOG_SB: float = 1.304
    # Form factor parameters from 2207.12468 table V [f+_0, f+_1, f+_2]
    FF_PARAMS: Tuple[float, ...] = (0.2545, -0.71, 0.32)
    # Prior sigma multiplier for uniform bounds
    PRIOR_SIGMA: int = 15


@dataclass(frozen=True)
class EOSFormFactorPriors:
    """Form factor prior parameters for EOS analysis."""
    
    # Central values for 8 form factor parameters
    PARAMETERS: Tuple[float, ...] = (
        0.33772497529184886, -0.87793473613271, -0.07935870922121949,
        0.3719622997220613, 0.07388594710238389, 0.327935912834808,
        -0.9490004115927961, -0.23146429907794228
    )
    
    # Uncertainties for 8 form factor parameters
    ERRORS: Tuple[float, ...] = (
        0.010131234226468245, 0.09815140228051167, 0.26279803480131697,
        0.07751034526769873, 0.14588095119443809, 0.019809720318176644,
        0.16833757660616938, 0.36912754148836896
    )


def analysis() -> eos.Analysis:
    """
    Create and optimize an EOS analysis with form factor constraints.
    
    Sets up uniform priors for 8 form factor parameters (f+_0,1,2, f0_1,2, fT_0,1,2)
    using the BSZ2015 parameterization. Includes likelihoods from FLAG:2021A and
    HPQCD:2022A lattice QCD calculations.
    
    Note:
        f0_0 is not included due to a constraint that removes one parameter.
    
    Returns:
        Optimized EOS analysis instance with form factor posteriors.
    """
    priors_config = EOSFormFactorPriors()
    ff_config = FormFactorConfig()
    
    parameters = priors_config.PARAMETERS
    errors = priors_config.ERRORS
    sigma = ff_config.PRIOR_SIGMA
    
    # Parameter names for the 8 form factor expansion coefficients
    param_names = [
        'B->K::alpha^f+_0@BSZ2015',
        'B->K::alpha^f+_1@BSZ2015',
        'B->K::alpha^f+_2@BSZ2015',
        'B->K::alpha^f0_1@BSZ2015',
        'B->K::alpha^f0_2@BSZ2015',
        'B->K::alpha^fT_0@BSZ2015',
        'B->K::alpha^fT_1@BSZ2015',
        'B->K::alpha^fT_2@BSZ2015'
    ]
    
    # Build uniform priors centered on parameters with sigma*error bounds
    priors = []
    for param_name, central, error in zip(param_names, parameters, errors):
        priors.append({
            'parameter': param_name,
            'min': central - sigma * error,
            'max': central + sigma * error,
            'type': 'uniform'
        })
    
    analysis_args = {
        'priors': priors,
        'likelihood': [
            'B->K::f_0+f_++f_T@FLAG:2021A',
            'B->K::f_0+f_++f_T@HPQCD:2022A'
        ]
    }
    
    ana = eos.Analysis(**analysis_args)
    ana.optimize()
    return ana


"""
Define the kinematic distributions
"""


class NullPrediction:
    """
    Standard Model (null hypothesis) prediction for B->Knunu decay.
    
    Implements the theoretical prediction from arXiv:2207.12468 and 2207.13371.
    Computes the differential decay rate dsigma/dq2.
    """
    
    def __init__(self, config: FormFactorConfig = None, constants: PhysicalConstants = None):
        """
        Initialize the Standard Model prediction.
        
        Args:
            config: Form factor configuration parameters
            constants: Physical constants
        """
        self.config = config or FormFactorConfig()
        self.constants = constants or PhysicalConstants()
        
        # Store commonly used values
        self.mB_par = self.constants.M_B
        self.mK_par = self.constants.M_K
        self.mBsstar_par = self.constants.M_BS_STAR
        self.N_par = self.config.N_FF_PARAMS
        self.logsB_par = self.config.LOG_SB
        self.ff_par = list(self.config.FF_PARAMS)
        
        # Calculate normalization factor from eq. 16 in 2207.13371
        # (everything except |p_k|^3 f_+^2(q^2))
        alpha_ew = 1 / self.constants.ALPHA_EW_INV
        self.f_par = (
            self.constants.G_FERMI**2
            * alpha_ew**2
            * self.constants.XT**2
            / (32 * np.pi**5 * self.constants.SIN2_THETA**2)
            * self.constants.TAU_B
            * self.constants.VV**2
        )

    def z_B_par(self, qsq: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute the z parameter for form factor expansion (arXiv:2207.12468).
        
        Args:
            qsq: Momentum transfer squared [GeV^2]
            
        Returns:
            The z parameter for form factor calculation
        """
        t_plus = (self.mB_par + self.mK_par) ** 2
        t_0 = 0.0
        
        sqrt_diff_qsq = np.sqrt(t_plus - qsq)
        sqrt_diff_t0 = np.sqrt(t_plus - t_0)
        
        z = (sqrt_diff_qsq - sqrt_diff_t0) / (sqrt_diff_qsq + sqrt_diff_t0)
        
        # Handle q2 = 0 case
        if isinstance(z, np.ndarray):
            z[qsq == 0] = 0
        elif qsq == 0:
            z = 0
            
        return z

    def fplus_par(self, qsq: Union[float, np.ndarray], apB: List[float]) -> Union[float, np.ndarray]:
        """
        Compute the B->K form factor using z-expansion (arXiv:2207.12468).
        
        Args:
            qsq: Momentum transfer squared [GeV^2]
            apB: Form factor parameters [f+_0, f+_1, f+_2]
            
        Returns:
            The B->K form factor f_+(q^2)
        """
        # Pole structure
        pole = 1 / (1 - qsq / self.mBsstar_par**2)
        
        # z-expansion
        z = self.z_B_par(qsq)
        fp = 0.0
        
        for n in range(self.N_par):
            constraint_term = (n / self.N_par) * (-1) ** (n - self.N_par) * z**self.N_par
            fp += apB[n] * (z**n - constraint_term)
            
        return fp * pole * self.logsB_par

    def lam(self, a: float, b: float, c: float) -> float:
        """
        Compute the kinematic lambda function for phase space calculations.
        
        This is the Källén function λ(a,b,c) used in two-body decay kinematics,
        from eq. 102 in arXiv:1409.4557.
        
        Args:
            a: First invariant mass squared
            b: Second invariant mass squared
            c: Third invariant mass squared
            
        Returns:
            The Källén function value λ(a,b,c)
        """
        return a**2 + b**2 + c**2 - 2 * (a * b + b * c + a * c)

    def distribution(self, q2: Union[float, np.ndarray], a: List[float] = None) -> Union[float, np.ndarray]:
        """
        Compute the differential branching ratio for B->Knunu (eq. 16 from arXiv:2207.13371).
        
        Args:
            q2: Momentum transfer squared [GeV^2]
            a: Form factor parameters. If None, uses default values.
            
        Returns:
            Differential decay rate dΓ/dq² [GeV^-1]
        """
        if a is None:
            a = self.ff_par
            
        # Three-body phase space factor
        phase_space = self.lam(self.mB_par**2, self.mK_par**2, q2) ** (3/2)
        phase_space /= (2 * self.mB_par)**3
        
        # Form factor contribution
        ff_squared = self.fplus_par(q2, a) ** 2
        
        return self.f_par * phase_space * ff_squared


class AlternativePrediction:
    """
    Beyond Standard Model (BSM) prediction using EOS framework.
    
    Implements predictions for B->Knunu using the Weak Effective Theory (WET)
    with Wilson coefficients (cVL, cSL, cTL) and form factor parameters.
    Uses the EOS library for precise calculations including BSM contributions.
    """
    
    def __init__(self):
        """
        Initialize the BSM prediction using EOS.
        
        Sets up EOS Analysis, Parameters, Kinematics, and Observable for computing
        B->Knunu differential branching ratio with BSM Wilson coefficients and
        form factor parameters.
        """
        # Initialize EOS analysis with form factor constraints
        self.ana = analysis()
        p = self.ana.parameters
        k = eos.Kinematics({'q2': 0.0})
        o = eos.Options(**{'form-factors': 'BSZ2015', 'model': 'WET'})
        
        # Store kinematic variable handle
        self.kv1 = k['q2']
        
        # Store Wilson coefficient handles
        self.wc1 = p['sbnunu::Re{cVL}']  # Vector left-handed
        self.wc3 = p['sbnunu::Re{cSL}']  # Scalar left-handed
        self.wc5 = p['sbnunu::Re{cTL}']  # Tensor left-handed
        
        # Store form factor parameter handles (8 parameters total)
        self.hv1 = p['B->K::alpha^f+_0@BSZ2015']
        self.hv2 = p['B->K::alpha^f+_1@BSZ2015']
        self.hv3 = p['B->K::alpha^f+_2@BSZ2015']
        self.hv4 = p['B->K::alpha^f0_1@BSZ2015']
        self.hv5 = p['B->K::alpha^f0_2@BSZ2015']
        self.hv6 = p['B->K::alpha^fT_0@BSZ2015']
        self.hv7 = p['B->K::alpha^fT_1@BSZ2015']
        self.hv8 = p['B->K::alpha^fT_2@BSZ2015']
        
        # Create the EOS observable for differential branching ratio
        self.obs = eos.Observable.make('B->Knunu::dBR/dq2', p, k, o)

    def distribution(
        self,
        q2: Union[float, np.ndarray, List[float]],
        cvl: float,
        csl: float,
        ctl: float,
        fp0: float,
        fp1: float,
        fp2: float,
        f01: float,
        f02: float,
        fT0: float,
        fT1: float,
        fT2: float
    ) -> Union[float, List[float]]:
        """
        Compute the BSM differential branching ratio for B->Knunu.
        
        Args:
            q2: Momentum transfer squared [GeV^2], can be scalar or array
            cvl: Wilson coefficient Re{cVL} (vector left-handed)
            csl: Wilson coefficient Re{cSL} (scalar left-handed)
            ctl: Wilson coefficient Re{cTL} (tensor left-handed)
            fp0: Form factor parameter alpha^f+_0
            fp1: Form factor parameter alpha^f+_1
            fp2: Form factor parameter alpha^f+_2
            f01: Form factor parameter alpha^f0_1
            f02: Form factor parameter alpha^f0_2
            fT0: Form factor parameter alpha^fT_0
            fT1: Form factor parameter alpha^fT_1
            fT2: Form factor parameter alpha^fT_2
            
        Returns:
            Differential decay rate dΓ/dq² [GeV^-1]
        """
        # Set Wilson coefficients
        self.wc1.set(cvl)
        self.wc3.set(csl)
        self.wc5.set(ctl)
        
        # Set form factor parameters
        self.hv1.set(fp0)
        self.hv2.set(fp1)
        self.hv3.set(fp2)
        self.hv4.set(f01)
        self.hv5.set(f02)
        self.hv6.set(fT0)
        self.hv7.set(fT1)
        self.hv8.set(fT2)

        # Evaluate at single q2 point or multiple points
        if isinstance(q2, numbers.Number):
            self.kv1.set(q2)
            return self.obs.evaluate()
        else:
            obs = []
            for q in q2:
                self.kv1.set(q)
                obs.append(self.obs.evaluate())
            return obs


"""
Utility functions for parameter inference
"""


def parameter_cov(ana: eos.Analysis, chains: int = 5, samples: int = 5000) -> List[List[float]]:
    """
    Compute covariance matrix of parameters from EOS analysis MCMC samples.
    
    Runs multiple MCMC chains and combines samples to estimate the posterior
    covariance matrix for the analysis parameters.
    
    Args:
        ana: EOS Analysis instance with defined priors and likelihoods
        chains: Number of independent MCMC chains to run
        samples: Number of samples per chain
        
    Returns:
        Covariance matrix as a list of lists (can be used in pyhf/redist)
    """
    pars = []
    
    for n in range(chains):
        # Use different random seed for each chain
        rng = np.random.mtrand.RandomState(74205 + n)
        
        # Sample from posterior
        p, _ = ana.sample(
            N=samples,
            stride=5,
            pre_N=1000,
            preruns=5,
            rng=rng
        )
        
        pars += p.tolist()
    
    # Combine all samples and compute covariance
    pars = np.array(pars)
    cov = np.cov(pars.T).tolist()
    
    return cov