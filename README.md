# $B^+ \to K^+ \nu \bar \nu$ Reinterpretation Mini

A minimal working example for reinterpreting the Belle II $B^+ \to K^+ \nu \bar \nu$ measurement using Weak Effective Theory (WET) and kinematic reweighting techniques implemented in [redist](https://github.com/lorenzennio/redist).

This repository accompanies the paper [*"A model-agnostic likelihood for the reinterpretation of the $B^+ \to K^+ \nu \bar \nu$ measurement at Belle II"*](https://arxiv.org/abs/2507.12393) and provides practical implementations of the Wilson coefficient analysis described therein.

## Overview

The Belle II collaboration recently measured the branching fraction of the $B^+ \to K^+ \nu \bar \nu$ decay, providing the first evidence for this rare process. This repository demonstrates how to reinterpret that measurement within the Weak Effective Theory framework using Wilson coefficients.

**Key features:**
- **Minimal fit example**: Standard likelihood fit using the published results
- **WET reinterpretation**: Wilson coefficient analysis with kinematic reweighting
- **Form factor constraints**: EOS library integration for lattice QCD inputs
- **Statistical inference**: Complete uncertainty propagation and correlation analysis

## Setup

### Prerequisites

This project uses [Pixi](https://pixi.sh/) for dependency management. Install Pixi first if you haven't already:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/lorenzennio/knunu-reinterpretation-mini.git
cd knunu-reinterpretation-mini
```

2. Install dependencies using Pixi:
```bash
pixi install
```

This will create a virtual environment and install all required packages:
- `python >=3.13.7`
- `pyhf >=0.7.6` - Statistical inference library
- `iminuit >=2.31.1` - Minimization library
- `redistpy >=1.0.4` - Kinematic reweighting
- `eoshep >=1.0.17` - EOS physics library

3. Activate the environment:
```bash
pixi shell
```

## Usage

### Performing a Simple Fit

The simplest example demonstrates how to fit the published Belle II likelihood:

```bash
pixi run python scripts/fit-mini.py
```

This script:
1. Loads the combined ITA+HTA likelihood from `data/combined_likelihood.json`
2. Configures the `iminuit` optimizer for parameter uncertainties
3. Performs a maximum likelihood estimation fit
4. Extracts and displays the signal strength parameter

**Expected output:**
```
The best-fit signal strength is X.X +/- X.X.
```

### Performing WET Reinterpretation

For Wilson coefficient analysis within the Weak Effective Theory framework:

```bash
pixi run python scripts/reinterpretation-mini.py
```

This comprehensive analysis:
1. **Form factor extraction**: Uses EOS library for lattice QCD constraints from FLAG and HPQCD
2. **Wilson coefficients**: Fits $C_{\rm VL}+C_{\rm VR}$, $C_{\rm SL}+C_{\rm SR}$, $C_{\rm TL}$ parameters
3. **Kinematic reweighting**: Maps theory predictions to experimental observables with [redist](https://github.com/lorenzennio/redist)
4. **Systematic uncertainties**: Includes form factor covariance matrix
5. **Model construction**: Creates reinterpretable likelihood specification

**Key parameters fitted:**
- `cvl`: Vector left-handed Wilson coefficient for $C_{\rm VL}+C_{\rm VR}$, range [0.0, 20.0]
- `csl`: Scalar left-handed Wilson coefficient for $C_{\rm SL}+C_{\rm SR}$, range [0.0, 20.0]  
- `ctl`: Tensor left-handed Wilson coefficient for $C_{\rm TL}$, range [0.0, 20.0]
- `FFK`: 8 form factor parameters with Gaussian constraint from lattice QCD

**Expected output:**
```
Building model...
Saving model to data/reinterpretation_likelihood_wet.json...
Performing fit...
Fit results:
  C_VL+C_VR = X.X +/- X.X
  C_SL+C_SR = X.X +/- X.X
  C_TL      = X.X +/- X.X
  -2log(L) = X.X
```

## Project Structure

```
├── LICENSE                # MIT License
├── pixi.toml              # Project dependencies and configuration
├── pixi.lock              # Locked dependency versions
├── scripts/
│   ├── fit-mini.py            # Simple likelihood fit
│   ├── reinterpretation-mini.py  # WET Wilson coefficient analysis
│   ├── utils.py               # Utility functions and classes
│   └── data/
│       ├── combined_likelihood.json           # Published Belle II likelihood
│       ├── number_density_ita_regA.csv       # ITA Region A data
│       ├── number_density_ita_regB.csv       # ITA Region B data  
│       ├── number_density_hta.csv            # HTA data
│       └── reinterpretation_likelihood_wet.json  # Generated WET model
```

## Data Files

The `scripts/data/` directory contains:

- **`combined_likelihood.json`**: Complete Belle II published likelihood specification in `pyhf` format
- **Number density files**: Experimental kinematic distributions for reweighting
  - `number_density_ita_regA.csv`: Inclusive Tagging Analysis, Signal Region  
  - `number_density_ita_regB.csv`: Inclusive Tagging Analysis, Control Region
  - `number_density_hta.csv`: Hadronic Tagging Analysis
- **`reinterpretation_likelihood_wet.json`**: Generated likelihood for WET analysis

## Scientific Context

### The $B^+ \to K^+ \nu \bar \nu$ Decay

The $B^+ \to K^+ \nu \bar \nu$ decay is a rare process that:
- Occurs at O(10⁻⁶) level in the Standard Model
- Is sensitive to new physics through loop corrections
- Provides clean theoretical predictions with small uncertainties
- Was first observed by Belle II with 362 fb⁻¹ of data

### Reinterpretation Strategy

This repository implements the kinematic reweighting method implemented in [redist](https://github.com/lorenzennio/redisthttps://github.com/lorenzennio/redist) for WET analysis:

1. **Standard Model baseline**: Uses form factor predictions from lattice QCD via EOS
2. **Wilson coefficients**: Parameterizes new physics through effective field theory
3. **Statistical reweighting**: Maps theory predictions to experimental observables
4. **Likelihood construction**: Creates modified `pyhf` specifications with custom modifiers
5. **Parameter inference**: Extracts constraints on Wilson coefficients with full correlations

## Contributing

This is a research code repository accompanying a scientific publication. For questions or issues:

1. Check the [paper](https://arxiv.org/abs/2507.12393) for background
2. Review the code documentation in individual scripts
3. Open an issue for technical problems or bugs

## Citation

If you use this code in your research, please cite:

```bibtex
@article{Belle-II:2025lfq,
    author = "Abumusabh, Merna and others",
    collaboration = "Belle-II",
    title = "{A model-agnostic likelihood for the reinterpretation of the $\boldsymbol{B^{+}\to K^{+} ν\barν}$ measurement at Belle II}",
    eprint = "2507.12393",
    archivePrefix = "arXiv",
    primaryClass = "hep-ex",
    reportNumber = "Belle II Preprint 2025-021 KEK Preprint 2025-20",
    month = "7",
    year = "2025"
}

```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

This code accompanies the Belle II collaboration's measurement and the associated paper. While the code is freely available under the MIT License, please cite the original paper when using this code in your research.
