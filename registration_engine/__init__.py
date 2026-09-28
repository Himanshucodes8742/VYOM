import warnings

# Suppress pyfftw fallback warning from phasepack package
warnings.filterwarnings("ignore", category=UserWarning, module="phasepack")
