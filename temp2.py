import numpy as np

def generate_parameters_from_snr(snr):
    # SNR-based scaling
    scale_factor = np.interp(snr, [0, 100], [1.0, 0.1])
    scale = np.random.uniform(0.1, 1.0) * scale_factor

    # SNR-based octaves
    max_octaves = int(np.interp(snr, [0, 100], [1, 6]))
    octaves = np.random.randint(1, max_octaves + 1)

    # SNR-based persistence
    persistence = np.interp(snr, [0, 100], [0.9, 0.1])

    return scale, octaves, persistence

# Example usage
snr = 0.001  # Example signal-to-noise ratio
scale, octaves, persistence = generate_parameters_from_snr(snr)

print("Scale:", scale)
print("Octaves:", octaves)
print("Persistence:", persistence)
