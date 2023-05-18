import numpy as np

class NoiseGenerator:
    def __init__(self, snr=None):
        self.snr = snr
    
    def add_gaussian_noise(self, data, magnitude=None):
        """
        Adds Gaussian noise to the data.

        Args:
            data (numpy.ndarray): Input data.
            magnitude (float or None): Magnitude of the noise. If None, uses the SNR value specified during initialization. Default: None.

        Returns:
            numpy.ndarray: Noisy data.
        """
        if magnitude is None:
            magnitude = self._calculate_noise_magnitude(data)
        noise = np.random.normal(0, magnitude, data.shape)
        return data + noise
    
    def add_uniform_noise(self, data, magnitude=None):
        """
        Adds uniform noise to the data.

        Args:
            data (numpy.ndarray): Input data.
            magnitude (float or None): Magnitude of the noise. If None, uses the SNR value specified during initialization. Default: None.

        Returns:
            numpy.ndarray: Noisy data.
        """
        if magnitude is None:
            magnitude = self._calculate_noise_magnitude(data)
        noise = np.random.uniform(-magnitude, magnitude, data.shape)
        return data + noise
    
    def add_salt_and_pepper_noise(self, data, probability, magnitude=None):
        """
        Adds salt and pepper noise to the data.

        Args:
            data (numpy.ndarray): Input data.
            probability (float): Probability of a pixel being affected by noise.
            magnitude (float or None): Magnitude of the noise. If None, uses the SNR value specified during initialization. Default: None.

        Returns:
            numpy.ndarray: Noisy data.
        """
        if magnitude is None:
            magnitude = self._calculate_noise_magnitude(data)
        mask = np.random.random(data.shape) < probability
        noise = np.random.choice([-magnitude, magnitude], data.shape) * mask
        return data + noise
    
    def add_quantization_noise(self, data, levels=None):
        """
        Adds quantization noise to the data.

        Args:
            data (numpy.ndarray): Input data.
            levels (int or None): Number of quantization levels. If None, uses the SNR value specified during initialization. Default: None.

        Returns:
            numpy.ndarray: Noisy data.
        """
        if levels is None:
            levels = self._calculate_noise_levels(data)
        bin_width = (np.max(data) - np.min(data)) / levels
        noise = np.random.uniform(-bin_width / 2, bin_width / 2, data.shape)
        quantized_data = np.round(data / bin_width) * bin_width
        return quantized_data + noise
    
    def add_sparse_coding_noise(self, data, sparsity=None):
        """
        Adds sparse coding noise to the data.

        Args:
            data (numpy.ndarray): Input data.
            sparsity (float or None): Proportion of elements set to zero. If None, uses the SNR value specified during initialization. Default: None.

        Returns:
            numpy.ndarray: Noisy data.
        """
        if sparsity is None:
            sparsity = self._calculate_noise_sparsity(data)
        mask = np.random.random(data.shape) < sparsity
        noise = np.random.normal(0, 1, data.shape) * mask
        return data + noise
    
    def add_perlin_noise(self, data, scale, octaves, persistence):
        """
        Adds Perlin noise to the data.

        Args:
            data (numpy.ndarray): Input data.
            scale (scale (float): Scale factor for the noise.
            octaves (int): Number of octaves in the noise.
            persistence (float): Persistence of the noise.
        Returns:
            numpy.ndarray: Noisy data.
        """
        noise = np.zeros(data.shape)
        for octave in range(octaves):
            freq = 2 ** octave
            noise += self.perlin_noise(data * freq / scale) / freq ** persistence
        return data + noise

    def perlin_noise(self, coord):
        """
        Generates Perlin noise.

        Args:
            coord (numpy.ndarray): Input coordinate.

        Returns:
            numpy.ndarray: Perlin noise.
        """
        coord_floor = np.floor(coord).astype(int)
        p = self.get_permutation_table()
        gradient_vectors = self.get_gradient_vectors()
        
        diff = coord - coord_floor
        dot_products = np.zeros_like(coord)
        
        for offset in np.ndindex(2, 2):
            index = coord_floor + offset
            hash_val = p[p[index[0] % 256] + index[1] % 256]
            gradient = gradient_vectors[hash_val % len(gradient_vectors)]
            dot_product = np.dot(gradient, diff - offset)
            dot_products += dot_product * (1 - np.abs(diff - offset))

        return dot_products

    def get_permutation_table(self):
        """
        Generates a permutation table for Perlin noise.

        Returns:
            numpy.ndarray: Permutation table.
        """
        p = np.arange(256, dtype=int)
        np.random.shuffle(p)
        p = np.tile(p, 2)
        return p

    def get_gradient_vectors(self):
        """
        Generates gradient vectors for Perlin noise.

        Returns:
            numpy.ndarray: Gradient vectors.
        """
        angles = 2 * np.pi * np.random.random(256)
        gradients = np.column_stack((np.cos(angles), np.sin(angles)))
        return gradients

    def _calculate_noise_magnitude(self, data):
        """
        Calculates the noise magnitude based on the specified SNR.

        Args:
            data (numpy.ndarray): Input data.

        Returns:
            float: Noise magnitude.
        """
        if self.snr is None:
            raise ValueError("SNR value is not specified.")
        signal_power = np.mean(data ** 2)
        noise_power = signal_power / (10 ** (self.snr / 10))
        return np.sqrt(noise_power)

    def _calculate_noise_levels(self, data):
        """
        Calculates the number of quantization levels based on the specified SNR.

        Args:
            data (numpy.ndarray): Input data.

        Returns:
            int: Number of quantization levels.
        """
        if self.snr is None:
            raise ValueError("SNR value is not specified.")
        max_level = np.ceil(np.log2(np.max(data) - np.min(data)))
        noise_power = np.var(data) / (10 ** (self.snr / 10))
        return int(np.sqrt(12 * noise_power) / (max_level + 1))

    def _calculate_noise_sparsity(self, data):
        """
        Calculates the noise sparsity based on the specified SNR.

        Args:
            data (numpy.ndarray): Input data.

        Returns:
            float: Noise sparsity.
        """
        if self.snr is None:
            raise ValueError("SNR value is not specified.")
        signal_power = np.mean(data ** 2)
        noise_power = signal_power / (10 ** (self.snr / 10))
        return noise_power / np.var(data)

