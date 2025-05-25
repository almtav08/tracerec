"""
Base class for samplers in the Tracerec package.
"""

class Sampler:
    def sample(self, num_samples=1):
        """
        Sample negative triples based on the specific sampling strategy.
        
        Args:
            num_samples: Number of negative samples per positive triple
            
        Returns:
            torch.Tensor: Tensor of shape (num_triples, num_samples, 3) containing negative triples
        """
        raise NotImplementedError("This method should be overridden by subclasses.")