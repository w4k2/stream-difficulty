from concepts import cp2evidence
import numpy as np

class ConditionalEvidenceStream():
    def __init__(self, X, y, 
                 condition_map, 
                 concept_proba,
                 chunk_size,
                 fragile=False):
        self.X = X
        self.y = y
        self.n_samples = X.shape[0]
        self.fragile = fragile
        
        self.condition_map = condition_map.T
        self.concept_proba = concept_proba
        self.chunk_size = chunk_size
        self.n_concepts = concept_proba.shape[1]
        
        self.evidence = cp2evidence(concept_proba, 
                                    chunk_size=self.chunk_size)
        self.n_chunks = self.evidence.shape[0]

        self.mutable_condition_map = self.condition_map.copy()
        self.idx = np.arange(self.n_samples)
        
        self.chunk_idx = 0
        
        self.usage = np.zeros_like(self.idx)
        
    def get_chunk(self):
        if self.chunk_idx >= self.n_chunks:
            return None
        
        # concept_probability = self.concept_proba[self.chunk_idx]
        counts = self.evidence[self.chunk_idx]
        
        # Identify concepts with necessary samples
        ccm = np.array(np.unique(counts, return_counts=True)).T

        # Prepare storage for concept indices
        concept_selected = []
        for concept, count in ccm:        
            # Establish condition mask for existing concept
            cond_mask = self.mutable_condition_map[:,concept]        
            
            # If not enough samples
            if np.sum(cond_mask) < count:
                # If fragile, stop iteration
                if self.fragile:
                    raise StopIteration

                # Restart evidence otherwise
                self.mutable_condition_map = self.condition_map.copy()
                cond_mask = self.mutable_condition_map[:,concept]
            
            # Select samples from condition mask
            selected = np.random.choice(self.idx[cond_mask], size=count)
            
            # Take off selected samples from all concepts
            self.mutable_condition_map[selected, :] = False

            # Append selected samples to concept_selected
            concept_selected.append(selected)
            
        # Clean-up
        concept_selected = np.concatenate(concept_selected)
        self.usage[concept_selected] += 1        
        self.chunk_idx += 1

        # Sample adressing
        cX, cy = self.X[concept_selected], self.y[concept_selected]
    
        return cX, cy
    