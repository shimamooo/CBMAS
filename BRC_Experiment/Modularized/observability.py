"""Observability utilities for experiment progress tracking."""

from typing import Iterator, TypeVar, Sequence, Optional
from tqdm import tqdm
import contextlib

T = TypeVar('T')


class ExperimentProgressTracker:
    """Provides progress tracking for BRC experiments with nested loops."""
    
    def __init__(self, show_progress: bool = True) -> None:
        """Initialize the progress tracker.
        
        Args:
            show_progress: Whether to show progress bars. Set to False for silent mode.
        """
        self.show_progress = show_progress
        self._main_pbar = None
        self._current_layer_pbar = None
    
    def track_injection_layers(self, inject_layers: Sequence[T], desc: str = "Processing injection layers") -> Iterator[T]:
        """Track progress through injection layers."""
        if not self.show_progress:
            yield from inject_layers
            return
            
        with tqdm(inject_layers, desc=desc, unit="layer", position=0, leave=True) as pbar:
            self._main_pbar = pbar
            for layer in pbar:
                pbar.set_postfix(layer=layer)
                yield layer
            self._main_pbar = None
    
    def track_read_layers(self, read_layers: Sequence[T], desc: str = "Processing read layers") -> Iterator[T]:
        """Track progress through read layers for current injection layer."""
        if not self.show_progress:
            yield from read_layers
            return
            
        with tqdm(read_layers, desc=desc, unit="layer", position=1, leave=False) as pbar:
            self._current_layer_pbar = pbar
            for layer in pbar:
                pbar.set_postfix(layer=layer)
                yield layer
            self._current_layer_pbar = None
    
    def track_vector_types(self, vector_types: Sequence[T], desc: str = "Processing vectors") -> Iterator[T]:
        """Track progress through vector types for current layer combination."""
        if not self.show_progress:
            yield from vector_types
            return
            
        with tqdm(vector_types, desc=desc, unit="vector", position=2, leave=False) as pbar:
            for vector_type in pbar:
                pbar.set_postfix(type=vector_type)
                yield vector_type
    
    def track_plotting(self, results: Sequence[T], desc: str = "Generating plots") -> Iterator[T]:
        """Track progress through plotting phase."""
        if not self.show_progress:
            yield from results
            return
            
        with tqdm(results, desc=desc, unit="plot", position=0, leave=True) as pbar:
            for result in pbar:
                # Extract layer info for display if it's a dict
                if isinstance(result, dict) and 'inj' in result and 'read' in result:
                    pbar.set_postfix(inj=result['inj'], read=result['read'])
                yield result
    
    def update_status(self, message: str) -> None:
        """Update the status message of the current progress bar."""
        if not self.show_progress:
            return
            
        if self._current_layer_pbar is not None:
            self._current_layer_pbar.set_description(message)
        elif self._main_pbar is not None:
            self._main_pbar.set_description(message)
    
    @contextlib.contextmanager
    def track_model_loading(self, model_name: str):
        """Context manager for tracking model loading progress."""
        if not self.show_progress:
            yield
            return
            
        with tqdm(total=100, desc=f"Loading model {model_name}", unit="step", position=0, leave=True) as pbar:
            # Update progress at key stages
            pbar.set_postfix(stage="Initializing")
            yield ModelLoadingProgress(pbar)
            pbar.update(100 - pbar.n)  # Complete the bar
            pbar.set_postfix(stage="Complete")


class ModelLoadingProgress:
    """Helper class to update model loading progress."""
    
    def __init__(self, pbar: tqdm):
        self.pbar = pbar
        
    def update(self, amount: int = 10, stage: Optional[str] = None):
        """Update progress by the given amount."""
        self.pbar.update(min(amount, 100 - self.pbar.n))
        if stage:
            self.pbar.set_postfix(stage=stage)


def create_progress_tracker(enabled: bool = True) -> ExperimentProgressTracker:
    """Factory function to create a progress tracker.
    
    Args:
        enabled: Whether progress tracking should be enabled.
        
    Returns:
        ExperimentProgressTracker instance.
    """
    return ExperimentProgressTracker(show_progress=enabled)
