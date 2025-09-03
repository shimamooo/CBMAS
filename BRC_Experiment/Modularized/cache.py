"""Vector caching system for bias response curves."""

import os
from pathlib import Path
from typing import Dict, Optional
import torch


class VectorCache:
    """Simple file-based cache for steering vectors."""
    
    def __init__(self, cache_dir: str = "cache/vectors"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, model_name: str, dataset: str, layer: int, inject_site: str) -> Path:
        """Get hierarchical path to cache file: cache_dir/model/dataset/layer_site.pt"""
        # Clean model name for directory (replace slashes and dashes with underscores)
        clean_model_name = model_name.replace("/", "_").replace("-", "_")
        
        # Create hierarchical path: model/dataset/filename
        model_dir = self.cache_dir / clean_model_name
        dataset_dir = model_dir / dataset
        filename = f"layer{layer}_{inject_site}.pt"
        
        return dataset_dir / filename
    
    def exists(self, model_name: str, dataset: str, layer: int, inject_site: str) -> bool:
        """Check if vectors are cached for given parameters."""
        cache_path = self._get_cache_path(model_name, dataset, layer, inject_site)
        return cache_path.exists()
    
    def load(self, model_name: str, dataset: str, layer: int, inject_site: str, device: torch.device) -> Optional[Dict[str, torch.Tensor]]:
        """Load cached vectors if they exist."""
        cache_path = self._get_cache_path(model_name, dataset, layer, inject_site)
        
        if not cache_path.exists():
            return None
        
        try:
            vectors = torch.load(cache_path, map_location=device)
            # Ensure we have all expected vector types
            if all(key in vectors for key in ["bias", "random", "orth"]):
                return vectors
            else:
                print(f"Warning: Cached vectors missing expected keys, recomputing for {cache_path.name}")
                return None
        except Exception as e:
            print(f"Warning: Failed to load cached vectors from {cache_path}: {e}")
            return None
    
    def save(self, vectors: Dict[str, torch.Tensor], model_name: str, dataset: str, layer: int, inject_site: str) -> None:
        """Save vectors to cache."""
        cache_path = self._get_cache_path(model_name, dataset, layer, inject_site)
        
        try:
            # Create directories if they don't exist
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Move tensors to CPU for storage
            cpu_vectors = {k: v.cpu() for k, v in vectors.items()}
            torch.save(cpu_vectors, cache_path)
            
            # Show relative path for cleaner output
            relative_path = cache_path.relative_to(self.cache_dir)
            print(f"Cached vectors saved to {relative_path}")
        except Exception as e:
            print(f"Warning: Failed to save vectors to cache: {e}")
    
    def clear_cache(self, model_name: Optional[str] = None, dataset: Optional[str] = None) -> int:
        """Clear cached vectors. If model_name/dataset specified, only clear matching directories/files."""
        cleared_count = 0
        
        if model_name and dataset:
            # Clear specific model/dataset combination
            clean_model = model_name.replace("/", "_").replace("-", "_")
            target_dir = self.cache_dir / clean_model / dataset
            if target_dir.exists():
                for cache_file in target_dir.glob("*.pt"):
                    try:
                        cache_file.unlink()
                        cleared_count += 1
                    except Exception as e:
                        print(f"Warning: Failed to delete {cache_file}: {e}")
                # Remove empty directory
                try:
                    target_dir.rmdir()
                    # Try to remove model directory if it's empty
                    if not any(target_dir.parent.iterdir()):
                        target_dir.parent.rmdir()
                except OSError:
                    pass  # Directory not empty, that's fine
        
        elif model_name:
            # Clear all datasets for a specific model
            clean_model = model_name.replace("/", "_").replace("-", "_")
            model_dir = self.cache_dir / clean_model
            if model_dir.exists():
                for cache_file in model_dir.rglob("*.pt"):
                    try:
                        cache_file.unlink()
                        cleared_count += 1
                    except Exception as e:
                        print(f"Warning: Failed to delete {cache_file}: {e}")
                # Remove the entire model directory
                try:
                    import shutil
                    shutil.rmtree(model_dir)
                except Exception as e:
                    print(f"Warning: Failed to remove model directory {model_dir}: {e}")
        
        elif dataset:
            # Clear specific dataset across all models
            for model_dir in self.cache_dir.iterdir():
                if model_dir.is_dir():
                    dataset_dir = model_dir / dataset
                    if dataset_dir.exists():
                        for cache_file in dataset_dir.glob("*.pt"):
                            try:
                                cache_file.unlink()
                                cleared_count += 1
                            except Exception as e:
                                print(f"Warning: Failed to delete {cache_file}: {e}")
                        # Remove empty dataset directory
                        try:
                            dataset_dir.rmdir()
                            # Try to remove model directory if it's empty
                            if not any(model_dir.iterdir()):
                                model_dir.rmdir()
                        except OSError:
                            pass  # Directory not empty
        
        else:
            # Clear all cache
            for cache_file in self.cache_dir.rglob("*.pt"):
                try:
                    cache_file.unlink()
                    cleared_count += 1
                except Exception as e:
                    print(f"Warning: Failed to delete {cache_file}: {e}")
            # Remove all directories
            try:
                import shutil
                for item in self.cache_dir.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item)
            except Exception as e:
                print(f"Warning: Failed to remove cache directories: {e}")
        
        return cleared_count
    
    def list_cached_vectors(self) -> list[str]:
        """List all cached vector files with hierarchical paths."""
        cached_files = []
        for cache_file in self.cache_dir.rglob("*.pt"):
            # Get relative path from cache directory
            relative_path = cache_file.relative_to(self.cache_dir)
            cached_files.append(str(relative_path))
        return sorted(cached_files)
