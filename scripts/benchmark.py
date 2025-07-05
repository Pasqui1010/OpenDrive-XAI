#!/usr/bin/env python3
"""
Performance benchmarking script for OpenDrive-XAI.

This script performs comprehensive performance testing including:
- Model inference speed
- Memory usage analysis
- Throughput testing
- Latency profiling
- Bottleneck identification
- Optimization recommendations
"""

import os
import sys
import time
import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging
import torch
import numpy as np
import psutil
import gc

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from opendrive_xai.causal_world_model import CausalWorldModel, WorldState
from opendrive_xai.modular_components import InterpretableDrivingSystem
from opendrive_xai.monitoring import PerformanceMetrics, get_performance_monitor
from opendrive_xai.config import get_config


class PerformanceBenchmark:
    """Comprehensive performance benchmarking for OpenDrive-XAI."""
    
    def __init__(self, device: str = "auto", verbose: bool = False):
        """
        Initialize performance benchmark.
        
        Args:
            device: Device to run benchmarks on
            verbose: Enable verbose logging
        """
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        
        # Setup logging
        logging.basicConfig(
            level=logging.DEBUG if verbose else logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = get_config()
        
        # Performance monitor
        self.performance_monitor = get_performance_monitor()
        
        # Benchmark results
        self.results: Dict[str, Any] = {}
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all performance benchmarks."""
        self.logger.info("Starting comprehensive performance benchmarks...")
        
        benchmarks = [
            ("system_info", self.benchmark_system_info),
            ("model_loading", self.benchmark_model_loading),
            ("inference_speed", self.benchmark_inference_speed),
            ("memory_usage", self.benchmark_memory_usage),
            ("throughput", self.benchmark_throughput),
            ("latency_profile", self.benchmark_latency_profile),
            ("batch_processing", self.benchmark_batch_processing),
            ("concurrent_processing", self.benchmark_concurrent_processing)
        ]
        
        for name, benchmark_func in benchmarks:
            self.logger.info(f"Running benchmark: {name}")
            try:
                result = benchmark_func()
                self.results[name] = result
                self.logger.info(f"Benchmark '{name}' completed")
            except Exception as e:
                self.logger.error(f"Benchmark '{name}' failed: {e}")
                self.results[name] = {"error": str(e)}
        
        # Generate optimization recommendations
        self.results["recommendations"] = self.generate_recommendations()
        
        return self.results
    
    def benchmark_system_info(self) -> Dict[str, Any]:
        """Benchmark system information."""
        info = {
            "device": self.device,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "torch_version": torch.__version__,
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available
        }
        
        if self.device == "cuda":
            info.update({
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_device_name": torch.cuda.get_device_name(),
                "cuda_memory_total": torch.cuda.get_device_properties(0).total_memory,
                "cuda_memory_allocated": torch.cuda.memory_allocated(),
                "cuda_memory_reserved": torch.cuda.memory_reserved()
            })
        
        return info
    
    def benchmark_model_loading(self) -> Dict[str, Any]:
        """Benchmark model loading times."""
        results = {}
        
        # Causal World Model
        start_time = time.time()
        causal_model = CausalWorldModel(device=self.device)
        causal_load_time = time.time() - start_time
        results["causal_world_model"] = {
            "load_time": causal_load_time,
            "parameters": sum(p.numel() for p in causal_model.parameters()),
            "trainable_parameters": sum(p.numel() for p in causal_model.parameters() if p.requires_grad)
        }
        
        # Interpretable Driving System
        start_time = time.time()
        interpretable_system = InterpretableDrivingSystem(device=self.device)
        interpretable_load_time = time.time() - start_time
        results["interpretable_system"] = {
            "load_time": interpretable_load_time,
            "parameters": sum(p.numel() for p in interpretable_system.parameters()),
            "trainable_parameters": sum(p.numel() for p in interpretable_system.parameters() if p.requires_grad)
        }
        
        return results
    
    def benchmark_inference_speed(self) -> Dict[str, Any]:
        """Benchmark inference speed."""
        results = {}
        
        # Causal World Model inference
        causal_model = CausalWorldModel(device=self.device)
        
        # Create test world state
        world_state = WorldState(
            vehicle_positions=torch.randn(3, 3, device=self.device),
            vehicle_velocities=torch.randn(3, 2, device=self.device),
            traffic_lights=torch.randn(2, 3, device=self.device),
            road_geometry=torch.randn(5, 4, device=self.device),
            ego_state=torch.randn(6, device=self.device)
        )
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                causal_model(world_state)
        
        # Benchmark
        times = []
        for _ in range(100):
            start_time = time.time()
            with torch.no_grad():
                causal_model(world_state)
            torch.cuda.synchronize() if self.device == "cuda" else None
            times.append(time.time() - start_time)
        
        results["causal_world_model"] = {
            "mean_inference_time": statistics.mean(times),
            "std_inference_time": statistics.stdev(times),
            "min_inference_time": min(times),
            "max_inference_time": max(times),
            "inference_fps": 1.0 / statistics.mean(times)
        }
        
        # Interpretable Driving System inference
        interpretable_system = InterpretableDrivingSystem(device=self.device)
        input_tensor = torch.randn(1, 512, device=self.device)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                interpretable_system(input_tensor)
        
        # Benchmark
        times = []
        for _ in range(100):
            start_time = time.time()
            with torch.no_grad():
                interpretable_system(input_tensor)
            torch.cuda.synchronize() if self.device == "cuda" else None
            times.append(time.time() - start_time)
        
        results["interpretable_system"] = {
            "mean_inference_time": statistics.mean(times),
            "std_inference_time": statistics.stdev(times),
            "min_inference_time": min(times),
            "max_inference_time": max(times),
            "inference_fps": 1.0 / statistics.mean(times)
        }
        
        return results
    
    def benchmark_memory_usage(self) -> Dict[str, Any]:
        """Benchmark memory usage."""
        results = {}
        
        # Get initial memory state
        initial_cpu_memory = psutil.virtual_memory().used
        initial_gpu_memory = torch.cuda.memory_allocated() if self.device == "cuda" else 0
        
        # Load models
        causal_model = CausalWorldModel(device=self.device)
        interpretable_system = InterpretableDrivingSystem(device=self.device)
        
        # Memory after loading
        after_load_cpu_memory = psutil.virtual_memory().used
        after_load_gpu_memory = torch.cuda.memory_allocated() if self.device == "cuda" else 0
        
        # Run inference
        world_state = WorldState(
            vehicle_positions=torch.randn(3, 3, device=self.device),
            vehicle_velocities=torch.randn(3, 2, device=self.device),
            traffic_lights=torch.randn(2, 3, device=self.device),
            road_geometry=torch.randn(5, 4, device=self.device),
            ego_state=torch.randn(6, device=self.device)
        )
        
        input_tensor = torch.randn(1, 512, device=self.device)
        
        with torch.no_grad():
            causal_model(world_state)
            interpretable_system(input_tensor)
        
        # Memory after inference
        after_inference_cpu_memory = psutil.virtual_memory().used
        after_inference_gpu_memory = torch.cuda.memory_allocated() if self.device == "cuda" else 0
        
        results["memory_usage"] = {
            "cpu_memory_increase": after_load_cpu_memory - initial_cpu_memory,
            "gpu_memory_increase": after_load_gpu_memory - initial_gpu_memory,
            "inference_memory_overhead": after_inference_gpu_memory - after_load_gpu_memory if self.device == "cuda" else 0,
            "total_cpu_memory": after_inference_cpu_memory,
            "total_gpu_memory": after_inference_gpu_memory
        }
        
        return results
    
    def benchmark_throughput(self) -> Dict[str, Any]:
        """Benchmark throughput (samples per second)."""
        results = {}
        
        # Causal World Model throughput
        causal_model = CausalWorldModel(device=self.device)
        world_state = WorldState(
            vehicle_positions=torch.randn(3, 3, device=self.device),
            vehicle_velocities=torch.randn(3, 2, device=self.device),
            traffic_lights=torch.randn(2, 3, device=self.device),
            road_geometry=torch.randn(5, 4, device=self.device),
            ego_state=torch.randn(6, device=self.device)
        )
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                causal_model(world_state)
        
        # Throughput test
        num_samples = 1000
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_samples):
                causal_model(world_state)
        
        torch.cuda.synchronize() if self.device == "cuda" else None
        total_time = time.time() - start_time
        
        results["causal_world_model"] = {
            "samples_per_second": num_samples / total_time,
            "total_time": total_time,
            "num_samples": num_samples
        }
        
        # Interpretable Driving System throughput
        interpretable_system = InterpretableDrivingSystem(device=self.device)
        input_tensor = torch.randn(1, 512, device=self.device)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                interpretable_system(input_tensor)
        
        # Throughput test
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_samples):
                interpretable_system(input_tensor)
        
        torch.cuda.synchronize() if self.device == "cuda" else None
        total_time = time.time() - start_time
        
        results["interpretable_system"] = {
            "samples_per_second": num_samples / total_time,
            "total_time": total_time,
            "num_samples": num_samples
        }
        
        return results
    
    def benchmark_latency_profile(self) -> Dict[str, Any]:
        """Profile latency distribution."""
        results = {}
        
        # Causal World Model latency profile
        causal_model = CausalWorldModel(device=self.device)
        world_state = WorldState(
            vehicle_positions=torch.randn(3, 3, device=self.device),
            vehicle_velocities=torch.randn(3, 2, device=self.device),
            traffic_lights=torch.randn(2, 3, device=self.device),
            road_geometry=torch.randn(5, 4, device=self.device),
            ego_state=torch.randn(6, device=self.device)
        )
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                causal_model(world_state)
        
        # Latency profile
        times = []
        for _ in range(1000):
            start_time = time.time()
            with torch.no_grad():
                causal_model(world_state)
            torch.cuda.synchronize() if self.device == "cuda" else None
            times.append(time.time() - start_time)
        
        # Calculate percentiles
        times_sorted = sorted(times)
        percentiles = [50, 90, 95, 99, 99.9]
        
        results["causal_world_model"] = {
            "percentiles": {f"p{p}": times_sorted[int(p * len(times_sorted) / 100)] for p in percentiles},
            "mean": statistics.mean(times),
            "std": statistics.stdev(times),
            "min": min(times),
            "max": max(times)
        }
        
        return results
    
    def benchmark_batch_processing(self) -> Dict[str, Any]:
        """Benchmark batch processing performance."""
        results = {}
        
        # Test different batch sizes
        batch_sizes = [1, 2, 4, 8, 16, 32]
        
        # Causal World Model batch processing
        causal_model = CausalWorldModel(device=self.device)
        batch_results = {}
        
        for batch_size in batch_sizes:
            # Create batch of world states
            world_states = [
                WorldState(
                    vehicle_positions=torch.randn(3, 3, device=self.device),
                    vehicle_velocities=torch.randn(3, 2, device=self.device),
                    traffic_lights=torch.randn(2, 3, device=self.device),
                    road_geometry=torch.randn(5, 4, device=self.device),
                    ego_state=torch.randn(6, device=self.device)
                )
                for _ in range(batch_size)
            ]
            
            # Warmup
            for _ in range(5):
                with torch.no_grad():
                    for world_state in world_states:
                        causal_model(world_state)
            
            # Benchmark
            times = []
            for _ in range(50):
                start_time = time.time()
                with torch.no_grad():
                    for world_state in world_states:
                        causal_model(world_state)
                torch.cuda.synchronize() if self.device == "cuda" else None
                times.append(time.time() - start_time)
            
            batch_results[batch_size] = {
                "mean_time": statistics.mean(times),
                "samples_per_second": batch_size / statistics.mean(times),
                "efficiency": batch_size / statistics.mean(times) / (1 / statistics.mean(times[:10]))  # Relative to batch size 1
            }
        
        results["causal_world_model"] = batch_results
        
        return results
    
    def benchmark_concurrent_processing(self) -> Dict[str, Any]:
        """Benchmark concurrent processing performance."""
        results = {}
        
        # This would typically test multi-threading/multi-processing
        # For now, we'll simulate concurrent processing with multiple models
        
        # Load multiple models
        models = [
            CausalWorldModel(device=self.device),
            CausalWorldModel(device=self.device),
            CausalWorldModel(device=self.device),
            CausalWorldModel(device=self.device)
        ]
        
        world_state = WorldState(
            vehicle_positions=torch.randn(3, 3, device=self.device),
            vehicle_velocities=torch.randn(3, 2, device=self.device),
            traffic_lights=torch.randn(2, 3, device=self.device),
            road_geometry=torch.randn(5, 4, device=self.device),
            ego_state=torch.randn(6, device=self.device)
        )
        
        # Warmup
        for model in models:
            for _ in range(5):
                with torch.no_grad():
                    model(world_state)
        
        # Benchmark concurrent processing
        times = []
        for _ in range(100):
            start_time = time.time()
            with torch.no_grad():
                for model in models:
                    model(world_state)
            torch.cuda.synchronize() if self.device == "cuda" else None
            times.append(time.time() - start_time)
        
        results["concurrent_processing"] = {
            "num_models": len(models),
            "mean_time": statistics.mean(times),
            "total_samples_per_second": len(models) / statistics.mean(times),
            "efficiency": len(models) / statistics.mean(times) / (1 / statistics.mean(times[:10]))
        }
        
        return results
    
    def generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on benchmark results."""
        recommendations = []
        
        # Check inference speed
        if "inference_speed" in self.results:
            causal_fps = self.results["inference_speed"]["causal_world_model"]["inference_fps"]
            if causal_fps < 10:
                recommendations.append("Consider model optimization or hardware upgrade for real-time performance")
            
            interpretable_fps = self.results["inference_speed"]["interpretable_system"]["inference_fps"]
            if interpretable_fps < 10:
                recommendations.append("Interpretable system needs optimization for real-time operation")
        
        # Check memory usage
        if "memory_usage" in self.results:
            gpu_memory = self.results["memory_usage"]["memory_usage"]["total_gpu_memory"]
            if gpu_memory > 4e9:  # 4GB
                recommendations.append("High GPU memory usage - consider model compression or gradient checkpointing")
        
        # Check throughput
        if "throughput" in self.results:
            causal_throughput = self.results["throughput"]["causal_world_model"]["samples_per_second"]
            if causal_throughput < 50:
                recommendations.append("Low throughput - consider batch processing or model optimization")
        
        # Check batch processing efficiency
        if "batch_processing" in self.results:
            batch_results = self.results["batch_processing"]["causal_world_model"]
            optimal_batch_size = max(batch_results.keys(), key=lambda x: batch_results[x]["efficiency"])
            if optimal_batch_size > 1:
                recommendations.append(f"Use batch size {optimal_batch_size} for optimal throughput")
        
        # General recommendations
        if self.device == "cpu":
            recommendations.append("Consider using GPU for significant performance improvement")
        
        if not recommendations:
            recommendations.append("System performance meets requirements")
        
        return recommendations
    
    def save_results(self, output_file: str) -> None:
        """Save benchmark results to file."""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        self.logger.info(f"Benchmark results saved to {output_file}")
    
    def print_summary(self) -> None:
        """Print benchmark summary."""
        print("\n" + "="*60)
        print("PERFORMANCE BENCHMARK SUMMARY")
        print("="*60)
        
        # System info
        if "system_info" in self.results:
            info = self.results["system_info"]
            print(f"Device: {info['device']}")
            print(f"Python: {info['python_version']}")
            print(f"PyTorch: {info['torch_version']}")
            if self.device == "cuda":
                print(f"GPU: {info['cuda_device_name']}")
        
        # Inference speed
        if "inference_speed" in self.results:
            print("\nInference Speed:")
            for model_name, metrics in self.results["inference_speed"].items():
                print(f"  {model_name}: {metrics['inference_fps']:.1f} FPS")
        
        # Throughput
        if "throughput" in self.results:
            print("\nThroughput:")
            for model_name, metrics in self.results["throughput"].items():
                print(f"  {model_name}: {metrics['samples_per_second']:.1f} samples/sec")
        
        # Memory usage
        if "memory_usage" in self.results:
            mem = self.results["memory_usage"]["memory_usage"]
            print(f"\nMemory Usage:")
            print(f"  GPU Memory: {mem['total_gpu_memory'] / 1e9:.2f} GB")
        
        # Recommendations
        if "recommendations" in self.results:
            print("\nOptimization Recommendations:")
            for i, rec in enumerate(self.results["recommendations"], 1):
                print(f"  {i}. {rec}")
        
        print("="*60)


def main():
    """Main benchmark script."""
    parser = argparse.ArgumentParser(description="Benchmark OpenDrive-XAI performance")
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run benchmarks on"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="benchmark_results.json",
        help="Output file for benchmark results"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Create benchmark instance
    benchmark = PerformanceBenchmark(device=args.device, verbose=args.verbose)
    
    try:
        # Run benchmarks
        results = benchmark.run_all_benchmarks()
        
        # Save results
        benchmark.save_results(args.output)
        
        # Print summary
        benchmark.print_summary()
        
        print(f"\n✅ Benchmark completed successfully!")
        print(f"📊 Results saved to: {args.output}")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 