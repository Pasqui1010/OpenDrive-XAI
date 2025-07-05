#!/usr/bin/env python3
"""
Deployment script for OpenDrive-XAI.

This script handles:
- Environment setup and validation
- Dependency installation
- Configuration validation
- System deployment
- Health checks
- Performance testing
"""

import os
import sys
import subprocess
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from opendrive_xai.config import get_config_manager, validate_config
from opendrive_xai.monitoring import start_monitoring, stop_monitoring, get_system_status


class DeploymentManager:
    """Manages the deployment process for OpenDrive-XAI."""
    
    def __init__(self, config_path: Optional[str] = None, verbose: bool = False):
        """
        Initialize deployment manager.
        
        Args:
            config_path: Path to configuration file
            verbose: Enable verbose logging
        """
        self.config_path = config_path
        self.verbose = verbose
        self.project_root = Path(__file__).parent.parent
        
        # Setup logging
        logging.basicConfig(
            level=logging.DEBUG if verbose else logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config_manager = get_config_manager(config_path)
        self.config = self.config_manager.get_config()
    
    def deploy(self, steps: List[str] = None) -> bool:
        """
        Run the complete deployment process.
        
        Args:
            steps: List of deployment steps to run (None for all)
            
        Returns:
            True if deployment successful, False otherwise
        """
        if steps is None:
            steps = [
                "validate_environment",
                "install_dependencies",
                "validate_configuration",
                "setup_directories",
                "run_tests",
                "start_monitoring",
                "health_check"
            ]
        
        self.logger.info("Starting OpenDrive-XAI deployment...")
        
        for step in steps:
            self.logger.info(f"Running step: {step}")
            
            try:
                if step == "validate_environment":
                    success = self.validate_environment()
                elif step == "install_dependencies":
                    success = self.install_dependencies()
                elif step == "validate_configuration":
                    success = self.validate_configuration()
                elif step == "setup_directories":
                    success = self.setup_directories()
                elif step == "run_tests":
                    success = self.run_tests()
                elif step == "start_monitoring":
                    success = self.start_monitoring()
                elif step == "health_check":
                    success = self.health_check()
                else:
                    self.logger.error(f"Unknown deployment step: {step}")
                    return False
                
                if not success:
                    self.logger.error(f"Deployment step '{step}' failed")
                    return False
                
                self.logger.info(f"Step '{step}' completed successfully")
                
            except Exception as e:
                self.logger.error(f"Error in deployment step '{step}': {e}")
                return False
        
        self.logger.info("Deployment completed successfully!")
        return True
    
    def validate_environment(self) -> bool:
        """Validate the deployment environment."""
        self.logger.info("Validating environment...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 11):
            self.logger.error(f"Python 3.11+ required, got {python_version.major}.{python_version.minor}")
            return False
        
        self.logger.info(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # Check for required tools
        required_tools = ["pip", "git"]
        for tool in required_tools:
            try:
                subprocess.run([tool, "--version"], capture_output=True, check=True)
                self.logger.info(f"Found {tool}")
            except (subprocess.CalledProcessError, FileNotFoundError):
                self.logger.error(f"Required tool '{tool}' not found")
                return False
        
        # Check for GPU if CUDA is requested
        if self.config.device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    self.logger.warning("CUDA requested but not available")
                else:
                    self.logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
            except ImportError:
                self.logger.warning("PyTorch not installed, cannot check CUDA")
        
        return True
    
    def install_dependencies(self) -> bool:
        """Install project dependencies."""
        self.logger.info("Installing dependencies...")
        
        try:
            # Install from requirements.txt
            requirements_file = self.project_root / "requirements.txt"
            if requirements_file.exists():
                self.logger.info("Installing from requirements.txt...")
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
                    capture_output=not self.verbose,
                    text=True
                )
                
                if result.returncode != 0:
                    self.logger.error(f"Failed to install requirements: {result.stderr}")
                    return False
            
            # Install the package in development mode
            self.logger.info("Installing package in development mode...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", str(self.project_root)],
                capture_output=not self.verbose,
                text=True
            )
            
            if result.returncode != 0:
                self.logger.error(f"Failed to install package: {result.stderr}")
                return False
            
            self.logger.info("Dependencies installed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error installing dependencies: {e}")
            return False
    
    def validate_configuration(self) -> bool:
        """Validate the configuration."""
        self.logger.info("Validating configuration...")
        
        # Validate configuration
        issues = validate_config()
        if issues:
            self.logger.error("Configuration validation failed:")
            for issue in issues:
                self.logger.error(f"  - {issue}")
            return False
        
        self.logger.info("Configuration validation passed")
        return True
    
    def setup_directories(self) -> bool:
        """Setup required directories."""
        self.logger.info("Setting up directories...")
        
        try:
            # Create checkpoint directory
            checkpoint_dir = Path(self.config.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created checkpoint directory: {checkpoint_dir}")
            
            # Create log directory
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            self.logger.info(f"Created log directory: {log_dir}")
            
            # Create data directory
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            self.logger.info(f"Created data directory: {data_dir}")
            
            # Create models directory
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)
            self.logger.info(f"Created models directory: {models_dir}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting up directories: {e}")
            return False
    
    def run_tests(self) -> bool:
        """Run the test suite."""
        self.logger.info("Running tests...")
        
        try:
            # Run core fixes tests
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/test_core_fixes.py", "-v"],
                capture_output=not self.verbose,
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode != 0:
                self.logger.error(f"Core fixes tests failed: {result.stderr}")
                return False
            
            # Run integration tests if they exist
            integration_test_file = self.project_root / "tests" / "test_integration.py"
            if integration_test_file.exists():
                self.logger.info("Running integration tests...")
                result = subprocess.run(
                    [sys.executable, "-m", "pytest", "tests/test_integration.py", "-v"],
                    capture_output=not self.verbose,
                    text=True,
                    cwd=self.project_root
                )
                
                if result.returncode != 0:
                    self.logger.error(f"Integration tests failed: {result.stderr}")
                    return False
            
            self.logger.info("All tests passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error running tests: {e}")
            return False
    
    def start_monitoring(self) -> bool:
        """Start the monitoring system."""
        self.logger.info("Starting monitoring system...")
        
        try:
            start_monitoring()
            self.logger.info("Monitoring system started")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting monitoring: {e}")
            return False
    
    def health_check(self) -> bool:
        """Perform health check on the deployed system."""
        self.logger.info("Performing health check...")
        
        try:
            # Wait a moment for monitoring to start
            time.sleep(2)
            
            # Get system status
            status = get_system_status()
            
            if status.get("status") == "healthy":
                self.logger.info("Health check passed")
                self.logger.info(f"System status: {json.dumps(status, indent=2)}")
                return True
            else:
                self.logger.error("Health check failed")
                self.logger.error(f"System status: {json.dumps(status, indent=2)}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error during health check: {e}")
            return False
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        self.logger.info("Cleaning up...")
        stop_monitoring()


def main():
    """Main deployment script."""
    parser = argparse.ArgumentParser(description="Deploy OpenDrive-XAI")
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--steps", "-s",
        nargs="+",
        choices=[
            "validate_environment",
            "install_dependencies", 
            "validate_configuration",
            "setup_directories",
            "run_tests",
            "start_monitoring",
            "health_check"
        ],
        help="Specific deployment steps to run"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually doing it"
    )
    
    args = parser.parse_args()
    
    # Create deployment manager
    manager = DeploymentManager(config_path=args.config, verbose=args.verbose)
    
    if args.dry_run:
        print("DRY RUN - Would execute the following steps:")
        steps = args.steps or [
            "validate_environment",
            "install_dependencies",
            "validate_configuration", 
            "setup_directories",
            "run_tests",
            "start_monitoring",
            "health_check"
        ]
        for step in steps:
            print(f"  - {step}")
        return
    
    try:
        # Run deployment
        success = manager.deploy(args.steps)
        
        if success:
            print("✅ Deployment completed successfully!")
            sys.exit(0)
        else:
            print("❌ Deployment failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Deployment interrupted by user")
        manager.cleanup()
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error during deployment: {e}")
        manager.cleanup()
        sys.exit(1)


if __name__ == "__main__":
    main() 