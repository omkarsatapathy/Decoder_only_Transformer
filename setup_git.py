# setup_git.py
import os
import subprocess
from typing import List, Optional


class GitSetup:
    def __init__(self, project_path: str):
        self.project_path = project_path

    def run_command(self, command: List[str], cwd: Optional[str] = None) -> None:
        """Run a shell command"""
        try:
            subprocess.run(command, cwd=cwd or self.project_path, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running command {' '.join(command)}: {e}")
            raise

    def create_gitignore(self) -> None:
        """Create .gitignore file"""
        gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Model files and data
*.pth
*.pt
*.pkl
*.h5
data/
models/
checkpoints/

# Logs
logs/
*.log

# Jupyter Notebook
.ipynb_checkpoints
"""
        with open(os.path.join(self.project_path, '.gitignore'), 'w') as f:
            f.write(gitignore_content.strip())

    def create_readme(self) -> None:
        """Create README.md file"""
        readme_content = """# Transformer-Based Sentiment Analysis

## Project Overview
This project implements a transformer-based model for sentiment analysis using PyTorch.

## Features
- Custom transformer implementation
- Multi-head attention mechanism
- Configurable architecture
- Support for CPU/GPU/MPS training
- Comprehensive logging and monitoring

## Project Structure
```
project/
├── main.py            # Training script
├── utils.py           # Utility functions
├── config.py          # Configuration classes
├── tokenizer.py       # Tokenizer implementation
└── README.md         # Documentation
```

## Requirements
- Python 3.8+
- PyTorch 1.9+
- transformers
- numpy
- pandas

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python main.py --config config.json
```

## Results
- Achieved 80% validation accuracy on IMDB dataset
- Successfully handles both positive and negative sentiments
- Efficient training with configurable parameters

## License
MIT License
"""
        with open(os.path.join(self.project_path, 'README.md'), 'w') as f:
            f.write(readme_content.strip())

    def initialize_repository(self) -> None:
        """Initialize git repository"""
        self.run_command(['git', 'init'])

    def create_initial_commit(self) -> None:
        """Create initial commit"""
        self.run_command(['git', 'add', '.'])
        self.run_command(['git', 'commit', '-m', 'Initial commit: Project structure and basic implementation'])

    def add_remote(self, remote_url: str) -> None:
        """Add remote repository"""
        self.run_command(['git', 'remote', 'add', 'origin', remote_url])
        self.run_command(['git', 'branch', '-M', 'main'])

    def push_to_remote(self) -> None:
        """Push to remote repository"""
        self.run_command(['git', 'push', '-u', 'origin', 'main'])

    def setup(self, remote_url: Optional[str] = None) -> None:
        """Complete setup process"""
        print("Creating .gitignore...")
        self.create_gitignore()

        print("Creating README.md...")
        self.create_readme()

        print("Initializing git repository...")
        self.initialize_repository()

        print("Creating initial commit...")
        self.create_initial_commit()

        if remote_url:
            print("Adding remote repository...")
            self.add_remote(remote_url)

            print("Pushing to remote repository...")
            self.push_to_remote()

        print("Git setup completed successfully!")


def main():
    # Get current directory
    project_path = os.getcwd()

    # Create GitSetup instance
    git_setup = GitSetup(project_path)

    # Get remote URL from user
    remote_url = input("Enter GitHub repository URL (or press Enter to skip): ").strip()

    # Run setup
    git_setup.setup(remote_url if remote_url else None)


if __name__ == "__main__":
    main()