# AI PROJECT

## Author
**Adrian Kwaśnik**

## Setup Instructions

To run this project properly, make sure to follow the steps below carefully:

### 1. Install System Dependencies

Before installing Python dependencies, you **must** install `swig`. This is required for `gymnasium[box2d]` to work correctly.

```bash
sudo apt update
sudo apt install swig
```

### 2. Create a Virtual Environment (optional but recommended)

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python Dependencies

Once `swig` is installed, install the required Python packages:

```bash
pip install -r requirements.txt
```

### 4. How to run .py files

Different scripts have different args however all of them are run with --mode flag

--mode train (default)
--mode resume (resumes training)
--mode play (plays saved model)

```bash
python3 PPO_agent.py --mode play
```