# AML System - Quick Start Guide

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Redis (Required)

```bash
# macOS
brew install redis
brew services start redis

# Ubuntu/Debian
sudo apt-get install redis-server
sudo systemctl start redis

# Or use Docker
docker run -d -p 6379:6379 redis:7-alpine
```

### 3. Optional: Install Kafka (for streaming)

```bash
# Using Docker
docker-compose up -d zookeeper kafka
```

### 4. Optional: Install MLflow (for tracking)

```bash
# Using Docker
docker-compose up -d mlflow
```

## Running the Enhanced System

### Standalone Demo (Minimum Requirements)

```bash
# Requires only Redis
python code/scripts/run_enhanced_system.py
```

### Full Stack Demo

```bash
# Start all services
docker-compose up -d

# Run demo
docker-compose exec aml-system python code/scripts/run_enhanced_system.py

# Access dashboards:
# - Explainability: http://localhost:5002
# - MLflow: http://localhost:5001
# - Grafana: http://localhost:3000
```

## Key Features

### 1. Real Data Validation

```python
from code.validation.data_validator import DataValidator

validator = DataValidator()
comparison = validator.compare_distributions(synthetic, real_data)
```

### 2. Adversarial Testing

```python
from code.adversarial.adversarial_tester import AdversarialTester

tester = AdversarialTester()
results = tester.run_adversarial_test_suite(model, data)
```

### 3. Cost-Benefit Analysis

```python
from code.analysis.cost_benefit import CostBenefitAnalyzer

analyzer = CostBenefitAnalyzer()
analysis = analyzer.calculate_costs(confusion_matrix, volumes)
optimal = analyzer.optimize_threshold(y_true, y_proba)
```

### 4. Explainability Dashboard

```bash
python -m code.dashboard.explainability_dashboard
# Access at http://localhost:5001
```

## Configuration

Edit configuration in `code/scripts/run_enhanced_system.py`:

```python
config = {
    'redis_host': 'localhost',
    'redis_port': 6379,
    'kafka_servers': ['localhost:9092'],
    'mlflow_uri': 'http://localhost:5000',
    'enable_streaming': False,
    'enable_prometheus': False,
    'enable_dashboard': False
}
```

## Troubleshooting

### Redis Connection Error

- Check if Redis is running: `redis-cli ping`
- Should return: `PONG`

### Import Errors

- Ensure you're in the project root
- Activate virtual environment

### Memory Issues

- Reduce `num_attacks` in adversarial testing
- Use smaller dataset for demo

## Next Steps

1. Review `README.md` for complete documentation
2. Check `results/` folder for generated reports
3. Customize cost parameters in `code/analysis/cost_benefit.py`
4. Add your own data sources in `code/validation/data_validator.py`
