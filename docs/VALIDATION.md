# Validation

Run the tests with:

```bash
python -m unittest discover -s tests -v
```

Tests cover the model, preprocessing, data splits, metrics, checkpoint loading and five-fold training on synthetic data.

The full BiomedCLIP checkpoint, RTX 4090 execution and clinical results have not been verified with this implementation.
