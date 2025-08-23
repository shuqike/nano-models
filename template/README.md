TL;DR flow

1. Compose config with Hydra → `DictConfig`.

2. Generate run name with coolname (if none).

3. Validate with Pydantic → `Config` (typed).

4. Initialize W&B with the validated config.

5. Train, log metrics/artefacts.

6. Override everything from CLI; use multirun for quick sweeps.