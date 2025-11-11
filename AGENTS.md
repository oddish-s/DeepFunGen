# Repository Guidelines

## Project Structure & Module Organization
Source lives under scripts/ with core modules for data handling, models, training, and visualization. Sequence datasets and sampling logic reside in scripts/core/dataset.py, while neural architectures sit in scripts/core/model.py. Training and inference entry-points are scripts/train.py and scripts/inference.py; reusable configs go in configs/, and trained artefacts are written to models/<config_name>/. Tests under 	ests/ exercise dataset splits, model wiring, and trainer behaviour.

## Build, Test, and Development Commands
Run python -m pytest from the repo root to execute the full test suite. Launch training with python scripts/train.py <config_name> (for example python scripts/train.py conv_tcn), and add --force-preprocess if frame caches should be refreshed. For quick smoke checks during development, target individual tests such as python -m pytest tests/test_model_factory.py.

## Coding Style & Naming Conventions
Follow PEP 8 and favour explicit, descriptive names (e.g., sequence_encoder_cfg). Prefer type hints for public functions and keep modules ASCII-only unless data requires otherwise. New frame or sequence encoder classes should inherit the existing base mixins and expose an output_dim attribute so they plug into the factory cleanly.

## Testing Guidelines
Add unit tests in 	ests/ whenever introducing sampling logic, encoders, or schedulers. Validate new model registry entries with a forward-pass test on synthetic data, mirroring the existing smoke tests. For data changes, craft fixtures that cover ignore ranges and edge-case splits. Use pytest -k <keyword> to iterate quickly and ensure python -m pytest remains green before submitting changes.

## Commit & Pull Request Guidelines
Write commits in the imperative mood ("Add ConvNeXt frame encoder") and keep them scoped by concern. Pull requests should summarise intention, list the configs touched, and call out any new dependencies or download requirements (e.g., DINO checkpoints). Include before/after metrics or screenshots for training or visualization changes when relevant, and link tracking issues so history stays traceable.

## Model Options & Recommended Configs
The model factory now supports multiple frame encoders (conv, esidual_cnn, convnext, dinov3_stack, dinov3_two_tower) and sequence encoders (	ransformer, lstm, 	cn). Curated presets live in configs/, including esidual_lstm.yaml, conv_tcn.yaml, convnext_transformer.yaml, and DINO-based recipes that point to ./models/dinov3_vits16_pretrain_lvd1689m-08c60483.pth. Use these as baselines when benchmarking architectures or tuning new datasets.
