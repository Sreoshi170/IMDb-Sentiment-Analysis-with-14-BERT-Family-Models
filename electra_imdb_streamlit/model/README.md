# Local model folder

Do not commit model weights to GitHub. To run locally, extract the supplied model
ZIP here with:

```bash
python scripts/setup_local_model.py --zip /path/to/ELECTRA-base_IMDb_deployment.zip
```

The app automatically chooses this local folder when `model.safetensors` exists.
On Streamlit Community Cloud, it loads the same files from `HF_MODEL_REPO` instead.

