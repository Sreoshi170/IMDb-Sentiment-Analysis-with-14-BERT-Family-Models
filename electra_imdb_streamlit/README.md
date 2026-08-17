# ELECTRA IMDb Sentiment — Streamlit Deployment

This beginner-friendly project deploys your fine-tuned **ELECTRA-base** IMDb
sentiment classifier with Streamlit. It supports one-review prediction, confidence
visualization, batch CSV prediction, and downloadable results.

Your model archive is approximately 388 MB and contains a 438 MB weight file.
GitHub's normal per-file limit is smaller than that, so the recommended cloud setup is:

```text
GitHub repository (small app code) → Streamlit Community Cloud
                                      ↓
Hugging Face model repository (large model files)
```

For local testing, the app can load the weights directly from `model/` without
using Hugging Face.

## Project files

- `app.py` — Streamlit user interface and prediction code.
- `requirements.txt` — Python packages installed locally and in the cloud.
- `scripts/setup_local_model.py` — safely extracts your model ZIP for local use.
- `scripts/upload_model_to_hf.py` — uploads the model ZIP to Hugging Face.
- `sample_reviews.csv` — example file for batch prediction.
- `.streamlit/secrets.toml.example` — safe template for app configuration.

## Option 1: Run locally

### 1. Place the two ZIP files together

Download this Streamlit project and unzip it. Keep your existing
`ELECTRA-base_IMDb_deployment.zip` somewhere easy to find.

### 2. Open a terminal in this project

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Install the packages:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Install the model from your ZIP

Replace the example path with the actual path on your computer:

```bash
python scripts/setup_local_model.py --zip "/path/to/ELECTRA-base_IMDb_deployment.zip"
```

On Windows, an example may look like:

```powershell
python scripts/setup_local_model.py --zip "C:\Users\YourName\Downloads\ELECTRA-base_IMDb_deployment.zip"
```

### 4. Start Streamlit

```bash
streamlit run app.py
```

The terminal displays a local address, normally `http://localhost:8501`.

## Option 2: Deploy on Streamlit Community Cloud

### Step 1. Create a Hugging Face account and model repository

Create an account at Hugging Face. In your terminal, install the project
requirements and authenticate:

```bash
pip install -r requirements.txt
hf auth login
```

Create and upload the repository with the included helper. Replace the username:

```bash
python scripts/upload_model_to_hf.py \
  --model-zip "/path/to/ELECTRA-base_IMDb_deployment.zip" \
  --repo-id "YOUR_USERNAME/electra-imdb-sentiment"
```

The helper safely extracts the archive, creates the repository if necessary, and
uploads all tokenizer, configuration, metadata, and weight files. A public model
repository is simplest. Add `--private` if the files must remain private.

### Step 2. Put this app code on GitHub

Create a new GitHub repository and upload this project, **but not the original
model ZIP and not the extracted `model.safetensors` file**. The included
`.gitignore` already excludes them.

If you use Git from the terminal:

```bash
git init
git add .
git commit -m "Add ELECTRA IMDb Streamlit app"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/electra-imdb-streamlit.git
git push -u origin main
```

### Step 3. Create the Streamlit app

1. Open Streamlit Community Cloud and choose **Create app**.
2. Select your GitHub repository and the `main` branch.
3. Set the entrypoint file to `app.py`.
4. Open **Advanced settings**, choose Python 3.11 when available, and paste the
   secrets shown below.

For a public Hugging Face model repository:

```toml
HF_MODEL_REPO = "YOUR_USERNAME/electra-imdb-sentiment"
MAX_LENGTH = 128
```

For a private model repository, create a read-only Hugging Face access token and add:

```toml
HF_MODEL_REPO = "YOUR_USERNAME/electra-imdb-sentiment"
HF_TOKEN = "hf_your_read_only_token"
MAX_LENGTH = 128
```

Click **Deploy**. The first startup is slower because Streamlit downloads and loads
the model. Later Streamlit reruns reuse the model through `st.cache_resource`.

## How the prediction code works

1. `clean_review` converts HTML entities, removes HTML tags and URLs, and normalizes spaces.
2. The tokenizer converts text into token IDs understood by ELECTRA.
3. Inputs are padded and truncated to 128 tokens, matching the training notebook.
4. The model produces two logits. Softmax turns them into negative and positive probabilities.
5. The larger probability becomes the predicted label and is shown as confidence.

Batch predictions use groups of 16 reviews to keep memory consumption manageable.
The app limits one upload to 5,000 rows so a public deployment is less likely to run
out of memory.

## Expected model performance

The saved training run reported:

| Metric | Score |
|---|---:|
| Accuracy | 0.905 |
| Precision | 0.898 |
| Recall | 0.914 |
| F1 | 0.906 |
| ROC AUC | 0.959 |

These scores describe the held-out test split used in the notebook. Real-world
performance may differ, especially for sarcasm, mixed opinions, non-English text,
or reviews very different from IMDb.

## Why ELECTRA-base ranked first in this experiment

ELECTRA learns by finding tokens that a generator replaced. Its discriminator gets
a learning signal at every token position, while masked-language models such as BERT
learn directly from only the masked positions. This makes ELECTRA's pre-training
sample-efficient and often gives a strong accuracy-to-compute tradeoff.

The task is also a favorable match: IMDb is a large, balanced English dataset with
a clear binary objective, and ELECTRA-base was fully fine-tuned. However, the result
is specific to this experiment. Data split, seed, cleaning, maximum length, learning
rate, number of epochs, and checkpoint choice can change the ranking.

## Troubleshooting

**“No model was found”**

- Locally: run `scripts/setup_local_model.py` and verify `model/model.safetensors` exists.
- Cloud: add `HF_MODEL_REPO` to the Streamlit app's Secrets, not to GitHub.

**Hugging Face says the repository is unauthorized**

- Make the model repository public, or add a read-only `HF_TOKEN` to Streamlit Secrets.
- Confirm that the token can read the exact repository in `HF_MODEL_REPO`.

**The app runs out of memory**

- Restart the app from Streamlit Cloud.
- Use smaller CSV files. Single-review predictions require much less temporary memory.
- Do not increase the batch size or maximum token length unless the host has enough memory.

**The first prediction is slow**

- This is expected during cold start. The app must download and load ELECTRA once.
  Subsequent reruns reuse the cached model.

## Security

Never commit a Hugging Face token or `.streamlit/secrets.toml` to GitHub. The example
secrets file contains placeholders only, and the real secrets file is ignored by Git.

