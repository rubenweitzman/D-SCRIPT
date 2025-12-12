#!/usr/bin/env python3
"""
PPI Prediction Evaluation Script (Profluent-style) for D-SCRIPT

Evaluates D-SCRIPT PPI prediction on datasets with BATCHED processing
to avoid disk space issues with large datasets.

Key features:
- Processes data in batches (default 10K pairs per batch)
- Deletes embeddings after each batch to save disk space
- Combines partial results at the end

Usage:
    python eval_profluent_style/eval_profluent_style.py \
        --dataset-name alignment_intact_ppi \
        --output-dir ./results/alignment_intact_ppi \
        --batch-size 10000
"""

import os
import sys
import logging
import click
import pickle
import subprocess
import shlex
import tempfile
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

# Get the project root (parent of eval_profluent_style folder)
PROJECT_ROOT = Path(__file__).parent.parent

# Add project root to path for D-SCRIPT imports
sys.path.insert(0, str(PROJECT_ROOT))

# Try to import streaming (will be available if using pixi or if installed)
try:
    from streaming import StreamingDataset
except ImportError:
    logging.error("streaming package not found. Install with: pip install mosaicml-streaming")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Dataset paths - supports both MDS and CSV formats
DATASET_PATHS_MDS = {
    "alignment_skempi": "gs://profluent-rweitzman/alignment/test_dataset_mds_round_2/alignment_skempi",
    "alignment_mutational_ppi": "gs://profluent-rweitzman/alignment/test_dataset_mds_round_2/alignment_mutational_ppi",
    "alignment_yeast_ppi_combined": "gs://profluent-rweitzman/alignment/test_dataset_mds_round_2/alignment_yeast_ppi_combined",
    "alignment_human_ppi_combined": "gs://profluent-rweitzman/alignment/test_dataset_mds_round_2/alignment_human_ppi_combined",
    "alignment_intact_ppi": "gs://profluent-rweitzman/alignment/test_dataset_mds_round_2/alignment_intact_ppi",
    "validation_high_score_20_species": "gs://profluent-rweitzman/alignment/test_dataset_mds_round_2/validation_high_score_20_species",
    "alignment_bindinggym_combined": "gs://profluent-rweitzman/alignment/test_dataset_mds_round_2/alignment_bindinggym_combined",
    "alignment_gold_combined": "gs://profluent-rweitzman/alignment/test_dataset_mds_round_2/alignment_gold_combined",
    "human_validation_with_negatives": "gs://profluent-rweitzman/alignment/test_dataset_mds_round_2/human_validation_with_negatives",
}

DATASET_PATHS_CSV = {
    "alignment_intact_covid": "gs://profluent-rweitzman/alignment/test_dataset_csv_round_2/alignment_intact_covid.csv",
    "alignment_virus_human": "gs://profluent-rweitzman/alignment/test_dataset_csv_round_2/alignment_virus_human.csv",
}

DATASET_PATHS = {**DATASET_PATHS_MDS, **DATASET_PATHS_CSV}


def load_csv_dataset(csv_path: str, max_samples: Optional[int] = None) -> Tuple[pd.DataFrame, List[Dict]]:
    """Load CSV dataset from GCS or local path."""
    logger.info(f"Loading CSV dataset from: {csv_path}")
    
    local_csv = None
    if csv_path.startswith("gs://"):
        local_csv = tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name
        logger.info(f"Downloading to: {local_csv}")
        cmd = f"gcloud storage cp {shlex.quote(csv_path)} {shlex.quote(local_csv)}"
        subprocess.run(shlex.split(cmd), check=True)
        csv_path = local_csv
    
    df = pd.read_csv(csv_path)
    logger.info(f"CSV contains {len(df)} rows, columns: {list(df.columns)}")
    
    if max_samples and max_samples < len(df):
        df = df.head(max_samples)
        logger.info(f"Limited to {max_samples} samples")
    
    samples = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading samples"):
        samples.append({
            'sequence': row.get('sequence', ''),
            'value': float(row.get('value', 0.0)),
            'data_source': row.get('data_source', 'default')
        })
    
    # Cleanup downloaded CSV
    if local_csv and os.path.exists(local_csv):
        os.remove(local_csv)
    
    return df, samples


def load_mds_dataset(gcs_path: str, max_samples: Optional[int] = None) -> List[Dict]:
    """Load MDS dataset from GCS."""
    logger.info(f"Loading MDS dataset from: {gcs_path}")
    
    local_cache_dir = tempfile.mkdtemp(prefix="mds_cache_")
    logger.info(f"Using temporary cache directory: {local_cache_dir}")
    
    dataset = StreamingDataset(
        remote=gcs_path,
        local=local_cache_dir,
        batch_size=1000,
        shuffle=False,
        num_canonical_nodes=1,
        download_timeout=600,
    )
    
    total_samples = len(dataset)
    logger.info(f"Dataset contains {total_samples} samples")
    
    num_to_load = min(max_samples, total_samples) if max_samples else total_samples
    
    samples = []
    with tqdm(total=num_to_load, desc="Loading samples", unit="samples") as pbar:
        for i, sample in enumerate(dataset):
            if max_samples and i >= max_samples:
                break
            samples.append({
                'sequence': sample.get('sequence', ''),
                'value': float(sample.get('value', 0.0)),
                'data_source': sample.get('data_source', 'default')
            })
            pbar.update(1)
    
    # Cleanup MDS cache
    shutil.rmtree(local_cache_dir, ignore_errors=True)
    
    logger.info(f"Loaded {len(samples)} samples")
    return samples


def extract_protein_pairs(samples: List[Dict]) -> List[Dict]:
    """Extract protein pairs from samples."""
    pairs = []
    
    for i, sample in enumerate(samples):
        seq = sample['sequence']
        parts = seq.split(',', 1)
        if len(parts) == 2:
            seq1, seq2 = parts[0].strip(), parts[1].strip()
            if seq1 and seq2:
                pairs.append({
                    'key': f'protein_{i}',
                    'protein1': seq1,
                    'protein2': seq2,
                })
            else:
                pairs.append({'key': f'protein_{i}', 'protein1': '', 'protein2': ''})
        else:
            pairs.append({'key': f'protein_{i}', 'protein1': '', 'protein2': ''})
    
    return pairs


def create_fasta_file(pairs: List[Dict], output_path: str) -> str:
    """Create a FASTA file from pairs for D-SCRIPT embedding."""
    with open(output_path, 'w') as f:
        seen_keys = set()
        for pair in pairs:
            key = pair['key']
            if key not in seen_keys:
                f.write(f">{key}_1\n{pair['protein1']}\n")
                seen_keys.add(key)
            key2 = f"{key}_2"
            if key2 not in seen_keys:
                f.write(f">{key2}\n{pair['protein2']}\n")
                seen_keys.add(key2)
    return output_path


def create_tsv_file(pairs: List[Dict], output_path: str) -> str:
    """Create a TSV file from pairs in D-SCRIPT format."""
    with open(output_path, 'w') as f:
        for pair in pairs:
            key = pair['key']
            f.write(f"{key}_1\t{key}_2\n")
    return output_path


def run_batch_prediction(
    pairs: List[Dict],
    model: str,
    batch_dir: Path,
    device: str = "0",
) -> np.ndarray:
    """
    Run D-SCRIPT prediction on a batch of pairs.
    Creates embeddings, runs prediction, then cleans up.
    
    Returns:
        Array of prediction scores for this batch
    """
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    # Create FASTA and TSV files for this batch
    fasta_file = str(batch_dir / "sequences.fasta")
    pairs_tsv = str(batch_dir / "pairs.tsv")
    embeddings_file = str(batch_dir / "embeddings.h5")
    predictions_file = str(batch_dir / "predictions.tsv")
    
    create_fasta_file(pairs, fasta_file)
    create_tsv_file(pairs, pairs_tsv)
    
    # Step 1: Generate embeddings
    try:
        from dscript.commands.embed import add_args as embed_add_args, main as embed_main
        import argparse
        
        parser = argparse.ArgumentParser()
        embed_add_args(parser)
        embed_args = parser.parse_args([
            "--seqs", fasta_file,
            "--outfile", embeddings_file,
            "--device", device if device.lower() != "cpu" else "cpu"
        ])
        
        embed_main(embed_args)
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        raise
    
    # Step 2: Run predictions
    try:
        from dscript.commands.predict_serial import add_args as predict_add_args, main as predict_main
        import argparse
        
        parser = argparse.ArgumentParser()
        predict_add_args(parser)
        predict_args = parser.parse_args([
            "--pairs", pairs_tsv,
            "--embeddings", embeddings_file,
            "--model", model,
            "--outfile", predictions_file,
            "--device", device if device.lower() != "cpu" else "cpu"
        ])
        
        predict_main(predict_args)
    except Exception as e:
        logger.error(f"Failed to generate predictions: {e}")
        raise
    
    # Load predictions
    pred_file = Path(predictions_file + ".tsv")
    if pred_file.exists():
        predictions_df = pd.read_csv(pred_file, sep="\t", header=None, names=['protein1_key', 'protein2_key', 'score'])
        predictions = predictions_df['score'].values
    else:
        logger.error(f"Prediction file not found: {pred_file}")
        predictions = np.array([np.nan] * len(pairs))
    
    # Cleanup batch files (especially embeddings which are huge!)
    for f in [fasta_file, pairs_tsv, embeddings_file, predictions_file, str(pred_file)]:
        if os.path.exists(f):
            os.remove(f)
    
    return predictions


@click.command()
@click.option("--dataset-name", type=str, help="Dataset name (e.g., 'alignment_intact_ppi')")
@click.option("--gcs-path", type=str, help="GCS path to MDS dataset (overrides dataset-name)")
@click.option("--csv-path", type=str, help="GCS or local path to CSV file")
@click.option("--model", type=str, default="samsl/topsy_turvy_human_v1", help="D-SCRIPT model")
@click.option("--output-dir", type=str, required=True, help="Output directory")
@click.option("--max-samples", type=int, default=None, help="Max samples (for testing)")
@click.option("--device", type=str, default="0", help="GPU device")
@click.option("--batch-size", type=int, default=10000, help="Batch size for processing (default: 10000 pairs)")
def main(
    dataset_name: Optional[str],
    gcs_path: Optional[str],
    csv_path: Optional[str],
    model: str,
    output_dir: str,
    max_samples: Optional[int],
    device: str,
    batch_size: int,
) -> None:
    """Run D-SCRIPT PPI prediction with batched processing to save disk space."""
    
    # Determine data source
    use_csv = False
    original_df = None
    
    if csv_path:
        data_path = csv_path
        use_csv = True
    elif gcs_path:
        data_path = gcs_path
        use_csv = gcs_path.endswith('.csv')
    elif dataset_name and dataset_name in DATASET_PATHS:
        data_path = DATASET_PATHS[dataset_name]
        use_csv = dataset_name in DATASET_PATHS_CSV or data_path.endswith('.csv')
    else:
        logger.error(f"Must provide --csv-path, --gcs-path, or --dataset-name")
        sys.exit(1)
    
    logger.info("="*80)
    logger.info("D-SCRIPT PPI Prediction (BATCHED)")
    logger.info("="*80)
    logger.info(f"Dataset: {dataset_name or 'custom'}")
    logger.info(f"Data Path: {data_path}")
    logger.info(f"Format: {'CSV' if use_csv else 'MDS'}")
    logger.info(f"Model: {model}")
    logger.info(f"Batch Size: {batch_size}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info("="*80)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load dataset
    logger.info("\n[Step 1] Loading dataset...")
    if use_csv:
        original_df, samples = load_csv_dataset(data_path, max_samples=max_samples)
    else:
        samples = load_mds_dataset(data_path, max_samples=max_samples)
    
    # Step 2: Extract protein pairs
    logger.info("\n[Step 2] Extracting protein pairs...")
    pairs = extract_protein_pairs(samples)
    logger.info(f"Total pairs: {len(pairs)}")
    
    if len(pairs) == 0:
        logger.error("No protein pairs extracted!")
        sys.exit(1)
    
    # Step 3: Process in batches
    logger.info(f"\n[Step 3] Processing in batches of {batch_size}...")
    
    num_batches = (len(pairs) + batch_size - 1) // batch_size
    all_predictions = []
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(pairs))
        batch_pairs = pairs[start_idx:end_idx]
        
        logger.info(f"\n--- Batch {batch_idx + 1}/{num_batches} (samples {start_idx}-{end_idx}) ---")
        
        batch_dir = output_path / f"batch_{batch_idx}"
        
        try:
            batch_predictions = run_batch_prediction(
                pairs=batch_pairs,
                model=model,
                batch_dir=batch_dir,
                device=device,
            )
            all_predictions.extend(batch_predictions)
            logger.info(f"✓ Batch {batch_idx + 1} complete: {len(batch_predictions)} predictions")
        except Exception as e:
            logger.error(f"Batch {batch_idx + 1} failed: {e}")
            # Fill with NaN for failed batch
            all_predictions.extend([np.nan] * len(batch_pairs))
        finally:
            # Cleanup batch directory
            if batch_dir.exists():
                shutil.rmtree(batch_dir, ignore_errors=True)
    
    predictions = np.array(all_predictions)
    
    # Step 4: Create output CSV
    logger.info("\n[Step 4] Creating output CSV...")
    
    if use_csv and original_df is not None:
        output_df = original_df.copy()
        if len(predictions) == len(output_df):
            output_df['dscript_prediction'] = predictions
        else:
            output_df['dscript_prediction'] = np.nan
            output_df.loc[:len(predictions)-1, 'dscript_prediction'] = predictions
    else:
        output_rows = []
        for i, sample in enumerate(samples):
            row = {
                'data_source': sample.get('data_source', ''),
                'sequence': sample['sequence'],
                'value': sample['value'],
                'dscript_prediction': float(predictions[i]) if i < len(predictions) else np.nan,
            }
            output_rows.append(row)
        output_df = pd.DataFrame(output_rows)
    
    csv_output_file = output_path / "results.csv"
    output_df.to_csv(csv_output_file, index=False)
    logger.info(f"Saved CSV results to {csv_output_file}")
    
    # Save pickle
    results_file = output_path / "ppi_results.pkl"
    results = {
        'predictions': predictions,
        'num_pairs': len(pairs),
        'dataset_name': dataset_name or 'custom',
        'data_path': data_path,
        'num_samples': len(samples),
    }
    with open(results_file, "wb") as f:
        pickle.dump(results, f)
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("Evaluation Complete!")
    logger.info("="*80)
    logger.info(f"Total samples: {len(samples)}")
    logger.info(f"Total batches: {num_batches}")
    
    valid_preds = output_df['dscript_prediction'].dropna()
    if len(valid_preds) > 0:
        logger.info(f"Prediction range: {valid_preds.min():.4f} - {valid_preds.max():.4f}")
        logger.info(f"Prediction mean: {valid_preds.mean():.4f}")
    
    # Upload to GCS
    gcs_bucket = "profluent-rweitzman"
    method_name = "dscript"
    gcs_base_path = f"gs://{gcs_bucket}/baseline_results/{method_name}/{dataset_name or 'custom'}"
    
    logger.info(f"\nUploading results to GCS: {gcs_base_path}")
    try:
        csv_gcs_path = f"{gcs_base_path}/results.csv"
        cmd = f"gcloud storage cp {shlex.quote(str(csv_output_file))} {shlex.quote(csv_gcs_path)}"
        subprocess.run(shlex.split(cmd), check=True)
        logger.info(f"✓ Uploaded CSV to {csv_gcs_path}")
        
        pkl_gcs_path = f"{gcs_base_path}/ppi_results.pkl"
        cmd = f"gcloud storage cp {shlex.quote(str(results_file))} {shlex.quote(pkl_gcs_path)}"
        subprocess.run(shlex.split(cmd), check=True)
        logger.info(f"✓ Uploaded pickle to {pkl_gcs_path}")
        
    except Exception as e:
        logger.error(f"Failed to upload to GCS: {e}")
    
    logger.info("="*80)


if __name__ == "__main__":
    main()
