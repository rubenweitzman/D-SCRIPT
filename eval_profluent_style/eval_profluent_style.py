#!/usr/bin/env python3
"""
PPI Prediction Evaluation Script (Profluent-style) for D-SCRIPT

Evaluates D-SCRIPT PPI prediction on datasets from dataset_to_eval.md.
Loads MDS datasets from GCS and runs PPI prediction pipeline using D-SCRIPT.

D-SCRIPT uses embeddings from Bepler+Berger protein language model and predicts
interaction probabilities based on contact maps.

Usage:
    python eval_profluent_style/eval_profluent_style.py \
        --dataset-name alignment_skempi \
        --model samsl/topsy_turvy_human_v1 \
        --output-dir ./results/alignment_skempi
"""

import os
import sys
import logging
import click
import pickle
import subprocess
import shlex
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
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


# Dataset paths from dataset_to_eval.md
DATASET_PATHS = {
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


def load_mds_dataset(gcs_path: str, max_samples: Optional[int] = None, local_cache_dir: Optional[str] = None) -> List[Dict]:
    """
    Load MDS dataset from GCS.
    
    Args:
        gcs_path: GCS path to MDS dataset
        max_samples: Maximum number of samples to load (None for all)
        local_cache_dir: Optional local directory for caching (auto-generated if None)
    
    Returns:
        List of samples, each with 'sequence' and 'value' fields
    """
    logger.info(f"Loading MDS dataset from: {gcs_path}")
    
    # Use temp directory for caching if not provided
    if local_cache_dir is None:
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
    
    # Determine how many samples to load
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
    
    logger.info(f"Loaded {len(samples)} samples")
    return samples


def extract_protein_pairs(samples: List[Dict]) -> List[Dict]:
    """
    Extract protein pairs from samples.
    
    D-SCRIPT expects pairs in TSV format with protein keys (no header).
    
    Returns:
        List of dicts with 'protein1' and 'protein2' sequences
    """
    pairs = []
    
    for i, sample in enumerate(tqdm(samples, desc="Extracting protein pairs", unit="samples")):
        seq = sample['sequence']
        
        # Split by comma (always the separator in these datasets)
        parts = seq.split(',', 1)
        if len(parts) == 2:
            seq1, seq2 = parts[0].strip(), parts[1].strip()
            if seq1 and seq2:
                # Use index as protein key (D-SCRIPT uses keys from FASTA)
                pairs.append({
                    'key': f'protein_{i}',
                    'protein1': seq1,
                    'protein2': seq2,
                })
            else:
                logger.warning(f"Empty sequence in sample {i}")
                pairs.append({
                    'key': f'protein_{i}',
                    'protein1': '',
                    'protein2': ''
                })
        else:
            logger.warning(f"Sample {i} does not contain comma-separated pair: {seq[:50]}...")
            pairs.append({
                'key': f'protein_{i}',
                'protein1': '',
                'protein2': ''
            })
    
    logger.info(f"Extracted {len(pairs)} protein pairs from {len(samples)} samples")
    
    return pairs


def create_fasta_file(pairs: List[Dict], output_path: str) -> str:
    """Create a FASTA file from pairs for D-SCRIPT embedding."""
    with open(output_path, 'w') as f:
        seen_keys = set()
        for pair in pairs:
            key = pair['key']
            # Write protein1
            if key not in seen_keys:
                f.write(f">{key}_1\n{pair['protein1']}\n")
                seen_keys.add(key)
            # Write protein2 (use _2 suffix)
            key2 = f"{key}_2"
            if key2 not in seen_keys:
                f.write(f">{key2}\n{pair['protein2']}\n")
                seen_keys.add(key2)
    
    logger.info(f"Created FASTA file: {output_path} with {len(seen_keys)} sequences")
    return output_path


def create_tsv_file(pairs: List[Dict], output_path: str) -> str:
    """Create a TSV file from pairs in D-SCRIPT format (no header, protein1_key, protein2_key)."""
    with open(output_path, 'w') as f:
        for pair in pairs:
            key = pair['key']
            # D-SCRIPT expects keys matching FASTA headers
            f.write(f"{key}_1\t{key}_2\n")
    
    logger.info(f"Created TSV file: {output_path} with {len(pairs)} pairs")
    return output_path


def run_ppi_prediction(
    fasta_file: str,
    pairs_tsv: str,
    model: str,
    output_dir: Path,
    device: str = "0",
    embeddings_file: Optional[str] = None,
) -> Dict:
    """
    Run D-SCRIPT PPI prediction pipeline.
    
    Args:
        fasta_file: Path to FASTA file with sequences
        pairs_tsv: Path to TSV file with protein pairs
        model: D-SCRIPT model name (e.g., 'samsl/topsy_turvy_human_v1') or path
        output_dir: Output directory for results
        device: Device for inference ('cpu' or GPU index like '0')
        embeddings_file: Optional path to pre-computed embeddings (h5 file)
    
    Returns:
        Dictionary with results including predictions
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Generate embeddings if not provided
    if embeddings_file is None:
        logger.info("Generating embeddings from FASTA file...")
        embeddings_file = str(output_dir / "embeddings.h5")
        
        # Use Python module directly instead of CLI to avoid argument parsing issues
        try:
            from dscript.commands.embed import add_args, main as embed_main
            import argparse
            
            # Create args object
            parser = argparse.ArgumentParser()
            add_args(parser)
            embed_args = parser.parse_args([
                "--seqs", fasta_file,
                "--outfile", embeddings_file,
                "--device", device if device.lower() != "cpu" else "cpu"
            ])
            
            embed_main(embed_args)
            logger.info("✓ Embeddings generated successfully")
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    else:
        logger.info(f"Using pre-computed embeddings: {embeddings_file}")
    
    # Step 2: Run predictions
    logger.info("Running D-SCRIPT predictions...")
    predictions_file = str(output_dir / "predictions.tsv")
    
    # Use Python module directly instead of CLI to avoid argument parsing issues
    try:
        from dscript.commands.predict_serial import add_args, main as predict_main
        import argparse
        
        # Create args object
        parser = argparse.ArgumentParser()
        add_args(parser)
        predict_args = parser.parse_args([
            "--pairs", pairs_tsv,
            "--embeddings", embeddings_file,
            "--model", model,
            "--outfile", predictions_file,
            "--device", device if device.lower() != "cpu" else "cpu"
        ])
        
        predict_main(predict_args)
        logger.info("✓ Predictions generated successfully")
    except Exception as e:
        logger.error(f"Failed to generate predictions: {e}")
        raise
    
    # Load predictions
    pred_file = Path(predictions_file + ".tsv")
    if pred_file.exists():
        # D-SCRIPT outputs TSV with: protein1_key, protein2_key, score
        predictions_df = pd.read_csv(pred_file, sep="\t", header=None, names=['protein1_key', 'protein2_key', 'score'])
        predictions = predictions_df['score'].values
        logger.info(f"Loaded {len(predictions)} predictions")
        logger.info(f"Prediction range: {predictions.min():.4f} - {predictions.max():.4f}")
    else:
        logger.error(f"Prediction file not found: {pred_file}")
        predictions = np.array([])
    
    return {
        'predictions': predictions,
        'num_pairs': len(predictions),
    }


@click.command()
@click.option(
    "--dataset-name",
    type=str,
    help="Dataset name from dataset_to_eval.md (e.g., 'alignment_skempi')"
)
@click.option(
    "--gcs-path",
    type=str,
    help="GCS path to MDS dataset (overrides dataset-name if provided)"
)
@click.option(
    "--model",
    type=str,
    default="samsl/topsy_turvy_human_v1",
    help="D-SCRIPT model name or path. Options: "
         "'samsl/topsy_turvy_human_v1' (recommended), "
         "'samsl/dscript_human_v1', "
         "'samsl/tt3d_human_v1', "
         "or path to .pt/.sav file [default: samsl/topsy_turvy_human_v1]"
)
@click.option(
    "--output-dir",
    type=str,
    required=True,
    help="Output directory for results"
)
@click.option(
    "--max-samples",
    type=int,
    default=None,
    help="Maximum number of samples to process (for testing, None = all)"
)
@click.option(
    "--device",
    type=str,
    default="0",
    help="Device for D-SCRIPT model ('cpu' or GPU index like '0') [default: 0]"
)
@click.option(
    "--embeddings",
    type=str,
    default=None,
    help="Optional path to pre-computed embeddings (h5 file). If not provided, embeddings will be generated."
)
def main(
    dataset_name: Optional[str],
    gcs_path: Optional[str],
    model: str,
    output_dir: str,
    max_samples: Optional[int],
    device: str,
    embeddings: Optional[str],
) -> None:
    """Run D-SCRIPT PPI prediction evaluation on MDS dataset."""
    
    # Determine GCS path
    if gcs_path:
        dataset_gcs_path = gcs_path
    elif dataset_name and dataset_name in DATASET_PATHS:
        dataset_gcs_path = DATASET_PATHS[dataset_name]
    else:
        logger.error(f"Must provide either --gcs-path or --dataset-name (one of: {list(DATASET_PATHS.keys())})")
        sys.exit(1)
    
    logger.info("="*80)
    logger.info("D-SCRIPT PPI Prediction Evaluation")
    logger.info("="*80)
    logger.info(f"Dataset: {dataset_name or 'custom'}")
    logger.info(f"GCS Path: {dataset_gcs_path}")
    logger.info(f"Model: {model}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info("="*80)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load MDS dataset
    logger.info("\n[Step 1/5] Loading MDS dataset...")
    samples = load_mds_dataset(dataset_gcs_path, max_samples=max_samples)
    
    # Step 2: Extract protein pairs
    logger.info("\n[Step 2/5] Extracting protein pairs...")
    pairs = extract_protein_pairs(samples)
    
    if len(pairs) == 0:
        logger.error("No protein pairs extracted from samples!")
        sys.exit(1)
    
    # Step 3: Create FASTA file
    logger.info("\n[Step 3/5] Creating FASTA file...")
    fasta_file = str(output_path / "sequences.fasta")
    create_fasta_file(pairs, fasta_file)
    
    # Step 4: Create TSV file for pairs
    logger.info("\n[Step 4/5] Creating TSV file for pairs...")
    pairs_tsv = str(output_path / "pairs.tsv")
    create_tsv_file(pairs, pairs_tsv)
    
    # Step 5: Run PPI prediction
    logger.info("\n[Step 5/5] Running D-SCRIPT PPI prediction...")
    results = run_ppi_prediction(
        fasta_file=fasta_file,
        pairs_tsv=pairs_tsv,
        model=model,
        output_dir=output_path,
        device=device,
        embeddings_file=embeddings,
    )
    
    # Step 6: Create CSV output with predictions
    logger.info("\n[Step 6/6] Creating CSV output with predictions...")
    
    predictions = results['predictions']
    
    # Build output rows preserving original structure
    output_rows = []
    for i, sample in enumerate(tqdm(samples, desc="Creating output", unit="samples")):
        # Prediction score (always present)
        if i < len(predictions):
            prediction_score = float(predictions[i])
        else:
            logger.warning(f"Could not find prediction for sample {i}")
            prediction_score = np.nan
        
        # Preserve original structure and add prediction
        row = {
            'data_source': sample.get('data_source', ''),
            'sequence': sample['sequence'],
            'value': sample['value'],
            'prediction': prediction_score,
        }
        
        output_rows.append(row)
    
    # Create DataFrame and save to CSV
    output_df = pd.DataFrame(output_rows)
    csv_output_file = output_path / "results.csv"
    output_df.to_csv(csv_output_file, index=False)
    logger.info(f"Saved CSV results to {csv_output_file}")
    logger.info(f"CSV contains {len(output_df)} rows with columns: {list(output_df.columns)}")
    
    # Also save pickle for detailed analysis
    results_file = output_path / "ppi_results.pkl"
    logger.info(f"\nSaving detailed results to {results_file}")
    
    # Add metadata to results
    results['dataset_name'] = dataset_name or 'custom'
    results['dataset_gcs_path'] = dataset_gcs_path
    results['num_samples'] = len(samples)
    results['num_pairs'] = len(pairs)
    
    with open(results_file, "wb") as f:
        pickle.dump(results, f)
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("Evaluation Complete!")
    logger.info("="*80)
    logger.info(f"Total samples processed: {len(samples)}")
    logger.info(f"Total pairs: {len(pairs)}")
    
    # Log predictions
    if 'prediction' in output_df.columns and not output_df['prediction'].isna().all():
        valid_preds = output_df['prediction'].dropna()
        logger.info(f"Prediction range: {valid_preds.min():.4f} - {valid_preds.max():.4f}")
        logger.info(f"Prediction mean: {valid_preds.mean():.4f}")
    
    logger.info(f"CSV results saved to: {csv_output_file}")
    logger.info(f"Detailed results saved to: {results_file}")
    
    # Upload results to GCS
    gcs_bucket = "profluent-rweitzman"
    method_name = "dscript"
    gcs_base_path = f"gs://{gcs_bucket}/baseline_results/{method_name}/{dataset_name or 'custom'}"
    
    logger.info(f"\nUploading results to GCS: {gcs_base_path}")
    try:
        # Upload CSV file
        csv_gcs_path = f"{gcs_base_path}/results.csv"
        logger.info(f"Uploading {csv_output_file} -> {csv_gcs_path}")
        cmd = f"gcloud storage cp {shlex.quote(str(csv_output_file))} {shlex.quote(csv_gcs_path)}"
        subprocess.run(shlex.split(cmd), check=True)
        logger.info(f"✓ Successfully uploaded CSV to {csv_gcs_path}")
        
        # Upload pickle file (optional, but useful for detailed analysis)
        pkl_gcs_path = f"{gcs_base_path}/ppi_results.pkl"
        logger.info(f"Uploading {results_file} -> {pkl_gcs_path}")
        cmd = f"gcloud storage cp {shlex.quote(str(results_file))} {shlex.quote(pkl_gcs_path)}"
        subprocess.run(shlex.split(cmd), check=True)
        logger.info(f"✓ Successfully uploaded pickle to {pkl_gcs_path}")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to upload to GCS: {e}")
        logger.error("Results are still saved locally")
    except Exception as e:
        logger.error(f"Unexpected error uploading to GCS: {e}")
        logger.error("Results are still saved locally")
    
    logger.info("="*80)


if __name__ == "__main__":
    main()

