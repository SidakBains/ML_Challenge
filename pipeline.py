#!/usr/bin/env python3
"""
Main Pipeline Orchestration for Amazon ML Challenge 2026
End-to-end: Data -> Blocking -> Features -> Training -> Inference -> Submission
"""

import argparse
import sys
import os

def run_eda():
    """Run exploratory data analysis"""
    print("=== RUNNING EDA ===")
    os.system("cd student_resource && python ../eda_fast.py")
    os.system("cd student_resource && python ../eda_matched.py")

def run_training(sample_frac=1.0, data_dir='dataset/train', output_dir='models'):
    """Run training pipeline"""
    print("=== RUNNING TRAINING ===")
    cmd = f"cd student_resource && python ../train.py --sample {sample_frac} --data-dir {data_dir} --output-dir {output_dir}"
    os.system(cmd)

def run_inference(test_dir='dataset/test', model_dir='models', output_dir='output', max_candidates=100, validate=False):
    """Run inference on test set"""
    print("=== RUNNING INFERENCE ===")
    cmd = (f"cd student_resource && python ../predict.py "
           f"--test-dir {test_dir} --model-dir {model_dir} --output-dir {output_dir} "
           f"--max-candidates {max_candidates}")
    if validate:
        cmd += " --validate"
    os.system(cmd)

def run_full_pipeline(sample_frac=1.0, validate=True):
    """Run complete pipeline"""
    print("=" * 60)
    print("AMAZON ML CHALLENGE 2026 - FULL PIPELINE")
    print("=" * 60)
    
    run_eda()
    run_training(sample_frac=sample_frac)
    run_inference(validate=validate)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print("Output files in student_resource/output/:")
    print("  - matching_results.tsv (for leaderboard)")
    print("  - candidate_pairs.tsv (for audit)")
    print("\nTo create final submission zip:")
    print("  cd student_resource && zip -r ../<team_name>_submission.zip \\")
    print("    output/matching_results.tsv \\")
    print("    output/candidate_pairs.tsv \\")
    print("    code/business_entity_resolution/ \\")
    print("    Documentation_template.md")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Amazon ML Challenge 2026 Pipeline')
    parser.add_argument('--stage', choices=['eda', 'train', 'predict', 'full'], 
                        default='full', help='Pipeline stage to run')
    parser.add_argument('--sample', type=float, default=1.0, help='Fraction of data for training (quick test)')
    parser.add_argument('--data-dir', default='dataset/train', help='Training data directory')
    parser.add_argument('--test-dir', default='dataset/test', help='Test data directory')
    parser.add_argument('--model-dir', default='models', help='Model output directory')
    parser.add_argument('--output-dir', default='output', help='Prediction output directory')
    parser.add_argument('--max-candidates', type=int, default=100, help='Max candidates per S1')
    parser.add_argument('--validate', action='store_true', help='Run validator after inference')
    
    args = parser.parse_args()
    
    if args.stage == 'eda':
        run_eda()
    elif args.stage == 'train':
        run_training(args.sample, args.data_dir, args.model_dir)
    elif args.stage == 'predict':
        run_inference(args.test_dir, args.model_dir, args.output_dir, args.max_candidates, args.validate)
    elif args.stage == 'full':
        run_full_pipeline(args.sample, args.validate)