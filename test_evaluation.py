"""Quick test script to validate evaluation logging system."""

import subprocess
import sys
from pathlib import Path
import pandas as pd

def main():
    """Run quick test and validate CSV outputs."""
    print("="*60)
    print("Testing Evaluation System")
    print("="*60)

    # Run Flower with test config
    print("\nRunning FL experiment with 5 rounds...")
    print("Command: flwr run . --run-config pyproject_test.toml")

    try:
        result = subprocess.run(
            ["flwr", "run", ".", "--run-config", "pyproject_test.toml"],
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout
        )

        if result.returncode != 0:
            print(f"\nError running experiment:")
            print(result.stderr)
            return False

        print("\nExperiment completed successfully!")

    except subprocess.TimeoutExpired:
        print("\nError: Experiment timed out (5 minutes)")
        return False
    except FileNotFoundError:
        print("\nError: 'flwr' command not found. Make sure Flower is installed.")
        return False

    # Check CSV files
    print("\n" + "="*60)
    print("Validating CSV outputs...")
    print("="*60)

    results_dir = Path("results")
    csv_files = list(results_dir.glob("test_evaluation_C2_*.csv"))

    if not csv_files:
        print("\nError: No CSV files found!")
        return False

    # Find latest files
    global_csv = sorted(results_dir.glob("test_evaluation_C2_global_*.csv"))[-1]
    client_csv = sorted(results_dir.glob("test_evaluation_C2_client_*.csv"))[-1]
    weight_csv = sorted(results_dir.glob("test_evaluation_C2_weight_*.csv"))[-1]

    print(f"\nFound CSV files:")
    print(f"  - Global: {global_csv}")
    print(f"  - Client: {client_csv}")
    print(f"  - Weight: {weight_csv}")

    # Validate global CSV
    print("\n1. Validating Global CSV:")
    df_global = pd.read_csv(global_csv)
    print(f"   Columns: {list(df_global.columns)}")
    print(f"   Rows: {len(df_global)}")
    print(f"   Expected columns: ['round', 'loss', 'accuracy', 'precision', 'recall', 'f1']")

    if len(df_global) > 0:
        print(f"\n   Sample data (first 3 rounds):")
        print(df_global.head(3).to_string(index=False))

        # Check for NaN
        nan_counts = df_global.isna().sum()
        if nan_counts.sum() > 0:
            print(f"\n   ⚠️  Warning: Found NaN values:")
            print(nan_counts[nan_counts > 0])
        else:
            print("\n   ✅ No NaN values found")
    else:
        print("   ❌ Error: Global CSV is empty!")

    # Validate client CSV
    print("\n2. Validating Client CSV:")
    df_client = pd.read_csv(client_csv)
    print(f"   Columns: {list(df_client.columns)}")
    print(f"   Rows: {len(df_client)}")
    print(f"   Expected columns: ['round', 'client_id', 'phase', 'loss', 'accuracy', 'precision', 'recall', 'f1', 'num_examples']")

    if len(df_client) > 0:
        print(f"\n   Sample data (first 5 entries):")
        print(df_client.head(5).to_string(index=False))

        # Check phases
        phases = df_client['phase'].unique()
        print(f"\n   Phases logged: {phases}")
        print(f"   Train entries: {len(df_client[df_client['phase'] == 'train'])}")
        print(f"   Evaluate entries: {len(df_client[df_client['phase'] == 'evaluate'])}")
    else:
        print("   ❌ Error: Client CSV is empty!")

    # Validate weight CSV
    print("\n3. Validating Weight CSV:")
    df_weight = pd.read_csv(weight_csv)
    print(f"   Columns: {list(df_weight.columns)}")
    print(f"   Rows: {len(df_weight)}")
    print(f"   Expected columns: ['round', 'weight_norm', 'weight_change', 'weight_relative_change']")

    if len(df_weight) > 0:
        print(f"\n   Sample data (first 3 rounds):")
        print(df_weight.head(3).to_string(index=False))

        # Check for NaN
        nan_counts = df_weight.isna().sum()
        if nan_counts.sum() > 0:
            print(f"\n   ⚠️  Warning: Found NaN values:")
            print(nan_counts[nan_counts > 0])
        else:
            print("\n   ✅ No NaN values found")
    else:
        print("   ❌ Error: Weight CSV is empty!")

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    success = True
    if len(df_global) == 0:
        print("❌ Global CSV is empty")
        success = False
    else:
        print(f"✅ Global CSV has {len(df_global)} rounds")

    if len(df_client) == 0:
        print("❌ Client CSV is empty")
        success = False
    else:
        print(f"✅ Client CSV has {len(df_client)} entries")

    if len(df_weight) == 0:
        print("❌ Weight CSV is empty")
        success = False
    else:
        print(f"✅ Weight CSV has {len(df_weight)} rounds")

    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed")

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
