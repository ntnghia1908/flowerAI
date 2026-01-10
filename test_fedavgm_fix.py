"""Test script to verify FedAvgM receives correct parameters."""

from pytorchexample.strategies import get_strategy

# Test FedAvgM parameter passing
print("=" * 60)
print("Testing FedAvgM Parameter Fix")
print("=" * 60)

# Create strategy with parameters
strategy = get_strategy(
    strategy_name="FedAvgM",
    fraction_train=1.0,
    fraction_evaluate=1.0,
    min_train_nodes=6,
    min_evaluate_nodes=6,
    min_available_nodes=6,
    server_momentum=0.9,
    server_learning_rate=0.5,  # This should be passed now
)

print(f"\nStrategy created: {type(strategy).__name__}")
print(f"Strategy base class: {type(strategy).__bases__}")

# Check if parameters are set correctly
if hasattr(strategy, 'server_momentum'):
    print(f"[OK] server_momentum: {strategy.server_momentum}")
else:
    print(f"[FAIL] server_momentum: NOT FOUND")

if hasattr(strategy, 'server_learning_rate'):
    print(f"[OK] server_learning_rate: {strategy.server_learning_rate}")
else:
    print(f"[FAIL] server_learning_rate: NOT FOUND")

# Expected values
print(f"\n{'=' * 60}")
print("Expected vs Actual:")
print(f"{'=' * 60}")
print(f"server_momentum:      Expected=0.9  | Actual={getattr(strategy, 'server_momentum', 'NOT SET')}")
print(f"server_learning_rate: Expected=0.5  | Actual={getattr(strategy, 'server_learning_rate', 'NOT SET')}")

# Verify
expected_momentum = 0.9
expected_lr = 0.5

actual_momentum = getattr(strategy, 'server_momentum', None)
actual_lr = getattr(strategy, 'server_learning_rate', None)

print(f"\n{'=' * 60}")
print("Verification:")
print(f"{'=' * 60}")

if actual_momentum == expected_momentum:
    print("[PASS] server_momentum: PASS")
else:
    print(f"[FAIL] server_momentum: FAIL (expected {expected_momentum}, got {actual_momentum})")

if actual_lr == expected_lr:
    print("[PASS] server_learning_rate: PASS")
else:
    print(f"[FAIL] server_learning_rate: FAIL (expected {expected_lr}, got {actual_lr})")

# Overall result
if actual_momentum == expected_momentum and actual_lr == expected_lr:
    print(f"\n*** ALL TESTS PASSED! Bug is fixed! ***")
    print(f"\nFedAvgM will now use:")
    print(f"  - server_learning_rate = 0.5 (not 1.0 default)")
    print(f"  - server_momentum = 0.9")
    print(f"\nThis should prevent the model collapse observed in previous experiments.")
else:
    print(f"\n*** TESTS FAILED! Bug still exists! ***")

print(f"\n{'=' * 60}")
