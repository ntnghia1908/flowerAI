@echo off
REM Script to run all test cases with 10 rounds each
REM Author: Claude Code
REM Date: 2025-12-26

echo ================================================================
echo Running All Test Cases (10 rounds each)
echo ================================================================
echo.

REM 1. Homogeneous (IID)
echo [1/10] Running test_homo (Homogeneous/IID)...
conda run -n flwr flwr run . --run-config configs/test_homo.toml
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: test_homo failed!
    pause
    exit /b 1
)
echo.

REM 2. Label Skew C1
echo [2/10] Running test_C1 (1 class per client)...
conda run -n flwr flwr run . --run-config configs/test_C1.toml
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: test_C1 failed!
    pause
    exit /b 1
)
echo.

REM 3. Label Skew C2
echo [3/10] Running test_C2 (2 classes per client)...
conda run -n flwr flwr run . --run-config configs/test_C2.toml
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: test_C2 failed!
    pause
    exit /b 1
)
echo.

REM 4. Label Skew C3
echo [4/10] Running test_C3 (3 classes per client)...
conda run -n flwr flwr run . --run-config configs/test_C3.toml
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: test_C3 failed!
    pause
    exit /b 1
)
echo.

REM 5. Label Skew C4
echo [5/10] Running test_C4 (4 classes per client)...
conda run -n flwr flwr run . --run-config configs/test_C4.toml
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: test_C4 failed!
    pause
    exit /b 1
)
echo.

REM 6. Label Skew C5
echo [6/10] Running test_C5 (5 classes per client)...
conda run -n flwr flwr run . --run-config configs/test_C5.toml
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: test_C5 failed!
    pause
    exit /b 1
)
echo.

REM 7. Dirichlet(0.1)
echo [7/10] Running test_Dir0p1 (Dirichlet alpha=0.1 - Very non-IID)...
conda run -n flwr flwr run . --run-config configs/test_Dir0p1.toml
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: test_Dir0p1 failed!
    pause
    exit /b 1
)
echo.

REM 8. Dirichlet(0.5)
echo [8/10] Running test_Dir0p5 (Dirichlet alpha=0.5 - Moderate non-IID)...
conda run -n flwr flwr run . --run-config configs/test_Dir0p5.toml
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: test_Dir0p5 failed!
    pause
    exit /b 1
)
echo.

REM 9. Dirichlet(1.0)
echo [9/10] Running test_Dir1p0 (Dirichlet alpha=1.0 - Mild non-IID)...
conda run -n flwr flwr run . --run-config configs/test_Dir1p0.toml
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: test_Dir1p0 failed!
    pause
    exit /b 1
)
echo.

REM 10. Dirichlet(10.0)
echo [10/10] Running test_Dir10p0 (Dirichlet alpha=10.0 - Nearly IID)...
conda run -n flwr flwr run . --run-config configs/test_Dir10p0.toml
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: test_Dir10p0 failed!
    pause
    exit /b 1
)
echo.

echo ================================================================
echo All tests completed successfully!
echo ================================================================
echo.
echo Results saved in: results/
echo.
echo CSV files:
dir /b results\test_*.csv
echo.
pause
