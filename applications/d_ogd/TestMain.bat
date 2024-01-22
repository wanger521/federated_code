@echo off
REM TestMain.bat

REM Define the common arguments
set "common_args=--model softmax_regression --lr_controller ConstantThenDecreasingLr --centralized True"

REM Define the list of attack types
set "attack_types=NoAttack SignFlipping Gaussian SampleDuplicating"

REM Define the list of aggregation rules
REM set "aggregation_rules=Mean Median GeometricMedian Krum TrimmedMean Faba Phocas CenteredClipping"
set "aggregation_rules=CenteredClipping TrimmedMean Phocas"

REM Run Python Script with Arguments for each attack type
for %%a in (%attack_types%) do (
    for %%b in (%aggregation_rules%) do (
        python main.py --attack_type %%a --aggregation_rule %%b %common_args%
    )
)