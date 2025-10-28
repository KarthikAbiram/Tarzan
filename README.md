# Tarzan
Tarzan helps to analyze waveform data (like from oscilloscope) and compare against a reference file. It then calculates the timing for the different segments for the waveform and generates an output CSV report.

# Usage
```
uv run tarzan.py analyze --input=sample\input.csv --ref=sample\reference.csv
```
Optionally, can also specify the output file location and tolerance as below: 
```
uv run tarzan.py --input=sample\input.csv --ref=sample\reference.csv --output=sample\output.csv --tolerance 0.1
```

# Sample File Formats
## Input File
Sample input CSV file contents:
| Time (s) | CH0  |
| -------- | ---- |
| 0        | 0.1  |
| 0.2      | 0.1  |
| 0.4      | 0.1  |
| 0.6      | -0.1 |
| 0.8      | 0.1  |
| 1        | 1.01 |
| 1.2      | 1    |
| 1.4      | 1    |
| 1.6      | 1    |
| 1.8      | 0.99 |
| 2        | 0    |
| 2.2      | 0.1  |
| 2.4      | 0.1  |
| 2.6      | 0.1  |
| 2.8      | 0.1  |
| 3        | 0.1  |

## Reference File
Sample reference CSV file contents:
| CH0 | Expected Time (s) |
| --- | ----------------- |
| 0   | X                 |
| 1   | 1                 |
| 0   | X                 |

## Output File
Sample output CSV file contents:
| CH0 | Start Time (s) | Stop Time (s) | Start Index | Stop Index |
| --- | -------------- | ------------- | ----------- | ---------- |
| 0   | 0.0            | 1.0           | 0.0         | 10.0       |
| 1   | 1.0            | 2.0           | 10.0        | 20.0       |
| 0   | 2.0            | 3.0           | 20.0        | 30.0       |