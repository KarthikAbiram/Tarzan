# Problem Statement
Tarzan helps to analyze waveform data (like from oscilloscope) and compare against a reference file. It then calculates the amplitude & timing for the different segments for the waveform and generates an output CSV report.

## Inputs
Expected input waveform CSV file:
| CH0 | Expected Time (s) |
| --- | ----------------- |
| 0   | X                 |
| 1   | 1                 |
| 0   | X                 |

Input waveforms, either as a single consolidated CSV file or as individual Tek wfm file formats.
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
| 2.8      | 0.1  |
| 3        | 0.1  |

## Outputs
Sample output report format and metrics:
| CH0 | Start Time (s) | Stop Time (s) | Start Index | Stop Index |
| --- | -------------- | ------------- | ----------- | ---------- |
| 0   | 0.0            | 1.0           | 0.0         | 10.0       |
| 1   | 1.0            | 2.0           | 10.0        | 20.0       |
| 0   | 2.0            | 3.0           | 20.0        | 30.0       |

# Design
1. Breakdown the expected waveform into segments.
2. Find the transition point for each of the segments to know the starting and ending of each segment
3. Find the next segment
4. For each segment, then calculate the required metrics related to amplitude and timing for each channel.
5. Output the metrics into a CSV file

## Step 1: Breakdown Waveform to Segments
Considering a single channel square pulse of 0,1,0, the resultant waveform has below segments
1. State 0
2. Transition from State 0 to State 1
3. State 1
4. Transition from State 1 to State 0
5. State 0

## Step 2: Finding the Transition Segment
From one state to next state:
1. Find the transition ending point
2. Backtrack and find the transition starting point

Say, current state is 0. Next expected state is 1. Now, we need to find the transition segment between 0 and 1, by finding the transition starting and ending points.

### Find the Transition Ending Point
1. Initial zoomed out spotting based on maximum tolerance specified by user. Say, if tolerance specified by the user is 0.1, from the state 0, find the point where the value is next expected value +/- tolerance (1st point whose value lies between 0.9 to 1.1)
2. Next, zoomed in finding of more accurate transition ending point. For better accuracy, we zoom in and keep going as long as the error difference keeps reducing. Say, the values are 0.9, 0.95, 1.02, 1.01, 1.04..., we can find that the error difference keeps reducing when going through 0.9, 0.95, 1.02, 1.01 and stops reducing (or maintains the same error difference) after that. From this, we determine the transition point is 1.01.

### Find the Transition Starting Point
From the transition ending point, backtrack and find the transition starting point. This is found by finding the 1st point backtracking from the transition ending point, which meets the criteria previous state +/- tolerance.

This helps in finding the transition segment. 

## Step 3: Finding the Next Segment
Next, it tries to find the next transition segment's ending and starting point. Once this is found, the state segments between the two transition segments can be determined - the starting point would be the previous transition's ending point and the ending point would be the current transition's starting point.

## Step 4: Calculate Metrics
For each of the segments, then calculate the required metrics related to amplitude and timing for each channel.

## Step 5: Output the Segment metrics to a CSV report file.