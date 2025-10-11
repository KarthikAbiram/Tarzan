import pandas as pd
import argparse
import csv

class OutputResults():
    def __init__(self):
        self.start_time = None
        self.start_index = None
        self.stop_time = None
        self.stop_index = None
        self.is_transition = False

    def to_dict(self):
        return {
        "Start Index": self.start_index,
        "Stop Index": self.stop_index,
        "Start Time": self.start_time,
        "Stop Time": self.stop_time,
        "Is Transition?": self.is_transition
    }


def analyze_segment(segment_df, channel_names, output_results_dict: dict):
    for channel in channel_names:
        # Calculate mean, min, max for each channel
        output_results_dict[f"{channel}.Min"]=segment_df[channel].min()
        output_results_dict[f"{channel}.Max"]=segment_df[channel].max()
        output_results_dict[f"{channel}.Mean"]=segment_df[channel].mean()

    return output_results_dict

def process_file(input_file, ref_file, output_file, tolerance, chunksize):
    ref_df = pd.read_csv(ref_file)
    # print(ref_df)

    chunk_iter = pd.read_csv(input_file, chunksize=chunksize)
    input_df = next(chunk_iter)
    # print(first_chunk)

    # Find channel names from ref data
    # All columns except 'Expected Time (s)' are channel names
    channel_names = [column for column in ref_df.columns.tolist() if column not in ['Expected Time (s)']]
    # print(channel_names)

    # Find input file time column name
    input_time_column_name = 'Time (s)'

    # Initialize variables
    offset = 0
    output = []
    output_results = OutputResults()
    current_target_series = pd.Series(dtype=float) # Just for initialization and not used

    # Get 1st expected ref data
    for index, row in ref_df.iterrows():
        # Create previous and current target series
        previous_target_series = current_target_series.copy()
        current_target_series = row[channel_names]
        # print(target_series)

        # Input subset
        input_subset = input_df.iloc[offset:][channel_names]
        # print(input_subset)

        # Find the match for the 1st expected ref data across all channel columns within tolerance limit
        # Generate boolean mask for the search criteria
        mask = (input_subset - current_target_series).abs().le(tolerance).all(axis=1)
        # print(mask)

        # When match is found, set current Time as 'Start Time'
        if mask.any():
            match_index = mask[mask].index[0]
            match_time = input_df.iloc[match_index][input_time_column_name]
            # print(offset, match_index)
            if index == 0:
                # First match, update start time & offset
                output_results.start_time = match_time
                output_results.start_index = match_index
                offset = match_index
                # print(output_start_time)
            else:
                # Second or further matches
                # Identify transition segment and add row for previous segment
                # The last match of previous series is the stop index of previous segment
                segment_with_transition = input_df.iloc[output_results.start_index:match_index][channel_names]
                transition_mask = (segment_with_transition - previous_target_series).abs().le(tolerance).all(axis=1)
                prev_segment_stop_index = transition_mask[transition_mask].index[-1]
                prev_segment_stop_time = input_df.iloc[prev_segment_stop_index][input_time_column_name]
                output_results.is_transition = False
                output_results.stop_time = prev_segment_stop_time
                output_results.stop_index = prev_segment_stop_index
                output_results_dict = analyze_segment(input_df.iloc[output_results.start_index:output_results.stop_index], channel_names, output_results.to_dict())
                results = pd.Series(output_results_dict)
                output_row = pd.concat([previous_target_series.copy(), results])
                output.append(output_row)

                if match_index - prev_segment_stop_index > 1:
                    # Transition segment exists
                    # Identify end of transition segment and add row for transition segment
                    transition_segment_start_index = prev_segment_stop_index + 1
                    transition_segment_start_time = input_df.iloc[transition_segment_start_index][input_time_column_name]
                    transition_segment_stop_index = match_index - 1
                    transition_segment_stop_time = input_df.iloc[transition_segment_stop_index][input_time_column_name]
                    # Update 'Stop Time' & append this entry to output
                    output_results.start_index = transition_segment_start_index
                    output_results.start_time = transition_segment_start_time
                    output_results.stop_index = transition_segment_stop_index
                    output_results.stop_time = transition_segment_stop_time
                    output_results.is_transition = True

                    output_results_dict = analyze_segment(input_df.iloc[output_results.start_index:output_results.stop_index], channel_names, output_results.to_dict())
                    results = pd.Series(output_results_dict)
                    output_row = pd.concat([previous_target_series.copy(), results])
                    output.append(output_row)
                else:
                    # No transition segment exists
                    pass

                # Then update 'Start Time' & 'Offset' for next segment from the current match
                output_results.start_time = match_time
                output_results.start_index = match_index
                output_results.is_transition = False
                offset = match_index
        else:
            print("No matching row found.")
            return
        
    # Use last row as the last match
    match_index = mask[mask].index[-1]
    match_time = input_df.iloc[match_index][input_time_column_name]
    output_results.stop_time = match_time
    output_results.stop_index = match_index
    # Append to output
    output_results_dict = analyze_segment(input_df.iloc[output_results.start_index:output_results.stop_index], channel_names, output_results.to_dict())
    results = pd.Series(output_results_dict)
    output_row = pd.concat([current_target_series.copy(), results])
    output.append(output_row)
    # print("output",output_start_time, output_stop_time)

    # Write output to csv
    output_df = pd.DataFrame(output)
    output_df.to_csv(output_file, index=False)
    print(f'Output generated at: {output_file}')

def main():
    parser = argparse.ArgumentParser(description="Process multi-channel signal CSV with actual column names.")
    parser.add_argument('--input', default=r'sample\input.csv', help="Input CSV with header. First column = Time, others = channels.")
    parser.add_argument('--ref', default=r'sample\reference.csv', help="Expected pattern CSV with columns: CH, Expected Time (s)")
    parser.add_argument('--output', default=r'sample\output.csv', help="Output CSV file")
    parser.add_argument('--tolerance', type=float, default=0.1, help="Tolerance (default: 0.1)")
    parser.add_argument('--chunksize', type=int, default=50000, help="Chunk size (default: 50000)")
    args = parser.parse_args()

    process_file(args.input, args.ref, args.output, args.tolerance, args.chunksize)
    

if __name__ == '__main__':
    main()
