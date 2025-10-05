import pandas as pd
import argparse
import csv

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
    output_start_time = 0
    output_start_index = 0
    output_stop_time = 0
    output_stop_index = 0
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
                output_start_time = match_time
                output_start_index = match_index
                offset = match_index
                # print(output_start_time)
            else:
                # Second or further matches
                # Update 'Stop Time' & append this entry to output
                output_stop_time = match_time
                output_stop_index = match_index
                results = pd.Series({'Start Time (s)':output_start_time, 'Stop Time (s)':output_stop_time, 'Start Index':output_start_index, 'Stop Index':output_stop_index})
                output_row = pd.concat([previous_target_series.copy(), results])
                output.append(output_row)
                # print(output_row)
                # print("output",output_start_time, output_stop_time)

                # Then update 'Start Time' & 'Offset'
                output_start_time = match_time
                output_start_index = match_index
                offset = match_index
        else:
            print("No matching row found.")
            return
        
    # Use last row as the last match
    match_index = mask[mask].index[-1]
    match_time = input_df.iloc[match_index][input_time_column_name]
    output_stop_time = match_time
    output_stop_index = match_index
    # Append to output
    results = pd.Series({'Start Time (s)':output_start_time, 'Stop Time (s)':output_stop_time, 'Start Index':output_start_index, 'Stop Index':output_stop_index})
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
