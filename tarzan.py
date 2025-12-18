import pandas as pd
import argparse
from tm_data_types import read_file
from pathlib import Path
import fire

class OutputResults():
    def __init__(self):
        self.start_time = None
        self.start_index = None
        self.stop_time = None
        self.stop_index = None
        self.is_transition = False

    def to_dict(self):
        return {
        "Time Taken":self.stop_time-self.start_time,
        "Is Transition?": self.is_transition,
        "Start Index": self.start_index,
        "Stop Index": self.stop_index,
        "Start Time": self.start_time,
        "Stop Time": self.stop_time
    }

class Tarzan:
    def _analyze_segment(self, segment_df, channel_names, output_results_dict: dict):
        for channel in channel_names:
            # Calculate mean, min, max for each channel
            if not segment_df.empty:
                output_results_dict[f"{channel}.Start"] = segment_df[channel].iloc[0]
                output_results_dict[f"{channel}.End"] = segment_df[channel].iloc[-1]
            else:
                output_results_dict[f"{channel}.Start"] = None
                output_results_dict[f"{channel}.End"] = None
            output_results_dict[f"{channel}.Min"]=segment_df[channel].min()
            output_results_dict[f"{channel}.Max"]=segment_df[channel].max()
            output_results_dict[f"{channel}.Mean"]=segment_df[channel].mean()
            output_results_dict[f"{channel}.Pk to Pk"]=output_results_dict[f"{channel}.Max"]-output_results_dict[f"{channel}.Min"]

        return output_results_dict

    def analyze(
        self,
        input_file=r"sample/analyze/input/all_channels.csv",
        ref_file=r"sample/analyze/input/reference.csv",
        output_file=r"sample/analyze/output/all_channel_analysis.csv",
        tolerance=0.01
    ):
        """
        Analyze waveform data against a reference.

        Args:
            input_file (str, optional): 
                Path to the CSV file containing waveform data from multiple channels.
                Default: 'sample\\analyze\\input\\all_channels.csv'

            ref_file (str, optional): 
                Path to the reference CSV file for comparison.
                Default: 'sample\\analyze\\input\\reference.csv'

            output_file (str, optional): 
                Path where the analysis result will be saved.
                Default: 'sample\\analyze\\output\\all_channel_analysis.csv'

            tolerance (float, optional): 
                Acceptable tolerance level between the input and reference values.
                Default: 0.1

        Returns:
            None. Outputs analysis results are logged to the specified CSV file.
        """
        # Create output folder if not present
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        ref_df = pd.read_csv(ref_file)
        # print(ref_df)

        input_df = pd.read_csv(input_file)

        # Find channel names from ref data
        # All columns except 'Expected Time (s)' are channel names
        channel_names = [column for column in ref_df.columns.tolist() if column not in ['Label','Expected Time (s)']]
        # print(channel_names)

        # Find input file time column name
        input_time_column_name = 'Time (s)'

        # Initialize variables
        offset = 0
        output = []
        output_results = OutputResults()
        current_ref_target_series = pd.Series(dtype=float) # Just for initialization and not used
        current_ref_row = pd.Series(dtype=float) # Just for initialization and not used
        break_on_no_match_flag = False

        # Get 1st expected ref data
        for index, row in ref_df.iterrows():
            # Create previous and current target series
            previous_target_series = current_ref_target_series.copy()
            previous_row = current_ref_row.copy()
            current_ref_target_series = row[channel_names]
            current_ref_row = row.copy()

            # Input subset
            input_subset = input_df.loc[offset:][channel_names]
            # print(input_subset)

            # Find the match for the 1st expected ref data across all channel columns within tolerance limit

            # Generate boolean mask for the search criteria
            mask = (input_subset - current_ref_target_series).abs().le(tolerance).all(axis=1)
            # When match is found, set current Time as 'Start Time'
            # if mask.any():
                # match_index = mask[mask].index[0]

            match_index = self._get_match_index(input_subset, current_ref_target_series, tolerance)
            # print(match_index)
            if match_index is not None:
                match_time = input_df.loc[match_index][input_time_column_name]
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
                    segment_with_transition = input_df.loc[output_results.start_index:match_index][channel_names]
                    transition_mask = (segment_with_transition - previous_target_series).abs().le(tolerance).all(axis=1)
                    prev_segment_stop_index = transition_mask[transition_mask].index[-1]
                    prev_segment_stop_time = input_df.loc[prev_segment_stop_index][input_time_column_name]
                    output_results.is_transition = False
                    output_results.stop_time = prev_segment_stop_time
                    output_results.stop_index = prev_segment_stop_index
                    output_results_dict = self._analyze_segment(input_df.loc[output_results.start_index:output_results.stop_index], channel_names, output_results.to_dict())
                    results = pd.Series(output_results_dict)
                    output_row = pd.concat([previous_row.copy(), results])
                    output.append(output_row)

                    if match_index - prev_segment_stop_index > 1:
                        # Transition segment exists
                        # Identify end of transition segment and add row for transition segment
                        transition_segment_start_index = prev_segment_stop_index + 1
                        transition_segment_start_time = input_df.loc[transition_segment_start_index][input_time_column_name]
                        transition_segment_stop_index = match_index - 1
                        transition_segment_stop_time = input_df.loc[transition_segment_stop_index][input_time_column_name]
                        # Update 'Stop Time' & append this entry to output
                        output_results.start_index = transition_segment_start_index
                        output_results.start_time = transition_segment_start_time
                        output_results.stop_index = transition_segment_stop_index
                        output_results.stop_time = transition_segment_stop_time
                        output_results.is_transition = True

                        output_results_dict = self._analyze_segment(input_df.loc[output_results.start_index:output_results.stop_index], channel_names, output_results.to_dict())
                        results = pd.Series(output_results_dict)
                        output_row = pd.concat([current_ref_row.copy(), results])
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
                print(f"Match not found at index {index}: {dict(current_ref_row)}")
                break_on_no_match_flag = True
                break
        
        if not break_on_no_match_flag:
            # Use last row as the last match
            match_index = mask[mask].index[-1]
            match_time = input_df.loc[match_index][input_time_column_name]
            output_results.stop_time = match_time
            output_results.stop_index = match_index
            # Append to output
            output_results_dict = self._analyze_segment(input_df.loc[output_results.start_index:output_results.stop_index], channel_names, output_results.to_dict())
            results = pd.Series(output_results_dict)
            output_row = pd.concat([current_ref_row.copy(), results])
            output.append(output_row)
            # print("output",output_start_time, output_stop_time)

        # Write output to csv
        output_df = pd.DataFrame(output)
        output_df.to_csv(output_file, index=False, float_format="%.4g") # Using 4 significant digits for float
        print(f'Output generated at: {output_file}')

    def convert(
        self,
        wfm_folder_path=r"sample/convert/input",
        output_csv_file_path=r"sample/convert/output/output_wfm_to_csv.csv"
    ):
        """
        Convert all .wfm files in a folder to a single consolidated CSV file.
        
        Args:
            wfm_folder_path (str, optional): 
                Path to the folder containing .wfm files to convert.
                Default: 'sample\\convert\\input'

            output_csv_file_path (str, optional): 
                Path to the output CSV file where the merged data will be saved.
                Default: 'sample\\convert\\output\\output_wfm_to_csv.csv'

        Returns:
            None. Writes the merged waveform data to the specified CSV file.
        """
        # Create output folder if not present
        Path(output_csv_file_path).parent.mkdir(parents=True, exist_ok=True)

        time_column_name = "Time (s)"
        # Get list of *.wfm files
        wfm_files = list(Path(wfm_folder_path).glob("*.wfm"))
        # Get the last string after "_" or if not present, the entire file name as channel name
        # channel_names = [Path(file).stem.split("_")[-1].upper() for file in wfm_files]
        all_channel_df = None

        for file in wfm_files:
            # Get the last string after "_" or if not present, the entire file name as channel name
            channel_name = Path(file).stem.split("_")[-1].upper()
            # Read waveform data and convert to pandas data frame
            waveform = read_file(file)
            channel_df = pd.DataFrame({
                time_column_name: waveform.normalized_horizontal_values,
                channel_name: waveform.normalized_vertical_values
            })
            # Merge channel data with output
            if all_channel_df is None:
                all_channel_df = channel_df
            else:
                all_channel_df = pd.merge(all_channel_df, channel_df, on=time_column_name, how="outer")
        # Log to csv
        all_channel_df.to_csv(output_csv_file_path, index=False)
        print(f'Output generated at: {output_csv_file_path}')

    def analyze_wfm(self, 
                    wfm_folder_path=r"sample/analyze_wfm/input/wfm",
                    ref_file=r"sample/analyze_wfm/input/reference.csv",
                    output_file=r"sample/analyze_wfm/output/all_channel_analysis.csv",
                    tolerance=0.01):
        """
        Takes in a folder containing channel data as individual *.wfm files, converts it to csv and analyzes it
        """
        wfm_csv_file_path = Path(output_file).parent / "all_channel_wfm.csv" 
        self.convert(wfm_folder_path, wfm_csv_file_path)
        self.analyze(wfm_csv_file_path, ref_file, output_file, tolerance)

    def _get_match_index(self, dataframe, ref_target_series, tolerance):
        # Find list of values within tolerance
        mask = (dataframe - ref_target_series).abs().le(tolerance).all(axis=1)
        # print(mask)
        if mask.any():
            # Find start index and end index of tolerance match
            idx = mask[mask].index
            start = idx[0]
            end = idx[-1]
            for index, id in enumerate(idx):
                if (start + index) != id:
                    end = id
                    break

            diff = (dataframe.loc[start:end] - ref_target_series).abs().sum(axis=1)
            # print(diff)
            if diff.empty:
                return None
            index, min_diff = diff.index[0], diff.iloc[0]

            for i, d in diff.items():
                if i == index: # First Index/Match
                    min_diff = d
                elif min_diff <= d:
                    break
                else:
                    min_diff = d
                    index = i

            # print(index, min_diff)
            return index
        else:
            return None  # If no match, return None
    
if __name__ == '__main__':
    debug = False
    # debug = True
    if not debug:
        fire.Fire(Tarzan)
    else:
        df = pd.read_csv(r'sample\analyze\input\all_channels.csv')[['CH0']]
        ref_target_series = pd.Series({'CH0':0})
        tolerance = 0.2
        result = Tarzan()._get_match_index(df, ref_target_series, tolerance)
        print(result)