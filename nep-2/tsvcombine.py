import pandas as pd

def combine_tsv(file1, file2, output_file):
    # Read the TSV files into DataFrames
    df1 = pd.read_csv(file1, sep='\t')
    df2 = pd.read_csv(file2, sep='\t')
    
    # Combine them (stack one after the other)
    combined_df = pd.concat([df1, df2], ignore_index=True)
    
    # Save to a new TSV file
    combined_df.to_csv(output_file, sep='\t', index=False)

if __name__ == "__main__":
    # Example usage
    file1 = "/home/sahil_duwal/MajorProject/Dataset/male-female-data/MaleVoice.tsv"
    file2 = "/home/sahil_duwal/MajorProject/Dataset/male-female-data/FemaleVoice.tsv"
    output_file = "/home/sahil_duwal/MajorProject/Dataset/male-female-data/combined.tsv"

    combine_tsv(file1, file2, output_file)
    print(f"Combined file saved as {output_file}")
