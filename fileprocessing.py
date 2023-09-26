import splitfolders

'''
    INPUT : Specify path to root directory containing subdirectory (named as labels)
    OUTPUT : Path to output directory
'''
input_folder = "/Users/kayle/Projects/Python/audio/data_dir/cwt_scalograms_data"
output_folder = "/Users/kayle/Projects/Python/audio/data_dir/cwt_scalograms_dataprocessed"

splitfolders.ratio(input_folder, output_folder, seed = 42, ratio=(.7,.1,.2))

