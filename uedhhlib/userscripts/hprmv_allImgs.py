from uedhhlib.tools import remove_hpx_from_dataset, copy_all_txt_files

input_dir=r"Z:\Users\Emma\Zyla1\2026_03\Netztraeger46\260325\perylene\Meas1"
output_dir=r"Z:\Users\Emma\Zyla1\2026_03\Netztraeger46\260325\perylene\Meas1_wohtpx"

if __name__ == '__main__':
    print("Starting hot pixel removal...")
    stats = remove_hpx_from_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        cycles=[c+1 for c in range(6)]
        )
    
    print("\nCompleted!")

#this copied all txt files into the new folder with hotpixels removed to be able to create a PumpedDataset from there
copy_all_txt_files(input_dir, output_dir)


