from uedhhlib.tools import remove_hpx_from_dataset, copy_txt_files

input_dir=r"Z:\Users\Emma\Zyla1\2026\2026_11\NT59\261119\TcFF4TCNQ\Meas2"
output_dir=r"Z:\Users\Emma\Zyla1\2026\2026_11\NT59\261119\TcFF4TCNQ\Meas2_temp"

cycles = 7 #[c+1 for c in range(12)]

if __name__ == '__main__':
    print("Starting hot pixel removal...")
    stats = remove_hpx_from_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        cycles=cycles
        )
    
    print("\nCompleted!")

#this copys all txt files into the new folder with hotpixels removed to be able to create a PumpedDataset from there
# copy_txt_files(input_dir, output_dir)


