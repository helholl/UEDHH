from uedhhlib.tools import remove_hpx_from_dataset

if __name__ == '__main__':
    print("Starting hot pixel removal...")
    stats = remove_hpx_from_dataset(
        input_dir="./Meas1",
        output_dir="./Meas1_hprmv",
        cycles=1
        )
    
    print("\nCompleted!")

