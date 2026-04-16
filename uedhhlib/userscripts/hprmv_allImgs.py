from uedhhlib.tools import remove_hpx_from_dataset

if __name__ == '__main__':
    print("Starting hot pixel removal...")
    stats = remove_hpx_from_dataset(
        input_dir=r"Z:\Users\Emma\Zyla1\2026_03\Netztraeger46\260325\perylene\Meas2",
        output_dir=r"Z:\Users\Emma\Zyla1\2026_03\Netztraeger46\260325\perylene\Meas2_wohtpx",
        cycles=[c for c in range(26)]
        )
    
    print("\nCompleted!")

