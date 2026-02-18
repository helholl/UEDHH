"""
Preprocessing tools for raw UED data.
Run hot pixel removal over every image before reading in the data and analysis
"""
import numpy as np
from typing import Union, List
from pathlib import Path
from os import listdir
from tqdm import tqdm

from ..processing.hotpixels import hotpixel_filter

def remove_hpx_from_dataset(
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        cycles: Union[int, List[int], None] = None,
        tolerance: float = 3,
        size: int = 10,
        file_patterns: List[str] = None,
        progress: bool = True
    ):
    r"""
    Remove hot pixels from all .npy images in a dataset and save to new directory.
    
    Parameters
    ----------
    input_dir : Path
        Base directory containing "Cycle X" folders
    output_dir : Path
        Output directory (will mirror input structure)
    cycles : int or list of int, optional
        Specific cycles to process. If None, processes all cycles.
    tolerance : float
        Hot pixel detection tolerance (default: 3)
    size : int
        Median filter size (default: 10)
    file_patterns : list of str, optional
        Only process files containing these patterns (e.g., ["ProbeOnPumpOn_", "ProbeOnPumpOff_"]).
        If None, processes all .npy files.
    progress : bool
        Show progress bar (default: True)
    
    Returns
    -------
    dict
        Statistics about processed files
    
    Examples
    --------
    >>> from uedhhlib.tools.preprocessing import remove_hot_pixels_from_dataset
    >>> 
    >>> remove_hpx_from_dataset(
    ...     input_dir=r"Z:\\Users\Emma\Zyla1\2026_01\Netztraeger42\260112\perylene\Meas1",
    ...     output_dir=rZ:\\Users\Emma\Zyla1\2026_01\Netztraeger42\260112\perylene\Meas1_hprmv,
    ...     cycles=[1, 2, 3, 4, 5],
    ...     tolerance=3
    ... )
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # detect cycles if not specified
    if cycles is None:
        cycles = _detect_cycles(input_dir)
    elif isinstance(cycles, int):
        cycles = [cycles]

    # prepare statistics
    stats = {
            'total_files_per_cyc': [],
            'processed_files_per_cyc': [np.zeros(len(cycles))],
            'skipped_files_per_cyc': [np.zeros(len(cycles))],
            'hot_pixels_found_per_cyc': []
    }

    # process each cycle
    for cycle in cycles:
        cycle_input = input_dir / f"Cycle {cycle}"
        cycle_output = output_dir / f"Cycle {cycle}"

        if not cycle_input.exists():
            print(f"Warning: {cycle_input} does not exist, skipping")
            continue
        
        #create output directory
        cycle_output.mkdir(parents=True, exist_ok=True)

        #get all files
        files = [f for f in listdir(cycle_input) if f.endswith('.npy')]

        #filter by pattern if specified
        if file_patterns:
            files = [f for f in files
                    if any(pattern in f for pattern in file_patterns)]
        
        stats['total_files_per_cyc'].append(len(files))

        #process files
        if progress:
            iterator = tqdm(files, desc=f"Cycle {cycle}")
        else:
            iterator = files

        for filename in iterator:
            input_path = cycle_input / filename
            output_path = cycle_output / filename
            
            try:
                #load image
                img = np.load(input_path)

                #remove hot pixels
                outliers, cleaned_img = hotpixel_filter(
                    img,
                    tolerance=tolerance,
                    size=size
                )

                #save cleaned image
                np.save(output_path, cleaned_img)

                #statistics
                stats['processed_files_per_cyc'][cycle] += 1
                stats['hot_pixels_found'].append(len(outliers[0]))

            except Exception as e:
                print(f"Error processing {filename}: {e}")
                stats['skipped_files_per_cyc'][cycle] += 1

    #print summary
    if progress:
        print("\n" + "="*50)
        print("Hot Pixel Removal Summary")
        print("="*50)
        print(f"Total files found: {np.sum(stats['total_files_per_cyc'])}")
        print(f"Successfully processed: {np.sum(stats['processed_files_per_cyc'])}")
        print(f"Skipped (errors): {np.sum(stats['skipped_files_per_cyc'])}")
        if np.sum(stats['hot_pixels_found'])>0:
            avg_hot_pixels = np.mean(stats['hot_pixels_found'])
            max_hot_pixels = np.max(stats['hot_pixels_found'])
            print(f"Average hot pixels per image: {avg_hot_pixels:.1f}")
            print(f"Maximum hot pixels in single image: {max_hot_pixels}")

        print(f"\nCleaned data saved to: {output_dir}")
        print("="*50)

    return stats

            


        



def _detect_cycles(basedir: Path) -> List[int]:
    """Auto-detect available cycle folders"""
    cycle_dirs = [d for d in listdir(basedir) if d.startswith("Cycle ")]
    return sorted([int(d.split()[-1]) for d in cycle_dirs])
            
