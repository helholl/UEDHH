"""
Preprocessing tools for raw UED data.
Run hot pixel removal over every image before reading in the data and analysis
"""

from multiprocessing import Pool, cpu_count
from functools import partial
import numpy as np
from typing import Union, List, Tuple
from pathlib import Path
from os import listdir
from tqdm import tqdm

from ..processing.hotpixels import hotpixel_filter

def _processing_single_file(
        args: Tuple[Path, Path, float, int]
        ) -> dict:
    """
    Process a single file (used by multiprocessing).
    
    Parameters
    ----------
    args : tuple
        (input_path, output_path, tolerance, size)
    
    Returns
    -------
    dict
        Result statistics
    """
    input_path, output_path, tolerance, size = args

    try:
        img = np.load(input_path)

        outliers, cleaned_img = hotpixel_filter(
            img,
            tolerance=tolerance,
            size=size
        )
        
        np.save(output_path, cleaned_img)

        return {
            'success': True,
            'file': input_path.name,
            'hot_pixels': len(outliers[0])
        }
    
    except Exception as e:
        return {
            'success': False,
            'file': input_path.name,
            'error': str(e)
        }



def remove_hpx_from_dataset(
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        cycles: Union[int, List[int], None] = None,
        tolerance: float = 3,
        size: int = 10,
        file_patterns: List[str] = None,
        progress: bool = True,
        n_workers: int = None
    ):
    """
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
    n_workers : int, optional
        Number of parallel workers. If None, uses cpu_count() - 2.
        Recommended: 12-14 for 16-core CPU.
    
    Returns
    -------
    dict
        Statistics about processed files
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # determine number of workers
    if n_workers is None:
        n_workers = max(1, cpu_count()-2)
    
    if progress:
        print(f"Using {n_workers} parallel workers")

    # detect cycles if not specified
    if cycles is None:
        cycles = _detect_cycles(input_dir)
    elif isinstance(cycles, int):
        cycles = [cycles]

    # prepare statistics
    stats = {
            'total_files_per_cyc': [0]*len(cycles),
            'processed_files_per_cyc': [0]*len(cycles),
            'skipped_files_per_cyc': [0]*len(cycles),
            'hot_pixels_found_per_cyc': [[] for _ in cycles]
    }

    # collect all files to process
    all_tasks = []

    # process each cycle
    for cycle_idx, cycle in enumerate(cycles):
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
        
        stats['total_files_per_cyc'][cycle_idx]=len(files)

        #create tasks for this cycle
        for filename in files:
            input_path = cycle_input / filename
            output_path = cycle_output / filename
            all_tasks.append((
                input_path,
                output_path,
                tolerance,
                size,
                cycle_idx
            ))
        
    if not all_tasks:
            print("No files to process")
            return stats

    #process files in parallel
    if progress:
        print(f"\nProcessing {len(all_tasks)} files...")

    with Pool(processes=n_workers) as pool:
        if progress:
            results = list(tqdm(
                pool.imap(
                    _processing_single_file, 
                    [(t[0], t[1], t[2], t[3]) for t in all_tasks]
                    ),
                total=len(all_tasks),
                desc="Processing files"
            ))
        else:
            results = pool.map(
                _processing_single_file,
                [(t[0], t[1], t[2], t[3]) for t in all_tasks]
            )

    for task, result in zip(all_tasks, results):
        cycle_idx = task[4]

        if result['success']:
            stats['processed_files_per_cyc'][cycle_idx] += 1
            stats['hot_pixels_found_per_cyc'][cycle_idx].append(result['hot_pixels'])
        else:
            stats['skipped_files_per_cyc'][cycle_idx] += 1
            if progress:
                print(f"Error in {result['file']}: {result['error']}")


    #print summary
    if progress:
        print("\n" + "="*50)
        print("Hot Pixel Removal Summary")
        print("="*50)
        print(f"Total files found: {np.sum(stats['total_files_per_cyc'])}")
        print(f"Successfully processed: {np.sum(stats['processed_files_per_cyc'])}")
        print(f"Skipped (errors): {np.sum(stats['skipped_files_per_cyc'])}")

        hotpixels_max_per_cyc = []
        hotpixels_ave_per_cyc = []
        for cycle_idx, cycle in enumerate(cycles):
            hotpixels_max_per_cyc.append(np.max(stats['hot_pixels_found_per_cyc'][cycle_idx]))
            hotpixels_ave_per_cyc.append(np.mean(stats['hot_pixels_found_per_cyc'][cycle_idx]))

        avg_hot_pixels = np.mean(hotpixels_ave_per_cyc)
        max_hot_pixels = np.max(hotpixels_max_per_cyc)
        print(f"Average hot pixels per image: {avg_hot_pixels:.1f}")
        print(f"Maximum hot pixels in single image: {max_hot_pixels}")

        print(f"\nCleaned data saved to: {output_dir}")
        print("="*50)

    return stats


def _detect_cycles(basedir: Path) -> List[int]:
    """Auto-detect available cycle folders"""
    cycle_dirs = [d for d in listdir(basedir) if d.startswith("Cycle ")]
    return sorted([int(d.split()[-1]) for d in cycle_dirs])
            
