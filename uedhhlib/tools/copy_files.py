import shutil
from pathlib import Path
from tqdm import tqdm
from typing import Union, List

def copy_all_txt_files(source_file_dir: str, target_file_dir:str):
    """
    copies all txt files from a MeasX folder to another MeasY folder 
    (e.g. for hotpixel removed folders, which don't have the txt files, which are needed for creating PumpedDataset)
    """

    cycle_folders = [f for f in Path(source_file_dir).iterdir() if f.is_dir() and f.name.startswith("Cycle ")]

    for cyc_folder in tqdm(cycle_folders, desc="Going through cycle folders"):
        target_cyc_folder = Path(target_file_dir)/cyc_folder.name
        target_cyc_folder.mkdir(exist_ok=True)
        for file in cyc_folder.glob("*.txt"):
            shutil.copy2(str(file), str(target_cyc_folder/file.name))

def copy_txt_files(source_file_dir: str, target_file_dir:str, cycles: Union[int, List[int], None] = None):
    """
    copies all txt files from a MeasX folder to another MeasY folder 
    (e.g. for hotpixel removed folders, which don't have the txt files, which are needed for creating PumpedDataset)
    """

    
    if isinstance(cycles, (int, list)):
        if isinstance(cycles, int):
            cycles = [cycles]

        for cyc in cycles:
         
            cyc_folder = Path(source_file_dir) / f"Cycle {cyc}"

            target_cyc_folder = Path(target_file_dir)/cyc_folder.name
            target_cyc_folder.mkdir(exist_ok=True)
            for file in cyc_folder.glob("*.txt"):
                shutil.copy2(str(file), str(target_cyc_folder/file.name))

    elif cycles is None:
        copy_all_txt_files(source_file_dir=source_file_dir, target_file_dir=target_file_dir)