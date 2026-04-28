import shutil
from pathlib import Path
from tqdm import tqdm

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
            shutil.move(str(file), str(target_cyc_folder/file.name))