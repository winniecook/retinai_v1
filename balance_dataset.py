#balance_dataset.py

import os
from pathlib import Path
import shutil
import random

def balance_classes(src_dir='../processed_data', dest_dir='../balanced_processed_data', target_size=80):
    src_dir = Path(src_dir).resolve()
    dest_dir = Path(dest_dir).resolve()
    
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    
    for cls in ['normal', 'cataract', 'glaucoma']:
        src_class_dir = src_dir / cls
        dest_class_dir = dest_dir / cls
        os.makedirs(dest_class_dir)
        
        images = list(src_class_dir.glob('*.png'))
        selected = random.sample(images, target_size)
        for img in selected:
            shutil.copy2(img, dest_class_dir / img.name)
    
    print(f"Created balanced dataset: {target_size} images per class")

if __name__ == '__main__':
    balance_classes()