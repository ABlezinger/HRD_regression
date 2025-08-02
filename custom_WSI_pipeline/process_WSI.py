import sys
import os
import argparse
from pathlib import Path
import glob
import logging
import time
from typing import Dict, Tuple
from concurrent import futures

import PIL
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm

import openslide
import cv2


import stainNorm_Macenko
from concurrent_canny_rejection import reject_background
from feature_extraction import extract_and_save_features

## Sortiert bis hier 


def main(args):
    
    # Settings
    print(f"Number of CPUs in the system: {os.cpu_count()}")
    has_gpu=torch.cuda.is_available()
    print(f"GPU is available: {has_gpu}")
    if has_gpu:
        print(f"Number of GPUs in the system: {torch.cuda.device_count()}")
        
    Image.MAX_IMAGE_PIXELS = None  # Disable PIL's image size limit to avoid errors with large images
    
    #initialize Normalizer 
    norm=True
    if norm:
        print("\nInitialising Macenko normaliser...")
        target = cv2.imread('custom_WSI_pipeline/normalization_template.jpg') #TODO: make scaleable with path
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        normalizer = stainNorm_Macenko.Normalizer()
        normalizer.fit(target)
        logging.info('Running WSI to normalised feature extraction...')
    else:
        logging.info('Running WSI to raw feature extraction...')
    
    
    # create list of svs files 
    total_start_time = time.time()
    svs_dir = glob.glob(f"{args.wsi_dir}/*.svs", recursive=False)
    if len(svs_dir) == 0:
        print(f"No .svs files found in {args.wsi_dir}. Please check the path.")
        return
    else:
        print(f"Found {len(svs_dir)} .svs files in {args.wsi_dir}.")
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    for slide_url in (progress := tqdm(svs_dir, leave=False)):
        slide_name = Path(slide_url).stem
        progress.set_description(slide_name)
    
        #create jpg and feature directories     
        slide_jpg_dir = args.jpg_dir/slide_name
        slide_jpg_dir.mkdir(parents=True, exist_ok=True)
    
        if not (os.path.exists((f'{args.output_dir}/{slide_name}.h5'))):
            # If the slide is not already processed with the model
            
            #save as png to avoid information loss due to jpg compression
            if (slide_jpg := slide_jpg_dir/'norm_slide.png').exists():
                print("WSI has been normalized and converted into jpg, now extracting features...")
                # If the normalised slide png already exists, load it
                img_norm_wsi_jpg = PIL.Image.open(slide_jpg)
                image_array = np.array(img_norm_wsi_jpg)
                canny_patch_list = []
                coords_list=[]
                patch_saved=0
                
                
                coords_path = slide_jpg_dir / "coords.npy"
                if coords_path.exists():
                    coords_list = np.load(coords_path, allow_pickle=True)
                    canny_patch_list = [image_array[y:y+224, x:x+224, :] for (x, y) in coords_list]
                    patch_saved = len(canny_patch_list)
                else:
                    raise FileNotFoundError(f"Missing coords.npy for {slide_name}")
                
                print(f"Loaded {patch_saved} patches from {slide_name}")
                
                del img_norm_wsi_jpg, image_array  # Clean up memory
                                         
            else:
                #if normalized slide png does not exist, load the slide and normalise it
                print(f"\nLoading {slide_name}")
                try:
                    slide = openslide.OpenSlide(str(slide_url))
                except Exception as e:
                    print(f"Failed loading {slide_name}, error: {e}")
                    continue

                #measure time performance
                start_time = time.time()
                # load slide svs and convert to np.array
                slide_array = load_slide(slide)
                if slide_array is None:
                    print(f"Skipping slide and not deleting {slide_url} due to missing MPP...")
                    # os.remove(str(slide_url))
                    continue
                #save original WSI as jpg
                (Image.fromarray(slide_array)).save(f'{slide_jpg_dir}/slide.jpg')

                #remove .SVS from memory (couple GB)
                del slide
                
                print("\n--- Loaded slide: %s seconds ---" % (time.time() - start_time))
                #########################

                #########################
                #Do edge detection here and reject unnecessary tiles BEFORE normalisation
                # hier ist patch save evtl mit drin 
                bg_reject_array, rejected_tile_array, patch_shapes = reject_background(img = slide_array, patch_size=(224,224), step=224, outdir=args.jpg_dir, save_tiles=False)

                
                #measure time performance
                start_time = time.time()
                #pass raw slide_array for getting the initial concentrations, bg_reject_array for actual normalisation
                if norm:
                    print(f"Normalising {slide_name}...")
                    canny_img, img_norm_wsi_jpg, canny_patch_list, coords_list = normalizer.transform(slide_array, bg_reject_array, rejected_tile_array, patch_shapes)
                    print(f"\n--- Normalised slide {slide_name}: {(time.time() - start_time)} seconds ---")
                    
                    # save normalised image as "norm_slide.jpg"
                    img_norm_wsi_jpg.save(slide_jpg) #save WSI.svs -> WSI.jpg

                else:
                    canny_img, canny_patch_list, coords_list = get_raw_tile_list(slide_array.shape, bg_reject_array, rejected_tile_array, patch_shapes)

                print("Saving Canny background rejected image...")
                canny_img.save(f'{slide_jpg_dir}/canny_slide.jpg')
                np.save(slide_jpg_dir / "coords.npy", coords_list)
                
                #remove original slide jpg from memory
                del slide_array


            #FEATURE EXTRACTION
            print(f"Extracting {args.model} features from {slide_name}")
            #measure time performance
            start_time = time.time()
            
            extract_and_save_features(norm_wsi_img=np.asarray(canny_patch_list), wsi_name=slide_name, coords=coords_list, extraction_model=args.model, outdir=args.output_dir)
            print("\n--- Extracted features from slide: %s seconds ---" % (time.time() - start_time))
            #########################
            # print(f"Deleting slide {slide_name} from local folder...")
            # os.remove(str(slide_url))

        else:
            print(f"{slide_name}_loaded.h5 already exists. Skipping...")
            # print(f"Deleting slide {slide_name} from local folder...")
            # os.remove(str(slide_url))


def _load_tile(
    slide: openslide.OpenSlide, pos: Tuple[int, int], stride: Tuple[int, int], target_size: Tuple[int, int]
) -> np.ndarray:
    # Loads part of a WSI. Used for parallelization with ThreadPoolExecutor
    tile = slide.read_region(pos, 0, stride).convert('RGB').resize(target_size)
    return np.array(tile)


def load_slide(slide: openslide.OpenSlide, target_mpp: float = 256/224) -> np.ndarray:
    """Loads a slide into a numpy array."""
    # We load the slides in tiles to
    #  1. parallelize the loading process
    #  2. not use too much data when then scaling down the tiles from their
    #     initial size
    steps = 8
    stride = np.ceil(np.array(slide.dimensions)/steps).astype(int)
    try:
        slide_mpp = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
        print(f"Read slide MPP of {slide_mpp} from meta-data")
    except:
        print(f"Error: couldn't load MPP from slide!")
        return None
    tile_target_size = np.round(stride*slide_mpp/target_mpp).astype(int)
    #changed max amount of threads used
    with futures.ThreadPoolExecutor(122) as executor:
        # map from future to its (row, col) index
        future_coords: Dict[futures.Future, Tuple[int, int]] = {}
        for i in range(steps):  # row
            for j in range(steps):  # column
                future = executor.submit(
                    _load_tile, slide, (stride*(j, i)), stride, tile_target_size)
                future_coords[future] = (i, j)

        # write the loaded tiles into an image as soon as they are loaded
        im = np.zeros((*(tile_target_size*steps)[::-1], 3), dtype=np.uint8)
        for tile_future in tqdm(futures.as_completed(future_coords), total=steps*steps, desc='Reading WSI tiles', leave=False):
            i, j = future_coords[tile_future]
            tile = tile_future.result()
            x, y = tile_target_size * (j, i)
            im[y:y+tile.shape[0], x:x+tile.shape[1], :] = tile

    return im

def get_raw_tile_list(I_shape: tuple, bg_reject_array: np.array, rejected_tile_array: np.array, patch_shapes: np.array):
    canny_output_array=[]
    for i in range(len(bg_reject_array)):
        if not rejected_tile_array[i]:
            canny_output_array.append(np.array(bg_reject_array[i]))

    canny_img = Image.new("RGB", (I_shape[1], I_shape[0]))
    coords_list=[]
    i_range = range(I_shape[0]//patch_shapes[0][0])
    j_range = range(I_shape[1]//patch_shapes[0][1])

    for i in i_range:
        for j in j_range:
            idx = i*len(j_range) + j
            canny_img.paste(Image.fromarray(np.array(bg_reject_array[idx])), (j*patch_shapes[idx][1], 
            i*patch_shapes[idx][0],j*patch_shapes[idx][1]+patch_shapes[idx][1],i*patch_shapes[idx][0]+patch_shapes[idx][0]))
            
            if not rejected_tile_array[idx]:
                coords_list.append((j*patch_shapes[idx][1], i*patch_shapes[idx][0]))

    return canny_img, canny_output_array, coords_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description='Normalise WSI directly.')

    parser.add_argument('-o', '--output-dir', type=Path, required=True,
                        help='Path to save results to.')
    parser.add_argument('--wsi-dir', metavar='DIR', type=Path, required=True,
                        help='Path of where the whole-slide images are.')
    parser.add_argument('-m', '--model', type=str, required=True, choices=["UNI", "UNI_2", "RetCCL", "GPFM", "CONCH"],
                        help='The model to use for feature extraction. Choose between ["UNI", "UNI_2", "RetCCL", "GPFM", "CONCH"]')
    parser.add_argument('--jpg-dir', type=Path, default=None,
        help='Directory to cache extracted features etc. in.')


    args = parser.parse_args()
    # Call the main function with command line arguments
    print("----------------------------------------------------------")
    print(f"Starting WSI Pipeline with the following parameters:\n")
    
    print(f"WSI Directory:\t \t{args.wsi_dir}")
    print(f"Extraction Model: \t{args.model}")
    print(f"JPG Directory: \t{args.jpg_dir}")
    print(f"Output Path: \t\t{args.output_dir}")
    print(f"Amount of CPUs: \t{os.cpu_count()}")
    print("----------------------------------------------------------")
    
    
    main(args)