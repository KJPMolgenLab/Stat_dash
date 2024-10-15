# ------------------ Imports ------------------
# Core Modules
import numpy as np
import pandas as pd
import os
import datetime as dt

# Plotting
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from PIL import Image

# Image Processing
import cv2
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from skimage.feature import graycomatrix, graycoprops
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter
from skimage.segmentation import watershed

# Clustering
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import base64

# Multithreading
import concurrent.futures
from dask import delayed, compute
from dask.diagnostics import ProgressBar

# Loading Bar
from tqdm import tqdm


# debgu helper

image_file = '20240229_DH_2x_B22_H9_Q2.2_4_dish2_6.bmp'
image_path = '/Users/andreas_chiocchetti/develop/Organoid_Morphology/data/OneDrive_75_12-10-2024'


# ------------------ Helper Functions ------------------
def get_data_folders(verbose=True):
    """Get the data folders from the data directory

    Args:
    verbose (bool, optional): If True, prints the data folders found in the current directory. Defaults to True.#

    Returns:
    data_folders (list): List of data folders found in the data directory
    """
    # Define the paths relative to the current working directory
    path = os.getcwd()
    parent_path = os.path.abspath(os.path.join(path, os.pardir))
    data_path = os.path.join(parent_path, 'data')
    code_path = os.path.join(parent_path, 'code')
    output_folder = os.path.join(parent_path, 'output')

    # Load all the datafiles into a list if it does not start with a dot for iteration later
    data_folders = [f for f in os.listdir(data_path) if not f.startswith('.')]

    if verbose:
        print(data_folders)

    return data_folders, data_path, code_path, output_folder



def process_image(image_file, image_path):
    """
    Process an image file by reading it, converting it to grayscale, scaling it and performing analysis.

    Args:
    image_file (str): The filename of the image file.
    image_path (str): The path to the directory containing the image file.

    Returns:
    The result of the analysis performed on the grayscale image.
    """
    print(image_file)
    image = cv2.imread(os.path.join(image_path, image_file))
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_image = cv2.normalize(gray_image, None, alpha=50, beta=250, norm_type=cv2.NORM_MINMAX)
    return get_analysis(gray_image, inflection_threshold=0.01)


def get_analysis(gray_image, inflection_threshold=0.01):

    """
    Perform analysis on a gray-scale image of an organoid.

    Args:
    gray_image (ndarray): The gray-scale image of the organoid.
    k (int): The number of clusters for image segmentation. Default is 5.
    inflection_threshold (float): The threshold for detecting inflection points in the curvature. Default is 0.01.

    Returns:
    tuple: A tuple containing the following elements:
        stat_array (dict): A dictionary containing various statistical features of the organoid.
        final_segmented_organoid (ndarray): The segmented organoid image.
        hu_moments (ndarray): The Hu moments of the organoid contour.
        flags (list): A list of flags indicating the success or failure of certain operations.
        curvature (ndarray): The curvature values along the organoid contour.
        inflection_points (ndarray): The indices of inflection points in the curvature.
        smooth_contour (ndarray): The smoothed organoid contour.

    """
    final_segmented_organoid, bg_uniformity, flags, bg_mean = extract_organoid(
        gray_image)

    # Shape features
    contours, _ = cv2.findContours(
        final_segmented_organoid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        contour = max(contours, key=cv2.contourArea)
        curvature, smooth_contour = compute_curvature(contour)
        smooth_contour = smooth_contour.astype(int)
        area = cv2.contourArea(smooth_contour)
        perimeter = cv2.arcLength(smooth_contour, True)
        moments = cv2.moments(smooth_contour)
        hu_moments = cv2.HuMoments(moments).flatten()

        # Calculate roundness and circularity
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        roundness = (4 * area) / (np.pi *
                                  max([cv2.minAreaRect(smooth_contour)[1][0], cv2.minAreaRect(smooth_contour)[1][1]]) ** 2)

        # Calculate Feret diameters
        try:
            (x, y), (MA, ma), angle = cv2.fitEllipse(smooth_contour)
            min_feret = min(MA, ma)
            max_feret = max(MA, ma)
            aspect_ratio = min_feret / max_feret
        except:
            min_feret = max_feret = aspect_ratio = 0

        # Calculate curvature and DNE
        
        dne = np.sum(curvature**2)

        # Plot the smooth contour over the final centered organoid

        # Find inflection points (where curvature changes sign)
        inflection_points = find_inflection(curvature, inflection_threshold)

    else:
        area = perimeter = roundness = circularity = min_feret = max_feret = aspect_ratio = dne = 0
        hu_moments = np.zeros(7)
        curvature = np.array([])
        inflection_points = np.array([])

    # Extract the non-zero parts of the segmented organoid for texture analysis
    non_zero_coords = np.nonzero(final_segmented_organoid)
    if non_zero_coords[0].size > 0:
        min_row, max_row = np.min(
            non_zero_coords[0]), np.max(non_zero_coords[0])
        min_col, max_col = np.min(
            non_zero_coords[1]), np.max(non_zero_coords[1])

        roi = final_segmented_organoid[min_row:max_row+1, min_col:max_col+1]

        glcm = graycomatrix(roi, [1], [0], symmetric=True, normed=True)
        glcm = glcm[1:, 1:, :, :] # This is to remove the 0th row and column as they account for the value 0 which in this case is the background

        contrast = graycoprops(glcm, 'contrast')[0, 0]
        dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]
    else:
        contrast = dissimilarity = homogeneity = energy = correlation = 0

    # Naive measure of background transparency, the difference between the mean of the background and the mean of the organoid
    background_transparency = bg_mean - \
        np.median(final_segmented_organoid[final_segmented_organoid != 0])

    # put all the statistics in a dictionary with keys
    stat_array = {
        'circularity': circularity,
        'roundness': roundness,
        'area': area,
        'perimeter': perimeter,
        'feret max': max_feret,
        'feret min': min_feret,
        'contrast': contrast,
        'dissimilarity': dissimilarity,
        'homogeneity': homogeneity,
        'energy': energy,
        'correlation': correlation,
        'aspect_ratio': aspect_ratio,
        'background_uniformity': bg_uniformity,
        'dne': dne,
        'inflection_points': len(inflection_points),
        'background_transparency': background_transparency
    }

    return stat_array, final_segmented_organoid, hu_moments, flags, curvature, inflection_points, smooth_contour


def extract_organoid(gray_image, k=2):
    """
    Extracts the organoid from a grayscale image using image processing techniques.

    Args:
    gray_image (ndarray): The grayscale image from which the organoid is to be extracted.
    k (int): The number of clusters for KMeans clustering. Default is 5.

    Returns:
    tuple: A tuple containing the following elements:
        centered_organoid (ndarray): The extracted organoid centered in the image.
        bg_uniformity (bool): A flag indicating whether the background is uniform.
        flags (list): A list of flags indicating any issues with the extracted organoid.
        bg_mean (float): The mean intensity value of the background pixels.
    """

    flags = []

    pixel_values = gray_image.reshape((-1, 1))
    pixel_values = np.float32(pixel_values)

    # Define criteria and apply KMeans clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.5)
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None,
                                      criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    ret, thresh = cv2.threshold(gray_image,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers+1
    markers[unknown==255] = 0
    
    gray_image = np.uint8(gray_image)
    markers = watershed(gray_image,markers)



    # Convert the labels to the same shape as the original image
    segmented_image = markers

    # Calculate the average value of the gray_image for each label in the segmented_image
    label_averages = {}
    for label in np.unique(segmented_image):
        label_mask = segmented_image == label
        label_average = np.median(gray_image[label_mask])
        label_averages[label] = label_average
    
    # Find the background label (the label with the highest average value = lightest area)
    background = max(label_averages, key=label_averages.get)

    bg_uniformity = is_background_uniform(gray_image, segmented_image, background)

    bg_mean = label_averages[background]

    segmented_image[segmented_image == background] = 600

    # Apply Canny edge detection
    gray_image_blur =  cv2.GaussianBlur(gray_image, (5, 5), 0)
    edges = cv2.Canny(gray_image_blur, bg_mean/10, bg_mean)

    # Dilate the edges to make them more pronounced
    kernel = np.ones((10, 10), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    combined_mask = np.zeros_like(segmented_image)
    combined_mask[dilated_edges == 255] = 255
    combined_mask[segmented_image != 600] = 255

    cleaned_mask = cv2.morphologyEx(
        np.uint8(combined_mask), cv2.MORPH_CLOSE, kernel, iterations=4)
    
    # Shrink the mask by 2 pixels
    shrink_kernel = np.ones((2, 2), np.uint8)
    cleaned_mask = cv2.erode(cleaned_mask, shrink_kernel, iterations=1)

    # Remove small speckles from the cleaned mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned_mask, connectivity=8)
    sizes = stats[1:, -1]  # Get the size of each component, excluding the background
    min_size = 2500  # Minimum size of speckles to keep

    # Create a new mask to store the cleaned result
    cleaned_mask = np.zeros_like(cleaned_mask)

    # Keep only the components that are larger than the minimum size
    for i in range(1, num_labels):
        if sizes[i - 1] >= min_size:
            cleaned_mask[labels == i] = 255



    centered_organoid = cv2.bitwise_and(
        gray_image, gray_image, mask=cleaned_mask)

    # Find the largest contour in the centered organoid and only keep that
    contours, _ = cv2.findContours(
        centered_organoid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        # find largest contour that has more than 1 color value, as there is a big cluster of pixels with the same value
        largest_contour = max(contours, key=lambda x: len(
            np.unique(centered_organoid[x[:, 0, 1], x[:, 0, 0]])))
        mask = np.zeros_like(centered_organoid)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)
        centered_organoid = cv2.bitwise_and(
            centered_organoid, centered_organoid, mask=mask)

    # Check for issues with the extracted organoid
    if is_organoid_at_edge(centered_organoid):
        flags.append('Organoid is at the edge')
        print('Organoid is not in the center')

    centered_organoid = center_organoid(centered_organoid)

    if not is_single_object(centered_organoid):
        flags.append('Multiple objects detected')
        print('Multiple objects detected')

    if contains_holes(centered_organoid):
        flags.append('Organoid contains holes')
        print('Organoid contains holes')

    return centered_organoid, bg_uniformity, flags, bg_mean



def remove_flagged_from_metadata(metadata, flags):
    """
    Removes flagged images from the metadata dictionary.

    Args:
    metadata (dict): A dictionary containing image files as keys and their corresponding metadata as values.
    flags (list): A list of boolean values indicating whether an image should be flagged for removal.

    Returns:
    dict: The updated metadata dictionary with flagged images removed.
    """
    image_files = list(metadata.keys())
    for idx, image_file in enumerate(image_files):
        if flags[idx]:
            metadata.pop(image_file)
    return metadata


def center_organoid(gray_image):
    """
    Shifts the given gray-scale image so that the center of mass of the organoid is in the center of the image.

    Args:
    gray_image (ndarray): The gray-scale image of the organoid.

    Returns:
    ndarray: The shifted image with the center of mass at the center.
    """
    # find the center of mass of the organoid and shift the image so that the center of mass is in the center of the image
    contours, _ = cv2.findContours(
        gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    contour = contours[max_index]
    center_of_mass = np.mean(contour, axis=0)[0]

    rows, cols = gray_image.shape
    shift_x = int(cols / 2 - center_of_mass[0])
    shift_y = int(rows / 2 - center_of_mass[1])

    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted_image = cv2.warpAffine(gray_image, M, (cols, rows))

    return shifted_image


def is_background_uniform(image, segmentation, backgroundid, uniformity_threshold=15):
    """
    Check if the background signal in the image is uniform.

    Args:
    image (ndarray): The input image.
    uniformity_threshold (float): The threshold for considering the background as uniform. Default is 10.

    Returns:
    bool: True if the background is considered uniform, False otherwise.
    """
    background = image[segmentation == backgroundid]
    if background.size == 0:
        return False
    std = np.std(background)
    return std < uniformity_threshold


def is_organoid_at_edge(image):
    """
    Check if the organoid in the image is located at the edge.

    Args:
    image (ndarray): The input image.

    Returns:
    bool: True if the organoid is located at the edge, False otherwise.
    """
    # Check if one of the pixels near the edge is non-zero
    # If the organoid is not in the center, the image is not valid
    rows, cols = image.shape
    return np.any(image[0, :]) or np.any(image[rows - 1, :]) or np.any(image[:, 0]) or np.any(image[:, cols - 1])


def is_single_object(image):
    """
    Check if there is only one object in the image.

    Args:
    image (ndarray): The input image.

    Returns:
    bool: True if there is only one object in the image, False otherwise.
    """
    contours, _ = cv2.findContours(
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours) == 1


def contains_holes(image):
    """
    Check if an image contains any holes (contours with a parent).

    Args:
    image (ndarray): The input image.

    Returns:
    bool: True if the image contains holes, False otherwise.
    """
    # Find contours with hierarchy information
    contours, hierarchy = cv2.findContours(
        image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contour has a parent, indicating a hole
    if hierarchy is not None:
        for i in range(len(contours)):
            # If a contour has a parent, it's a hole
            if (hierarchy[0][i][3] != -1):
                print(cv2.contourArea(contours[i]))
                return True
    return False




def compute_curvature(contour, s=2500):
    """
    Compute the curvature of a contour.

    Args:
    contour (ndarray): The contour points.
    s (float): The smoothing factor for the spline fit. Default is 2500.

    Returns:
    ndarray: The curvature values.
    ndarray: The smoothed contour points.
    """
    # Fit a spline to the contour points to smooth it
    contour = contour[:, 0, :]
    tck, u = splprep([contour[:, 0], contour[:, 1]], s=s, per=True)
    u_new = np.linspace(u.min(), u.max(), len(contour) * 10)
    x_new, y_new = splev(u_new, tck, der=0)
    smooth_contour = np.column_stack((x_new, y_new))

    # Compute derivatives on the smoothed contour
    dx, dy = splev(u_new, tck, der=1)
    ddx, ddy = splev(u_new, tck, der=2)

    # Compute curvature on the smoothed contour
    curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5

    return curvature, smooth_contour


def find_inflection(curvature, threshold=0.01):
    """
    Find the indices of significant positive inflection points in a curvature array.

    Args:
    curvature (ndarray): Array containing curvature values.
    threshold (float, optional): Threshold value to determine significance of positive points. Default is 0.01.

    Returns:
    ndarray: Array containing the indices of significant positive inflection points.
    """
    significant_positive_points_indices = np.where(curvature > threshold)[0]

    if len(significant_positive_points_indices) == 0:
        # Return an empty array if no significant positive points are found
        return np.array([])

    groups = np.split(significant_positive_points_indices, np.where(
        np.diff(significant_positive_points_indices) != 1)[0] + 1)

    peak_positive_points_indices = []
    for group in groups:
        if len(group) > 0:
            peak_index = group[np.argmax(curvature[group])]
            peak_positive_points_indices.append(peak_index)

    return np.array(peak_positive_points_indices)



def get_image_metadata(folder_path, extra_metadata=None, condition_rules=None):
    """
    Retrieves metadata for images in a given folder.

    Args:
        folder_path (str): The path to the folder containing the images.
        extra_metadata (dict, optional): Additional metadata to include for all images. Defaults to None.
        condition_rules (dict, optional): Condition rules to apply for specific metadata values based on filename. Defaults to None.

    Returns:
        dict: A dictionary containing the metadata for each image, with the filename as the key.
    """
    if extra_metadata is None:
        extra_metadata = {}

    if condition_rules is None:
        condition_rules = {}

    metadata = {}

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
            try:
                with Image.open(file_path) as img:
                    width, height = img.size
                    file_size = os.path.getsize(file_path)

                    # Initialize image metadata with basic details
                    image_metadata = {
                        'file_size': file_size,
                        'width': width,
                        'height': height
                    }

                    # Add extra metadata
                    for key, value in extra_metadata.items():
                        image_metadata[key] = value

                    # Apply condition rules with default values
                    for key, rules in condition_rules.items():
                        default_value = rules.get('default', None)
                        condition_applied = False
                        for keyword, rule_value in rules.items():
                            if keyword != 'default' and keyword in filename:
                                image_metadata[key] = rule_value
                                condition_applied = True
                                break
                        if not condition_applied:
                            image_metadata[key] = default_value

                    metadata[filename] = image_metadata
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    return metadata


def save_metadata_as_xlsx(metadata, output_path):
    """
    Save metadata as an Excel file.

    Args:
        metadata (dict): A dictionary containing the metadata.
        output_path (str): The path to save the Excel file.

    Returns:
        None
    """
    df = pd.DataFrame(metadata).T
    df.to_excel(output_path)


def read_metadata_from_xlsx(input_path):
    """
    Read metadata from an Excel file.

    Args:
    input_path (str): The path to the Excel file.

    Returns:
    dict: A dictionary containing the metadata, where the keys are the row indices and the values are dictionaries representing each row.
    """
    df = pd.read_excel(input_path, index_col=0)
    metadata = df.to_dict(orient='index')
    return metadata


def get_df_from_xlsx(input_path):
    """
    Read an Excel file and return a pandas DataFrame.

    Args:
    input_path (str): The path to the Excel file.

    Returns:
    pandas.DataFrame: The DataFrame containing the data from the Excel file.
    """
    df = pd.read_excel(input_path, index_col=0)
    return df


def save_df_as_xlsx(df, output_path):
    """
    Save a pandas DataFrame as an Excel file.

    Args:
    df: pandas DataFrame
        The DataFrame to be saved.
    output_path: str
        The path where the Excel file will be saved.

    Returns:
    None
    """
    df.to_excel(output_path)



def update_metadata_with_analysis_old(image_path, metadata, image_files):
    """Old version of the function that uses ThreadPoolExecutor for parallel processing. Got updated with dask. This is for fallback only"""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_image, image_file, image_path): idx for idx, image_file in enumerate(image_files)}
        results = [None] * len(image_files)

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            idx = futures[future]
            stats, segmented_organoid, hu_moment, flags, curvature, inflection_points, smooth_contour = future.result()
            image_name = image_files[idx]
            metadata[image_name].update(stats)
            metadata[image_name]['hu_moments'] = hu_moment.tolist()
            results[idx] = (stats, segmented_organoid, hu_moment,
                            flags, curvature, inflection_points, smooth_contour)

    # Extract the ordered results
    statistics = [result[0] for result in results]
    segmented_organoids = [result[1] for result in results]
    hu_moments = [result[2] for result in results]
    flags = [result[3] for result in results]
    curvatures = [result[4] for result in results]
    inflection_points = [result[5] for result in results]
    smooth_contours = [result[6] for result in results]

    return metadata, statistics, segmented_organoids, hu_moments, flags, curvatures, inflection_points, smooth_contours


def update_metadata_with_analysis(image_path, metadata, image_files):
    """
    Updates the metadata dictionary with analysis results for each image file using Dask. This reduces memory load and speeds up the process.

    Args:
    image_path (str): The path to the directory containing the image files.
    metadata (dict): The metadata dictionary to be updated.
    image_files (list): A list of image file names.

    Returns:
    tuple: A tuple containing the updated metadata dictionary, statistics, segmented organoids,
            Hu moments, flags, curvatures, inflection points, and smooth contours.
    """
    tasks = []

    for idx, image_file in enumerate(image_files):
        task = delayed(process_image)(image_file, image_path)
        tasks.append(task)

    with ProgressBar():
        results = compute(*tasks)

    for idx, result in enumerate(results):
        stats, segmented_organoid, hu_moment, flags, curvature, inflection_points, smooth_contour = result
        image_name = image_files[idx]
        metadata[image_name].update(stats)
        metadata[image_name]['hu_moments'] = hu_moment.tolist()

    # Extract the ordered results
    statistics = [result[0] for result in results]
    segmented_organoids = [result[1] for result in results]
    hu_moments = [result[2] for result in results]
    flags = [result[3] for result in results]
    curvatures = [result[4] for result in results]
    inflection_points = [result[5] for result in results]
    smooth_contours = [result[6] for result in results]

    return metadata, statistics, segmented_organoids, hu_moments, flags, curvatures, inflection_points, smooth_contours


def cluster_hu_moments(hu_moments, num_clusters=5):
    """
    Cluster the given Hu moments using K-means algorithm.

    Args:
    hu_moments (array-like): The Hu moments to be clustered.
    num_clusters (int): The number of clusters to create (default: 5).

    Returns:
    labels (array-like): The cluster labels for each Hu moment.
    cluster_centers (array-like): The centroids of the clusters.
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(hu_moments)
    return labels, kmeans.cluster_centers_


def create_image_with_transparent_background(image):
    """
    Create an image with a transparent background.

    Args:
    image (ndarray): The input image.

    Returns:
    ndarray: The image with a transparent background.

    """
    # Create a 4-channel image with the same shape as the input, adding an alpha channel
    transparent_image = np.zeros((*image.shape, 4), dtype=np.uint8)
    # Set the RGB channels to the image values
    transparent_image[:, :, :3] = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # Set the alpha channel: 255 where the image is not zero, 0 where it is zero
    transparent_image[:, :, 3] = np.where(image > 0, 255, 0)
    return transparent_image


def image_to_base64(image):
    """
    Convert an image to base64 encoding.

    Args:
    image (ndarray): The image to be converted.

    Returns:
    str: The base64 encoded image.

    """
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')


def plot_with_images(df, target, segmented_organoids, output_folder, outputname):

    """
    Plots a scatter plot with images at the corresponding data points.

    Args:
    X (ndarray): The input data array.
    images (list): A list of images.
    conditions (list): A list of conditions corresponding to each data point.
    output_folder (str): The path to the output folder.

    Returns:
    None
    """ 

    fig = go.Figure()

    scatter_traces = []
    image_traces = []

    unique_conditions = df[target].unique()

    df["image"] = df.index[0]

    for condition in unique_conditions:
        indices = np.where(df[target] == condition)[0]

        scatter = go.Scatter(
            x=df.iloc[indices,]["umap_x"], y=df.iloc[indices,]["umap_y"],
            mode='markers',
            marker=dict(size=5, opacity=0.5),
            text=df.iloc[indices,]["image"],
            name=condition,
            showlegend=True
        )
        scatter_traces.append(scatter)

        for i in indices:
            image_with_transparency = create_image_with_transparent_background(segmented_organoids[i])
            image_base64 = image_to_base64(image_with_transparency)

            image_trace = dict(
                source=f'data:image/png;base64,{image_base64}',
                xref="x", yref="y",
                x=df.iloc[i,]["umap_x"], y=df.iloc[i,]["umap_y"],
                sizex=0.5, sizey=0-5,
                xanchor="center", yanchor="middle",
                layer="below",
                visible=True
            )
            image_traces.append(image_trace)
            fig.add_layout_image(image_trace)

    for scatter in scatter_traces:
        fig.add_trace(scatter)

    fig.update_layout(
        title="Organoids PCA with Conditions",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        width=800, height=800,
        legend_title_text='Conditions'
    )

    fig.update_layout(
        legend=dict(
            itemsizing='constant',
            itemclick='toggleothers',  # Ensure that clicking one label hides others
            itemdoubleclick='toggle'   # Double clicking toggles the selected label
        )
    )

    fig.write_image(os.path.join(
        output_folder, outputname+"umap_plot_w_images.pdf"))

    fig.show()


def plot_contour_with_dne(gray_image, smooth_contour, curvature, inflection_points, output_folder, output_name):
    """
    Plots the contour of a gray image with colors based on curvature and marks inflection points.

    Args:
    gray_image (ndarray): The gray image to be plotted.
    smooth_contour (ndarray): The smooth contour points.
    curvature (ndarray): The curvature values for each contour point.
    inflection_points (list): The indices of the inflection points in the contour.

    Returns:
    None
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(gray_image, cmap='gray')

    # Use scatter plot to visualize the contour with colors based on curvature
    plt.scatter(smooth_contour[:, 0], smooth_contour[:, 1],
                c=curvature, cmap='RdBu_r', s=4, vmin=-0.01, vmax=0.01)

    if len(inflection_points) > 0:
        # Mark inflection points in red
        plt.scatter(smooth_contour[inflection_points, 0], smooth_contour[inflection_points,
                    1], color='red', s=50, facecolors='none', label='Inflection Points')

    plt.title('Contour with Dirichlet Normal Energy and Inflection Points')
    plt.legend()
    plt.savefig(os.path.join(output_folder, output_name + 'contour.pdf'))
    plt.show()


def plot_analysis(segmented_organoid, stats, flags, output_folder, output_name=None):
    """
    Plot the segmented organoid image and display statistics in a table as well as saving the plot in an svg format.

    Args:
    segmented_organoid: numpy array
        The segmented organoid image.
    stats: dict
        A dictionary containing the statistics to be displayed in the table.
    flags: list
        A list of boolean flags indicating whether a specific condition is met (In this case revealing that the picture of the organoids has some flaws).
    output_folder: str
        The path to the folder where the output image will be saved.
    output_name: str, optional
        The name of the output image file. If not provided, a default name will be used.

    Returns:
    None
    """
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    # Plot the segmented organoid image
    ax[0].imshow(segmented_organoid, cmap='gray')
    ax[0].axis('off')
    ax[0].set_title('Segmented Organoid')

    # Create a table with statistics
    cell_text = [[f"{value:.2f}" if isinstance(
        value, float) else str(value)] for value in stats.values()]
    rows = list(stats.keys())

    ax[1].axis('off')
    table = ax[1].table(cellText=cell_text, rowLabels=rows,
                        colLabels=["Value"], loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(0.35, 1.5)

    # Save the figure
    if output_name:
        plt.savefig(os.path.join(output_folder, output_name + '.pdf'))
    else:
        plt.savefig(os.path.join(output_folder, 'analysis.pdf'))

    # Show the flags
    for flag in flags:
        if flag:
            # If there is a flag also add a red border to the segmented organoid
            if len(flags) > 1:
                ax[0].add_patch(plt.Rectangle((0, 0), segmented_organoid.shape[1],
                                segmented_organoid.shape[0], fill=False, edgecolor='red', linewidth=5))
                print(flag)
            else:
                ax[0].add_patch(plt.Rectangle((0, 0), segmented_organoid.shape[1],
                                segmented_organoid.shape[0], fill=False, edgecolor='pink', linewidth=5))
                print(flag)

    plt.show()
    plt.close()

