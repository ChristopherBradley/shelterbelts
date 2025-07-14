

def class_metrics(geometry, folder):
    """Calculate the percentage cover in each class.
    
    Parameters
    ----------
        geometry: The region to calculate metrics
        folder: The folder containing tif files with shelter categories for that region
            
    Returns
    -------
        df: A dataframe with total areas and percentage areas in each class.

    Downloads
    ---------
        class_metrics.csv: A csv with total areas and percentage areas in each class.

    """


def patch_metrics(geometry, folder):
    """Calculate the length, width, height, direction, area, perimeter for each patch.
        Also calculates an overall mean and standard deviation for each attribute across all patches.
    
    Parameters
    ----------
        geometry: The region to calculate metrics
        folder: The folder containing tif files with shelter categories for that region
            
    Returns
    -------
        df_individual: Indvidual attributes for each patch.
        df_aggregates: Aggregated attributes for all patches.

    Downloads
    ---------
        patch_metrics_individual.csv
        patch_metrics_aggregated.csv

    """