import logging

import pandas as pd

logger = logging.getLogger(__name__)


def check_zero_firstPosition(df):
    return (df.loc[0, ["dx", "dy", "dz"]] == 0).all()


def calc_shard_environment(shard: pd.DataFrame, receptorData, pdb, shard_name) -> pd.DataFrame:
    """ """
    # create one big DataFrame

    # Use a cross join to combine each chart row with each receptor
    df = receptorData.copy()
    # Calculate the differences
    df["dx"] = df["x"] - shard["x"]
    df["dy"] = df["y"] - shard["y"]

    df["dz"] = df["z"] - shard["z"]

    # Compute Euclidean distance
    df["dr_squared"] = df["dx"] ** 2 + df["dy"] ** 2 + df["dz"] ** 2

    # Sort by distance and reset index
    df.sort_values(by="dr_squared", ascending=True, inplace=True, ignore_index=True)

    # Check for zero vector in first entry
    if check_zero_firstPosition(df):
        logger.info(f"Zero length vector: {pdb} for shard {shard_name}")

    # Return only the desired columns
    return df[["target_atom_id", "dx", "dy", "dz", "dr_squared"]]


def find_interacting(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Filters the DataFrame by keeping only rows where the 'dr' value is below the given threshold.

    Parameters:
        df (pd.DataFrame): The input DataFrame that must include a 'dr' column.
        threshold (float): The threshold value. Only rows with a 'dr' value less than this threshold are retained.

    Returns:
        pd.DataFrame or None: A DataFrame containing only the rows where 'dr' is below the threshold,
                              or None if no such rows exist.
    """
    # Check if 'dr' column exists
    if "dr_squared" not in df.columns:
        raise KeyError("The DataFrame does not contain a 'dr_squared' column.")

    # Filter rows where 'dr' is below the threshold
    filtered_df = df[df["dr_squared"] < threshold**2]

    # Return None if the filtered DataFrame is empty
    if filtered_df.empty:
        return None
    else:
        return filtered_df


def calc_interface_shardwise(
    chart_name, chart_members: pd.DataFrame, receptor_data: pd.DataFrame, threshold, pdb_id: str
) -> list:
    """ """
    return [
        (chart_name, chart_member, interaction_env)
        for _, chart_member in chart_members.iterrows()
        if (
            interaction_env := find_interacting(
                calc_shard_environment(chart_member, receptor_data, pdb_id, chart_name), threshold
            )
        )
        is not None
    ]
