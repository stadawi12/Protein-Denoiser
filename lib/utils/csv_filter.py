import pandas as pd

def get_entries(data, min_dim, max_dim, min_res, max_res):
    """
        This  function  takes in a data-frame as an input
        and filters it according to the input parameters.
        This function returns a new, filtered data frame. 
            min_dim = minimum dimension (size)
            max_dim = maximum dimension (size)
            min_res = minimum resolution
            max_res = maximum resolution
    """
    # Ensure I only have two half maps of each structure
    searchfor = ['half_map_1', 'half_map_2']
    df2 = data.loc[
            data[" Tail"].str.contains('|'.join(searchfor))]

    # Ensure height = width = depth
    df2 = df2.loc[(data["Height"] == data["Width"]) &
                  (data["Width"]  == data["Depth"])]

    # Filter by max_dim
    df2 = df2.loc[df2["Height"].between(min_dim, max_dim)]

    # Filter by max_res
    df2 = df2.loc[df2["Resolution"].between(min_res, max_res)]

    print(f"Found {len(df2)} half maps, that's {int(len(df2)/2)} training examples.")

    return df2

def ge_example():
    print("Loading halfMaps.csv (all half-maps we have)")
    path = '../data/halfMaps.csv'
    data = pd.read_csv(path)
    df = pd.DataFrame(data)
    print(f"Number of half-maps before applying filter: {len(df)}")
    print("Filtering...")
    df2 = get_entries(df,0,128,3,4)
    
