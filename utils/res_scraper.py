from urllib.request import urlopen as uReq
from bs4 import BeautifulSoup as soup
import bs4
import pandas as pd

def get_entries():
    # url of page with all structures
    global url
    url = 'https://www.ebi.ac.uk/pdbe/emdb/'

    # load the csv file with all maps found
    path = '../data/halfMaps.csv'
    global data
    data = pd.read_csv(path)
    # select the Entry collumn only
    df = pd.DataFrame(data, columns = ['Entry']) 
    Entries = df.to_numpy().flatten()
    return Entries

def get_res(entry):
    """
        This function takes entry as an input and uses   it
        to enter the url containing information  about that
        structure, it searches the page for resolution   of 
        the structure and returns it.
    """
    # Compose url containing structure information
    total_url = url + entry
    # Enter page
    uClient = uReq(total_url)
    pg_html = uClient.read()
    uClient.close()

    # store soup of the page
    pg_soup = soup(pg_html, "html.parser")

    # Find all divs with class grid_24
    divs = pg_soup.findAll("div", {"class":"grid_24"})

    lst = []
    for div in divs:
        spans = div.findAll("span", {"class":"grid_12"})
        for span in spans:
            lst.append(span.text.strip())

    return lst[1][0:-1]

def get_reses(n=None):
    # wrapper for the get_res function.
    # Run the function get_res for all entries in our csv file
    # add resolution as an extra collumn to the csv file and
    # append the scraped resolutions.
    Entries = get_entries()
    res_data = []
    for i, entry in enumerate(Entries):
        res = get_res(entry)
        res_data.append(res)
        print(f'{i+1}/{len(Entries)}: {res_data[-1]}')
        if i == n-1:
            break

    df2 = pd.DataFrame(data)

    # Make sure we scraped the same number of resolutions
    # as the length of our csv file
    if n == None:
        if len(res_data) in [len(df2)]:
            df2["Resolution"] = res_data
        else:
            print(f'The length of res_data ({len(res_data)}) must be the same as length of data frame ({len(df2)})')

        # save the new dataframe to a new csv file
        df2.to_csv('with_res.csv', index=False)
    else:
        return res_data
