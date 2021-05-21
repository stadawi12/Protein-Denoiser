from urllib.request import urlopen as uReq
from bs4 import BeautifulSoup as soup
import bs4
import pandas as pd

def get_entries():
    # url of page with all structures
    global url
    url = 'https://www.ebi.ac.uk/pdbe/emdb/'

    # load csv file containing all half maps found
    global path
    path = '../data/halfMaps.csv'
    global data
    data = pd.read_csv(path)
    # select Entry collumn only
    df = pd.DataFrame(data, columns = ['Entry']) 
    Entries = df.to_numpy().flatten()
    return Entries

def get_dim(entry):
    tail = '/experiment'
    total_url = url + entry + tail

    uClient = uReq(total_url)
    pg_html = uClient.read()
    uClient.close()

    pg_soup = soup(pg_html, "html.parser")

    div16s = pg_soup.findAll('div', {'class':'grid_16'})

    div24s = []
    for div16 in div16s:
        x = div16.findAll('div', {'class':'grid_24'})
        div24s.append(x)


    box = div24s[2][1]

    NofG = box.findAll('span', {'class':'grid_14'})

    string = NofG[0].text.strip()
    new_string = ''

    for char in string:
        if char != " ":
            new_string += char

    dims = new_string.splitlines()

    for i, dim in enumerate(dims[0:2]):
        dims[i] = dim[0:-1]

    return dims

def get_dims(n=None):
    Entries = get_entries()
    dim_data = []
    for i, entry in enumerate(Entries):
        try:
            res = get_dim(entry)
            dim_data.append(res)
            print(f'{i+1}/{len(Entries)}: {entry}: {dim_data[-1]}')
        except:
            dim_data.append("None")
            print(f'{i+1}/{len(Entries)}: {entry}: {dim_data[-1]}')
        if i == n-1:
            break

    df2 = pd.DataFrame(data)

    if n == None:
        if len(dim_data) == len(df2):
            df2["Dimensions"] = dim_data
        else:
            print(f'The length of dim_data ({len(dim_data)}) must be the same as length of data frame ({len(df2)})')

        df2.to_csv('with_res.csv', index=False)
    else:
        return dim_data


