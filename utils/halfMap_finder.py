from ftplib import FTP
from dateutil import parser
import os

write_to_csv = True

if write_to_csv:
    with open('data_temp.csv', 'w') as wf:
        headers = "Entry, Tail, Size, Date\n"
        wf.write(headers)

def get_structures():
    """
    Accesses the ftp and returns all structures
    found in the structures directory as a list.
    """
    # Accessing ftp
    address = 'ftp.ebi.ac.uk'
    global ftp
    ftp = FTP(address)
    ftp.login()

    # Path to all structures
    structure_path = '/pub/databases/emdb/structures'

    # Enter structure directory
    ftp.cwd(structure_path)

    # Save all structures to variable
    structures = ftp.nlst()

    return structures

def find_string(filename, string):
    """
    Check if string is in the filename. 
    """
    if string in filename:
        return True
    else:
        return None


def find_dir(contents, target):
    """
    looks for a directory inside the directory you are in,
    Enters it and returns the contents of that directory.
    If target not found, takes one step back.
    """
    # Check if 'target/' is inside this directory
    if target in contents:
        # If True, enter 'target/'
        ftp.cwd(target)
        # Save contents of target to variable
        target_contents = ftp.nlst()
        # And Return that variable
        return target_contents

    # if contain_target is False, back to structures
    else:
        ftp.cwd('../')
        return None




def find_halfMaps(amount=50):
    """
    This script looks through the structure directories
    and looks for half maps,  when found,  it saves the
    entry, tail, size and date of the half_map to a csv
    file.
    
    amount means how many structures do you want to look
    through, default is set to 50,  if amount='all' then
    the script will look inside all structures. As there
    are over 14000 structures, this might take some time.
    """

    if amount == 'all':
        amount = len(structures)

    # Access the ftp and get all structures
    structures = get_structures()

    structures_searched = 0 # Counter
    halfMaps_found = 0      # Counter

    for i, structure in enumerate(structures):
        # Enter structure directory
        ftp.cwd(structure)
        # Save the contents of the directory to variable
        structure_content = ftp.nlst()
        # Check if other/ is inside structure_contents
        other_contents = find_dir(structure_content, 'other')
        # If find_dir has found 'other/' directory, we are
        # now inside 'other/' if not, find_dir will take us
        # back to structures

        # Now we check contents of 'other/'
        if other_contents != None:
            # Check if the files in 'other/'
            # contain the string 'half'
            for filename in other_contents:
                is_half = find_string(filename, 'half')
                # if file contains string 'half' 
                # save entry, tail, size and date to csv file
                if is_half:
                    # If found half map, increase counter
                    halfMaps_found += 1
                    # if write_to_csv = True, write to csv
                    if write_to_csv:
                        entry = structure
                        tail = filename
                        size = ftp.size(filename)

                        # obtaining date of file
                        m = 'MDTM '
                        pwd = ftp.pwd()
                        tail = filename
                        file_path = pwd + "/" + tail
                        timestamp = ftp.voidcmd(
                                m + file_path)[4:].strip()
                        time = parser.parse(timestamp)
                        date = str(time.date())

                        # writing data to csv file
                        with open('data_temp.csv', 'a') as af:
                            af.write(entry + "," 
                                    + tail + "," + str(size) + 
                                    "," + date + "\n")
                    print(structure)

            # At the end of search, go back to structures/
            ftp.cwd('../')
            ftp.cwd('../')
            
        # Increase structures_searched counter
        structures_searched += 1 

        # if we reach i = amount stop the search
        if i == amount-1:
            break

    # After searching, print findings
    print(f'Structures searched: {structures_searched}')
    print(f'Half maps found: {halfMaps_found}')

    # Rename temp_data to data to prevent overwriting
    dir_contents = os.listdir()
    if 'data.csv' in dir_contents:
        print("Warning, found 'data.csv' inside directory.")
        answer = input("Do you want to overwrite 'data.csv'? (Y/n)\n")
        # Possible 'yes' inputs
        yeses = ['Y', 'y', 'yes', 'Yes']
        if answer in yeses:
            os.rename("data_temp.csv", "data.csv")
        else:
            print("Output has been saved as 'data_temp.csv'")

    # If didn't find data.csv in directory, rename.
    else:
        os.rename("data_temp.csv", "data.csv")

# Run script
find_halfMaps()
