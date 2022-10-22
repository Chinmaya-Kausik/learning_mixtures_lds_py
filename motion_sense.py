import csv

def motion_sense_preprocessing():

    # Open the files    
    file1 = open('jog.csv')
    file2 = open('wlk_sub_23.csv')

    # Initialize the readers and skip the headers
    jog_reader = csv.reader(file1)
    walk_reader = csv.reader(file2)
    jog_header = next(jog_reader)
    walk_header = next(walk_reader)

    # Accumulate data from the rows, skip the serial number (first element)
    jog_rows = []
    for row in jog_reader:
        jog_rows.append([float(i) for i in row[1:]])
    walk_rows = []
    for row in walk_reader:
        walk_rows.append([float(i) for i in row[1:]])

    # Convert to arrays
    jog_rows = np.array(jog_rows)
    walk_rows = np.array(walk_rows)

    # Initialize d, K, M, T
    d = 12
    K = 2
    T = 400
    M=12

    # Initialize a numpy array containing 24 400 times 12 matrices 
    # Shape (24, 200, 12)
    combined_data = np.zeros([2*M,T,d])

    # Take blocks from jog_rows and walk_rows to add to the combined data
    for i in range(M):
        combined_data[i, :, :] = jog_rows[400*i:400*(i+1), :]
    for i in range(M):
        combined_data[(M+i), :, :] = walk_rows[400*i:400*(i+1), :]
        
    return combined_data.transpose(0,2,1)

