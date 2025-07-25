import h5py

#Open the H5 file in read mode
with h5py.File('test_output/UNI/TCGA-AA-3527-01A-01-BS1.5aa0bd9a-172a-4109-81e2-5293fba47e7c_loaded.h5', 'r') as file:
    print(file)
    print("KEYS: " + str(file.keys()))
    print("number of records ", len(file['feats']))
    a_group_key = list(file.keys())[0]
    
    # Getting the data
    data_1 = list(file['feats'])
    coords_1 = list(file['coords'])
    
with h5py.File('test_output/UNI/TCGA-AA-3527-01A-01-BS1.5aa0bd9a-172a-4109-81e2-5293fba47e7c.h5', 'r') as file:
    print(file)
    print("KEYS: " + str(file.keys()))
    print("number of records ", len(file['feats']))
    a_group_key = list(file.keys())[0]
    
    # Getting the data
    data_2 = list(file['feats'])
    coords_2 = list(file['coords'])

print(len(data_1))
print(len(data_2))
data_2_set = set(map(lambda x: tuple(x), data_2))

print(data_1[0])
print(data_2[0])
print(coords_1[0])
print(coords_2[0])
# Compare and print coordinates of features in data_1 not in data_2
missing_coords = []
for i, feat in enumerate(data_1):
    if tuple(feat) not in data_2_set:
        missing_coords.append(coords_1[i])
        print(f"Missing feature at coords: {coords_1[i]}")

print(f"Total missing features: {len(missing_coords)}")
# print(data_1)
# print(coords_1)
# print(data_2)
# print(coords_2)
# if data_2 in data_2:
#     print("JAAAAAA")
#     # print(data)