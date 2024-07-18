import csv

# Example lists of different lengths
list1 = ['A', 'B', 'C']
list2 = [1, 2]
list3 = [4.5, 5.5, 6.5, 7.5]

# Find the maximum length of the lists
max_length = max(len(list1), len(list2), len(list3))

# Pad lists to the maximum length with None (or any placeholder)
list1 += [None] * (max_length - len(list1))
list2 += [None] * (max_length - len(list2))
list3 += [None] * (max_length - len(list3))

# Combine lists into rows
rows = zip(list1, list2, list3)

# Write rows to CSV file
with open('output.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Column1', 'Column2', 'Column3'])  # Writing headers (optional)
    writer.writerows(rows)
