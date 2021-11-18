def binary_search(iter_list,centroid_x,centroid_y):
    lower_bound=1
    upper_bound=len(iter_list)
    
    for x in iter_list:
        median=lower_bound + ( upper_bound - lower_bound ) / 2
        if iter_list[median] == centroid_x:
            return True
        
