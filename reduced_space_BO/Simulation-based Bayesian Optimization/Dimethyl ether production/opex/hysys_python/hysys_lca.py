

def substring_exist(search_str, option):
    all_found = True
    if search_str not in option.lower():
        all_found = False

    return all_found


def db_search(search, list, limit=10, print_output=False):
    # Recursively shrinks the list size
    options = list
    if type(search) == str:
        search_str = search.lower()
        search_strs = [x for x in search_str.split(' ')]  

        for s in search_strs:
            options = [m for m in options if substring_exist(s, str(m))]

    elif type(search) == dict:
        for key in search.keys():
            search_str = search[key].lower()
            search_strs = [x for x in search_str.split(' ')]  

            for s in search_strs:
                options = [m for m in options if substring_exist(s, str(m[key]))]
                
    else:
        raise ValueError("Search must be either a string or a dict")
    
    if print_output:
        print("Found {} options:".format(len(options)))
        for i, option in enumerate(options):
            if i <= limit:
                print(i, option)

    return options