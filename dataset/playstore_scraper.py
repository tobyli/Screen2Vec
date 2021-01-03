import play_scraper
from os import path
DEFAULT_APP_DESCRIPTION_PATH = './PlayStoreDescriptions/'

# handles querying for app description

def get_app_description(packageName):
    try:
        return play_scraper.details(packageName)['description']
    except ValueError as ve:
        try:
            potential_app = play_scraper.search(packageName, detailed=True)
            return potential_app['description']
        except Exception as e:
            return ''
    except Exception as e:
        return ''

# scrape the playstore description for packageName and store it in path as a txt file
def update_app_description_file(packageName, dir_path):
    if (path.exists(dir_path + "unrecognized.text")):
        unrecognizedPackageNamesFile = open(dir_path + "unrecognized.text", "r")
        unrecognizedPackageNames = unrecognizedPackageNamesFile.readlines()
        if (packageName + '\n') in unrecognizedPackageNames:
            return None
    try:
        description = play_scraper.details(packageName)['description']
    except ValueError as e:
        if '404 Client Error: Not Found for url' in str(e):
            with open(dir_path + "unrecognized.text", "a") as f:
                f.write(packageName + '\n')
                f.close()
        raise e
    with open(dir_path + packageName + ".txt", "w") as f:
        f.write(description)
        f.close()
    return description

def update_app_description_file_in_batch(packageNameList, dir_path):
    for packageName in packageNameList:
        update_app_description_file(packageName, dir_path)


