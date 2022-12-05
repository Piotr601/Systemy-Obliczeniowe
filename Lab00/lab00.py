from pkgutil import iter_modules
from threading import Thread
import urllib.request
from faker import Faker

MAX_RANGE = 4
PATH = 'folder/'

fake = Faker()
url_list = []

for _ in range(MAX_RANGE):
   url_list.append('http://python.org/')

# Function to send url request
def downloading(url_path, name):
    print(f"Downloading filme from: {url_path}")
    urllib.request.urlretrieve(url_path, name)

# Function to write file in direction, each per new thread
def writing(names, direct):
    threads = []
    for nmb, item in enumerate(names):
        file_name = direct + str(nmb) + ".xml"
        t = Thread(target=downloading, args = (url_list[nmb], file_name))
        threads.append(t)
        t.start()
    return threads

# Main function - loop
if __name__ == '__main__':
    k = writing(url_list, PATH)
    for pos, t in enumerate(k):
        t.join()
        print(f"Running {pos} thread")