from bs4 import BeautifulSoup
import os
import time
import pickle
import tarfile
import multiprocessing
from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_boolean('extract', True, "Extract the zipped")  # set to False if extraction not required

def extract_zips():
    # unzips the compressed files in same directory
    for zipfile in os.listdir():
        if '.tgz' in zipfile:  # considers only zip extensions
            print(f'Extracting: {zipfile}')
            path = os.path.join(os.getcwd(), zipfile)
            tar = tarfile.open(path, 'r:gz')
            tar.extractall()
            tar.close()

def parser(xml: str):
    '''
    Reads a XML document and extracts the textual content from abstract tag of each patent
    :param xml: XML document filename
    :return: XML document filename, text content
    '''

    xml_pth = os.path.join(os.getcwd(), xml)
    with open(xml_pth, 'r') as f:
        data = f.read()
    # parse the xml content
    soup = BeautifulSoup(data, "html.parser")
    b_unique = soup.find_all('abstract')
    # extract all <p> tags
    content = []
    for data in b_unique:
        content.append(data.text) # get the textual content
    content = ''.join(content)  # convert into a string object
    return xml, content

def main(argv):

    root_dir = os.path.join(os.getcwd(), '../data')
    os.chdir(os.path.join(root_dir, 'patents'))
    if FLAGS.extract:
        extract_zips()

    directories = [f for f in list(os.listdir()) if '.' not in f]  # considers directories unzipped

    # make a new 'raw' directory if does not exists
    raw_dir = os.path.join(root_dir, 'raw')
    if not os.path.isdir(raw_dir):
        os.makedirs(raw_dir)

    for dir in directories:  # iterate on each directory
        os.chdir(os.path.join(root_dir, 'patents', dir))
        print(f'Parsing {dir} ...')
        start = time.time()
        xml_list = os.listdir()
        # utilize all the cores for faster processing
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        data = pool.map(parser, xml_list)
        pool.close()
        pool.join()
        print('Time:', time.time() - start)

        # dictionary data structure saves memory
        my_dict = {doc[0]: doc[1] for doc in data}
        with open(os.path.join(raw_dir, str(dir)+'.json'), 'wb') as fp:  # pickle as binary object
            pickle.dump(my_dict, fp)
        print(f'Saved parsed {dir} successfully')

if __name__=='__main__':
    app.run(main)