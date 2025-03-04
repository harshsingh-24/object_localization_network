# For explanation of this code, you can visit this video: https://www.youtube.com/watch?v=8AyKJxBxx0M&t=172s
import os
import requests # pip install requests #to sent GET requests
from bs4 import BeautifulSoup # pip install bs4 #to parse html(getting data out from html, xml or other markup languages)
from argparse import ArgumentParser # Command Line Argument Parsing  

# user can input a search keyword and the count of images required
# download images from google search image
Google_Image = \
    'https://www.google.com/search?site=&tbm=isch&source=hp&biw=1873&bih=990&'

# The User-Agent request header contains a characteristic string 
# that allows the network protocol peers to identify the application type, 
# operating system, and software version of the requesting software user agent.
# needed for google search
u_agnt = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
    'Accept-Encoding': 'none',
    'Accept-Language': 'en-US,en;q=0.8',
    'Connection': 'keep-alive',
} #write: 'my user agent' in browser to get your browser user agent details

# Image_Folder = 'data/custom/google_search_images'

def main():
    parser = ArgumentParser()
    parser.add_argument('keyword', help='Search Keyword')
    parser.add_argument('count', type=int, help='Number of Images you want')
    parser.add_argument('--outputdir', type=str, default='data/custom/google_search_images', help='Output Directory for Image to be downloaded')
    args = parser.parse_args()

    if not os.path.exists(args.outputdir):
        os.mkdir(args.outputdir)
    
    print('Searching Images....')
    
    search_url = Google_Image + 'q=' + args.keyword #'q=' because its a query
    
    # request url, without u_agnt the permission gets denied
    response = requests.get(search_url, headers=u_agnt)
    html = response.text #To get actual result i.e. to read the html data in text mode
    
    # find all img where class='rg_i Q4LuWd'
    b_soup = BeautifulSoup(html, 'html.parser') #html.parser is used to parse/extract features from HTML files
    results = b_soup.findAll('img', {'class': 'rg_i Q4LuWd'})
    
    #extract the links of requested number of images with 'data-src' attribute and appended those links to a list 'imagelinks'
    #allow to continue the loop in case query fails for non-data-src attributes
    count = 0
    imagelinks= []
    for res in results:
        try:
            link = res['data-src']
            imagelinks.append(link)
            count = count + 1
            if (count >= args.count):
                break
            
        except KeyError:
            continue
    
    print(f'Found {len(imagelinks)} images')
    print('Start downloading...')

    for i, imagelink in enumerate(imagelinks):
        # open each image link and save the file
        response = requests.get(imagelink)
        
        imagename = args.outputdir + '/' + args.keyword + str(i+1) + '.jpg'
        with open(imagename, 'wb') as file:
            file.write(response.content)

    print('Download Completed!')
    

if __name__ == '__main__':
    main()