import xml.etree.cElementTree as ET
from collections import defaultdict
import pprint
import re
#regular expression for identifying strings with problematic characters
problemchars = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')
#Created a mapping dictionary to replace inconsistent street names
mapping = { "Ave" : "Avenue",
"Avenuen" : "Avenue",
"Saint" : "St",
"Boadway" : "Broadway",
"HIll" : "Hill",
"By-pass" : "Bypass",
"Center" : "Centre",
"Cresent" : "Crescent",
"Eastways" : "Eastway",
"Field" : "Fields",
"Garden" : "Gardens",
"Garen" : "Gardens",
"James" : "James'",
"James's" : "James'",
"Parage" : "Parade",
"Park," : "Park",
"Place?" : "Place",
"ROAD," : "Rd",
"ROAD" : "Rd",
"Road--" : "Rd",
"Road3" : "Rd",
"Road" : "Rd",
"Rpad" : "Rd",
"STREET" : "St",
"St." : "St",
"Steet" : "St",
"STABLE" : "Stable",
"VIEW" : "View",
"ave" : "Avenue",
"by-pass" : "Bypass",
 }
#Function to solve Capitalization problem
def capitalize(Text):
    return ' '.join(word[0].upper() + word[1:] for word in Text.split())
#Function to lookup proper street names from the mapping dictionary defined above            
def update_name(name, mapping):
     name_arr=name.split(' ')
     for idx,item in enumerate(name_arr):
         if item in mapping:
             name_arr[idx]=mapping[item]
     return ' '.join(name_arr)
#A separate function to handle all the address cleaning which calls all the cleaning functions defined above
def cleanse_address(value):
    if 'false>>' in value:
        value= value.split("'")[1]
    #cleansing improper street names
    value=update_name(value,mapping)
    #capitalizing
    clean_value=capitalize(value)
    return clean_value
#Fuction for transforming the data according to our datamodel for inserting it into MongoDb
CREATED = [ "version", "changeset", "timestamp", "user", "uid"]
def shape_element(element):
    node = {}
    created={}
    pos=[]
    #processing nodes and ways
    if element.tag == "node" or element.tag == "way" :
        node['id']=element.attrib['id']
        node['type']=element.tag
        #visible attribute is not present in all the elements
        if 'visible' in element.attrib:
            node['visible']=element.attrib['visible']
        for item in CREATED:
            if item in element.attrib:
                created[item]=element.attrib[item]
        node['created']=created
        pos=[]
        if 'lat' in element.attrib:
            lat=float(element.attrib['lat'])
            pos.append(lat)
        if 'lon' in element.attrib:
            lon=float(element.attrib['lon'])
            pos.append(lon)
        if len(pos)!=0:
            node['pos']=pos
        address={}
        for item in element.iter('tag'):
            
            if 'k' in item.attrib:
                key_list=item.attrib['k'].split(':')
                
                if key_list[0]=='addr':
                    #cleaning address
                    clean_value=cleanse_address(item.attrib['v'])
                    #ignoring second and further levels of address
                    if len(key_list)>2:
                        continue
                    #ignoring problematic characters
                    elif problemchars.match(item.attrib['k']):
                        continue
                    else:
                        address[key_list[1]]=clean_value
                else:
                    #tags with : other than address
                    #eliminating problematic keys
                        if '.' not in item.attrib['k']:
                    node[item.attrib['k']]=item.attrib['v']
                #testing for empty dictionary
                if bool(address):
                    node['address']=address
        #adding node references for ways        
        if element.tag=='way':
            node_refs=[]
            for item in element.iter('nd'):
                node_refs.append(item.attrib['ref'])
            node['node_refs']=node_refs
        return node
    else:
        return None
#connecting to MongoDB
def get_db():
    from pymongo import MongoClient
    client = MongoClient('localhost:27017')
    db=client['examples']
    return db
def get_element(osm_file, tags=('node', 'way', 'relation')):
    """Yield element if it is the right type of tag
    Reference:
    http://stackoverflow.com/questions/3095434/inserting-newlines-in-xml-file-generated-via-xml-etree-elementtree-in-python
    """
    context = ET.iterparse(osm_file, events=('start', 'end'))
    _, root = next(context)
    for event, elem in context:
        if event == 'end' and elem.tag in tags:
            yield elem
            root.clear()

def test():
    #path where the file is stored
    path="E:\udacity\MongoDB_project"

    SAMPLE_FILE = path+"\london_england.osm"
    db=get_db()
    #parsing the file
    for i, elem in enumerate(get_element(SAMPLE_FILE)):
            node=shape_element(elem)
            #insert in Mongodb
            if node is not None:
                db['openstreetmap'].insert(node)
            elem.clear()
            node={}
            
if __name__ == '__main__':
    test()     
        
