from Bio import Entrez

Entrez.email = "kevin.yar@outlook.com" # Always provide your email to NCBI

# Search for a GEO dataset (e.g., GSE ID)
handle = Entrez.esearch(db="gds", term="GSE126030")
record = Entrez.read(handle)
handle.close()
print(f"Found GEO IDs: {record['IdList']}")

handle = Entrez.efetch(db="geoprofiles", id='200126030', retmode="xml")
record = Entrez.read(handle)
record

# Fetch the data for a specific GEO ID
if record['IdList']:
    geo_id = record['IdList'][0]
    handle = Entrez.efetch(db="geo", id=geo_id, rettype="soft", retmode="text")
    geo_data = handle.read()
    handle.close()
    print(geo_data[:500]) # Print first 500 characters of the fetched data


"""
summary failed:
NCBI dodesn provide this option 
"""