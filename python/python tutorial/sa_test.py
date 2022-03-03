from azure.storage.blob import BlobServiceClient

#connect_str = 'https://dlstorageaccountsyn.blob.core.windows.net/blob-ohe?sv=2020-02-10&st=2022-02-28T18%3A10%3A01Z&se=2022-03-05T18%3A10%3A00Z&sr=c&sp=racwlme&sig=88rUz%2FbXooL%2FwXHrNSGXlyWouPJBl4pcpv1X%2BSaRGWE%3D'
connect_str = 'DefaultEndpointsProtocol=https;AccountName=dlstorageaccountsyn;AccountKey=a9+j2RFB9KPe9n80lyxFfAUHZnvSAE8wUa2wVG42lY2IjafiRb5UyFwTCJZFjhaxlNev/3CzT0yD2PDywJqtjw==;EndpointSuffix=core.windows.net'
blob_service_client = BlobServiceClient.from_connection_string(connect_str)
container_client=blob_service_client.get_container_client('blob-ohe')
blobs = container_client.list_blobs( )
for blob in blobs:
    print(blob.name)
    doc = open(blob.name)
    for word in doc:
        print(word)