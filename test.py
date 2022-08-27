
# from google.cloud import storage
# import os


# bucket_name = 'euphoric-point-358206-kubeflowpipelines-default'
# prefix = 'data/'
# dl_dir = './data/'
# os.mkdir(dl_dir)

# storage_client = storage.Client()
# print(bucketstorage_client)
# bucket = storage_client.bucket(bucket_name=bucket_name)
# print(bucket)
# blobs = bucket.list_blobs(prefix=prefix)  # Get list of files
# print(blobs)
# for blob in blobs:
#     print(blob)
#     filename = blob.name.replace(prefix, '') 
#     blob.download_to_filename(dl_dir + filename)  # Download

import dill
print(dill.__version__)