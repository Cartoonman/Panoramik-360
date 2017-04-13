import boto3
import os




def get_images():
    s3 = boto3.client('s3')
    #s3.download_file(os.environ.get("S3_BUCKET"), '360_stream/data', 'data')
    for x in filter(lambda x: "360_stream/data" in x['Key'], s3.list_objects(Bucket = os.environ.get("S3_BUCKET"))['Contents']):
        s3.download_file(os.environ.get("S3_BUCKET"), x['Key'], 'data/' + x['Key'].split('/')[2])

    
    
    
if __name__ == '__main__':
    get_images()
