import tensorflow as tf
import os
import base64
from google.protobuf.json_format import MessageToJson
import json
import wget
import pandas as pd
import requests
import shutil

convert_res = []

directory = "./frame_feature"
for filename in os.listdir(directory):
    if filename.startswith("validate"):
        raw_dataset = tf.data.TFRecordDataset(directory+'/'+filename)

        for raw_record in raw_dataset.take(1000):
            example = tf.train.Example()
            # If you want to print feature lists, use example = tf.train.SequenceExample()
            example.ParseFromString(raw_record.numpy())

            examplejson = json.loads(MessageToJson(example))
            videoid = examplejson['features']['feature']['id']['bytesList']['value'][0]
            videoid = base64.b64decode(videoid).decode("utf-8")

            url = "http://data.yt8m.org/2/j/i/" + videoid[0:2] + '/' + videoid + ".js"
            # wget.download(url, out="./videoid_convert/")
            r = requests.get(url, stream=True)
            if r.status_code == 200:
                with open("./videoid_convert/"+videoid+".js", 'wb') as f:
                    r.raw.decode_content = True
                    shutil.copyfileobj(r.raw, f)

                with open("./videoid_convert/"+videoid+".js", 'r') as f:
                    lines = f.readlines()
                    idconvert = lines[0][2:-2].split(",")
                    idconvert[0] = idconvert[0][1:-1]
                    idconvert[1] = idconvert[1][1:-1]
                    convert_res.append(idconvert)

df = pd.DataFrame(convert_res, columns=['videoid', 'youtubeid'])
df.to_csv("videoid_convert.csv", index=None)